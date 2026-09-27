"""Tests for the forked scan compute pool.

These start real worker processes, so they exercise pickling of calls and
outcomes, input ordering, and recovery when a worker dies.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from app.domain.scanning.ports import (
    SerialStockScanBatchRunner,
    StockScanCall,
)
from app.infra.tasks import scan_compute_pool
from app.infra.tasks.scan_compute_pool import (
    ProcessStockScanBatchRunner,
    process_stock_scan_batch_runner,
)
from app.use_cases.scanning.run_bulk_scan import (
    RunBulkScanCommand,
    RunBulkScanUseCase,
)
from tests.unit.use_cases.conftest import (
    FakeCancellationToken,
    FakeProgressSink,
    FakeScan,
    FakeScanRepository,
    FakeScanResultRepository,
    FakeStockDataProvider,
    FakeUnitOfWork,
    with_test_opportunity_projection,
)

_PARENT_PID = os.getpid()


class _Unpicklable(Exception):
    def __init__(self):
        super().__init__("cannot cross processes")
        self.lock = threading.Lock()


class _PidScanner:
    """Reports which process computed each symbol."""

    def scan_stock_multi(self, symbol, **kwargs):
        if symbol == "RAISE":
            raise ValueError("bad symbol")
        if symbol == "UNPICKLABLE":
            raise _Unpicklable()
        if symbol == "DIE" and os.getpid() != _PARENT_PID:
            os._exit(1)
        # Enough work per call that both workers pick up batches.
        time.sleep(0.01)
        return {"symbol": symbol, "pid": os.getpid(), "kwargs": sorted(kwargs)}


def _calls(*symbols):
    return [StockScanCall(symbol=s, kwargs={"screener_names": ["minervini"]}) for s in symbols]


def test_outcomes_are_computed_in_worker_processes_in_input_order():
    symbols = [f"S{i}" for i in range(40)]

    with ProcessStockScanBatchRunner(_PidScanner(), 2) as runner:
        outcomes = runner.scan_batch(_calls(*symbols))

    assert [outcome.result["symbol"] for outcome in outcomes] == symbols
    pids = {outcome.result["pid"] for outcome in outcomes}
    assert _PARENT_PID not in pids
    assert len(pids) == 2
    assert outcomes[0].result["kwargs"] == ["screener_names"]


def test_worker_exceptions_come_back_as_outcomes():
    with ProcessStockScanBatchRunner(_PidScanner(), 2) as runner:
        outcomes = runner.scan_batch(_calls("A", "RAISE", "UNPICKLABLE", "B"))

    assert outcomes[0].result["symbol"] == "A"
    assert isinstance(outcomes[1].error, ValueError)
    assert str(outcomes[1].error) == "bad symbol"
    assert isinstance(outcomes[2].error, RuntimeError)
    assert str(outcomes[2].error) == "_Unpicklable: cannot cross processes"
    assert outcomes[3].result["symbol"] == "B"


def test_a_dead_worker_degrades_to_in_process_without_losing_results():
    symbols = [f"S{i}" for i in range(12)] + ["DIE"] + [f"T{i}" for i in range(12)]

    with ProcessStockScanBatchRunner(_PidScanner(), 2) as runner:
        outcomes = runner.scan_batch(_calls(*symbols))
        after = runner.scan_batch(_calls("X", "Y"))

    assert [outcome.result["symbol"] for outcome in outcomes] == symbols
    assert all(outcome.error is None for outcome in outcomes)
    # DIE only dies inside a worker, so the parent computed it after the break.
    assert outcomes[symbols.index("DIE")].result["pid"] == _PARENT_PID
    assert [outcome.result["pid"] for outcome in after] == [_PARENT_PID, _PARENT_PID]


def test_pool_start_failure_computes_in_process(monkeypatch):
    def broken_context():
        raise OSError("fork unavailable")

    monkeypatch.setattr(scan_compute_pool, "_default_mp_context", broken_context)

    with ProcessStockScanBatchRunner(_PidScanner(), 2) as runner:
        outcomes = runner.scan_batch(_calls("A", "B"))

    assert [outcome.result["pid"] for outcome in outcomes] == [_PARENT_PID, _PARENT_PID]


def test_runner_releases_its_scanner_registration():
    with ProcessStockScanBatchRunner(_PidScanner(), 2) as runner:
        runner.scan_batch(_calls("A", "B"))
        assert len(scan_compute_pool._scanners_by_token) == 1

    assert scan_compute_pool._scanners_by_token == {}


@pytest.mark.parametrize("processes", [0, 1])
def test_factory_computes_in_process_for_a_single_process(processes):
    assert isinstance(
        process_stock_scan_batch_runner(_PidScanner(), processes),
        SerialStockScanBatchRunner,
    )


def test_runner_rejects_fewer_than_two_processes():
    with pytest.raises(ValueError, match="processes must be >= 2"):
        ProcessStockScanBatchRunner(_PidScanner(), 1)


class _StockDataScanner:
    """Needs the real prefetched StockData to arrive intact in the worker."""

    def get_merged_requirements(self, screener_names, criteria=None):
        return {"needs": "price+fundamentals"}

    def scan_stock_multi(self, symbol, pre_fetched_data=None, **_kwargs):
        closes = pre_fetched_data.price_data["Close"]
        return with_test_opportunity_projection(
            {
                "composite_score": 70.0,
                "rating": "Buy",
                "passes_template": symbol != "MSFT",
                "current_price": float(closes.iloc[-1]),
                "bars": len(closes),
                "computed_in_worker": os.getpid() != _PARENT_PID,
            }
        )


def test_bulk_scan_end_to_end_on_the_process_pool():
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
    scan_repo = FakeScanRepository()
    scan_repo.scans["s1"] = FakeScan(scan_id="s1", screener_types=["minervini"])
    result_repo = FakeScanResultRepository()
    progress = FakeProgressSink()

    result = RunBulkScanUseCase(
        scanner=_StockDataScanner(),
        data_provider=FakeStockDataProvider(price_days=300),
        scan_batch_runner_factory=process_stock_scan_batch_runner,
    ).execute(
        FakeUnitOfWork(scans=scan_repo, scan_results=result_repo),
        RunBulkScanCommand(
            scan_id="s1",
            symbols=symbols,
            chunk_size=3,
            cache_only=True,
            parallel_workers=2,
        ),
        progress,
        FakeCancellationToken(),
    )

    assert result.status == "completed"
    assert result.total_scanned == 5
    assert result.passed == 4
    persisted = result_repo._persisted_results
    assert [symbol for _scan_id, symbol, _result in persisted] == symbols
    assert all(row["computed_in_worker"] for _scan_id, _symbol, row in persisted)
    assert all(row["bars"] > 200 for _scan_id, _symbol, row in persisted)
    assert [(event.current, event.passed) for event in progress.events] == [(3, 2), (5, 4)]
