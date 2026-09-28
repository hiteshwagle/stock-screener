"""Process pool that computes scan results on several cores.

Scanning is CPU-bound Python, so threads serialize on the GIL; separate
processes are the only way one scan uses more than one core.

The parent keeps every stateful step — bulk prefetch, Market RS pinning,
cancellation, persistence and progress — and ships only
``StockScanner.scan_stock_multi`` calls to the workers, so a scan means the
same thing however many processes compute it.

Workers are forked through billiard (Celery's multiprocessing fork): Celery
prefork children are daemonic, and the standard library refuses to let a
daemonic process start children. Forking also hands each worker the parent's
fully wired scanner without rebuilding runtime services.
"""

from __future__ import annotations

import itertools
import logging
import math
import pickle
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Sequence

from app.domain.scanning.ports import (
    SerialStockScanBatchRunner,
    StockScanBatchRunner,
    StockScanCall,
    StockScanner,
    StockScanOutcome,
    run_stock_scan_call,
)

logger = logging.getLogger(__name__)

# Batches submitted per worker for each scan_batch() call. Several small
# batches per worker let fast workers pick up slack when some symbols (long
# histories, many pattern candidates) take much longer than others.
_BATCHES_PER_PROCESS = 4

# Scanners registered by the parent before forking, keyed by runner token.
_scanners_by_token: dict[int, StockScanner] = {}
_tokens = itertools.count(1)

# The scanner a forked worker computes with, set by _init_worker.
_worker_scanner: StockScanner | None = None


def _default_mp_context():
    import billiard

    return billiard.get_context("fork")


def _exit_with_parent() -> None:
    """Ask Linux to kill this worker if its parent dies (e.g. a hard time limit)."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        pr_set_pdeathsig = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            pr_set_pdeathsig, signal.SIGKILL
        )
    except (OSError, AttributeError):
        pass


def _init_worker(token: int) -> None:
    global _worker_scanner
    _exit_with_parent()
    # The parent owns shutdown: Celery's inherited handlers must not run here,
    # and Ctrl-C in a dev shell should stop the parent, which stops the pool.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from app.database import engine

        # Drop inherited pooled connections without closing the parent's sockets.
        engine.dispose(close=False)
    except Exception:
        logger.debug("Scan compute worker could not reset the DB pool", exc_info=True)
    _worker_scanner = _scanners_by_token[token]


def _warm_up() -> None:
    return None


def _picklable(outcome: StockScanOutcome) -> StockScanOutcome:
    if outcome.error is None:
        return outcome
    try:
        pickle.dumps(outcome.error)
    except Exception:
        return StockScanOutcome(
            error=RuntimeError(f"{type(outcome.error).__name__}: {outcome.error}")
        )
    return outcome


def _scan_calls(calls: Sequence[StockScanCall]) -> list[StockScanOutcome]:
    return [_picklable(run_stock_scan_call(_worker_scanner, call)) for call in calls]


def _split(calls: Sequence[StockScanCall], parts: int) -> list[Sequence[StockScanCall]]:
    size = max(1, math.ceil(len(calls) / parts))
    return [calls[start:start + size] for start in range(0, len(calls), size)]


class ProcessStockScanBatchRunner:
    """Computes scan calls on ``processes`` forked workers, in input order.

    Any pool failure degrades to in-process computation instead of failing
    the scan: a worker that dies breaks the pool, and every batch not yet
    collected is recomputed in the parent. Scan calls are pure, so
    recomputing one is safe.
    """

    def __init__(
        self,
        scanner: StockScanner,
        processes: int,
        *,
        mp_context=None,
    ) -> None:
        if processes < 2:
            raise ValueError("processes must be >= 2")
        self._scanner = scanner
        self._processes = processes
        self._mp_context = mp_context
        self._serial = SerialStockScanBatchRunner(scanner)
        self._executor: ProcessPoolExecutor | None = None
        self._token: int | None = None

    def __enter__(self) -> "ProcessStockScanBatchRunner":
        self._token = next(_tokens)
        _scanners_by_token[self._token] = self._scanner
        try:
            self._executor = ProcessPoolExecutor(
                max_workers=self._processes,
                mp_context=self._mp_context or _default_mp_context(),
                initializer=_init_worker,
                initargs=(self._token,),
            )
            # Fork every worker now, before the parent holds any chunk data.
            self._executor.submit(_warm_up).result()
            logger.info("Scan compute pool started with %d processes", self._processes)
        except Exception:
            logger.warning(
                "Scan compute pool failed to start; computing in-process",
                exc_info=True,
            )
            self._shutdown()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._shutdown()
        if self._token is not None:
            _scanners_by_token.pop(self._token, None)
            self._token = None

    def scan_batch(self, calls: Sequence[StockScanCall]) -> list[StockScanOutcome]:
        if self._executor is None or len(calls) < 2:
            return self._serial.scan_batch(calls)

        batches = _split(calls, self._processes * _BATCHES_PER_PROCESS)
        futures = [self._executor.submit(_scan_calls, batch) for batch in batches]
        outcomes: list[StockScanOutcome] = []
        for batch, future in zip(batches, futures):
            try:
                outcomes.extend(future.result())
            except BrokenProcessPool:
                if self._executor is not None:
                    logger.warning(
                        "Scan compute worker died; finishing this scan in-process",
                        exc_info=True,
                    )
                    self._shutdown()
                outcomes.extend(self._serial.scan_batch(batch))
            except Exception as exc:
                # Raised in the parent while waiting (e.g. Celery's soft time
                # limit), not by the worker: propagate it.
                if not future.done() or (
                    not future.cancelled() and future.exception() is not exc
                ):
                    raise
                logger.warning(
                    "Scan compute batch failed in a worker; recomputing it in-process",
                    exc_info=True,
                )
                outcomes.extend(self._serial.scan_batch(batch))
        return outcomes

    def _shutdown(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def process_stock_scan_batch_runner(
    scanner: StockScanner,
    processes: int,
) -> StockScanBatchRunner:
    """Production :data:`StockScanBatchRunnerFactory`."""
    if processes <= 1:
        return SerialStockScanBatchRunner(scanner)
    return ProcessStockScanBatchRunner(scanner, processes)
