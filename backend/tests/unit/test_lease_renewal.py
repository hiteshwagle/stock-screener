"""Heartbeat renewal keeps short workload leases alive only while the holder runs."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.tasks.lease_renewal import keep_leases_alive, lease_ttl_seconds


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_renewal_interval_stays_ahead_of_even_a_one_second_lease(monkeypatch):
    from app.tasks import lease_renewal

    monkeypatch.setattr(lease_renewal.settings, "data_fetch_lock_timeout", 1)
    assert lease_renewal.lease_renew_interval_seconds() < 1
    monkeypatch.setattr(lease_renewal.settings, "data_fetch_lock_timeout", 300)
    assert lease_renewal.lease_renew_interval_seconds() == 100


def test_lease_ttl_falls_back_for_missing_or_invalid_settings():
    assert lease_ttl_seconds(120) == 120
    assert lease_ttl_seconds(0) == 300
    assert lease_ttl_seconds(True) == 300
    assert lease_ttl_seconds(MagicMock()) == 300


def test_leases_are_renewed_while_running_and_not_after_exit():
    renew = MagicMock(return_value=True)

    with keep_leases_alive([("market_workload:us", renew)], interval_seconds=0.01):
        assert _wait_for(lambda: renew.call_count >= 3)

    calls_at_exit = renew.call_count
    time.sleep(0.05)
    assert renew.call_count == calls_at_exit


def test_leases_are_renewed_once_before_the_body_starts():
    # A lease inherited by a retried task (same id) may have less than one
    # interval left, so the first renewal cannot wait for the first beat.
    renew = MagicMock(return_value=True)

    with keep_leases_alive([("market_workload:us", renew)], interval_seconds=60):
        assert renew.call_count == 1


def test_a_lease_lost_while_running_stops_renewing_while_others_continue():
    lost = MagicMock(side_effect=[True, False])
    held = MagicMock(return_value=True)

    with keep_leases_alive(
        [("lost", lost), ("held", held)], interval_seconds=0.01
    ):
        assert _wait_for(lambda: held.call_count >= 4)

    assert lost.call_count == 2


def test_a_lease_already_lost_on_entry_prevents_the_body_from_starting():
    from app.tasks.lease_renewal import LeaseNotHeld

    body_ran = False
    with pytest.raises(LeaseNotHeld, match="market_workload:us"):
        with keep_leases_alive(
            [
                ("market_workload:us", MagicMock(return_value=False)),
                ("external_fetch_global", MagicMock(return_value=True)),
            ],
            interval_seconds=60,
        ):
            body_ran = True

    assert body_ran is False


@patch("app.wiring.bootstrap.get_workload_coordination")
def test_market_workload_retries_instead_of_running_when_lease_lost_on_entry(
    mock_get_coordination,
):
    from celery.exceptions import Retry

    from app.tasks.workload_coordination import serialized_market_workload

    coordination = MagicMock()
    # The same-id lease looked reentrant, but expired and was taken before
    # the first renewal.
    coordination.acquire_market_workload.return_value = (True, True)
    coordination.renew_market_workload.return_value = False
    coordination.get_market_workload_holder.return_value = {
        "task_name": "calculate_daily_group_rankings_with_gapfill",
        "task_id": "other-task",
    }
    mock_get_coordination.return_value = coordination
    retries = []

    def _retry(*, exc=None, countdown=None, max_retries=None):
        retries.append(str(exc))
        raise Retry(message=str(exc))

    task = SimpleNamespace(request=SimpleNamespace(id="task-1", retries=0), retry=_retry)
    body = MagicMock()

    @serialized_market_workload("calculate_daily_breadth_with_gapfill")
    def run(self, market=None):
        body()

    with pytest.raises(Retry):
        run(task, market="US")

    body.assert_not_called()
    assert "waiting_for_market_workload:US" in retries[0]


@patch("app.wiring.bootstrap.get_workload_coordination")
@patch("app.wiring.bootstrap.get_data_fetch_lock")
def test_data_fetch_retries_instead_of_running_when_lease_lost_on_entry(
    mock_get_lock, mock_get_coordination
):
    from celery.exceptions import Retry

    from app.tasks.data_fetch_lock import _serialized_data_fetch

    lock = MagicMock()
    lock.acquire.return_value = (True, True)
    lock.renew.return_value = False
    coordination = MagicMock()
    coordination.acquire_market_workload.return_value = (True, True)
    coordination.acquire_external_fetch.return_value = (True, True)
    mock_get_lock.return_value = lock
    mock_get_coordination.return_value = coordination
    retries = []

    def _retry(*, exc=None, countdown=None, max_retries=None):
        retries.append(str(exc))
        raise Retry(message=str(exc))

    task = SimpleNamespace(request=SimpleNamespace(id="task-9", retries=0), retry=_retry)
    body = MagicMock()

    @_serialized_data_fetch("refresh_cot")
    def run(self, market=None):
        body()

    with pytest.raises(Retry):
        run(task, market="US")

    body.assert_not_called()
    assert "lease_lost_before_start" in retries[0]


def test_renewal_errors_are_retried_on_the_next_beat():
    outcomes = iter([ConnectionError("redis blip"), True, True])
    renewed = threading.Event()

    def renew():
        outcome = next(outcomes, True)
        if isinstance(outcome, Exception):
            raise outcome
        renewed.set()
        return outcome

    with keep_leases_alive([("flaky", renew)], interval_seconds=0.01):
        assert renewed.wait(2.0)


def test_no_renewals_starts_no_thread():
    before = threading.active_count()
    with keep_leases_alive([]):
        assert threading.active_count() == before


@patch("app.tasks.lease_renewal.lease_renew_interval_seconds", return_value=0.01)
@patch("app.wiring.bootstrap.get_workload_coordination")
def test_serialized_market_workload_renews_its_lease_while_the_body_runs(
    mock_get_coordination, _interval
):
    from app.tasks.workload_coordination import serialized_market_workload

    coordination = MagicMock()
    coordination.acquire_market_workload.return_value = (True, False)
    coordination.renew_market_workload.return_value = True
    mock_get_coordination.return_value = coordination
    task = SimpleNamespace(request=SimpleNamespace(id="task-1", retries=0))

    @serialized_market_workload("calculate_daily_breadth_with_gapfill")
    def body(self, market=None):
        assert _wait_for(lambda: coordination.renew_market_workload.call_count >= 2)
        return "done"

    assert body(task, market="US") == "done"
    coordination.renew_market_workload.assert_called_with("task-1", market="US")
    coordination.release_market_workload.assert_called_once_with(
        "task-1", market="US"
    )


@patch("app.tasks.lease_renewal.lease_renew_interval_seconds", return_value=0.01)
@patch("app.wiring.bootstrap.get_workload_coordination")
def test_reentrant_market_workload_lease_is_still_renewed(
    mock_get_coordination, _interval
):
    # A Celery retry or redelivery reuses the task id, so a "reentrant" lease
    # can be a leftover from an earlier attempt that nothing else renews.
    from app.tasks.workload_coordination import serialized_market_workload

    coordination = MagicMock()
    coordination.acquire_market_workload.return_value = (True, True)
    coordination.renew_market_workload.return_value = True
    mock_get_coordination.return_value = coordination
    task = SimpleNamespace(request=SimpleNamespace(id="task-1", retries=0))

    @serialized_market_workload("calculate_market_exposure")
    def body(self, market=None):
        assert _wait_for(lambda: coordination.renew_market_workload.call_count >= 2)
        return "done"

    assert body(task, market="US") == "done"
    coordination.renew_market_workload.assert_called_with("task-1", market="US")
    coordination.release_market_workload.assert_not_called()


@patch("app.tasks.lease_renewal.lease_renew_interval_seconds", return_value=0.01)
@patch("app.wiring.bootstrap.get_workload_coordination")
@patch("app.wiring.bootstrap.get_data_fetch_lock")
def test_serialized_data_fetch_renews_all_three_leases(
    mock_get_lock, mock_get_coordination, _interval
):
    from app.tasks.data_fetch_lock import _serialized_data_fetch

    lock = MagicMock()
    lock.acquire.return_value = (True, False)
    lock.renew.return_value = True
    coordination = MagicMock()
    coordination.acquire_market_workload.return_value = (True, False)
    coordination.acquire_external_fetch.return_value = (True, False)
    coordination.renew_market_workload.return_value = True
    coordination.renew_external_fetch.return_value = True
    mock_get_lock.return_value = lock
    mock_get_coordination.return_value = coordination
    task = SimpleNamespace(request=SimpleNamespace(id="task-9", retries=0))

    @_serialized_data_fetch("refresh_cot")
    def body(self, market=None):
        assert _wait_for(
            lambda: lock.renew.call_count >= 2
            and coordination.renew_market_workload.call_count >= 2
            and coordination.renew_external_fetch.call_count >= 2
        )
        return "done"

    assert body(task, market="US") == "done"
    lock.renew.assert_called_with("task-9", market="US")
    coordination.renew_external_fetch.assert_called_with("task-9")
    lock.release.assert_called_once_with("task-9", market="US")


@pytest.mark.parametrize(
    ("module_path", "class_name"),
    [
        ("app.tasks.data_fetch_lock", "DataFetchLock"),
        ("app.tasks.workload_coordination", "WorkloadCoordination"),
    ],
)
def test_lease_redis_clients_have_bounded_timeouts(module_path, class_name):
    import importlib

    module = importlib.import_module(module_path)
    with patch(f"{module_path}.redis.Redis") as mock_redis_cls:
        mock_redis_cls.return_value = MagicMock()
        getattr(module, class_name)()

    kwargs = mock_redis_cls.call_args.kwargs
    assert 0 < kwargs["socket_connect_timeout"] <= 5
    assert 0 < kwargs["socket_timeout"] <= 10


def test_data_fetch_lock_renew_and_extend_are_capped_at_the_lease():
    with patch("app.tasks.data_fetch_lock.settings") as mock_settings:
        mock_settings.redis_host = "localhost"
        mock_settings.redis_port = 6379
        mock_settings.redis_db = 0
        mock_settings.data_fetch_lock_timeout = 300
        with patch("app.tasks.data_fetch_lock.redis.Redis") as mock_redis_cls:
            mock_redis = MagicMock()
            mock_redis_cls.return_value = mock_redis
            release_script, extend_script, renew_script = (
                MagicMock(),
                MagicMock(return_value=300),
                MagicMock(return_value=1),
            )
            mock_redis.register_script.side_effect = [
                release_script,
                extend_script,
                renew_script,
            ]
            from app.tasks.data_fetch_lock import DataFetchLock

            lock = DataFetchLock()

    assert lock.renew("task-1", market="HK") is True
    renew_script.assert_called_once_with(
        keys=["data_fetch_job_lock:hk"], args=[":task-1:", 300]
    )
    assert lock.extend_lock("task-1", 300, market="HK") is True
    extend_script.assert_called_once_with(
        keys=["data_fetch_job_lock:hk"], args=[":task-1:", 300, 300]
    )
