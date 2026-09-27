"""Heartbeat renewal for short-lived Redis workload leases.

Leases are taken with a short TTL (``settings.data_fetch_lock_timeout``) and
renewed by a daemon thread for as long as the holder runs. If the worker
process dies (OOM kill, container restart) the thread dies with it and the
lease expires within one TTL instead of blocking the market for hours.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import logging
import threading

from ..config import settings

logger = logging.getLogger(__name__)

DEFAULT_LEASE_TTL_SECONDS = 300
# Lease clients must never block indefinitely: one stalled call would stall
# every lease renewed on the same heartbeat thread. Three renewals at the
# worst case (connect + read) stay far below the default 100 s interval.
LEASE_REDIS_CONNECT_TIMEOUT_SECONDS = 2
LEASE_REDIS_SOCKET_TIMEOUT_SECONDS = 5

# Refresh the TTL only while the lease still names this holder, so a lease
# that expired and was taken by another task is never extended by us.
RENEW_LEASE_LUA = """
local val = redis.call('get', KEYS[1])
if val and string.find(val, ARGV[1], 1, true) then
    return redis.call('expire', KEYS[1], tonumber(ARGV[2]))
end
return 0
"""


def lease_ttl_seconds(value: object = None) -> int:
    """The configured lease TTL, or the default for a missing/invalid value."""
    if value is None:
        value = getattr(settings, "data_fetch_lock_timeout", DEFAULT_LEASE_TTL_SECONDS)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return DEFAULT_LEASE_TTL_SECONDS
    return int(value)


def lease_renew_interval_seconds() -> float:
    """Renew three times per TTL so one missed beat never drops the lease.

    No lower bound: even a one-second TTL must be renewed before it expires.
    """
    return lease_ttl_seconds() / 3


@contextmanager
def keep_leases_alive(
    renewals: Sequence[tuple[str, Callable[[], bool]]],
    *,
    interval_seconds: float | None = None,
) -> Iterator[None]:
    """Run each ``(name, renew)`` every interval until the block exits.

    ``renew`` returns whether the lease is still held. A lease that is no
    longer held is dropped and logged; the task is not interrupted, since
    aborting mid-write would be worse than finishing unserialized. Redis
    errors are logged and retried on the next beat.
    """
    if not renewals:
        yield
        return

    interval = (
        lease_renew_interval_seconds() if interval_seconds is None else interval_seconds
    )
    stop = threading.Event()
    active = list(renewals)

    def renew_all() -> None:
        for entry in tuple(active):
            name, renew = entry
            try:
                held = renew()
            except Exception:
                logger.warning("Lease renewal failed for %s", name, exc_info=True)
                continue
            if not held:
                logger.error(
                    "Lease %s is no longer held by this task; stopped renewing",
                    name,
                )
                active.remove(entry)

    def beat() -> None:
        while active and not stop.wait(interval):
            renew_all()

    # Renew once before the body starts: a lease reused by a retried task
    # (same id) may have less than one interval left and would otherwise
    # expire before the first beat.
    renew_all()
    thread = threading.Thread(target=beat, name="lease-renewal", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)
