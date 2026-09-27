"""Helpers for sizing scan compute process pools."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# Upper bound for the automatic process count. Every compute process holds
# its own copy of the chunk it is scanning, so memory grows with the count.
AUTO_SCAN_COMPUTE_PROCESSES_CAP = 4

_CGROUP_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_CGROUP_V1_CFS_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_CGROUP_V1_CFS_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def _cgroup_cpu_quota() -> float | None:
    """Return the container CPU quota in cores, or None when unlimited."""
    try:
        quota, period = _CGROUP_V2_CPU_MAX.read_text().split()[:2]
        if quota == "max":
            return None
        return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota_us = int(_CGROUP_V1_CFS_QUOTA.read_text().strip())
        period_us = int(_CGROUP_V1_CFS_PERIOD.read_text().strip())
    except (OSError, ValueError):
        return None
    if quota_us <= 0 or period_us <= 0:
        return None
    return quota_us / period_us


def available_cpu_count() -> int:
    """CPUs this process may actually use: affinity mask and cgroup quota aware.

    ``os.cpu_count()`` reports every host CPU even inside a container capped
    at half a core, which would oversubscribe the worker.
    """
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None:
        count = min(count, math.floor(quota))
    return max(1, count)


def resolve_scan_compute_processes(requested: int) -> int:
    """Resolve how many processes should compute scan results.

    ``requested`` of 0 means automatic: the available CPUs, capped at
    :data:`AUTO_SCAN_COMPUTE_PROCESSES_CAP`, and 1 (in-process) off Linux where
    forking a worker that has loaded native libraries is unsafe. A positive
    value is honoured up to the available CPUs. 1 always means in-process.
    """
    if requested < 0:
        raise ValueError("requested must be >= 0")
    cpus = available_cpu_count()
    if requested == 0:
        if not sys.platform.startswith("linux"):
            return 1
        return min(cpus, AUTO_SCAN_COMPUTE_PROCESSES_CAP)
    return min(requested, cpus)
