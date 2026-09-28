"""Tests for scan compute process sizing helpers."""

from __future__ import annotations

import pytest

from app.utils import parallelism


def _no_cgroup(monkeypatch, tmp_path):
    monkeypatch.setattr(parallelism, "_CGROUP_V2_CPU_MAX", tmp_path / "missing-v2")
    monkeypatch.setattr(parallelism, "_CGROUP_V1_CFS_QUOTA", tmp_path / "missing-quota")
    monkeypatch.setattr(parallelism, "_CGROUP_V1_CFS_PERIOD", tmp_path / "missing-period")


def _affinity(monkeypatch, count):
    monkeypatch.setattr(
        parallelism.os,
        "sched_getaffinity",
        lambda _pid: set(range(count)),
        raising=False,
    )


@pytest.mark.parametrize(
    ("cpu_max", "affinity", "expected"),
    [
        ("max 100000\n", 8, 8),
        ("200000 100000\n", 8, 2),
        ("50000 100000\n", 8, 1),
        ("250000 100000\n", 8, 2),
        ("400000 100000\n", 2, 2),
    ],
)
def test_available_cpu_count_honours_cgroup_v2_quota(
    monkeypatch, tmp_path, cpu_max, affinity, expected
):
    _no_cgroup(monkeypatch, tmp_path)
    cpu_max_path = tmp_path / "cpu.max"
    cpu_max_path.write_text(cpu_max)
    monkeypatch.setattr(parallelism, "_CGROUP_V2_CPU_MAX", cpu_max_path)
    _affinity(monkeypatch, affinity)

    assert parallelism.available_cpu_count() == expected


@pytest.mark.parametrize(
    ("quota_us", "expected"),
    [("-1", 6), ("300000", 3), ("20000", 1)],
)
def test_available_cpu_count_honours_cgroup_v1_quota(
    monkeypatch, tmp_path, quota_us, expected
):
    _no_cgroup(monkeypatch, tmp_path)
    quota = tmp_path / "cpu.cfs_quota_us"
    period = tmp_path / "cpu.cfs_period_us"
    quota.write_text(quota_us)
    period.write_text("100000")
    monkeypatch.setattr(parallelism, "_CGROUP_V1_CFS_QUOTA", quota)
    monkeypatch.setattr(parallelism, "_CGROUP_V1_CFS_PERIOD", period)
    _affinity(monkeypatch, 6)

    assert parallelism.available_cpu_count() == expected


def test_available_cpu_count_falls_back_to_cpu_count(monkeypatch, tmp_path):
    _no_cgroup(monkeypatch, tmp_path)
    monkeypatch.delattr(parallelism.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(parallelism.os, "cpu_count", lambda: None)

    assert parallelism.available_cpu_count() == 1


@pytest.mark.parametrize(
    ("platform", "cpus", "requested", "expected"),
    [
        ("linux", 8, 0, parallelism.AUTO_SCAN_COMPUTE_PROCESSES_CAP),
        ("linux", 3, 0, 3),
        ("linux", 1, 0, 1),
        ("darwin", 8, 0, 1),
        ("darwin", 8, 3, 3),
        ("linux", 4, 8, 4),
        ("linux", 8, 6, 6),
        ("linux", 8, 1, 1),
    ],
)
def test_resolve_scan_compute_processes(
    monkeypatch, platform, cpus, requested, expected
):
    monkeypatch.setattr(parallelism.sys, "platform", platform)
    monkeypatch.setattr(parallelism, "available_cpu_count", lambda: cpus)

    assert parallelism.resolve_scan_compute_processes(requested) == expected


def test_resolve_scan_compute_processes_rejects_negative_requests():
    with pytest.raises(ValueError, match="requested must be >= 0"):
        parallelism.resolve_scan_compute_processes(-1)
