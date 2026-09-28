from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from app.models.company_exposure import EvidenceTombstoneEvent, ResearchResourcePool
from app.services.company_exposure.storage import (
    OriginalStore,
    StorageUnavailable,
    blob_key,
)
from tests.fixtures.company_exposure.factory import (
    FixedClock,
    make_document,
    make_revision,
)

GIB = 1024**3


@pytest.fixture
def clock():
    return FixedClock()


def _store(db_session, tmp_path, clock, *, max_bytes=100, free=10 * GIB, min_free=GIB):
    return OriginalStore(
        db_session,
        tmp_path / "store",
        max_bytes=max_bytes,
        min_free_bytes=min_free,
        clock=clock.now,
        disk_free=lambda _path: free,
    )


def _pool(db_session):
    return db_session.query(ResearchResourcePool).filter_by(unit="blob_bytes").one()


def test_put_is_atomic_and_charges_actual_bytes(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    ticket = store.reserve(50, purpose="t", operation_key="a")
    blob = store.put(b"hello", "text/plain", ticket)
    assert store.path_for(blob.key).read_bytes() == b"hello"
    assert list((tmp_path / "store" / "tmp").iterdir()) == []
    assert _pool(db_session).reserved_amount == 5


def test_identical_content_is_not_charged_twice(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    store.put(b"same", "text/plain", store.reserve(10, purpose="t", operation_key="a"))
    again = store.put(
        b"same", "text/plain", store.reserve(10, purpose="t", operation_key="b")
    )
    assert again.deduplicated
    assert _pool(db_session).reserved_amount == 4


def test_full_store_pauses_before_io(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock, max_bytes=10)
    assert store.reserve(8, purpose="t", operation_key="a").allowed
    blocked = store.reserve(8, purpose="t", operation_key="b")
    assert (blocked.allowed, blocked.reason, blocked.available) == (
        False,
        "paused_storage",
        2,
    )


def test_free_space_floor_pauses_before_io(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock, free=GIB + 5)
    blocked = store.reserve(10, purpose="t", operation_key="a")
    assert (blocked.allowed, blocked.reason) == (False, "paused_storage")


def test_put_larger_than_reservation_is_refused(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    with pytest.raises(StorageUnavailable):
        store.put(
            b"x" * 20, "text/plain", store.reserve(10, purpose="t", operation_key="a")
        )
    assert _pool(db_session).reserved_amount == 0


def test_blob_key_is_content_addressed(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    blob = store.put(
        b"doc", "text/plain", store.reserve(10, purpose="t", operation_key="a")
    )
    assert blob.key == blob_key(blob.content_hash)


def _age(path, days):
    stamp = path.stat().st_mtime - days * 86400
    os.utime(path, (stamp, stamp))


def test_gc_keeps_referenced_and_recent_blobs_and_tombstones_removals(
    db_session, tmp_path, clock
):
    store = _store(db_session, tmp_path, clock, max_bytes=10_000)

    def put(data, key):
        return store.put(
            data, "text/plain", store.reserve(10, purpose="t", operation_key=key)
        )

    referenced, orphan, recent = (
        put(b"kept", "a"),
        put(b"orphan", "b"),
        put(b"recent", "c"),
    )
    make_revision(db_session, make_document(db_session, "sec:accession:gc"), b"kept")
    for blob in (referenced, orphan):
        _age(store.path_for(blob.key), 40)
    stale_temp = tmp_path / "store" / "tmp" / "abc.part"
    stale_temp.parent.mkdir(parents=True, exist_ok=True)
    stale_temp.write_bytes(b"partial")
    _age(stale_temp, 2)
    # File ages are real filesystem times, so GC runs "now".
    clock.advance_to(datetime.now(timezone.utc))

    preview = store.collect_unreferenced_blobs(dry_run=True)
    assert preview.unreferenced == (orphan.key,)
    assert store.path_for(orphan.key).exists()

    report = store.collect_unreferenced_blobs(dry_run=False)
    assert report.unreferenced == (orphan.key,)
    assert report.abandoned_temp == ("abc.part",)
    assert not store.path_for(orphan.key).exists()
    assert store.path_for(referenced.key).exists()
    assert store.path_for(recent.key).exists()
    tombstone = db_session.query(EvidenceTombstoneEvent).one()
    assert (tombstone.reason, tombstone.byte_length) == ("unreferenced_gc", 6)
    with pytest.raises(StorageUnavailable, match="evidence_removed"):
        store.read(orphan.key)


def test_gc_of_a_crash_orphan_does_not_credit_bytes_it_never_charged(
    db_session, tmp_path, clock
):
    store = _store(db_session, tmp_path, clock, max_bytes=10_000)
    kept = store.put(
        b"kept", "text/plain", store.reserve(10, purpose="t", operation_key="k")
    )
    make_revision(db_session, make_document(db_session, "sec:accession:kept"), b"kept")
    ticket = store.reserve(20, purpose="t", operation_key="crash")
    db_session.commit()
    # The worker writes the blob, then dies before its transaction commits.
    orphan = store.put(b"orphan!!", "text/plain", ticket)
    db_session.rollback()
    clock.advance_to(datetime.now(timezone.utc) + timedelta(hours=2))
    assert store.release_abandoned_reservations() == (ticket.reservation_id,)
    assert _pool(db_session).reserved_amount == 4  # only the kept blob

    _age(store.path_for(orphan.key), 40)
    clock.advance_to(datetime.now(timezone.utc))
    report = store.collect_unreferenced_blobs(dry_run=False)
    assert report.unreferenced == (orphan.key,)
    assert store.path_for(kept.key).exists()
    assert _pool(db_session).reserved_amount == 4


def test_reacquired_bytes_are_readable_and_charged_once_after_gc(
    db_session, tmp_path, clock
):
    store = _store(db_session, tmp_path, clock, max_bytes=10_000)

    def put(key):
        return store.put(
            b"filing", "text/plain", store.reserve(10, purpose="t", operation_key=key)
        )

    first = put("first")
    _age(store.path_for(first.key), 40)
    clock.advance_to(datetime.now(timezone.utc))
    store.collect_unreferenced_blobs(dry_run=False)
    db_session.commit()
    assert _pool(db_session).reserved_amount == 0
    with pytest.raises(StorageUnavailable, match="evidence_removed"):
        store.read(first.key)

    again = put("again")
    db_session.commit()
    assert store.read(again.key) == b"filing"
    assert _pool(db_session).reserved_amount == 6
    _age(store.path_for(again.key), 40)
    store.collect_unreferenced_blobs(dry_run=False)
    # Only the second incarnation's charge is credited, not both.
    assert _pool(db_session).reserved_amount == 0


def test_identical_put_adopts_an_uncharged_crash_orphan(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock, max_bytes=10_000)
    ticket = store.reserve(20, purpose="t", operation_key="crash")
    db_session.commit()
    # The blob reaches disk but the charging transaction never commits.
    store.put(b"orphan!!", "text/plain", ticket)
    db_session.rollback()
    clock.advance_to(datetime.now(timezone.utc) + timedelta(hours=2))
    store.release_abandoned_reservations()
    assert _pool(db_session).reserved_amount == 0

    def put(key):
        return store.put(
            b"orphan!!", "text/plain", store.reserve(20, purpose="t", operation_key=key)
        )

    adopted = put("adopt")
    assert adopted.deduplicated
    # The bytes on disk are now paid for, once.
    assert _pool(db_session).reserved_amount == 8
    put("again")
    assert _pool(db_session).reserved_amount == 8
    _age(store.path_for(adopted.key), 40)
    clock.advance_to(datetime.now(timezone.utc))
    store.collect_unreferenced_blobs(dry_run=False)
    assert _pool(db_session).reserved_amount == 0


def test_unreadable_blob_is_a_storage_condition(
    db_session, tmp_path, clock, monkeypatch
):
    store = _store(db_session, tmp_path, clock)
    blob = store.put(
        b"hello", "text/plain", store.reserve(10, purpose="t", operation_key="a")
    )

    def fail(_path):
        raise PermissionError("denied")

    monkeypatch.setattr(type(store.path_for(blob.key)), "read_bytes", fail)
    with pytest.raises(StorageUnavailable, match="evidence_unreadable"):
        store.read(blob.key)
