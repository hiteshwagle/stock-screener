from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep

import pytest
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.services.company_exposure.storage import OriginalStore

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql", reason="requires PostgreSQL row locks"
    ),
    pytest.mark.exposure_layer("postgres"),
]


def test_storage_last_capacity_is_reserved_once(tmp_path):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    barrier = Barrier(2)

    def contend(label):
        session = factory()
        try:
            store = OriginalStore(
                session,
                tmp_path / "store",
                max_bytes=100,
                min_free_bytes=0,
                disk_free=lambda _p: 10**12,
            )
            store.usage()  # create the shared pool row before contending
            session.commit()
            barrier.wait(timeout=10)
            ticket = store.reserve(60, purpose="t", operation_key=label)
            session.commit()
            return ticket
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(contend, ["a", "b"]))
    assert sum(result.allowed for result in outcomes) == 1
    assert {r.reason for r in outcomes if not r.allowed} == {"paused_storage"}


def test_concurrent_puts_of_identical_bytes_charge_the_store_once(
    tmp_path, monkeypatch
):
    """Every put's first ledger transition locks the one storage pool row
    until commit, so a second put of the same bytes finds the file and its
    charge; without that lock both writers would charge the blob."""

    from app.services.company_exposure import storage

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    barrier = Barrier(2)
    data = b"identical filing bytes"
    real_fsync = storage.os.fsync

    def slow_fsync(fd):
        # Widen the window between "no file yet" and the rename.
        sleep(0.5)
        real_fsync(fd)

    monkeypatch.setattr(storage.os, "fsync", slow_fsync)

    def store_for(session):
        return OriginalStore(
            session,
            tmp_path / "store",
            max_bytes=10_000,
            min_free_bytes=0,
            disk_free=lambda _p: 10**12,
        )

    setup = factory()
    before = store_for(setup).usage()["reserved_or_used_bytes"]
    setup.commit()
    setup.close()

    def put(label):
        session = factory()
        try:
            store = store_for(session)
            ticket = store.reserve(len(data), purpose="t", operation_key=label)
            session.commit()
            barrier.wait(timeout=10)
            ref = store.put(data, "text/html", ticket)
            session.commit()
            return ref
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        refs = list(executor.map(put, ["a", "b"]))

    assert sorted(ref.deduplicated for ref in refs) == [False, True]
    check = factory()
    after = store_for(check).usage()["reserved_or_used_bytes"]
    check.close()
    assert after - before == len(data)
