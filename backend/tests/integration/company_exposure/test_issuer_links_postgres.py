from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.domain.company_exposure.contracts import SERVICE_PRINCIPAL, RegistryMatch
from app.models.company_exposure import IssuerSecurityLinkRevision
from app.services.company_exposure.issuer_identity import IssuerIdentityAdapter
from tests.fixtures.company_exposure.factory import make_security

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql", reason="requires PostgreSQL row locks"
    ),
    pytest.mark.exposure_layer("postgres"),
]


@pytest.mark.case("I02")
def test_concurrent_registry_acceptance_creates_one_link():
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    security = make_security(session, "RACE")
    session.commit()
    session.close()
    match = RegistryMatch(
        security_id=security.id,
        market="US",
        scheme="cik",
        value="555",
        candidate_count=1,
        ticker_confirmed=True,
        matched_ticker="RACE",
        registry_capture_revision_id=None,
        official_record_capture_revision_id=None,
    )
    barrier = Barrier(2)

    def accept(_):
        worker = factory()
        try:
            barrier.wait(timeout=10)
            ref = IssuerIdentityAdapter(worker).accept_registry_match(
                match, SERVICE_PRINCIPAL
            )
            worker.commit()
            return ref
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        refs = list(executor.map(accept, [0, 1]))

    assert {ref.state for ref in refs} == {"accepted"}
    assert sorted(ref.created for ref in refs) == [False, True]
    check = factory()
    assert check.query(IssuerSecurityLinkRevision).count() == 1
    check.close()


def test_concurrent_revisions_of_one_identifier_get_distinct_numbers():
    from time import sleep

    from app.domain.company_exposure.contracts import LinkState
    from app.models.company_exposure import IssuerIdentifierRevision
    from tests.fixtures.company_exposure.factory import make_issuer

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    setup = factory()
    issuer = make_issuer(setup, "shared-identifier-issuer")
    setup.commit()
    setup.close()
    key = ("US", "cik", "8675309")
    barrier = Barrier(2)

    def revise(label):
        session = factory()
        try:
            adapter = IssuerIdentityAdapter(session)
            flush = session.flush

            def slow_flush(*args, **kwargs):
                # Widen the window between reading the maximum and inserting.
                sleep(0.5)
                return flush(*args, **kwargs)

            session.flush = slow_flush
            barrier.wait(timeout=10)
            row = adapter._add_identifier(
                issuer.id,
                key,
                state=LinkState.PROPOSED,
                policy=None,
                evidence={"reference": label},
                actor="test:admin",
                reason=label,
            )
            session.commit()
            return row.revision_number
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        numbers = sorted(executor.map(revise, ["a", "b"]))

    assert numbers == [1, 2]
    check = factory()
    assert (
        check.query(IssuerIdentifierRevision)
        .filter_by(market="US", scheme="cik", value="8675309")
        .count()
        == 2
    )
    check.close()


def test_two_listings_resolved_to_one_new_identifier_are_not_both_accepted():
    from time import sleep

    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    securities = [make_security(session, symbol) for symbol in ("TWINA", "TWINB")]
    session.commit()
    session.close()
    barrier = Barrier(2)

    def accept(security):
        worker = factory()
        try:
            flush = worker.flush

            def slow_flush(*args, **kwargs):
                # Widen the window between the ownership read and the writes.
                sleep(0.3)
                return flush(*args, **kwargs)

            worker.flush = slow_flush
            match = RegistryMatch(
                security_id=security.id,
                market="US",
                scheme="cik",
                value="4242",
                candidate_count=1,
                ticker_confirmed=True,
                matched_ticker=security.symbol,
                registry_capture_revision_id=None,
                official_record_capture_revision_id=None,
            )
            barrier.wait(timeout=10)
            ref = IssuerIdentityAdapter(worker).accept_registry_match(
                match, SERVICE_PRINCIPAL
            )
            worker.commit()
            return ref
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        refs = list(executor.map(accept, securities))

    # The second decision sees the first owner: a cross-listing to review.
    assert sorted(ref.state for ref in refs) == ["accepted", "review_required"]
    held = next(ref for ref in refs if ref.state == "review_required")
    assert held.reason == "cik_linked_to_other_issuer"
