from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.services.company_exposure.holds import HoldRegistry
from app.services.company_exposure.materiality import validate_measure
from tests.fixtures.company_exposure.factory import verified_claim

PARTICIPATION_DATE = datetime(2024, 3, 1, tzinfo=timezone.utc)
EXIT_DATE = datetime(2025, 6, 30, tzinfo=timezone.utc)
ORIGINAL_10K = datetime(2025, 2, 14, tzinfo=timezone.utc)
AMENDED_10K = datetime(2025, 4, 1, tzinfo=timezone.utc)


@pytest.fixture
def ended_dossier(dossier):
    participation = verified_claim(
        "participation",
        passage=dossier.passages["role"],
        supported_as_of=PARTICIPATION_DATE,
    )
    exit_claim = verified_claim(
        "exposure_end",
        passage=dossier.passages["exit"],
        supported_as_of=EXIT_DATE,
        status="discontinued",
    )
    result, ref = dossier.persist(dossier.attempt(participation, exit_claim))
    assert result.claim("participation").ended
    return dossier, ref


@pytest.mark.case("E14")
@pytest.mark.exposure_layer("unit")
def test_late_old_capture_cannot_restore_ended_exposure(ended_dossier):
    dossier, first = ended_dossier
    for late_date in (PARTICIPATION_DATE, datetime(2023, 3, 1, tzinfo=timezone.utc)):
        late = verified_claim(
            "participation", passage=dossier.passages["role"], supported_as_of=late_date
        )
        result = dossier.service.assess(dossier.attempt(late))
        participation = result.claim("participation")
        assert participation.action == "carried_forward"
        assert participation.ended
        assert result.claim("exposure_end").conclusion == "supported"
        ref = dossier.service.persist_assessment(result)
        # No new participation revision: the obsolete exposure is not restored.
        assert (
            ref.claim_revision_ids.get(
                participation.proposition_key,
                first.claim_revision_ids[participation.proposition_key],
            )
            == first.claim_revision_ids[participation.proposition_key]
        )

    holds = HoldRegistry(dossier.db).active("claim", participation.prior_claim_id)
    assert [h.hold_kind for h in holds] == ["exposure_end"]


@pytest.mark.case("E14")
@pytest.mark.exposure_layer("unit")
def test_original_filing_arriving_after_amendment_does_not_win(dossier):
    amended = verified_claim(
        "commercial_status",
        passage=dossier.passages["exit"],
        supported_as_of=AMENDED_10K,
        status="discontinued",
    )
    dossier.persist(dossier.attempt(amended))
    original = verified_claim(
        "commercial_status",
        passage=dossier.passages["role"],
        supported_as_of=ORIGINAL_10K,
    )
    result = dossier.service.assess(dossier.attempt(original))
    status = result.claim("commercial_status")
    assert (status.reason, status.commercial_status) == (
        "older_than_selected",
        "discontinued",
    )
    assert result.set_aside[0]["reason"] == "older_than_selected"


def test_same_date_disagreement_holds_the_proposition(dossier):
    shipping = verified_claim(
        "commercial_status",
        passage=dossier.passages["role"],
        supported_as_of=ORIGINAL_10K,
    )
    dossier.persist(dossier.attempt(shipping))
    contradicting = verified_claim(
        "commercial_status",
        passage=dossier.passages["exit"],
        supported_as_of=ORIGINAL_10K,
        status="discontinued",
    )
    result, _ = dossier.persist(dossier.attempt(contradicting))
    status = result.claim("commercial_status")
    assert status.commercial_status == "shipping_or_operating"
    assert status.hold_kinds == ("conflict",)
    assert result.conflicts == (
        {"proposition": status.proposition_key, "reason": "same_date_disagreement"},
    )


def test_exit_holds_only_the_same_product(dossier):
    unrelated = verified_claim(
        "role",
        passage=dossier.passages["other"],
        product_key="probe-cards",
        supported_as_of=PARTICIPATION_DATE,
    )
    related = verified_claim(
        "role", passage=dossier.passages["role"], supported_as_of=PARTICIPATION_DATE
    )
    exit_claim = verified_claim(
        "exposure_end",
        passage=dossier.passages["exit"],
        supported_as_of=EXIT_DATE,
        status="discontinued",
    )
    result = dossier.service.assess(dossier.attempt(unrelated, related, exit_claim))
    assert result.claim("role", "hbm-test-equipment").ended
    assert not result.claim("role", "probe-cards").ended


def _materiality_candidate(dossier, percent):
    passage = dossier.passages["materiality"]
    quote = f"Memory test was {percent}% of revenue in fiscal 2025."
    measure = validate_measure(
        metric="revenue_percent",
        value=Decimal(percent),
        unit="percent",
        period="FY2025",
        scope="segment_or_subsidiary",
        scope_label="Memory test",
        quote=quote,
        passage_id=str(passage.id),
    )
    assert not measure.held
    return verified_claim(
        "materiality",
        passage=passage,
        quote=quote,
        supported_as_of=ORIGINAL_10K,
        materiality=measure,
    )


def test_same_date_candidate_differing_from_the_selected_claim_conflicts(dossier):
    dossier.persist(dossier.attempt(_materiality_candidate(dossier, 10)))
    result = dossier.service.assess(
        dossier.attempt(_materiality_candidate(dossier, 20))
    )
    selected = result.claim("materiality")
    assert selected.reason == "same_date_disagreement"
    assert "conflict" in selected.hold_kinds
    # The same figure again is a duplicate, not a disagreement.
    same = dossier.service.assess(dossier.attempt(_materiality_candidate(dossier, 10)))
    assert same.claim("materiality").reason == "same_substantive_date"


def test_same_date_candidates_with_different_materiality_conflict(dossier):
    passage = dossier.passages["materiality"]

    def candidate(percent):
        quote = f"Memory test was {percent}% of revenue in fiscal 2025."
        measure = validate_measure(
            metric="revenue_percent",
            value=Decimal(percent),
            unit="percent",
            period="FY2025",
            scope="segment_or_subsidiary",
            scope_label="Memory test",
            quote=quote,
            passage_id=str(passage.id),
        )
        assert not measure.held
        return verified_claim(
            "materiality",
            passage=passage,
            quote=quote,
            supported_as_of=ORIGINAL_10K,
            materiality=measure,
        )

    result = dossier.service.assess(dossier.attempt(candidate(10), candidate(20)))
    selected = result.claim("materiality")
    assert "conflict" in selected.hold_kinds
    assert result.conflicts == (
        {"proposition": selected.proposition_key, "reason": "same_date_disagreement"},
    )


def test_displayed_measure_returns_the_passage_it_rests_on(dossier):
    from app.services.company_exposure.reads import ResearchJobReader

    candidate = _materiality_candidate(dossier, 20)
    _, ref = dossier.persist(dossier.attempt(candidate))
    (revision_id,) = ref.claim_revision_ids.values()
    shown = ResearchJobReader(dossier.db)._materiality(revision_id)
    passage = dossier.passages["materiality"]
    assert shown["value"] == "20"
    assert shown["evidence"] == [
        {
            "role": "disclosed",
            "passage_id": str(passage.id),
            "value": "20",
            "label": "Memory test",
            "quote": "Memory test was 20% of revenue in fiscal 2025.",
            "document_revision_id": str(passage.document_revision_id),
        }
    ]
