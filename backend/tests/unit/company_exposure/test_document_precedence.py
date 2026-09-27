from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.models.company_exposure import AssessmentRevision
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
    [conflict] = result.conflicts
    assert (conflict["proposition"], conflict["reason"]) == (
        status.proposition_key,
        "same_date_disagreement",
    )
    # The set-aside candidate's own wording is kept for the reviewer.
    assert [(e["quote"], e["commercial_status"]) for e in conflict["evidence"]] == [
        (dossier.passages["exit"].original_text, "discontinued")
    ]


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
    [conflict] = result.conflicts
    assert (conflict["proposition"], conflict["reason"]) == (
        selected.proposition_key,
        "same_date_disagreement",
    )
    assert len(conflict["evidence"]) == 1


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


def test_held_ratio_is_persisted_as_unknown(dossier):
    from app.models.company_exposure import MaterialityMeasure
    from app.services.company_exposure.materiality import (
        Operand,
        calculate_materiality,
    )

    passage = dossier.passages["materiality"]

    def operand(value, unit):
        return Operand(
            value=Decimal(value),
            unit=unit,
            period="FY2025",
            scope="issuer_consolidated",
            label="Memory test revenue",
            currency="USD",
            passage_id=str(passage.id),
            quote=f"Memory test revenue was USD {value} {unit} in FY2025",
        )

    held = calculate_materiality(
        metric="revenue_share",
        numerator=operand("20", "million"),
        denominator=operand("1", "billion"),
    )
    assert "unit_mismatch" in held.hold_reasons
    candidate = verified_claim(
        "materiality",
        passage=passage,
        supported_as_of=ORIGINAL_10K,
        materiality=held,
    )
    _, ref = dossier.persist(dossier.attempt(candidate))
    (revision_id,) = ref.claim_revision_ids.values()
    row = (
        dossier.db.query(MaterialityMeasure)
        .filter_by(claim_revision_id=revision_id)
        .one()
    )
    assert (row.basis, row.value_low) == ("unknown", None)
    assert "unit_mismatch" in row.hold_reasons

    # The preview shows no value, but keeps both operand quotes for review.
    from app.services.company_exposure.reads import ResearchJobReader

    shown = ResearchJobReader(dossier.db)._materiality(revision_id)
    assert shown["display"] and "value" not in shown
    assert [(e["role"], e["quote"]) for e in shown["evidence"]] == [
        ("numerator", "Memory test revenue was USD 20 million in FY2025"),
        ("denominator", "Memory test revenue was USD 1 billion in FY2025"),
    ]


def test_newly_detected_conflict_is_recorded_in_a_new_revision(dossier):
    from sqlalchemy import select

    from app.models.company_exposure import AssessmentRevision

    _, first = dossier.persist(dossier.attempt(_materiality_candidate(dossier, 10)))
    result, ref = dossier.persist(dossier.attempt(_materiality_candidate(dossier, 20)))
    # The selected figure is carried forward, but the disagreement is new
    # evidence for review and must not collapse into "unchanged".
    assert result.conflicts
    assert not ref.unchanged
    assert ref.revision_number == first.revision_number + 1
    latest = dossier.service.session.execute(
        select(AssessmentRevision).where(AssessmentRevision.id == ref.id)
    ).scalar_one()
    [conflict] = latest.conflicts
    assert (conflict["proposition"], conflict["reason"]) == (
        result.claim("materiality").proposition_key,
        "same_date_disagreement",
    )
    # The disagreeing 20% figure's quote is stored with the conflict.
    assert [e["quote"] for e in conflict["evidence"]] == [
        "Memory test was 20% of revenue in fiscal 2025."
    ]


def test_undated_candidate_does_not_displace_an_undated_primary_claim(dossier):
    shipping = verified_claim(
        "commercial_status", passage=dossier.passages["role"], supported_as_of=None
    )
    dossier.persist(dossier.attempt(shipping))
    contradicting = verified_claim(
        "commercial_status",
        passage=dossier.passages["exit"],
        supported_as_of=None,
        status="discontinued",
    )
    result = dossier.service.assess(dossier.attempt(contradicting))
    selected = result.claim("commercial_status")
    assert selected.reason == "undated_candidate"
    assert selected.commercial_status == "shipping_or_operating"


FY2024 = "2024-12-31"
FY2024_ANCHOR = datetime(2024, 12, 31, tzinfo=timezone.utc)


def _annual_status(passage, status, *, published, amendment=False):
    # An original 10-K and its 10-K/A share the period's date anchor.
    return verified_claim(
        "commercial_status",
        passage=passage,
        supported_as_of=FY2024_ANCHOR,
        period=FY2024,
        status=status,
        published=published,
        amendment=amendment,
    )


def _amendment_passage(dossier):
    from tests.fixtures.company_exposure.factory import (
        make_document,
        make_passage,
        make_revision,
    )

    document = make_document(dossier.db, "sec:10-K/A:dossier", issuer=dossier.issuer)
    revision = make_revision(
        dossier.db,
        document,
        b"amended annual report",
        published_at=AMENDED_10K,
        period=FY2024,
        correction={"form": "10-K/A", "is_amendment": True},
    )
    return make_passage(
        dossier.db, revision, "We discontinued HBM test equipment in 2024."
    )


def test_amendment_supersedes_its_original_at_the_same_date(dossier):
    original = _annual_status(
        dossier.passages["role"], "shipping_or_operating", published=ORIGINAL_10K
    )
    dossier.persist(dossier.attempt(original))
    amended = _annual_status(
        dossier.passages["exit"], "discontinued", published=AMENDED_10K, amendment=True
    )
    result = dossier.service.assess(dossier.attempt(amended))
    status = result.claim("commercial_status")
    assert (status.reason, status.commercial_status) == (
        "amendment_supersedes",
        "discontinued",
    )
    assert result.conflicts == ()


def test_original_arriving_after_its_amendment_is_set_aside_without_conflict(
    dossier,
):
    amended = _annual_status(
        _amendment_passage(dossier),
        "discontinued",
        published=AMENDED_10K,
        amendment=True,
    )
    dossier.persist(dossier.attempt(amended))
    original = _annual_status(
        dossier.passages["role"], "shipping_or_operating", published=ORIGINAL_10K
    )
    result = dossier.service.assess(dossier.attempt(original))
    status = result.claim("commercial_status")
    assert (status.reason, status.commercial_status) == (
        "superseded_by_amendment",
        "discontinued",
    )
    assert result.conflicts == ()
    assert "conflict" not in status.hold_kinds


def test_amendment_and_original_in_one_attempt_select_the_amendment(dossier):
    original = _annual_status(
        dossier.passages["role"], "shipping_or_operating", published=ORIGINAL_10K
    )
    amended = _annual_status(
        dossier.passages["exit"], "discontinued", published=AMENDED_10K, amendment=True
    )
    result = dossier.service.assess(dossier.attempt(original, amended))
    status = result.claim("commercial_status")
    assert status.commercial_status == "discontinued"
    assert result.conflicts == ()


def test_preview_freshness_is_evaluated_now_not_at_sealing(dossier):
    from datetime import timedelta

    from app.services.company_exposure.reads import ResearchJobReader

    status = verified_claim(
        "commercial_status",
        passage=dossier.passages["role"],
        supported_as_of=dossier.clock.now() - timedelta(days=10),
    )
    _, ref = dossier.persist(dossier.attempt(status))
    revision = dossier.db.get(AssessmentRevision, ref.id)
    sealed = ResearchJobReader(dossier.db, clock=dossier.clock.now)._claims(revision)
    assert sealed[0]["freshness_state"] == "current"

    later = dossier.clock.now() + timedelta(days=400)
    [claim] = ResearchJobReader(dossier.db, clock=lambda: later)._claims(revision)
    assert (claim["freshness_state"], claim["sealed_freshness_state"]) == (
        "stale",
        "current",
    )
