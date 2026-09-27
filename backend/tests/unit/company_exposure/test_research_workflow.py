from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from uuid import UUID

import pytest
from sqlalchemy import func, select

from app.domain.company_exposure.contracts import as_utc
from app.infra.db.repositories.company_exposure_work_repo import WorkLeaseError
from app.models.company_exposure import (
    AssessmentRevision,
    ExposureClaimRevision,
    ExposureDocument,
    ExposureDocumentRevision,
    ResearchArtifact,
    ResearchProviderAttempt,
    ResearchWorkItem,
)
from app.services.company_exposure.issuer_identity import (
    IssuerIdentityAdapter,
    LinkProposal,
)
from app.services.company_exposure.reads import ResearchJobReader
from app.services.company_exposure.research_requests import ResearchUnavailable
from app.tasks import company_exposure_tasks
from tests.fixtures.company_exposure.factory import make_theme
from tests.fixtures.company_exposure.research_harness import (
    ADMIN,
    REPORT_URL,
    SHADOW,
    TICKERS,
    Harness,
    claims_for,
)


@pytest.fixture
def harness(db_session, tmp_path, clock):
    return Harness(db_session, tmp_path, clock)


@pytest.mark.case("R05")
@pytest.mark.exposure_layer("unit")
def test_repeat_request_reuses_job(harness):
    first = harness.request()
    repeated = harness.request()
    assert first.id == repeated.id
    assert (first.created, repeated.created) == (True, False)
    assert harness.run_all()[0].stage == "resolve_issuer"


def test_disabled_research_and_discovery_are_refused(harness, db_session):
    harness.build(replace(SHADOW, research_mode="disabled"))
    with pytest.raises(ResearchUnavailable, match="research_disabled"):
        harness.request()
    harness.build(SHADOW)
    with pytest.raises(ResearchUnavailable, match="discovery_not_installed"):
        harness.request(kind="discover")


def test_offline_us_verify_resolves_cik_then_assesses(harness, db_session):
    harness.serve_sec()
    harness.go.queue_builder(claims_for)
    ref = harness.request()
    results = harness.run_all()
    assert [(r.stage, r.status) for r in results] == [
        ("resolve_issuer", "completed"),
        ("acquire", "completed"),
        ("verify", "completed"),
    ]
    assert results[0].detail["source"] == "official_registry"
    assert results[0].detail["acceptance_policy"] == "official_registry_single_listing"
    # The older 10-K is not served: coverage is partial, not an exposure change.
    assert results[-1].state == "partial"
    assert results[-1].detail["claims"] == 1
    assert len(harness.go.requests) == 1
    assert set(harness.rate.provider_names) == {"sec_edgar"}

    revision = db_session.get(
        AssessmentRevision, UUID(results[-1].detail["assessment_revision_id"])
    )
    assert revision.request_id == ref.id
    claim = db_session.execute(select(ExposureClaimRevision)).scalar_one()
    assert (claim.support_basis, claim.conclusion) == ("primary_explicit", "supported")
    assert ResearchJobReader(harness.db).read(ref.id)["state"] == "partial"


def test_identical_refresh_reuses_artifact_without_new_spend(harness, db_session):
    harness.serve_sec()
    harness.go.queue_builder(claims_for)
    harness.request()
    harness.run_all()
    attempts = db_session.execute(
        select(func.count()).select_from(ResearchProviderAttempt)
    ).scalar()
    harness.request(key="refresh-1", kind="refresh")
    results = harness.run_all()
    assert results[-1].detail["unchanged"] is True
    assert len(harness.go.requests) == 1
    assert (
        db_session.execute(
            select(func.count()).select_from(ResearchProviderAttempt)
        ).scalar()
        == attempts
    )


def test_link_change_after_acquire_reacquires_for_the_current_issuer(
    harness, db_session
):
    harness.serve_sec()
    harness.go.queue_builder(claims_for)
    harness.request()
    resolved, acquired = harness.step(), harness.step()
    assert acquired.detail["document_revision_ids"]
    identity = IssuerIdentityAdapter(db_session)
    relink = identity.propose_link(
        LinkProposal(
            security_id=harness.security.id,
            issuer_id=None,
            identifiers=(),
            evidence={"reference": "corrected listing"},
            requested_by="test:admin",
            reason="listing belongs to another issuer",
        )
    )
    ref = identity.apply_link(relink.link_revision_id, ADMIN, relink.proposal_hash)
    db_session.commit()
    assert ref.issuer_id != UUID(resolved.detail["issuer_id"])

    verified = harness.step()
    # The new issuer has no CIK yet: the stage pauses for it instead of
    # assessing with the old issuer's filings.
    assert (verified.stage, verified.status) == ("verify", "paused")
    assert verified.detail["condition"] == "issuer_cik_unresolved"
    assert harness.go.requests == []


def test_relink_during_provider_call_is_never_sealed(harness, db_session):
    harness.serve_sec()
    identity = IssuerIdentityAdapter(db_session)

    def relink_then_answer(request_json):
        # An administrator relinks the listing while the model call runs.
        relink = identity.propose_link(
            LinkProposal(
                security_id=harness.security.id,
                issuer_id=None,
                identifiers=(),
                evidence={"reference": "corrected listing"},
                requested_by="test:admin",
                reason="listing belongs to another issuer",
            )
        )
        identity.apply_link(relink.link_revision_id, ADMIN, relink.proposal_hash)
        db_session.commit()
        return claims_for(request_json)

    harness.go.queue_builder(relink_then_answer)
    harness.request()
    harness.step(), harness.step()
    verified = harness.step()
    assert (verified.stage, verified.status) == ("verify", "retryable")
    assert verified.detail["condition"] == "scope_changed_during_verification"
    revisions = select(func.count()).select_from(AssessmentRevision)
    assert db_session.execute(revisions).scalar() == 0


def test_relink_after_the_early_check_is_caught_inside_the_sealing_fence(
    harness, db_session, monkeypatch
):
    harness.serve_sec()
    identity = IssuerIdentityAdapter(db_session)
    real = harness.runner._scope_changed
    checks = []

    def relink_after_early_check(request, scope):
        # The relink lands after the pre-assessment check passed, so only
        # the recheck inside persist_assessment's fence can refuse it.
        checks.append(scope)
        if len(checks) == 1:
            relink = identity.propose_link(
                LinkProposal(
                    security_id=harness.security.id,
                    issuer_id=None,
                    identifiers=(),
                    evidence={"reference": "corrected listing"},
                    requested_by="test:admin",
                    reason="listing belongs to another issuer",
                )
            )
            identity.apply_link(relink.link_revision_id, ADMIN, relink.proposal_hash)
            db_session.commit()
            return False
        return real(request, scope)

    harness.go.queue_builder(claims_for)
    harness.request()
    harness.step(), harness.step()
    monkeypatch.setattr(harness.runner, "_scope_changed", relink_after_early_check)
    verified = harness.step()
    assert len(checks) == 2
    assert (verified.stage, verified.status) == ("verify", "retryable")
    assert verified.detail["condition"] == "scope_changed_during_verification"
    revisions = select(func.count()).select_from(AssessmentRevision)
    assert db_session.execute(revisions).scalar() == 0


def test_ambiguous_cik_pauses_for_review_then_resumes_after_admin_link(
    harness, db_session
):
    duplicated = {
        "fields": TICKERS["fields"],
        "data": [*TICKERS["data"], [7654321, "Other Corp", "EXMP", "NYSE"]],
    }
    harness.serve_sec()
    harness.sec.serve_json(
        "https://www.sec.gov/files/company_tickers_exchange.json", duplicated
    )
    ref = harness.request()
    paused = harness.step()
    assert (paused.status, paused.state) == ("paused", "review_required")
    assert paused.detail["condition"] == "multiple_ciks"
    assert (
        ResearchJobReader(harness.db).read(ref.id)["stages"][0]["pause_reason"]
        == "review_required"
    )
    assert harness.repo.claim_next(worker_id="w") is None

    identity = IssuerIdentityAdapter(db_session)
    proposal = identity.propose_link(
        LinkProposal(
            security_id=harness.security.id,
            issuer_id=None,
            identifiers=(("US", "cik", "1234567"),),
            evidence={"reference": "10-K cover page"},
            requested_by="test:admin",
            reason="administrator-resolved CIK",
        )
    )
    identity.apply_link(proposal.link_revision_id, ADMIN, proposal.proposal_hash)
    harness.requests.resume(ref.id)
    db_session.commit()
    resumed = harness.step()
    assert (resumed.stage, resumed.status) == ("resolve_issuer", "completed")
    assert resumed.detail["source"] == "accepted_link"


def test_missing_allocation_pauses_without_dispatch(harness, db_session):
    harness.build(replace(SHADOW, daily_request_limit=None))
    harness.serve_sec()
    harness.request()
    results = harness.run_all()
    assert (results[-1].status, results[-1].state) == ("paused", "paused_allowance")
    assert results[-1].detail["condition"] == "allocation_not_configured"
    assert harness.go.requests == []


def test_unapproved_route_is_an_unavailable_capability(harness):
    harness.build(replace(SHADOW, text_route_enabled=False))
    harness.serve_sec()
    harness.request()
    results = harness.run_all()
    assert (results[-1].state, results[-1].detail["condition"]) == (
        "unavailable_capability",
        "route_not_approved",
    )


def test_missing_user_agent_pauses_before_network(harness):
    harness.build(replace(SHADOW, sec_user_agent=""))
    harness.request()
    result = harness.step()
    assert (result.state, result.detail["condition"]) == (
        "unavailable_capability",
        "sec_user_agent_not_configured",
    )
    assert harness.sec.requests == []


def test_stale_lease_cannot_run_a_step(harness):
    harness.request()
    item = harness.repo.claim_next(worker_id="w")
    harness.db.commit()
    with pytest.raises(WorkLeaseError):
        harness.runner.run_step(item.id, item.id)


def test_provider_retry_after_delays_the_retry(harness, db_session):
    harness.serve_sec()
    harness.go.queue_status(429, {"retry-after": "3600"})
    harness.request()
    results = harness.run_all(limit=3)
    assert [(r.stage, r.status) for r in results][-1] == ("verify", "retryable")
    item = db_session.execute(
        select(ResearchWorkItem).where(ResearchWorkItem.stage == "verify")
    ).scalar_one()
    # The local backoff would retry after 2 minutes; the provider asked 1 hour.
    assert as_utc(item.available_at) >= harness.clock.now() + timedelta(hours=1)


def test_sec_retry_after_delays_the_retry(harness, db_session):
    for name in ("company_tickers_exchange", "company_tickers"):
        harness.sec.serve_status(
            f"https://www.sec.gov/files/{name}.json", 429, {"retry-after": "7200"}
        )
    harness.request()
    step = harness.step()
    assert (step.stage, step.status) == ("resolve_issuer", "retryable")
    item = db_session.execute(
        select(ResearchWorkItem).where(ResearchWorkItem.stage == "resolve_issuer")
    ).scalar_one()
    # The local backoff would retry after 2 minutes; SEC asked for 2 hours.
    assert as_utc(item.available_at) >= harness.clock.now() + timedelta(hours=2)


def test_unexamined_pdf_pages_make_coverage_partial(harness, db_session):
    harness.serve_sec()
    harness.go.queue_builder(claims_for)
    prepare = harness.runner.preparer.prepare

    def truncated(*args, **kwargs):
        prepared = prepare(*args, **kwargs)
        prepared.coverage["omitted_ranges"] = [[300, 412]]
        return prepared

    harness.runner.preparer.prepare = truncated
    harness.request()
    verified = harness.run_all()[-1]
    revision = db_session.get(
        AssessmentRevision, UUID(verified.detail["assessment_revision_id"])
    )
    gaps = [c for c in revision.coverage if c["reason"] == "pages_not_examined"]
    assert gaps and gaps[0]["detail"]["omitted_ranges"] == [[300, 412]]


def test_throttled_filing_fetch_retries_acquire_instead_of_sealing(harness, db_session):
    harness.serve_sec()
    harness.sec.serve_status(REPORT_URL, 429, {"retry-after": "900"})
    harness.request()
    harness.step()
    acquired = harness.step()
    assert (acquired.stage, acquired.status) == ("acquire", "retryable")
    item = db_session.execute(
        select(ResearchWorkItem).where(ResearchWorkItem.stage == "acquire")
    ).scalar_one()
    assert as_utc(item.available_at) >= harness.clock.now() + timedelta(seconds=900)
    assert (
        db_session.execute(
            select(ResearchWorkItem).where(ResearchWorkItem.stage == "verify")
        ).scalar_one_or_none()
        is None
    )


def test_missing_sec_configuration_pauses_acquire_for_resume(harness, db_session):
    harness.serve_sec()
    harness.request()
    harness.step()  # resolve with the configured user agent
    requests_before = len(harness.sec.requests)
    harness.build(replace(SHADOW, sec_user_agent=""))
    paused = harness.step()
    assert (paused.stage, paused.status) == ("acquire", "paused")
    assert paused.detail["condition"] == "sec_user_agent_not_configured"
    assert len(harness.sec.requests) == requests_before


def test_retained_filings_are_bound_to_the_resolved_issuer(harness, db_session):
    harness.serve_sec()
    harness.request()
    resolved, acquired = harness.step(), harness.step()
    documents = [
        db_session.get(
            ExposureDocument,
            db_session.get(ExposureDocumentRevision, UUID(r)).document_id,
        )
        for r in acquired.detail["document_revision_ids"]
    ]
    assert documents
    assert {str(d.issuer_id) for d in documents} == {resolved.detail["issuer_id"]}


def test_missing_retained_original_is_a_coverage_gap(harness, db_session):
    harness.serve_sec()
    harness.request()
    harness.step(), harness.step()
    for blob in (harness.store.root / "sha256").glob("*/*"):
        blob.unlink()
    verified = harness.step()
    assert verified.stage == "verify" and verified.status == "completed"
    revision = db_session.get(
        AssessmentRevision, UUID(verified.detail["assessment_revision_id"])
    )
    assert "evidence_blob_missing" in {c["reason"] for c in revision.coverage}


def test_rejected_verifier_output_is_a_coverage_gap(harness, db_session):
    harness.serve_sec()
    harness.go.queue_json({"unexpected": "shape"})
    harness.request()
    verified = harness.run_all()[-1]
    assert (verified.stage, verified.status) == ("verify", "completed")
    assert verified.state != "ready_for_publication"
    revision = db_session.get(
        AssessmentRevision, UUID(verified.detail["assessment_revision_id"])
    )
    assert "verifier_output_rejected" in {c["reason"] for c in revision.coverage}


def test_rejected_verifier_output_is_never_reused(harness, db_session):
    harness.serve_sec()
    harness.go.queue_json({"unexpected": "shape"})
    harness.request()
    rejected = harness.run_all()[-1]
    revision = db_session.get(
        AssessmentRevision, UUID(rejected.detail["assessment_revision_id"])
    )
    # Recorded for audit through its provider result, not as an artifact.
    refs = revision.input_manifest["model_attempt_refs"]
    assert [ref.split(":")[0] for ref in refs] == ["result"]
    assert db_session.execute(select(func.count(ResearchArtifact.id))).scalar() == 0

    # The same evidence is asked again, so one bad response can recover.
    harness.go.queue_builder(claims_for)
    harness.request(key="refresh-1", kind="refresh")
    refreshed = harness.run_all()[-1]
    assert len(harness.go.requests) == 2
    revision = db_session.get(
        AssessmentRevision, UUID(refreshed.detail["assessment_revision_id"])
    )
    assert "verifier_output_rejected" not in {c["reason"] for c in revision.coverage}
    assert db_session.execute(select(func.count(ResearchArtifact.id))).scalar() == 1


@pytest.mark.parametrize(
    ("status", "outcome"),
    [(403, ("acquire", "paused")), (400, ("acquire", "completed"))],
)
def test_filing_fetch_gaps_pause_only_when_an_operator_can_fix_them(
    harness, status, outcome
):
    harness.serve_sec()
    harness.sec.serve_status(REPORT_URL, status)
    harness.request()
    harness.step()
    acquired = harness.step()
    # 403 is an access problem to fix and resume; a refused document (400)
    # is a permanent, omitted gap rather than a retry that fails the job.
    assert (acquired.stage, acquired.status) == outcome


def test_slow_io_renews_the_lease_instead_of_losing_the_stage(harness):
    harness.serve_sec()
    pace = harness.rate.acquire

    def slow_pace(*args, **kwargs):
        # Each SEC request takes 4 of the lease's 5 minutes.
        harness.clock.advance(minutes=4)
        return pace(*args, **kwargs)

    harness.rate.acquire = slow_pace
    harness.request()
    step = harness.step()
    assert (step.stage, step.status) == ("resolve_issuer", "completed")
    assert len(harness.rate.provider_names) >= 2


def test_worker_task_runs_leased_stages(harness, monkeypatch):
    harness.serve_sec()
    harness.go.queue_builder(claims_for)
    harness.request()
    harness.db.commit()
    monkeypatch.setattr(
        "app.services.company_exposure.config.load_config", lambda *_a, **_k: SHADOW
    )
    outcome = company_exposure_tasks.process_exposure_work.run(
        max_steps=5, runner_factory=lambda _session, _config: harness.runner
    )
    assert outcome["status"] == "completed"
    assert [s["stage"] for s in outcome["steps"]] == [
        "resolve_issuer",
        "acquire",
        "verify",
    ]


def test_theme_context_reads_latest_sealed_definition(db_session):
    from datetime import datetime, timezone

    from app.models.economic_taxonomy import (
        EconomicThemeAlias,
        EconomicThemeRevision,
        TaxonomyVersion,
    )
    from app.services.company_exposure.research import load_theme_context

    theme = make_theme(db_session, "context")
    assert load_theme_context(db_session, theme.id) is None
    version = TaxonomyVersion(status="draft", created_by="test", reason="seed")
    db_session.add(version)
    db_session.flush()
    db_session.add(
        EconomicThemeRevision(
            taxonomy_version_id=version.id,
            theme_id=theme.id,
            display_name="AI Memory",
            definition="Memory for AI accelerators",
            mechanism="HBM demand",
            lifecycle="established",
            lifecycle_policy_version="v1",
            created_by="test",
        )
    )
    db_session.flush()
    db_session.add(
        EconomicThemeAlias(
            taxonomy_version_id=version.id,
            theme_id=theme.id,
            alias="HBM",
            normalized_alias="hbm",
            created_by="test",
        )
    )
    db_session.flush()
    assert load_theme_context(db_session, theme.id) is None  # drafts are not used
    version.status = "sealed"
    version.sealed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db_session.flush()
    context = load_theme_context(db_session, theme.id)
    assert (context.label, context.terms) == ("AI Memory", ("AI Memory", "HBM"))
    assert len(context.fingerprint) == 64


def test_live_mode_is_not_installed_in_this_slice(harness):
    harness.build(replace(SHADOW, research_mode="live"))
    with pytest.raises(ResearchUnavailable, match="live_mode_not_installed"):
        harness.request()
