from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models.company_exposure import IssuerSecurityLinkRevision
from scripts import company_exposure as cli
from tests.fixtures.company_exposure.research_harness import SHADOW, TICKERS, Harness

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNBOOK = REPO_ROOT / "docs/runbooks/company-exposure-map.md"


def test_runbook_covers_required_operations():
    text = RUNBOOK.read_text("utf-8")
    for required in (
        "mkdir -p ./data/exposure-evidence && sudo chown -R 1000:1000 ./data/exposure-evidence",
        "storage_not_writable",
        "review_required",
        "resolve-issuer",
        "official_registry_single_listing",
        "expired_uncertain",
        "tombstone",
        "EXPOSURE_RESEARCH_MODE=disabled",
        "shadow_preview",
        "not a live probe",
        "run_required_company_exposure_postgres.py",
    ):
        assert required in text, required


def test_documented_compose_commands_interpolate_from_the_production_env_file():
    # The overlay's ${EXPOSURE_*} values come from Compose's --env-file, not
    # from env_file: without it production would silently run "disabled".
    overlay = (REPO_ROOT / "docker-compose.exposure.yml").read_text("utf-8")
    for text in (RUNBOOK.read_text("utf-8"), overlay):
        command = text[text.index("docker-compose ") :].split("up -d")[0]
        assert "--env-file .env.docker" in command


def _run(capsys, argv, **kwargs):
    code = cli.main(argv, **kwargs)
    out = capsys.readouterr().out
    return code, json.loads(out) if out.strip() else None


@pytest.fixture
def cli_kwargs(db_session, tmp_path):
    return {
        "session_factory": lambda: db_session,
        "config_loader": lambda: replace(SHADOW, document_store=str(tmp_path)),
    }


def test_status_reports_activation_without_secrets(capsys, cli_kwargs):
    code, payload = _run(capsys, ["status"], **cli_kwargs)
    assert code == 0
    assert payload["activation"] == {
        "stage": "shadow_verify_us",
        "allowed": True,
        "reasons": [],
    }
    assert payload["configuration"]["subscription_key_present"] is True
    assert "api_key" not in json.dumps(payload["configuration"]).lower().replace(
        "subscription_key_present", ""
    )


def test_review_required_job_is_resolved_and_resumed(
    capsys, cli_kwargs, db_session, tmp_path, clock, monkeypatch
):
    harness = Harness(db_session, tmp_path, clock)
    harness.serve_sec()
    harness.sec.serve_json(
        "https://www.sec.gov/files/company_tickers_exchange.json",
        {
            "fields": TICKERS["fields"],
            "data": [*TICKERS["data"], [7654321, "Other", "EXMP", "NYSE"]],
        },
    )
    ref = harness.request()
    assert harness.step().state == "review_required"

    def links():
        return db_session.execute(
            select(func.count()).select_from(IssuerSecurityLinkRevision)
        ).scalar()

    before = links()
    args = [
        "resolve-issuer",
        "--security-id",
        str(harness.security.id),
        "--cik",
        "1234567",
    ]
    code, dry = _run(capsys, args, **cli_kwargs)
    assert (code, dry["state"]) == (0, "dry_run")
    assert links() == before

    from app.config import settings

    monkeypatch.setattr(settings, "admin_principal_id", "")
    code, blocked = _run(capsys, [*args, "--apply"], **cli_kwargs)
    assert (code, blocked["reason"]) == (2, "admin_principal_unbound")

    monkeypatch.setattr(settings, "admin_principal_id", "ops:alice")
    code, applied = _run(capsys, [*args, "--apply"], **cli_kwargs)
    assert (code, applied["state"]) == (0, "accepted")

    code, resumed = _run(capsys, ["resume", str(ref.id), "--apply"], **cli_kwargs)
    assert resumed == {"state": "queued", "previous_state": "review_required"}
    # Nothing is paused any more: a second resume must not re-announce the
    # job as queued, which would strand it with no claimable work.
    events = len(harness.requests.repo.events(ref.id))
    code, again = _run(capsys, ["resume", str(ref.id), "--apply"], **cli_kwargs)
    assert (code, again["state"]) == (2, "not_paused")
    assert len(harness.requests.repo.events(ref.id)) == events
    # The CLI resumes on the real clock; the harness claims on its own.
    harness.clock.advance_to(max(harness.clock.now(), datetime.now(timezone.utc)))
    step = harness.step()
    assert (step.stage, step.detail["source"]) == ("resolve_issuer", "accepted_link")


def test_cross_listing_resolves_to_the_issuer_owning_the_cik(
    capsys, cli_kwargs, db_session, tmp_path, clock, monkeypatch
):
    from app.config import settings
    from app.domain.company_exposure.contracts import (
        SERVICE_PRINCIPAL,
        RegistryMatch,
    )
    from app.services.company_exposure.issuer_identity import IssuerIdentityAdapter
    from tests.fixtures.company_exposure.factory import make_security

    harness = Harness(db_session, tmp_path, clock)
    harness.serve_sec()
    # Another listing of the same company already owns CIK 1234567.
    twin = make_security(db_session, "EXMPB", exchange="NYSE")
    identity = IssuerIdentityAdapter(db_session)
    owner = identity.accept_registry_match(
        RegistryMatch(
            security_id=twin.id,
            market="US",
            scheme="cik",
            value="1234567",
            candidate_count=1,
            ticker_confirmed=True,
            matched_ticker="EXMPB",
            matched_exchange="NYSE",
            registry_capture_revision_id=None,
            official_record_capture_revision_id=None,
        ),
        SERVICE_PRINCIPAL,
    ).issuer_id
    ref = harness.request()
    step = harness.step()
    assert step.state == "review_required"

    args = ["resolve-issuer", "--security-id", str(harness.security.id)]
    args += ["--cik", "1234567"]
    code, dry = _run(capsys, args, **cli_kwargs)
    assert (code, dry["issuer_id"]) == (0, str(owner))

    monkeypatch.setattr(settings, "admin_principal_id", "ops:alice")
    code, applied = _run(capsys, [*args, "--apply"], **cli_kwargs)
    assert (code, applied["state"], applied["issuer_id"]) == (0, "accepted", str(owner))
    assert identity.resolve_security(harness.security.id).issuer_id == owner
    code, resumed = _run(capsys, ["resume", str(ref.id), "--apply"], **cli_kwargs)
    assert resumed["state"] == "queued"


def test_process_reports_disabled_research_as_blocked(capsys, cli_kwargs):
    code, payload = _run(
        capsys,
        ["process"],
        **cli_kwargs,
        worker=lambda max_steps: {"status": "skipped", "reason": "research_disabled"},
    )
    assert (code, payload["reason"]) == (2, "research_disabled")


def test_bounded_arguments(capsys, cli_kwargs):
    assert cli.main(["process", "--max-steps", "99"], **cli_kwargs) == 64
    assert (
        cli.main(
            ["resolve-issuer", "--security-id", "1", "--cik", "12ab"], **cli_kwargs
        )
        == 64
    )


def test_refresh_holds_is_dry_run_by_default(capsys, cli_kwargs):
    code, payload = _run(capsys, ["refresh-holds"], **cli_kwargs)
    assert (code, payload["applied"], payload["new_holds"]) == (0, False, 0)
