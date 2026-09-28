"""Date-effective metadata tests for breadth contributor snapshots."""

from __future__ import annotations

import importlib
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily


def _metadata_module():
    try:
        return importlib.import_module(
            "app.services.breadth.contributor_metadata"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"breadth contributor metadata loader is missing: {exc}")


def _database():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[FeatureRun.__table__, StockFeatureDaily.__table__],
    )
    return sessionmaker(bind=engine)()


def _feature_run(
    db,
    *,
    as_of_date: date,
    status: str,
    group: str,
    company_name: str,
    published_at: datetime | None = None,
) -> None:
    run = FeatureRun(
        as_of_date=as_of_date,
        run_type="daily_snapshot",
        status=status,
        config_json={"universe": {"market": "US"}},
        published_at=published_at,
        completed_at=datetime(2026, 8, 20, 20, tzinfo=UTC),
    )
    db.add(run)
    db.flush()
    db.add(
        StockFeatureDaily(
            run_id=run.id,
            symbol="AAA",
            as_of_date=as_of_date,
            details_json={
                "company_name": company_name,
                "ibd_industry_group": group,
            },
        )
    )
    db.commit()


def test_historical_metadata_uses_exact_date_published_run_and_no_newer_group():
    """Catches historical snapshots silently adopting a later classification."""
    db = _database()
    old_date = date(2026, 8, 20)
    _feature_run(
        db,
        as_of_date=old_date,
        status="completed",
        group="Completed Group",
        company_name="Alpha Completed",
    )
    _feature_run(
        db,
        as_of_date=old_date,
        status="published",
        group="Old Group",
        company_name="Alpha Ltd",
        published_at=datetime(2026, 8, 20, 22, tzinfo=UTC),
    )
    _feature_run(
        db,
        as_of_date=date(2026, 8, 21),
        status="published",
        group="New Group",
        company_name="Alpha Inc",
        published_at=datetime(2026, 8, 21, 22, tzinfo=UTC),
    )

    metadata = _metadata_module().BreadthContributorMetadataLoader.historical(
        db,
        "US",
        {old_date: ("AAA", "MISSING")},
    )

    assert metadata[old_date]["AAA"].company_name == "Alpha Ltd"
    assert metadata[old_date]["AAA"].ibd_industry_group == "Old Group"
    assert metadata[old_date]["MISSING"].company_name is None
    assert metadata[old_date]["MISSING"].ibd_industry_group == "No Group"


def test_historical_metadata_reads_extended_fallback_without_loading_feature_rows():
    """A US run's full details documents OOM-killed the 512M breadth worker."""
    db = _database()
    as_of = date(2026, 8, 20)
    run = FeatureRun(
        as_of_date=as_of,
        run_type="daily_snapshot",
        status="published",
        config_json={"universe": {"market": "US"}},
        published_at=datetime(2026, 8, 20, 22, tzinfo=UTC),
    )
    db.add(run)
    db.flush()
    db.add_all(
        [
            StockFeatureDaily(
                run_id=run.id,
                symbol="EXT",
                as_of_date=as_of,
                details_json={
                    "company_name": None,
                    "extended": {"company_name": "Ext Co", "ibd_industry_group": "Ext Group"},
                    "ibd_industry_group": "Top Group",
                },
            ),
            StockFeatureDaily(run_id=run.id, symbol="NUL", as_of_date=as_of, details_json=None),
            StockFeatureDaily(run_id=run.id, symbol="ARR", as_of_date=as_of, details_json=[1, 2]),
        ]
    )
    db.commit()
    reader = sessionmaker(bind=db.get_bind())()
    statements = []
    event.listen(
        db.get_bind(),
        "before_cursor_execute",
        lambda _conn, _cursor, sql, *_args: statements.append(sql),
    )

    metadata = _metadata_module().BreadthContributorMetadataLoader.historical(
        reader, "us", {as_of: ("ext", "NUL", "ARR")}
    )[as_of]

    assert (metadata["EXT"].company_name, metadata["EXT"].ibd_industry_group) == (
        "Ext Co",
        "Top Group",
    )
    for symbol in ("NUL", "ARR"):
        assert (metadata[symbol].company_name, metadata[symbol].ibd_industry_group) == (
            None,
            "No Group",
        )
    # Only extracted values leave the database, never the whole document.
    assert not any("details_json AS" in sql for sql in statements)
