from datetime import UTC, date, datetime

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.infra.db.models.feature_store import (
    FeatureRun,
    FeatureRunPointer,
    FeatureRunUniverseSymbol,
    StockFeatureDaily,
)
from app.models.industry import IBDIndustryGroup
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse
from app.services.group_matrix_service import GroupMatrixService


@pytest.fixture(name="db")
def matrix_db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(
        engine,
        tables=[
            m.__table__
            for m in (
                FeatureRun,
                FeatureRunPointer,
                FeatureRunUniverseSymbol,
                StockFeatureDaily,
                StockFundamental,
                StockUniverse,
                IBDIndustryGroup,
            )
        ],
    )
    with Session(engine) as session:
        yield session
    engine.dispose()


def add_run(db, run_id, market, symbols, status="published"):
    run = FeatureRun(
        id=run_id,
        as_of_date=date(2026, 9, 11),
        run_type="daily_snapshot",
        status=status,
        published_at=datetime(2026, 9, 12, tzinfo=UTC),
        config_json={"universe": {"market": market}},
    )
    db.add(run)
    for symbol in symbols:
        db.add(FeatureRunUniverseSymbol(run_id=run_id, symbol=symbol))
        db.add(
            StockFeatureDaily(
                run_id=run_id,
                symbol=symbol,
                as_of_date=run.as_of_date,
                details_json={
                    "ibd_industry_group": "Local taxonomy",
                    "gics_sector": "Tech",
                    "price_change_1d": 2,
                    "perf_week": -1.25,
                    "perf_month": 9.5,
                    "rs_rating": 81,
                },
            )
        )
    db.commit()
    return run


def test_market_identity_actual_ibd_and_missing_caps(db):
    add_run(db, 1, "HK", ["A.HK", "B.HK"])
    add_run(db, 2, "US", ["US"])
    add_run(db, 3, "HK", ["NEW.HK"], status="running")
    db.add_all(
        [
            IBDIndustryGroup(
                symbol="A.HK",
                market="HK",
                industry_group="Software",
                source="crosswalk",
            ),
            IBDIndustryGroup(symbol="B.HK", market="US", industry_group="Wrong market"),
            StockUniverse(symbol="A.HK", market="HK", market_cap=100_000_000_000),
        ]
    )
    db.commit()
    result = GroupMatrixService().build(
        db, market="HK", generated_at="2026-09-13T00:00:00Z"
    )
    assert result["feature_run_id"] == 1
    assert result["stocks"][0]["ibd_industry_group"] == "Software"
    assert result["stocks"][0]["classification_source"] == "crosswalk"
    assert result["stocks"][0]["cap_tier"] == "unknown"
    assert result["stocks"][1]["ibd_industry_group"] is None
    assert result["stocks"][0]["price_change_1d"] == 2
    assert result["stocks"][0]["price_change_1w"] == -1.25
    assert result["stocks"][0]["price_change_1m"] == 9.5
    assert result["metadata_basis"] == "latest_stored"
    assert result["coverage"]["ibd_mapped_count"] == 1


def test_selected_run_is_not_replaced_and_mutable_cap_is_explicit(db):
    add_run(db, 1, "US", ["A"])
    add_run(db, 2, "US", ["B"])
    db.add_all(
        [
            IBDIndustryGroup(symbol="A", market="US", industry_group="Software"),
            StockFundamental(symbol="A", market_cap_usd=2_000_000_000),
        ]
    )
    db.commit()
    result = GroupMatrixService().build(
        db, market="US", feature_run_id=1, generated_at="now"
    )
    assert result["feature_run_id"] == 1
    assert result["stocks"][0]["cap_tier"] == "mid"
    assert result["stocks"][0]["fundamentals_updated_at"] is not None
    assert (
        GroupMatrixService().build(
            db, market="HK", feature_run_id=1, generated_at="now"
        )["reason"]
        == "publication_identity_mismatch"
    )


def test_partial_identity_and_unpublished_runs_fail_closed(db):
    run = add_run(db, 1, "US", ["A"])
    run.config_json = {
        "universe": {"market": "US"},
        "rs_formula_version": "balanced-quartile-v1",
    }
    # Use the actual canonical identifier so partial metadata is rejected.
    from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION

    run.config_json = {
        **run.config_json,
        "rs_formula_version": BALANCED_RS_FORMULA_VERSION,
    }
    db.commit()
    assert (
        GroupMatrixService().build(db, market="US", generated_at="now")["reason"]
        == "publication_identity_mismatch"
    )
    assert (
        GroupMatrixService().build(db, market="JP", generated_at="now")["reason"]
        == "no_published_run"
    )


def test_all_groups_loaded_in_constant_queries(db):
    symbols = [f"S{i}" for i in range(80)]
    add_run(db, 1, "US", symbols)
    db.add_all(
        [
            IBDIndustryGroup(symbol=s, market="US", industry_group=f"Group {s}")
            for s in symbols
        ]
    )
    db.commit()
    statements = []

    def record(*args):
        statements.append(args[2])

    event.listen(db.bind, "before_cursor_execute", record)
    try:
        result = GroupMatrixService().build(
            db, market="US", feature_run_id=1, generated_at="now"
        )
    finally:
        event.remove(db.bind, "before_cursor_execute", record)
    assert len(result["stocks"]) == 80
    assert len(statements) <= 4


def test_metadata_timestamp_tracks_the_read_not_export_start(db, monkeypatch):
    from datetime import datetime, timezone
    from unittest.mock import Mock

    import app.services.group_matrix_service as service_module

    add_run(db, 1, "US", ["A"])
    db.add(IBDIndustryGroup(symbol="A", market="US", industry_group="Software"))
    db.commit()
    service = GroupMatrixService()
    clock = Mock()
    clock.now.return_value = datetime(2026, 9, 13, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(service_module, "datetime", clock, raising=False)
    load_rows = service.repository.load_rows

    def read(*args, **kwargs):
        rows = load_rows(*args, **kwargs)
        clock.now.return_value = datetime(2026, 9, 13, 2, tzinfo=timezone.utc)
        return rows

    monkeypatch.setattr(service.repository, "load_rows", read)
    result = service.build(db, market="US", generated_at="2026-09-13T00:00:00Z")
    assert result["generated_at"] == "2026-09-13T00:00:00Z"
    assert result["metadata_read_at"] == "2026-09-13T02:00:00+00:00"


def test_detail_values_keep_json_types_and_documentless_rows_are_listed(db):
    from app.services.group_matrix_repository import GroupMatrixRepository

    add_run(db, 1, "US", ["A", "B", "C"])
    rows = {row.symbol: row for row in db.query(StockFeatureDaily)}
    rows["B"].details_json = None
    rows["C"].details_json = {"gics_sector": "", "rs_rating": "81", "perf_week": 1}
    db.add(StockUniverse(symbol="C", market="US", sector="Fallback"))
    db.commit()

    loaded = {
        row["symbol"]: row
        for row in GroupMatrixRepository().load_rows(db, run_id=1, market="US")
    }

    assert loaded["A"]["price_change_1d"] == 2
    assert isinstance(loaded["A"]["price_change_1d"], int)
    assert loaded["A"]["sector"] == "Tech"
    assert {
        key: loaded["B"][key]
        for key in (
            "sector",
            "price_change_1d",
            "price_change_1w",
            "price_change_1m",
            "rs_rating",
        )
    } == dict.fromkeys(
        ("sector", "price_change_1d", "price_change_1w", "price_change_1m", "rs_rating")
    )
    # Values arrive exactly as stored; the payload builder does the rejecting.
    assert loaded["C"]["rs_rating"] == "81"
    assert loaded["C"]["sector"] == "Fallback"
    assert loaded["C"]["price_change_1d"] is None
    assert loaded["C"]["price_change_1w"] == 1


@pytest.mark.parametrize("postgres", [True, False])
def test_matrix_query_never_selects_whole_details_documents(postgres):
    from sqlalchemy.dialects import postgresql, sqlite

    from app.services.group_matrix_repository import matrix_rows_statement

    statement = matrix_rows_statement(run_id=1, market="US", postgres=postgres)
    sql = str(
        statement.compile(dialect=postgresql.dialect() if postgres else sqlite.dialect())
    )

    details_column = StockFeatureDaily.__table__.c.details_json
    assert all(
        selected is not details_column for selected in statement.selected_columns
    )
    assert "stock_feature_daily.details_json AS" not in sql
    if postgres:
        # json, not jsonb: each -> would re-parse the whole document, so the
        # five values come from one json_to_record parse per row.
        assert sql.count("json_to_record(") == 1
        assert "->" not in sql
        assert "LEFT OUTER JOIN json_to_record(CASE WHEN" in sql
