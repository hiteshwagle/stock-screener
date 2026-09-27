from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.services.breadth_backfill as breadth_backfill_module
from app.database import Base
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.breadth_contributor import (
    MarketBreadthContributor,
    MarketBreadthContributorSnapshot,
)
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import UNIVERSE_STATUS_ACTIVE, StockUniverse
from app.services.breadth.contributors import contributor_calculation_signature
from app.services.breadth.types import BreadthUniverseMember, BreadthUniverseSnapshot
from app.services.breadth.universe import breadth_eligibility_signature
from app.services.breadth_backfill import (
    BreadthBackfillExecutor,
    BreadthBackfillPlan,
    BreadthContributorBackfillIncomplete,
)
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    DerivedDataExecutionPolicy,
    DerivedDataTargetKind,
)
from app.services.static_breadth_eligibility import (
    static_breadth_eligibility_signature,
)


def _flat_price_df(
    end_date: date,
    close: float = 100.0,
    periods: int = 80,
) -> pd.DataFrame:
    index = pd.bdate_range(end=end_date, periods=periods)
    closes = [close] * len(index)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Adj Close": closes,
            "Volume": [1_000_000] * len(index),
        },
        index=index,
    )


def _make_db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            StockUniverse.__table__,
            MarketBreadth.__table__,
            MarketBreadthContributorSnapshot.__table__,
            MarketBreadthContributor.__table__,
            FeatureRun.__table__,
            StockFeatureDaily.__table__,
        ],
    )
    return sessionmaker(bind=engine)()


@pytest.fixture(autouse=True)
def _use_current_rows_as_default_backfill_universe(monkeypatch):
    """Keep legacy unit tests focused on calculation rather than lifecycle data."""

    def snapshots(db, market, dates):
        rows = sorted(
            db.query(StockUniverse)
            .filter(
                StockUniverse.market == market,
                StockUniverse.is_active == True,
                StockUniverse.is_common_stock.is_(True),
            )
            .all(),
            key=lambda row: row.symbol,
        )
        members = tuple(BreadthUniverseMember(row.symbol, row.currency) for row in rows)
        signature = breadth_eligibility_signature(member.symbol for member in members)
        return {
            calculation_date: BreadthUniverseSnapshot(
                calculation_date=calculation_date,
                members=members,
                broad_signature=signature,
            )
            for calculation_date in dates
        }

    monkeypatch.setattr(
        breadth_backfill_module,
        "build_breadth_universe_snapshots",
        snapshots,
    )


def test_backfill_plan_canonicalizes_symbols_and_derives_signature():
    calculation_date = date(2026, 3, 20)
    expected_signature = static_breadth_eligibility_signature(("AAA", "BBB"))

    plan = BreadthBackfillPlan.from_legacy(
        dates=[calculation_date],
        eligible_symbols_by_date={
            calculation_date: ("BBB", "AAA", "AAA"),
        },
        eligibility_signatures_by_date={
            calculation_date: expected_signature,
        },
    )

    universe = plan.universe_for(calculation_date)
    assert universe is not None
    assert universe.symbols == ("AAA", "BBB")
    assert universe.eligibility_signature == expected_signature


@pytest.mark.parametrize(
    ("eligible_symbols_by_date", "eligibility_signatures_by_date"),
    [
        ({date(2026, 3, 20): ("AAA",)}, None),
        (None, {date(2026, 3, 20): "signature"}),
    ],
)
def test_backfill_plan_rejects_half_supplied_legacy_contract(
    eligible_symbols_by_date,
    eligibility_signatures_by_date,
):
    with pytest.raises(ValueError, match="must be supplied together"):
        BreadthBackfillPlan.from_legacy(
            dates=[date(2026, 3, 20)],
            eligible_symbols_by_date=eligible_symbols_by_date,
            eligibility_signatures_by_date=eligibility_signatures_by_date,
        )


def test_backfill_plan_rejects_missing_requested_date():
    with pytest.raises(ValueError, match="missing for 2026-03-21"):
        BreadthBackfillPlan.from_legacy(
            dates=[date(2026, 3, 20), date(2026, 3, 21)],
            eligible_symbols_by_date={
                date(2026, 3, 20): ("AAA",),
            },
            eligibility_signatures_by_date={
                date(2026, 3, 20): static_breadth_eligibility_signature(("AAA",)),
            },
        )


def test_backfill_plan_rejects_signature_for_different_symbols():
    calculation_date = date(2026, 3, 20)

    with pytest.raises(ValueError, match="signature does not match"):
        BreadthBackfillPlan.from_legacy(
            dates=[calculation_date],
            eligible_symbols_by_date={calculation_date: ("AAA",)},
            eligibility_signatures_by_date={
                calculation_date: static_breadth_eligibility_signature(("BBB",)),
            },
        )


def test_backfill_plan_without_explicit_eligibility_preserves_legacy_mode():
    calculation_date = date(2026, 3, 20)

    plan = BreadthBackfillPlan.from_legacy(
        dates=[calculation_date],
        eligible_symbols_by_date=None,
        eligibility_signatures_by_date=None,
    )

    assert plan.dates == (calculation_date,)
    assert plan.universe_for(calculation_date) is None


def test_backfill_without_explicit_universes_resolves_membership_per_date(
    monkeypatch,
):
    db = _make_db_session()
    first_date = date(2026, 3, 19)
    second_date = date(2026, 3, 20)
    db.add_all(
        [
            StockUniverse(
                symbol="AAA",
                market="US",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
            StockUniverse(
                symbol="RECENT",
                market="US",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
            ),
        ]
    )
    db.commit()

    def point_in_time_snapshots(_db, market, dates):
        assert market == "US"
        assert tuple(dates) == (first_date, second_date)
        return {
            first_date: BreadthUniverseSnapshot(
                calculation_date=first_date,
                members=(BreadthUniverseMember("AAA", "USD"),),
                broad_signature="first",
            ),
            second_date: BreadthUniverseSnapshot(
                calculation_date=second_date,
                members=(
                    BreadthUniverseMember("AAA", "USD"),
                    BreadthUniverseMember("RECENT", "USD"),
                ),
                broad_signature="second",
            ),
        }

    monkeypatch.setattr(
        breadth_backfill_module,
        "build_breadth_universe_snapshots",
        point_in_time_snapshots,
        raising=False,
    )
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _flat_price_df(second_date),
        "RECENT": _flat_price_df(second_date),
    }

    result = BreadthCalculatorService(db, price_cache).backfill_range(
        first_date,
        second_date,
        trading_dates=[first_date, second_date],
    )

    assert result["processed"] == 2
    stored = {
        row.date: row.broad_universe_count
        for row in db.query(MarketBreadth)
        .filter(MarketBreadth.date.in_((first_date, second_date)))
        .all()
    }
    assert stored == {first_date: 1, second_date: 2}
    assert db.query(MarketBreadthContributorSnapshot).count() == 2


def test_contributor_only_backfill_never_updates_aggregate_rows():
    db = _make_db_session()
    calculation_date = date(2026, 3, 20)
    db.add(
        StockUniverse(
            symbol="AAA",
            name="Alpha",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            is_common_stock=True,
        )
    )
    db.add(
        MarketBreadth(
            market="US",
            date=calculation_date,
            calculation_revision=3,
            stocks_up_4pct=0,
            stocks_down_4pct=0,
            stocks_up_25pct_quarter=0,
            stocks_down_25pct_quarter=0,
            stocks_up_25pct_month=0,
            stocks_down_25pct_month=0,
            stocks_up_50pct_month=0,
            stocks_down_50pct_month=0,
            stocks_up_13pct_34days=0,
            stocks_down_13pct_34days=0,
            atr_10x_extension_count=0,
            contributor_calculation_signature=(
                contributor_calculation_signature(())
            ),
            total_stocks_scanned=1,
        )
    )
    db.commit()
    aggregate = db.query(MarketBreadth).one()
    aggregate_id = aggregate.id
    original_created_at = aggregate.created_at
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _flat_price_df(calculation_date),
    }
    calculator = BreadthCalculatorService(db, price_cache, market="US")
    policy = DerivedDataExecutionPolicy(
        mode=DerivedDataExecutionMode.STRICT_CACHE_ONLY,
        target_kind=DerivedDataTargetKind.HISTORICAL,
    )

    result = (
        BreadthBackfillExecutor(calculator)
        .execute(
            BreadthBackfillPlan(dates=(calculation_date,)),
            policy=policy,
            require_complete_cache_coverage=True,
            contributor_only=True,
        )
        .to_legacy_dict()
    )

    db.expire_all()
    aggregate = db.query(MarketBreadth).one()
    assert result["processed"] == 1
    assert aggregate.id == aggregate_id
    assert aggregate.created_at == original_created_at
    assert aggregate.total_stocks_scanned == 1
    assert db.query(MarketBreadthContributorSnapshot).count() == 1


def test_contributor_only_backfill_fails_when_metadata_source_is_unavailable(
    monkeypatch,
):
    db = _make_db_session()
    calculation_date = date(2026, 3, 20)
    db.add(
        StockUniverse(
            symbol="AAA",
            name="Alpha",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            is_common_stock=True,
        )
    )
    db.add(
        MarketBreadth(
            market="US",
            date=calculation_date,
            calculation_revision=3,
            stocks_up_4pct=0,
            stocks_down_4pct=0,
            stocks_up_25pct_quarter=0,
            stocks_down_25pct_quarter=0,
            stocks_up_25pct_month=0,
            stocks_down_25pct_month=0,
            stocks_up_50pct_month=0,
            stocks_down_50pct_month=0,
            stocks_up_13pct_34days=0,
            stocks_down_13pct_34days=0,
            atr_10x_extension_count=0,
            total_stocks_scanned=1,
        )
    )
    db.commit()
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _flat_price_df(calculation_date),
    }
    monkeypatch.setattr(
        breadth_backfill_module.BreadthContributorMetadataLoader,
        "historical",
        MagicMock(side_effect=RuntimeError("metadata database unavailable")),
    )
    calculator = BreadthCalculatorService(db, price_cache, market="US")

    with pytest.raises(
        BreadthContributorBackfillIncomplete,
        match="metadata",
    ):
        BreadthBackfillExecutor(calculator).execute(
            BreadthBackfillPlan(dates=(calculation_date,)),
            policy=DerivedDataExecutionPolicy(
                mode=DerivedDataExecutionMode.STRICT_CACHE_ONLY,
                target_kind=DerivedDataTargetKind.HISTORICAL,
            ),
            require_complete_cache_coverage=True,
            contributor_only=True,
        )

    assert db.query(MarketBreadthContributorSnapshot).count() == 0


def test_aggregate_backfill_recovers_session_after_metadata_failure(monkeypatch):
    db = _make_db_session()
    calculation_date = date(2026, 3, 20)
    db.add(
        StockUniverse(
            symbol="AAA",
            name="Alpha",
            market="US",
            currency="USD",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
            is_common_stock=True,
        )
    )
    db.commit()
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _flat_price_df(calculation_date),
    }
    calculator = BreadthCalculatorService(db, price_cache, market="US")
    transaction = {"aborted": False, "rollbacks": 0}
    original_rollback = db.rollback

    def fail_metadata(*_args, **_kwargs):
        transaction["aborted"] = True
        raise RuntimeError("metadata database unavailable")

    def recover_session():
        transaction["rollbacks"] += 1
        transaction["aborted"] = False
        original_rollback()

    def load_ratio_context(_processed_dates):
        assert transaction["aborted"] is False, "session was not recovered"
        return []

    monkeypatch.setattr(
        breadth_backfill_module.BreadthContributorMetadataLoader,
        "historical",
        fail_metadata,
    )
    monkeypatch.setattr(db, "rollback", recover_session)
    monkeypatch.setattr(
        calculator,
        "_load_ratio_context_counts",
        load_ratio_context,
    )

    result = BreadthBackfillExecutor(calculator).execute(
        BreadthBackfillPlan(dates=(calculation_date,)),
        policy=DerivedDataExecutionPolicy(
            mode=DerivedDataExecutionMode.STRICT_CACHE_ONLY,
            target_kind=DerivedDataTargetKind.HISTORICAL,
        ),
        require_complete_cache_coverage=True,
    )

    assert result.to_legacy_dict()["processed"] == 1
    assert transaction["rollbacks"] == 1
    assert db.query(MarketBreadth).count() == 1
    assert db.query(MarketBreadthContributorSnapshot).count() == 0


def test_implicit_backfill_loads_members_at_final_requested_membership_date(
    monkeypatch,
):
    db = _make_db_session()
    first_date = date(2026, 3, 19)
    second_date = date(2026, 3, 20)

    def point_in_time_snapshots(_db, market, dates):
        assert market == "US"
        assert tuple(dates) == (first_date, second_date)
        return {
            first_date: BreadthUniverseSnapshot(
                calculation_date=first_date,
                members=(BreadthUniverseMember("DELISTED", "USD"),),
                broad_signature="first",
            ),
            second_date: BreadthUniverseSnapshot(
                calculation_date=second_date,
                members=(BreadthUniverseMember("CURRENT", "USD"),),
                broad_signature="second",
            ),
        }

    monkeypatch.setattr(
        breadth_backfill_module,
        "build_breadth_universe_snapshots",
        point_in_time_snapshots,
        raising=False,
    )
    expected_date_by_symbol = {
        "DELISTED": first_date,
        "CURRENT": second_date,
    }

    def cached_prices(symbols, period, *, required_as_of_date=None):
        assert period == "2y"
        return {
            symbol: (
                _flat_price_df(expected_date_by_symbol[symbol])
                if required_as_of_date == expected_date_by_symbol[symbol]
                else None
            )
            for symbol in symbols
        }

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.side_effect = cached_prices
    price_cache.get_historical_data.return_value = None

    result = BreadthCalculatorService(db, price_cache).backfill_range(
        first_date,
        second_date,
        trading_dates=[first_date, second_date],
    )

    assert result["processed"] == 2
    assert result["error_dates"] == []


def test_backfill_range_reuses_loaded_histories_and_computes_chronological_ratios(
    monkeypatch,
):
    db = _make_db_session()
    db.add_all(
        [
            StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
            StockUniverse(symbol="BBB", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        ]
    )

    prior_dates = [date(2026, 3, day) for day in range(2, 12)]
    for prior_date in prior_dates:
        db.add(
            MarketBreadth(
                date=prior_date,
                stocks_up_4pct=2,
                stocks_down_4pct=1,
                ratio_5day=2.0,
                ratio_10day=2.0,
                stocks_up_25pct_quarter=0,
                stocks_down_25pct_quarter=0,
                stocks_up_25pct_month=0,
                stocks_down_25pct_month=0,
                stocks_up_50pct_month=0,
                stocks_down_50pct_month=0,
                stocks_up_13pct_34days=0,
                stocks_down_13pct_34days=0,
                total_stocks_scanned=2,
                broad_universe_count=2,
                calculation_revision=3,
            )
        )
    db.commit()

    aaa_df = _flat_price_df(date(2026, 3, 13))
    bbb_df = _flat_price_df(date(2026, 3, 13))
    aaa_df.loc[pd.Timestamp(date(2026, 3, 12)), ["Close", "Adj Close"]] = 105.0
    aaa_df.loc[pd.Timestamp(date(2026, 3, 13)), ["Close", "Adj Close"]] = 110.0
    bbb_df.loc[pd.Timestamp(date(2026, 3, 12)), ["Close", "Adj Close"]] = 95.0
    bbb_df.loc[pd.Timestamp(date(2026, 3, 13)), ["Close", "Adj Close"]] = 100.0
    for frame in (aaa_df, bbb_df):
        frame.loc[pd.Timestamp(date(2026, 3, 12)), "Volume"] = 1_100_000
        frame.loc[pd.Timestamp(date(2026, 3, 13)), "Volume"] = 1_200_000

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": aaa_df, "BBB": None}
    price_cache.get_historical_data.return_value = bbb_df
    service = BreadthCalculatorService(db, price_cache)
    trading_dates = [date(2026, 3, 12), date(2026, 3, 13)]

    result = service.backfill_range(
        trading_dates[0], trading_dates[-1], trading_dates=trading_dates
    )

    assert result == {
        "total_dates": 2,
        "processed": 2,
        "errors": 0,
        "error_dates": [],
    }
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA", "BBB"],
        period="2y",
        required_as_of_date=trading_dates[-1],
    )
    price_cache.get_many_cached_only.assert_not_called()
    price_cache.get_historical_data.assert_called_once_with(symbol="BBB", period="2y")

    stored = (
        db.query(MarketBreadth)
        .filter(MarketBreadth.date.in_(trading_dates))
        .order_by(MarketBreadth.date.asc())
        .all()
    )

    assert len(stored) == 2
    assert stored[0].stocks_up_4pct == 1
    assert stored[0].stocks_down_4pct == 1
    assert stored[0].ratio_5day == 1.8
    assert stored[0].ratio_10day == 1.9
    assert stored[1].stocks_up_4pct == 2
    assert stored[1].stocks_down_4pct == 0
    assert stored[1].ratio_5day == 2.25
    assert stored[1].ratio_10day == 2.11


def test_backfill_range_excludes_explicit_non_common_instruments():
    db = _make_db_session()
    calculation_date = date(2026, 3, 20)
    db.add_all(
        [
            StockUniverse(
                symbol="AAA",
                market="US",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                is_common_stock=True,
            ),
            StockUniverse(
                symbol="SPY",
                market="US",
                is_active=True,
                status=UNIVERSE_STATUS_ACTIVE,
                is_common_stock=False,
            ),
        ]
    )
    db.commit()
    price_cache = MagicMock()
    histories = {
        "AAA": _flat_price_df(calculation_date),
        "SPY": _flat_price_df(calculation_date),
    }
    price_cache.get_many_cached_only_fresh.side_effect = (
        lambda symbols, period, *, required_as_of_date: {
            symbol: histories[symbol]
            for symbol in symbols
            if required_as_of_date == calculation_date
        }
    )
    service = BreadthCalculatorService(db, price_cache)

    service.backfill_range(
        calculation_date,
        calculation_date,
        trading_dates=[calculation_date],
    )

    stored = (
        db.query(MarketBreadth).filter(MarketBreadth.date == calculation_date).one()
    )
    assert stored.broad_universe_count == 1
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA"],
        period="2y",
        required_as_of_date=calculation_date,
    )


def test_backfill_range_scans_only_explicit_date_specific_eligible_symbols():
    db = _make_db_session()
    first_date = date(2026, 3, 19)
    second_date = date(2026, 3, 20)
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _flat_price_df(second_date),
        "BBB": _flat_price_df(second_date),
    }
    service = BreadthCalculatorService(db, price_cache)
    aaa_signature = static_breadth_eligibility_signature(("AAA",))
    bbb_signature = static_breadth_eligibility_signature(("BBB",))

    result = service.backfill_range(
        first_date,
        second_date,
        trading_dates=[first_date, second_date],
        eligible_symbols_by_date={
            first_date: ("AAA",),
            second_date: ("BBB",),
        },
        eligibility_signatures_by_date={
            first_date: aaa_signature,
            second_date: bbb_signature,
        },
    )

    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA", "BBB"], period="2y"
    )
    assert result["eligible_stocks_by_date"] == {
        "2026-03-19": 1,
        "2026-03-20": 1,
    }
    assert result["scanned_stocks_by_date"] == {
        "2026-03-19": 1,
        "2026-03-20": 1,
    }
    assert result["broad_universe_stocks_by_date"] == {
        "2026-03-19": 1,
        "2026-03-20": 1,
    }
    assert result["calculation_errors_by_date"] == {
        "2026-03-19": 0,
        "2026-03-20": 0,
    }
    stored = {
        row.date: row.eligibility_signature
        for row in db.query(MarketBreadth).filter(
            MarketBreadth.date.in_([first_date, second_date])
        )
    }
    assert stored == {first_date: aaa_signature, second_date: bbb_signature}


def test_backfill_range_validates_historical_symbols_on_their_eligible_date():
    db = _make_db_session()
    historical_date = date(2026, 3, 19)
    current_date = date(2026, 3, 20)
    eligible_date_by_symbol = {
        "CURRENT": current_date,
        "HISTORICAL": historical_date,
    }

    def cached_prices(symbols, period, *, required_as_of_date, minimum_rows):
        assert period == "2y"
        assert minimum_rows == 1
        return {
            symbol: (
                _flat_price_df(eligible_date_by_symbol[symbol])
                if required_as_of_date == eligible_date_by_symbol[symbol]
                else None
            )
            for symbol in symbols
        }

    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.side_effect = cached_prices
    service = BreadthCalculatorService(db, price_cache)

    result = service.backfill_range(
        historical_date,
        current_date,
        trading_dates=[historical_date, current_date],
        cache_only=True,
        required_as_of_date=current_date,
        eligible_symbols_by_date={
            historical_date: ("HISTORICAL",),
            current_date: ("CURRENT",),
        },
        eligibility_signatures_by_date={
            historical_date: static_breadth_eligibility_signature(("HISTORICAL",)),
            current_date: static_breadth_eligibility_signature(("CURRENT",)),
        },
    )

    assert result["scanned_stocks_by_date"] == {
        "2026-03-19": 1,
        "2026-03-20": 1,
    }
    assert result["broad_universe_stocks_by_date"] == {
        "2026-03-19": 1,
        "2026-03-20": 1,
    }


def test_backfill_range_reports_malformed_history_on_every_applicable_date():
    db = _make_db_session()
    first_date = date(2026, 3, 19)
    second_date = date(2026, 3, 20)
    malformed = _flat_price_df(second_date).drop(columns=["Adj Close"])
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": malformed}
    service = BreadthCalculatorService(db, price_cache)
    signature = static_breadth_eligibility_signature(("AAA",))

    result = service.backfill_range(
        first_date,
        second_date,
        trading_dates=[first_date, second_date],
        cache_only=True,
        eligible_symbols_by_date={
            first_date: ("AAA",),
            second_date: ("AAA",),
        },
        eligibility_signatures_by_date={
            first_date: signature,
            second_date: signature,
        },
    )

    assert result["calculation_errors_by_date"] == {
        "2026-03-19": 1,
        "2026-03-20": 1,
    }


def test_backfill_range_preserves_existing_failed_date_in_ratio_context():
    db = _make_db_session()
    failed_date = date(2026, 3, 19)
    processed_date = date(2026, 3, 20)
    for seed_date in pd.bdate_range(end="2026-03-18", periods=9).date:
        db.add(
            MarketBreadth(
                market="US",
                date=seed_date,
                stocks_up_4pct=1,
                stocks_down_4pct=1,
                total_stocks_scanned=1,
                broad_universe_count=1,
                calculation_revision=3,
            )
        )
    db.add(
        MarketBreadth(
            market="US",
            date=failed_date,
            stocks_up_4pct=0,
            stocks_down_4pct=10,
            total_stocks_scanned=10,
            broad_universe_count=10,
            calculation_revision=3,
        )
    )
    db.commit()

    history = _flat_price_df(processed_date)
    history = history.drop(index=pd.Timestamp(failed_date))
    history.loc[pd.Timestamp(processed_date), ["Close", "Adj Close"]] = 105.0
    history.loc[pd.Timestamp(processed_date), "Volume"] = 1_100_000
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": history}
    service = BreadthCalculatorService(db, price_cache)
    signature = static_breadth_eligibility_signature(("AAA",))

    result = service.backfill_range(
        failed_date,
        processed_date,
        trading_dates=[failed_date, processed_date],
        cache_only=True,
        eligible_symbols_by_date={
            failed_date: ("AAA",),
            processed_date: ("AAA",),
        },
        eligibility_signatures_by_date={
            failed_date: signature,
            processed_date: signature,
        },
    )

    stored = db.query(MarketBreadth).filter(MarketBreadth.date == processed_date).one()
    assert result["error_dates"] == ["2026-03-19"]
    assert stored.ratio_5day == 0.31
    assert stored.ratio_10day == 0.5


def test_strict_backfill_rejects_date_with_supported_missing_target_session():
    db = _make_db_session()
    missing_date = date(2026, 3, 19)
    complete_date = date(2026, 3, 20)
    complete_history = _flat_price_df(complete_date)
    incomplete_history = complete_history.drop(index=pd.Timestamp(missing_date))
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "COMPLETE": complete_history,
        "INCOMPLETE": incomplete_history,
    }
    service = BreadthCalculatorService(db, price_cache)
    symbols = ("COMPLETE", "INCOMPLETE", "WARRANT-W")
    signature = static_breadth_eligibility_signature(symbols)
    plan = BreadthBackfillPlan.from_legacy(
        dates=(missing_date, complete_date),
        eligible_symbols_by_date={
            missing_date: symbols,
            complete_date: symbols,
        },
        eligibility_signatures_by_date={
            missing_date: signature,
            complete_date: signature,
        },
    )

    result = (
        BreadthBackfillExecutor(service)
        .execute(
            plan,
            policy=DerivedDataExecutionPolicy(
                mode=DerivedDataExecutionMode.STRICT_CACHE_ONLY,
                target_kind=DerivedDataTargetKind.HISTORICAL,
            ),
            exclude_unsupported_price_symbols=True,
            required_as_of_date=complete_date,
            require_complete_cache_coverage=True,
        )
        .to_legacy_dict()
    )

    assert result["processed"] == 1
    assert result["error_dates"] == [missing_date.isoformat()]
    assert [row.date for row in db.query(MarketBreadth).all()] == [complete_date]


def test_backfill_range_rejects_signature_for_different_eligible_symbols():
    db = _make_db_session()
    calculation_date = date(2026, 3, 20)
    service = BreadthCalculatorService(db, MagicMock())

    with pytest.raises(ValueError, match="signature does not match"):
        service.backfill_range(
            calculation_date,
            calculation_date,
            trading_dates=[calculation_date],
            eligible_symbols_by_date={calculation_date: ("AAA",)},
            eligibility_signatures_by_date={calculation_date: "wrong"},
        )


def _walk_df(end_date: date, seed: int, periods: int = 320) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end_date, periods=periods)
    close = 40.0 * np.exp(np.cumsum(rng.normal(0.0, 0.03, len(index))))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.03,
            "Low": close * 0.97,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(500_000, 3_000_000, len(index)).astype(float),
        },
        index=index,
    )


_STREAM_SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE")


def _run_streaming_backfill(monkeypatch, trading_dates, *, batch_size):
    """Backfill five symbols; return stored rows and add_prices batch sizes."""
    monkeypatch.setattr(breadth_backfill_module, "PRICE_BATCH_SIZE", batch_size)
    db = _make_db_session()
    db.add_all(
        StockUniverse(symbol=symbol, is_active=True, status=UNIVERSE_STATUS_ACTIVE)
        for symbol in _STREAM_SYMBOLS
    )
    db.commit()
    frames = {
        symbol: _walk_df(date(2026, 3, 13), seed)
        for seed, symbol in enumerate(_STREAM_SYMBOLS)
    }
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.side_effect = lambda symbols, **_: {
        symbol: frames[symbol] for symbol in symbols
    }
    service = BreadthCalculatorService(db, price_cache)
    accumulators = []
    make_accumulator = service.engine.accumulator

    def recording_accumulator(**kwargs):
        accumulator = make_accumulator(**kwargs)
        add_prices = accumulator.add_prices
        sizes = []

        def add(batch):
            sizes.append(len(batch))
            add_prices(batch)

        accumulator.add_prices = add
        accumulators.append((kwargs["dates"], sizes))
        return accumulator

    monkeypatch.setattr(service.engine, "accumulator", recording_accumulator)
    result = service.backfill_range(
        trading_dates[0], trading_dates[-1], trading_dates=trading_dates
    )
    rows = {
        row.date: {
            column.name: getattr(row, column.name)
            for column in MarketBreadth.__table__.columns
            if column.name not in {"id", "created_at", "updated_at"}
            and "duration" not in column.name
        }
        for row in db.query(MarketBreadth).all()
    }
    return result, rows, accumulators


def test_backfill_streams_prices_one_batch_at_a_time(monkeypatch):
    trading_dates = [date(2026, 3, 11), date(2026, 3, 12), date(2026, 3, 13)]

    streamed, streamed_rows, accumulators = _run_streaming_backfill(
        monkeypatch, trading_dates, batch_size=2
    )
    whole, whole_rows, _ = _run_streaming_backfill(
        monkeypatch, trading_dates, batch_size=500
    )

    assert streamed == whole
    assert streamed["processed"] == 3
    assert streamed_rows == whole_rows
    [(dates, sizes)] = accumulators
    assert dates == tuple(trading_dates)
    assert sizes == [2, 2, 1]


def test_backfill_reevaluates_when_leading_dates_are_not_processed(monkeypatch):
    # No cached session exists on the Saturday, so it cannot be processed and
    # the feature warm-up must be re-anchored at the first processed date.
    saturday = date(2026, 3, 7)
    processed = [date(2026, 3, 12), date(2026, 3, 13)]

    result, rows, accumulators = _run_streaming_backfill(
        monkeypatch, [saturday, *processed], batch_size=2
    )
    direct, direct_rows, direct_accumulators = _run_streaming_backfill(
        monkeypatch, processed, batch_size=2
    )

    assert result["error_dates"] == [saturday.isoformat()]
    assert [dates for dates, _ in accumulators] == [
        (saturday, *processed),
        tuple(processed),
    ]
    assert [dates for dates, _ in direct_accumulators] == [tuple(processed)]
    assert rows == direct_rows
    assert set(rows) == set(processed)
