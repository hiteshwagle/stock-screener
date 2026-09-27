"""
Breadth Calculator Service for calculating market breadth indicators.

Calculates StockBee-style breadth metrics across all active stocks
in the universe, including daily movers, multi-period ratios, and
monthly/quarterly performance indicators.
"""

import logging
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session

from ..models.market_breadth import MarketBreadth
from .breadth.contributor_metadata import BreadthContributorMetadataLoader
from .breadth.engine import BreadthEngine
from .breadth.formulas import (
    BREADTH_FEATURE_WARMUP_SESSIONS,
    prices_for_feature_window,
    validate_price_frame,
)
from .breadth.market_policy import get_breadth_market_policy
from .breadth.persistence import BreadthPersistence
from .breadth.types import (
    CURRENT_BREADTH_CALCULATION_REVISION,
    BreadthContributorSnapshotResult,
    BreadthDailyCount,
    BreadthDailyResult,
    BreadthEligibilityCounts,
    BreadthIndicatorValues,
)
from .breadth.universe import build_breadth_universe_snapshots
from .breadth_backfill import (
    PRICE_BATCH_SIZE,
    BreadthBackfillExecutor,
    BreadthBackfillPlan,
)
from .breadth_coverage import (
    BreadthCalculationResult,
    BreadthCoverageReport,
    BreadthOutcomeCounter,
    BreadthPriceCoverageAccumulator,
)
from .derived_data_execution_policy import (
    DerivedDataExecutionMode,
    DerivedDataExecutionPolicy,
    DerivedDataTargetKind,
)
from .point_in_time_universe_service import PointInTimeUniverseService
from .price_cache_service import PriceCacheService

logger = logging.getLogger(__name__)
DEFAULT_BREADTH_EXECUTION_POLICY = DerivedDataExecutionPolicy.provider_allowed()


class BreadthCalculatorService:
    """
    Service for calculating market breadth indicators.

    Processes the point-in-time broad universe and calculates:
    - Daily 4%+ movers (up and down)
    - Today-inclusive 5-session and 10-session up/down ratios
    - Monthly 25%/50% movers versus exactly 20 sessions ago
    - Quarterly 25% movers versus trailing 65-session extrema
    - 34-day 13% movers versus trailing 34-session extrema
    - Advance/decline, 52-week high/low, T2108, and ATR extension context

    Each metric family applies its own history and data eligibility in the
    shared engine; there is no all-or-nothing history gate.
    """

    def __init__(
        self,
        db: Session,
        price_cache: PriceCacheService,
        market: str | None = None,
        *,
        engine: BreadthEngine | None = None,
    ):
        """
        Initialize breadth calculator service.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.price_cache = price_cache
        self.market = (market or "US").upper()
        self.market_policy = get_breadth_market_policy(self.market)
        self.engine = engine or BreadthEngine()
        self.persistence = BreadthPersistence(db)

    def calculate_daily_breadth(
        self,
        calculation_date: date | None = None,
        *,
        policy: DerivedDataExecutionPolicy = DEFAULT_BREADTH_EXECUTION_POLICY,
    ) -> BreadthCalculationResult:
        """
        Calculate and return all market breadth indicators for a given date.

        Args:
            calculation_date: Date to calculate breadth for (defaults to today)

        Returns:
            BreadthCalculationResult containing ``indicators`` and the
            authoritative ``coverage`` report. Use ``to_metrics_dict()`` when
            a merged persistence mapping is required. Task responses add
            execution-policy metadata at their serialization boundary.
        """
        if calculation_date is None:
            from ..utils.market_hours import get_eastern_now

            calculation_date = get_eastern_now().date()

        logger.info("Calculating breadth indicators for %s", calculation_date)
        snapshot = build_breadth_universe_snapshots(
            self.db,
            self.market,
            (calculation_date,),
        )[calculation_date]
        members = snapshot.members
        symbols = [member.symbol for member in members]
        history_period = (
            self._history_period_for_dates(
                (calculation_date,),
                cache_anchor_date=datetime.now(UTC).date(),
            )
            if symbols
            else "2y"
        )

        accumulator = self.engine.accumulator(
            market=self.market,
            dates=(calculation_date,),
            universes_by_date={calculation_date: snapshot},
            market_policy=self.market_policy,
        )
        price_coverage = BreadthPriceCoverageAccumulator()
        outcomes = BreadthOutcomeCounter()
        for offset in range(0, len(symbols), PRICE_BATCH_SIZE):
            batch_symbols = symbols[offset : offset + PRICE_BATCH_SIZE]
            loaded, cache_misses = self._load_price_data_for_batch(
                batch_symbols=batch_symbols,
                cache_only=policy.cache_only,
                required_as_of_date=(calculation_date if policy.cache_only else None),
                period=history_period,
            )
            price_coverage.record_batch(batch_symbols, cache_misses)
            usable: dict[str, pd.DataFrame] = {}
            for symbol in batch_symbols:
                history = loaded.get(symbol)
                if history is None or history.empty:
                    outcomes.record_cache_miss()
                    continue
                try:
                    validate_price_frame(history)
                except ValueError:
                    outcomes.record_error()
                    continue
                if not self._has_usable_target_session(history, calculation_date):
                    outcomes.record_insufficient()
                    continue
                usable[symbol] = history
                outcomes.record_scanned()
            # Reduce this batch to per-date counts before loading the next, so
            # only one batch of price histories is alive at a time.
            accumulator.add_prices(
                self._prices_for_feature_window(usable, (calculation_date,))
            )
            del loaded, usable

        seeds = self._load_ratio_seed_counts(calculation_date, limit=9)
        contributor_metadata_available = True
        try:
            if policy.target_kind is DerivedDataTargetKind.HISTORICAL:
                contributor_metadata = BreadthContributorMetadataLoader.historical(
                    self.db,
                    self.market,
                    {calculation_date: symbols},
                )[calculation_date]
            else:
                contributor_metadata = BreadthContributorMetadataLoader.current(
                    self.db,
                    self.market,
                    symbols,
                )
        except Exception as exc:  # noqa: BLE001 - metadata is optional for aggregate breadth.
            self.db.rollback()
            logger.warning(
                "Breadth contributor metadata unavailable for %s: %s",
                self.market,
                exc,
            )
            contributor_metadata = {}
            contributor_metadata_available = False
        batch = accumulator.finish(
            seed_counts=seeds,
            contributor_metadata_by_date={calculation_date: contributor_metadata},
        )
        canonical = batch.daily_results[calculation_date]
        coverage_report = BreadthCoverageReport.from_parts(
            price_coverage.report(),
            outcomes.report(),
        )
        metrics = canonical.to_record_mapping()
        metrics.pop("date", None)
        metrics.pop("market", None)
        return BreadthCalculationResult(
            indicators=metrics,
            coverage=coverage_report,
            daily_result=canonical,
            contributor_snapshot=(
                batch.contributor_snapshots[calculation_date]
                if contributor_metadata_available
                else None
            ),
        )

    @staticmethod
    def _has_usable_target_session(
        history: pd.DataFrame,
        calculation_date: date,
    ) -> bool:
        positions = [
            position
            for position, value in enumerate(history.index)
            if pd.Timestamp(value).date() == calculation_date
        ]
        if not positions:
            return False
        try:
            value = float(history["Adj Close"].iloc[positions[-1]])
        except (TypeError, ValueError):
            return False
        return math.isfinite(value) and value > 0

    def _history_period_for_dates(
        self,
        calculation_dates: tuple[date, ...],
        *,
        cache_anchor_date: date,
    ) -> str:
        if not calculation_dates:
            return "2y"
        from ..wiring.bootstrap import get_market_calendar_service

        calendar = get_market_calendar_service()
        first_calculation_date = min(calculation_dates)
        anchor_date = first_calculation_date
        if not calendar.is_trading_day(self.market, anchor_date):
            sessions: list[date] = []
            for search_days in (30, 90, 370):
                sessions = calendar.trading_days(
                    self.market,
                    anchor_date - timedelta(days=search_days),
                    anchor_date,
                )
                if sessions:
                    break
            if not sessions:
                raise ValueError(
                    f"No {self.market} trading session available on or before "
                    f"{anchor_date.isoformat()}"
                )
            anchor_date = sessions[-1]
        warmup_start = calendar.session_anchors(
            self.market,
            anchor_date,
            offsets=(BREADTH_FEATURE_WARMUP_SESSIONS,),
        )[BREADTH_FEATURE_WARMUP_SESSIONS]
        required_calendar_days = (cache_anchor_date - warmup_start).days
        if required_calendar_days <= 730:
            return "2y"
        if required_calendar_days <= 1825:
            return "5y"
        return "max"

    @staticmethod
    def _prices_for_feature_window(
        prices_by_symbol: Mapping[str, pd.DataFrame],
        calculation_dates: tuple[date, ...],
    ) -> dict[str, pd.DataFrame]:
        return prices_for_feature_window(prices_by_symbol, calculation_dates)

    def _load_ratio_seed_counts(
        self,
        calculation_date: date,
        *,
        limit: int,
    ) -> tuple[BreadthDailyCount, ...]:
        records = (
            self.db.query(MarketBreadth)
            .filter(
                MarketBreadth.date < calculation_date,
                MarketBreadth.market == self.market,
                MarketBreadth.calculation_revision
                == CURRENT_BREADTH_CALCULATION_REVISION,
            )
            .order_by(MarketBreadth.date.desc())
            .limit(limit)
            .all()
        )
        return tuple(
            BreadthDailyCount(
                date=record.date,
                stocks_up_4pct=record.stocks_up_4pct,
                stocks_down_4pct=record.stocks_down_4pct,
                market=self.market,
                calculation_revision=record.calculation_revision,
            )
            for record in reversed(records)
            if getattr(record, "calculation_revision", None)
            == CURRENT_BREADTH_CALCULATION_REVISION
        )

    def _load_ratio_context_counts(
        self,
        calculation_dates: list[date],
    ) -> tuple[BreadthDailyCount, ...]:
        if not calculation_dates:
            return ()
        requested = set(calculation_dates)
        prior = list(self._load_ratio_seed_counts(min(calculation_dates), limit=9))
        intervening = (
            self.db.query(MarketBreadth)
            .filter(
                MarketBreadth.date >= min(calculation_dates),
                MarketBreadth.date <= max(calculation_dates),
                MarketBreadth.market == self.market,
                MarketBreadth.calculation_revision
                == CURRENT_BREADTH_CALCULATION_REVISION,
            )
            .order_by(MarketBreadth.date.asc())
            .all()
        )
        prior.extend(
            BreadthDailyCount(
                date=record.date,
                stocks_up_4pct=record.stocks_up_4pct,
                stocks_down_4pct=record.stocks_down_4pct,
                market=self.market,
                calculation_revision=record.calculation_revision,
            )
            for record in intervening
            if record.date not in requested
            and getattr(record, "calculation_revision", None)
            == CURRENT_BREADTH_CALCULATION_REVISION
        )
        return tuple(sorted(prior, key=lambda item: item.date))

    def backfill_range(
        self,
        start_date: date,
        end_date: date,
        trading_dates: list[date] | None = None,
        *,
        policy: DerivedDataExecutionPolicy = DEFAULT_BREADTH_EXECUTION_POLICY,
        cache_only: bool | None = None,
        exclude_unsupported_price_symbols: bool = False,
        required_as_of_date: date | None = None,
        eligible_symbols_by_date: Mapping[date, tuple[str, ...]] | None = None,
        eligibility_signatures_by_date: Mapping[date, str] | None = None,
    ) -> dict:
        """Calculate and persist breadth for an entire historical range."""
        if cache_only is not None:
            policy = DerivedDataExecutionPolicy(
                mode=(
                    DerivedDataExecutionMode.STRICT_CACHE_ONLY
                    if cache_only
                    else DerivedDataExecutionMode.AUTO
                ),
                target_kind=DerivedDataTargetKind.HISTORICAL,
            )

        if trading_dates is None:
            from ..wiring.bootstrap import get_market_calendar_service

            calendar_service = get_market_calendar_service()
            trading_dates = [
                current_date
                for current_date in pd.date_range(
                    start=start_date,
                    end=end_date,
                    freq="D",
                ).date
                if calendar_service.is_trading_day(self.market, current_date)
            ]

        plan = BreadthBackfillPlan.from_legacy(
            dates=trading_dates,
            eligible_symbols_by_date=eligible_symbols_by_date,
            eligibility_signatures_by_date=eligibility_signatures_by_date,
        )
        if not plan.dates:
            return {
                "total_dates": 0,
                "processed": 0,
                "errors": 0,
                "error_dates": [],
            }

        return (
            BreadthBackfillExecutor(self)
            .execute(
                plan,
                policy=policy,
                exclude_unsupported_price_symbols=exclude_unsupported_price_symbols,
                required_as_of_date=required_as_of_date,
            )
            .to_legacy_dict()
        )

    def _load_price_data_for_batch(
        self,
        batch_symbols: list[str],
        cache_only: bool,
        *,
        required_as_of_date: date | None = None,
        period: str = "2y",
    ) -> tuple[dict[str, pd.DataFrame | None], list[str]]:
        """Load batch price histories once, with optional cache misses fetched a single time."""
        cache_kwargs = {"period": period}
        if required_as_of_date is not None:
            cache_kwargs["required_as_of_date"] = required_as_of_date
        if cache_only:
            cache_kwargs["minimum_rows"] = 1
        price_data_by_symbol = self.price_cache.get_many_cached_only_fresh(
            batch_symbols,
            **cache_kwargs,
        )
        cache_miss_symbols: list[str] = []

        if cache_only:
            for symbol in batch_symbols:
                price_history = price_data_by_symbol.get(symbol)
                if price_history is None or price_history.empty:
                    cache_miss_symbols.append(symbol)
            return price_data_by_symbol, cache_miss_symbols

        for symbol in batch_symbols:
            price_history = price_data_by_symbol.get(symbol)
            if price_history is None or price_history.empty:
                cache_miss_symbols.append(symbol)
                price_data_by_symbol[symbol] = self._calculate_stock_history(
                    symbol,
                    period=period,
                )

        return price_data_by_symbol, cache_miss_symbols

    def _calculate_stock_history(
        self,
        symbol: str,
        *,
        period: str = "2y",
    ) -> pd.DataFrame | None:
        """Fetch a symbol's full historical data once for reuse."""
        return self.price_cache.get_historical_data(
            symbol=symbol,
            period=period,
        )

    def find_missing_dates(
        self,
        lookback_days: int = 30,
        *,
        end_date: date | None = None,
    ) -> list[date]:
        """
        Find missing trading dates in the market_breadth table.

        Checks the lookback window for weekdays (excluding holidays)
        that don't have breadth records.

        Args:
            lookback_days: Number of days to look back for gaps

        Returns:
            List of missing trading dates (oldest first)
        """
        from sqlalchemy import func

        from ..wiring.bootstrap import get_market_calendar_service

        calendar_service = get_market_calendar_service()
        window_end = end_date or calendar_service.market_now(self.market).date()
        start_date = window_end - timedelta(days=lookback_days)

        # Get all dates that have breadth data for this market
        existing_dates = (
            self.db.query(func.distinct(MarketBreadth.date))
            .filter(
                MarketBreadth.date >= start_date,
                MarketBreadth.market == self.market,
            )
            .all()
        )

        existing_date_set = {d[0] for d in existing_dates}

        # Dates before any symbol was first seen resolve to an empty
        # point-in-time universe. They are unavailable rather than missing:
        # requesting them again every run can never produce a row.
        membership_floor = PointInTimeUniverseService().earliest_membership_date(
            self.db,
            market=self.market,
        )
        if membership_floor is None:
            logger.info(
                "No %s universe history exists; no breadth gaps to fill",
                self.market,
            )
            return []
        if membership_floor > start_date:
            logger.info(
                "Skipping %s breadth dates before %s: no point-in-time universe",
                self.market,
                membership_floor,
            )
            start_date = membership_floor

        # Generate all trading days in range using the per-market calendar
        missing_dates = []
        current_date = start_date

        while (
            current_date < window_end
        ):  # Exclude the target day; it is calculated separately.
            if (
                calendar_service.is_trading_day(self.market, current_date)
                and current_date not in existing_date_set
            ):
                missing_dates.append(current_date)
            current_date += timedelta(days=1)

        logger.info(
            f"Found {len(missing_dates)} missing breadth dates in last {lookback_days} days"
        )
        return sorted(missing_dates)

    def fill_gaps(
        self,
        missing_dates: list[date],
        *,
        policy: DerivedDataExecutionPolicy = DEFAULT_BREADTH_EXECUTION_POLICY,
    ) -> dict:
        """
        Fill gaps by calculating breadth for missing dates.

        Processes dates oldest first to ensure ratio calculations
        have prior data available.

        Args:
            missing_dates: List of dates to calculate breadth for

        Returns:
            Statistics about the gap-fill operation:
            {
                'total_dates': int,
                'processed': int,
                'errors': int,
                'error_dates': List[str]
            }
        """
        if not missing_dates:
            return {"total_dates": 0, "processed": 0, "errors": 0, "error_dates": []}

        ordered_dates = sorted(missing_dates)
        logger.info(f"Filling {len(ordered_dates)} missing breadth dates")
        stats = self.backfill_range(
            ordered_dates[0],
            ordered_dates[-1],
            trading_dates=ordered_dates,
            policy=policy,
        )

        logger.info(
            f"Gap-fill complete: {stats['processed']} processed, "
            f"{stats['errors']} errors"
        )

        return stats

    def store_daily_breadth(
        self,
        calculation_date: date,
        metrics: Mapping[str, object] | BreadthDailyResult,
        *,
        duration_seconds: float,
    ) -> None:
        result = (
            metrics
            if isinstance(metrics, BreadthDailyResult)
            else self._result_from_mapping(calculation_date, metrics)
        )
        self.persistence.upsert_daily(result, duration_seconds=duration_seconds)

    def store_daily_result(
        self,
        result: BreadthDailyResult,
        *,
        contributor_snapshot: BreadthContributorSnapshotResult | None = None,
        duration_seconds: float,
    ) -> None:
        self.persistence.upsert_daily(
            result,
            contributor_snapshot=contributor_snapshot,
            duration_seconds=duration_seconds,
        )

    def _result_from_mapping(
        self,
        calculation_date: date,
        metrics: Mapping[str, object],
    ) -> BreadthDailyResult:
        value_fields = BreadthIndicatorValues.__dataclass_fields__
        eligibility_fields = BreadthEligibilityCounts.__dataclass_fields__
        values = BreadthIndicatorValues(
            **{
                name: metrics.get(name, field.default)
                for name, field in value_fields.items()
            }
        )
        eligibility = BreadthEligibilityCounts(
            **{
                name: int(metrics.get(name, field.default) or 0)
                for name, field in eligibility_fields.items()
            }
        )
        broad_count = int(
            metrics.get(
                "broad_universe_count",
                metrics.get("total_stocks_scanned", 0),
            )
            or 0
        )
        return BreadthDailyResult(
            market=self.market,
            calculation_date=calculation_date,
            values=values,
            eligibility=eligibility,
            broad_universe_count=broad_count,
            eligibility_signature=str(metrics.get("eligibility_signature") or ""),
            stockbee_eligibility_signature=str(
                metrics.get("stockbee_eligibility_signature") or ""
            ),
        )
