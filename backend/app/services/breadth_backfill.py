"""Typed planning and execution boundary for historical breadth backfills."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..domain.providers.price_symbol_support import split_supported_price_symbols
from ..models.stock_universe import StockUniverse
from .breadth.contributor_metadata import BreadthContributorMetadataLoader
from .breadth.formulas import validate_price_frame
from .breadth.types import BreadthUniverseMember, BreadthUniverseSnapshot
from .breadth.universe import build_breadth_universe_snapshots
from .breadth_coverage import (
    BreadthCoverageReport,
    BreadthOutcomeCounter,
    BreadthOutcomeReport,
    BreadthPriceCoverageAccumulator,
)
from .derived_data_execution_policy import DerivedDataExecutionPolicy
from .point_in_time_universe_service import PointInTimeUniverseUnavailable
from .static_breadth_eligibility import static_breadth_eligibility_signature

if TYPE_CHECKING:
    from .breadth_calculator_service import BreadthCalculatorService


logger = logging.getLogger(__name__)
# Price histories are loaded, evaluated and released this many symbols at a
# time; holding a whole US universe of frames at once exhausted worker memory.
PRICE_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class BreadthEligibleUniverse:
    """The exact eligible universe and provenance for one calculation date."""

    calculation_date: date
    symbols: tuple[str, ...]
    eligibility_signature: str


@dataclass(frozen=True, slots=True)
class BreadthBackfillPlan:
    """Validated dates and optional explicit universes for one backfill."""

    dates: tuple[date, ...]
    universes: Mapping[date, BreadthEligibleUniverse] | None = None

    @classmethod
    def from_legacy(
        cls,
        *,
        dates: Sequence[date],
        eligible_symbols_by_date: Mapping[date, Sequence[str]] | None,
        eligibility_signatures_by_date: Mapping[date, str] | None,
    ) -> BreadthBackfillPlan:
        ordered_dates = tuple(sorted(set(dates)))
        has_symbols = eligible_symbols_by_date is not None
        has_signatures = eligibility_signatures_by_date is not None
        if has_symbols != has_signatures:
            raise ValueError(
                "eligible symbols and eligibility signatures must be supplied together"
            )
        if not has_symbols:
            return cls(dates=ordered_dates)

        assert eligible_symbols_by_date is not None
        assert eligibility_signatures_by_date is not None
        universes: dict[date, BreadthEligibleUniverse] = {}
        for calculation_date in ordered_dates:
            if calculation_date not in eligible_symbols_by_date:
                raise ValueError(
                    f"eligible symbols missing for {calculation_date.isoformat()}"
                )
            if calculation_date not in eligibility_signatures_by_date:
                raise ValueError(
                    f"eligibility signature missing for {calculation_date.isoformat()}"
                )
            symbols = tuple(sorted(set(eligible_symbols_by_date[calculation_date])))
            expected_signature = static_breadth_eligibility_signature(symbols)
            supplied_signature = eligibility_signatures_by_date[calculation_date]
            if supplied_signature != expected_signature:
                raise ValueError(
                    "eligibility signature does not match canonical symbols for "
                    f"{calculation_date.isoformat()}"
                )
            universes[calculation_date] = BreadthEligibleUniverse(
                calculation_date=calculation_date,
                symbols=symbols,
                eligibility_signature=expected_signature,
            )
        return cls(
            dates=ordered_dates,
            universes=MappingProxyType(universes),
        )

    def universe_for(
        self,
        calculation_date: date,
    ) -> BreadthEligibleUniverse | None:
        if self.universes is None:
            return None
        return self.universes[calculation_date]


@dataclass(frozen=True, slots=True)
class BreadthBackfillResult:
    values: Mapping[str, Any]

    def to_legacy_dict(self) -> dict[str, Any]:
        return dict(self.values)


class BreadthContributorBackfillIncomplete(RuntimeError):
    """Raised before persistence when any contributor backfill date is incomplete."""


def _history_fingerprint(history: pd.DataFrame) -> int:
    """Cheap identity of a price history, to detect a changed re-read."""
    return int(pd.util.hash_pandas_object(history, index=True).sum())


def _first_session_on_or_after(window: pd.DataFrame, first_date: date) -> date | None:
    for value in window.index:
        session = pd.Timestamp(value).date()
        if session >= first_date:
            return session
    return None


class BreadthBackfillExecutor:
    """Execute one validated historical breadth plan."""

    def __init__(self, calculator: BreadthCalculatorService) -> None:
        self._calculator = calculator

    def _replay_processed_window(
        self,
        batches,
        *,
        fingerprints: Mapping[str, int],
        processed_dates: tuple[date, ...],
        universes_by_date: Mapping[date, BreadthUniverseSnapshot],
    ):
        """Re-evaluate the first pass's histories with the processed window.

        Returns the new accumulator, or ``None`` when any history could not be
        re-read identically (the caller then keeps its first-pass results).
        """
        calculator = self._calculator
        accumulator = calculator.engine.accumulator(
            market=calculator.market,
            dates=processed_dates,
            universes_by_date=universes_by_date,
            market_policy=calculator.market_policy,
        )
        seen: set[str] = set()
        for batch_symbols, _, valid, _ in batches:
            for symbol in batch_symbols:
                history = valid.get(symbol)
                if history is None or _history_fingerprint(history) != fingerprints[symbol]:
                    logger.warning(
                        "Breadth warm-up replay for %s abandoned: cached history "
                        "for %s differs from the first pass; keeping the "
                        "first-pass window",
                        calculator.market,
                        symbol,
                    )
                    return None
                seen.add(symbol)
            accumulator.add_prices(
                calculator._prices_for_feature_window(valid, processed_dates)
            )
        if seen != set(fingerprints):
            return None
        return accumulator

    def execute(
        self,
        plan: BreadthBackfillPlan,
        *,
        policy: DerivedDataExecutionPolicy,
        exclude_unsupported_price_symbols: bool = False,
        required_as_of_date: date | None = None,
        require_complete_cache_coverage: bool = False,
        contributor_only: bool = False,
    ) -> BreadthBackfillResult:
        return self._execute_canonical(
            plan,
            policy=policy,
            exclude_unsupported_price_symbols=exclude_unsupported_price_symbols,
            required_as_of_date=required_as_of_date,
            require_complete_cache_coverage=require_complete_cache_coverage,
            contributor_only=contributor_only,
        )

    def _execute_canonical(
        self,
        plan: BreadthBackfillPlan,
        *,
        policy: DerivedDataExecutionPolicy,
        exclude_unsupported_price_symbols: bool,
        required_as_of_date: date | None,
        require_complete_cache_coverage: bool,
        contributor_only: bool,
    ) -> BreadthBackfillResult:
        calculator = self._calculator
        ordered_dates = list(plan.dates)
        started_at = datetime.now(UTC)
        explicit_symbols = (
            {
                calculation_date: plan.universe_for(calculation_date).symbols
                for calculation_date in ordered_dates
            }
            if plan.universes is not None
            else None
        )

        unavailable_dates: list[date] = []
        if explicit_symbols is None:
            # Membership is never fabricated: a date whose historical universe
            # cannot be reproduced, or had no members yet, is reported
            # unavailable instead of failing the range or counting as an error.
            try:
                resolved = dict(
                    build_breadth_universe_snapshots(
                        calculator.db,
                        calculator.market,
                        ordered_dates,
                    )
                )
            except PointInTimeUniverseUnavailable:
                resolved = {}
                for calculation_date in ordered_dates:
                    try:
                        resolved.update(
                            build_breadth_universe_snapshots(
                                calculator.db,
                                calculator.market,
                                (calculation_date,),
                            )
                        )
                    except PointInTimeUniverseUnavailable as exc:
                        logger.info("Breadth universe unavailable: %s", exc)
            universes_by_date = {}
            for calculation_date in ordered_dates:
                snapshot = resolved.get(calculation_date)
                if snapshot is None or not snapshot.members:
                    unavailable_dates.append(calculation_date)
                else:
                    universes_by_date[calculation_date] = snapshot
            if contributor_only and unavailable_dates:
                # Contributor backfills must cover every requested date or
                # write nothing; an unavailable date is incomplete coverage.
                raise BreadthContributorBackfillIncomplete(
                    "Contributor backfill has no point-in-time universe for: "
                    + ",".join(value.isoformat() for value in unavailable_dates)
                )
            ordered_dates = [
                calculation_date
                for calculation_date in ordered_dates
                if calculation_date in universes_by_date
            ]
            symbols_by_date = {
                calculation_date: tuple(
                    member.symbol
                    for member in universes_by_date[calculation_date].members
                )
                for calculation_date in ordered_dates
            }
            target_symbols = sorted(
                {symbol for symbols in symbols_by_date.values() for symbol in symbols}
            )
            currency_by_symbol = {
                member.symbol: member.currency
                for snapshot in universes_by_date.values()
                for member in snapshot.members
            }
        else:
            target_symbols = sorted(
                {symbol for symbols in explicit_symbols.values() for symbol in symbols}
            )
            stock_rows = (
                calculator.db.query(StockUniverse)
                .filter(StockUniverse.symbol.in_(target_symbols))
                .all()
                if target_symbols
                else []
            )
            rows_by_symbol = {row.symbol: row for row in stock_rows}
            symbols_by_date = dict(explicit_symbols)
            currency_by_symbol = {
                symbol: (
                    getattr(rows_by_symbol.get(symbol), "currency", None)
                    or calculator.market_policy.currency
                )
                for symbol in target_symbols
            }
            universes_by_date: dict[date, BreadthUniverseSnapshot] = {}
            for calculation_date in ordered_dates:
                symbols = tuple(sorted(symbols_by_date[calculation_date]))
                supplied = plan.universe_for(calculation_date)
                assert supplied is not None
                universes_by_date[calculation_date] = BreadthUniverseSnapshot(
                    calculation_date=calculation_date,
                    members=tuple(
                        BreadthUniverseMember(symbol, currency_by_symbol[symbol])
                        for symbol in symbols
                    ),
                    broad_signature=supplied.eligibility_signature,
                )

        price_symbols = target_symbols
        skipped_unsupported_symbols: list[str] = []
        if exclude_unsupported_price_symbols:
            price_symbols, skipped_unsupported_symbols = split_supported_price_symbols(
                target_symbols
            )
        unsupported_symbols = set(skipped_unsupported_symbols)

        price_coverage = BreadthPriceCoverageAccumulator()
        history_period = (
            calculator._history_period_for_dates(
                tuple(ordered_dates),
                cache_anchor_date=datetime.now(UTC).date(),
            )
            if price_symbols
            else "2y"
        )
        dates_by_symbol: dict[str, list[date]] = {}
        for calculation_date in ordered_dates:
            for symbol in symbols_by_date[calculation_date]:
                dates_by_symbol.setdefault(symbol, []).append(calculation_date)

        def load_batch(
            batch_symbols: list[str],
            *,
            cache_only: bool = policy.cache_only,
        ) -> tuple[dict[str, Any], list[str]]:
            if explicit_symbols is None or required_as_of_date is not None:
                grouped: dict[date, list[str]] = {}
                for symbol in batch_symbols:
                    grouped.setdefault(max(dates_by_symbol[symbol]), []).append(symbol)
                loaded: dict[str, Any] = {}
                cache_misses: list[str] = []
                for symbol_date, symbols in grouped.items():
                    group_kwargs: dict[str, Any] = {
                        "required_as_of_date": symbol_date,
                    }
                    if history_period != "2y":
                        group_kwargs["period"] = history_period
                    group_prices, group_misses = calculator._load_price_data_for_batch(
                        batch_symbols=symbols,
                        cache_only=cache_only,
                        **group_kwargs,
                    )
                    loaded.update(group_prices)
                    cache_misses.extend(group_misses)
                return loaded, cache_misses
            kwargs: dict[str, Any] = (
                {"required_as_of_date": required_as_of_date}
                if required_as_of_date is not None
                else {}
            )
            if history_period != "2y":
                kwargs["period"] = history_period
            return calculator._load_price_data_for_batch(
                batch_symbols=batch_symbols,
                cache_only=cache_only,
                **kwargs,
            )

        def price_batches(symbols: list[str], *, cache_only: bool = policy.cache_only):
            """Yield each batch's valid, non-empty histories, then drop them."""
            for offset in range(0, len(symbols), PRICE_BATCH_SIZE):
                batch_symbols = symbols[offset : offset + PRICE_BATCH_SIZE]
                loaded, cache_misses = load_batch(batch_symbols, cache_only=cache_only)
                valid: dict[str, Any] = {}
                invalid: set[str] = set()
                for symbol in batch_symbols:
                    history = loaded.get(symbol)
                    if history is None or history.empty:
                        continue
                    try:
                        validate_price_frame(history)
                    except ValueError:
                        invalid.add(symbol)
                        continue
                    valid[symbol] = history
                yield batch_symbols, cache_misses, valid, invalid

        # One streaming pass records coverage outcomes and evaluates every
        # planned date; only one batch of price histories is alive at a time.
        # The feature warm-up is anchored at the first planned date. It must
        # end up anchored at the first *processed* date (the recursive ATR
        # depends on where its window starts), so the pass also keeps, per
        # valid symbol, a fingerprint of the history it used, plus the
        # earliest in-window session: see the replay below.
        fingerprints: dict[str, int] = {}
        earliest_session: date | None = None
        outcomes_by_date = {
            calculation_date: BreadthOutcomeCounter()
            for calculation_date in ordered_dates
        }
        incomplete_target_session_dates: set[date] = set()
        accumulator = calculator.engine.accumulator(
            market=calculator.market,
            dates=tuple(ordered_dates),
            universes_by_date=universes_by_date,
            market_policy=calculator.market_policy,
        )
        for batch_symbols, cache_misses, valid, invalid in price_batches(price_symbols):
            price_coverage.record_batch(batch_symbols, cache_misses)
            for symbol in batch_symbols:
                history = valid.get(symbol)
                for calculation_date in dates_by_symbol[symbol]:
                    if symbol in invalid:
                        outcomes_by_date[calculation_date].record_error()
                    elif history is None:
                        outcomes_by_date[calculation_date].record_cache_miss()
                    elif calculator._has_usable_target_session(
                        history,
                        calculation_date,
                    ):
                        outcomes_by_date[calculation_date].record_scanned()
                    else:
                        outcomes_by_date[calculation_date].record_insufficient()
                        incomplete_target_session_dates.add(calculation_date)
            windowed = calculator._prices_for_feature_window(valid, tuple(ordered_dates))
            for symbol, window in windowed.items():
                fingerprints[symbol] = _history_fingerprint(valid[symbol])
                first_session = _first_session_on_or_after(window, ordered_dates[0])
                if first_session is not None and (
                    earliest_session is None or first_session < earliest_session
                ):
                    earliest_session = first_session
            accumulator.add_prices(windowed)
        for symbol in unsupported_symbols:
            for calculation_date in dates_by_symbol.get(symbol, ()):
                outcomes_by_date[calculation_date].record_insufficient()

        reports_by_date = {
            calculation_date: outcome.report()
            for calculation_date, outcome in outcomes_by_date.items()
        }
        processed_dates = [
            calculation_date
            for calculation_date in ordered_dates
            if (
                reports_by_date[calculation_date].scanned > 0
                and (
                    not require_complete_cache_coverage
                    or (
                        reports_by_date[calculation_date].cache_misses == 0
                        and reports_by_date[calculation_date].errors == 0
                        and calculation_date not in incomplete_target_session_dates
                    )
                )
            )
        ]
        if (
            processed_dates
            and earliest_session is not None
            and earliest_session < processed_dates[0]
        ):
            # A rejected leading date moved some symbol's warm-up window. Re-
            # evaluate over the processed dates so persisted results never
            # depend on rejected dates. The replay reads the cache only (no
            # provider access) and only the symbols the first pass used; any
            # history that is missing or differs from the first pass abandons
            # the replay, keeping the first-pass results rather than mixing
            # data from two reads.
            replayed = self._replay_processed_window(
                price_batches(sorted(fingerprints), cache_only=True),
                fingerprints=fingerprints,
                processed_dates=tuple(processed_dates),
                universes_by_date=universes_by_date,
            )
            if replayed is not None:
                accumulator = replayed
        contributor_metadata_available = True
        try:
            contributor_metadata_by_date = BreadthContributorMetadataLoader.historical(
                calculator.db,
                calculator.market,
                {
                    calculation_date: symbols_by_date[calculation_date]
                    for calculation_date in processed_dates
                },
            )
        except Exception as exc:
            calculator.db.rollback()
            logger.warning(
                "Historical breadth contributor metadata unavailable for %s: %s",
                calculator.market,
                exc,
            )
            if contributor_only:
                raise BreadthContributorBackfillIncomplete(
                    "Contributor metadata source is unavailable; no snapshots were written"
                ) from exc
            contributor_metadata_by_date = {}
            contributor_metadata_available = False
        canonical_batch = accumulator.finish(
            dates=tuple(processed_dates),
            seed_counts=calculator._load_ratio_context_counts(processed_dates),
            contributor_metadata_by_date=contributor_metadata_by_date,
        )
        canonical_by_date = canonical_batch.daily_results

        error_dates = [
            calculation_date.isoformat()
            for calculation_date in ordered_dates
            if calculation_date not in processed_dates
        ]
        if contributor_only and error_dates:
            raise BreadthContributorBackfillIncomplete(
                "Contributor backfill requires complete cached data for every date: "
                + ",".join(error_dates)
            )
        if processed_dates:
            elapsed = (datetime.now(UTC) - started_at).total_seconds()
            duration = round(elapsed / len(processed_dates), 2)
            snapshots_by_date = (
                {
                    value: canonical_batch.contributor_snapshots[value]
                    for value in processed_dates
                }
                if contributor_metadata_available
                else None
            )
            if contributor_only:
                assert snapshots_by_date is not None
                calculator.persistence.replace_contributor_snapshots(
                    snapshots_by_date.values(),
                    expected_aggregates=canonical_by_date,
                )
            else:
                calculator.persistence.upsert_many(
                    (canonical_by_date[value] for value in processed_dates),
                    contributor_snapshots_by_date=snapshots_by_date,
                    duration_seconds_by_date={
                        value: duration for value in processed_dates
                    },
                )

        result: dict[str, Any] = {
            "total_dates": len(plan.dates),
            "processed": len(processed_dates),
            "errors": len(error_dates),
            "error_dates": error_dates,
        }
        if unavailable_dates:
            result["unavailable"] = len(unavailable_dates)
            result["unavailable_dates"] = [
                value.isoformat() for value in unavailable_dates
            ]
        if explicit_symbols is not None:
            result.update(
                {
                    "eligible_stocks_by_date": {
                        value.isoformat(): len(symbols_by_date[value])
                        for value in ordered_dates
                    },
                    "scanned_stocks_by_date": {
                        value.isoformat(): outcomes_by_date[value].report().scanned
                        for value in ordered_dates
                    },
                    "broad_universe_stocks_by_date": {
                        value.isoformat(): len(universes_by_date[value].members)
                        for value in ordered_dates
                    },
                    "advance_decline_eligible_stocks_by_date": {
                        value.isoformat(): (
                            canonical_by_date[
                                value
                            ].eligibility.advance_decline_eligible_count
                            if value in canonical_by_date
                            else 0
                        )
                        for value in ordered_dates
                    },
                    "calculation_errors_by_date": {
                        value.isoformat(): outcomes_by_date[value].report().errors
                        for value in ordered_dates
                    },
                }
            )
        if exclude_unsupported_price_symbols:
            result.update(
                {
                    "skipped_unsupported_symbols": len(skipped_unsupported_symbols),
                    "unsupported_symbols_sample": sorted(
                        set(skipped_unsupported_symbols)
                    )[:20],
                }
            )
        if policy.cache_only:
            aggregate = sum(
                (counter.report() for counter in outcomes_by_date.values()),
                start=BreadthOutcomeReport(),
            )
            result.update(
                BreadthCoverageReport.from_parts(
                    price_coverage.report(),
                    aggregate,
                ).to_backfill_dict()
            )
        return BreadthBackfillResult(MappingProxyType(result))
