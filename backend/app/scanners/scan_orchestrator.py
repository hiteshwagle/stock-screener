"""
Scan orchestrator for multi-screener coordination.

Coordinates running multiple screeners on a single stock, with data
fetched once and shared across all screeners. Combines results and
calculates composite scores.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.analysis.patterns.config import (
    build_setup_engine_parameters,
)
from app.analysis.patterns.rs_line import (
    DEFAULT_BLUE_DOT_RECENT_DAYS,
    DEFAULT_LOOKBACK,
    RsLineLeadershipSnapshot,
    rs_line_leadership_snapshot,
)
from app.config import settings
from app.domain.scanning.models import CompositeMethod, ScreenerOutputDomain
from app.domain.scanning.ports import (
    CanonicalStockRsSource,
    MarketRsReader,
    MarketRsResolution,
    StockDataProvider,
)
from app.domain.scanning.scoring import (
    apply_quality_policy,
    calculate_composite_score,
    calculate_overall_rating,
)
from app.services.opportunity_state_service import (
    build_data_limited_projection,
)

from .base_screener import (
    BaseStockScreener,
    DataRequirements,
    PrecomputedScanContext,
    ScreenerResult,
    StockData,
)
from .criteria.relative_strength import RelativeStrengthCalculator
from .criteria.rs_resolution import CanonicalStockRsUnavailable, resolve_stock_rs
from .partial_history_metrics import partial_history_metrics
from .scan_result_assembler import (
    ScanResultAssembler,
    ScanResultAssemblyRequest,
    market_rs_audit_fields,
)
from .screener_registry import ScreenerRegistry

logger = logging.getLogger(__name__)

LISTING_ONLY_MIN_BARS = 30
FULL_SCAN_MIN_BARS = 252
IPO_BONUS_MIN_SCORE = 60.0
IPO_BONUS_MAX = 15.0
_DEFAULT_SCREENER_MIN_BARS = 100
_SCREENER_MIN_BARS: dict[str, int] = {
    "ipo": 30,
    "setup_engine": 100,
    "custom": 200,
    "minervini": 240,
    "canslim": 240,
    "volume_breakthrough": 252,
}


def _series_last_float(series) -> float | None:
    if series is None or len(series) == 0:
        return None
    value = series.iloc[-1]
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    return float(value)


def _build_precomputed_scan_context(
    stock_data: StockData,
) -> PrecomputedScanContext | None:
    """Build shared derived scan metrics once per symbol."""
    price_data = stock_data.price_data
    if price_data is None or price_data.empty or "Close" not in price_data.columns:
        return None

    close_indexed = price_data["Close"]
    close_chrono = close_indexed.reset_index(drop=True)
    close_rev = close_chrono[::-1].reset_index(drop=True)

    volume_chrono = None
    volume_rev = None
    if "Volume" in price_data.columns:
        volume_chrono = price_data["Volume"].reset_index(drop=True)
        volume_rev = volume_chrono[::-1].reset_index(drop=True)

    benchmark_close_indexed = None
    benchmark_close_chrono = None
    benchmark_close_rev = None
    if (
        stock_data.benchmark_data is not None
        and not stock_data.benchmark_data.empty
        and "Close" in stock_data.benchmark_data.columns
    ):
        benchmark_close_indexed = stock_data.benchmark_data["Close"]
        benchmark_close_chrono = benchmark_close_indexed.reset_index(drop=True)
        benchmark_close_rev = benchmark_close_chrono[::-1].reset_index(drop=True)

    ma_50_series = close_chrono.rolling(window=50, min_periods=50).mean()
    ma_150_series = close_chrono.rolling(window=150, min_periods=150).mean()
    ma_200_series = close_chrono.rolling(window=200, min_periods=200).mean()
    ema_10_series = close_chrono.ewm(span=10, adjust=False).mean()
    ema_20_series = close_chrono.ewm(span=20, adjust=False).mean()
    ema_50_series = close_chrono.ewm(span=50, adjust=False).mean()

    ma_200_month_ago = None
    if len(ma_200_series) > 220:
        ma_200_month_ago = _series_last_float(ma_200_series.iloc[:-20])
    if ma_200_month_ago is None:
        ma_200_month_ago = _series_last_float(ma_200_series)

    rs_ratings = None
    if isinstance(stock_data.rs_source, CanonicalStockRsSource) or (
        benchmark_close_rev is not None and not benchmark_close_rev.empty
    ):
        rs_ratings = resolve_stock_rs(
            stock_data,
            lambda: RelativeStrengthCalculator().calculate_all_rs_ratings(
                stock_data.symbol,
                close_rev,
                benchmark_close_rev,
                stock_data.rs_universe_performances,
            ),
        )

    rs_leadership = RsLineLeadershipSnapshot.empty()
    if benchmark_close_indexed is not None and not benchmark_close_indexed.empty:
        rs_leadership = rs_line_leadership_snapshot(
            close_indexed,
            benchmark_close_indexed,
            lookback=DEFAULT_LOOKBACK,
            recent_days=DEFAULT_BLUE_DOT_RECENT_DAYS,
        )

    return PrecomputedScanContext(
        close_chrono=close_chrono,
        close_rev=close_rev,
        volume_chrono=volume_chrono,
        volume_rev=volume_rev,
        benchmark_close_chrono=benchmark_close_chrono,
        benchmark_close_rev=benchmark_close_rev,
        current_price=_series_last_float(close_chrono),
        ma_50=_series_last_float(ma_50_series),
        ma_150=_series_last_float(ma_150_series),
        ma_200=_series_last_float(ma_200_series),
        ma_200_month_ago=ma_200_month_ago,
        ema_10=_series_last_float(ema_10_series),
        ema_20=_series_last_float(ema_20_series),
        ema_50=_series_last_float(ema_50_series),
        high_52w=float(close_rev.max()) if not close_rev.empty else None,
        low_52w=float(close_rev.min()) if not close_rev.empty else None,
        rs_ratings=rs_ratings,
        rs_line_leadership=rs_leadership,
    )


def _to_domain_output(name: str, result: ScreenerResult) -> ScreenerOutputDomain:
    """Map an infrastructure ScreenerResult to a domain ScreenerOutputDomain."""
    return ScreenerOutputDomain(
        screener_name=name,
        score=result.score,
        passes=result.passes,
        rating=result.rating,
        breakdown=result.breakdown,
        details=result.details,
    )


def _history_bar_count(stock_data: StockData) -> int:
    price_data = stock_data.price_data
    if price_data is None or getattr(price_data, "empty", True):
        return 0
    return len(price_data)


def _required_bars_for_screener(name: str) -> int:
    return _SCREENER_MIN_BARS.get(name, _DEFAULT_SCREENER_MIN_BARS)


def _requirements_with_opportunity_state_fields(
    requirements: DataRequirements,
) -> DataRequirements:
    """Opportunity rows need benchmark leadership and event-calendar context."""
    if requirements.needs_benchmark and requirements.needs_event_calendar:
        return requirements
    return DataRequirements(
        price_period=requirements.price_period,
        needs_fundamentals=requirements.needs_fundamentals,
        needs_quarterly_growth=requirements.needs_quarterly_growth,
        needs_benchmark=True,
        needs_earnings_history=requirements.needs_earnings_history,
        needs_event_calendar=True,
    )


def _compute_ipo_bonus(ipo_score: float | None, history_bars: int) -> float:
    if (
        ipo_score is None
        or ipo_score < IPO_BONUS_MIN_SCORE
        or history_bars < LISTING_ONLY_MIN_BARS
        or history_bars >= FULL_SCAN_MIN_BARS
    ):
        return 0.0
    raw_bonus = IPO_BONUS_MAX * (
        (FULL_SCAN_MIN_BARS - history_bars)
        / (FULL_SCAN_MIN_BARS - LISTING_ONLY_MIN_BARS)
    )
    return round(max(0.0, raw_bonus), 2)


def _scan_mode_for_history(history_bars: int) -> str:
    if history_bars >= FULL_SCAN_MIN_BARS:
        return "full"
    if history_bars >= LISTING_ONLY_MIN_BARS:
        return "ipo_weighted"
    return "listing_only"


def _insufficient_screener_reason(
    insufficient_screeners: dict[str, str | None],
) -> str:
    parts: list[str] = []
    for name in sorted(insufficient_screeners):
        reason = insufficient_screeners[name]
        parts.append(f"{name} ({reason})" if reason else name)
    return "Insufficient data from runnable screeners: " + ", ".join(parts)


@dataclass(frozen=True)
class _ScreenerExecution:
    results: dict[str, ScreenerResult]
    unavailable: list[str]
    hard_errors: list[str]
    insufficient: dict[str, str | None]


def _composite_method(value: str) -> CompositeMethod:
    try:
        return CompositeMethod(value)
    except ValueError:
        logger.warning(
            "Unknown composite method '%s', defaulting to weighted_average",
            value,
        )
        return CompositeMethod.WEIGHTED_AVERAGE


def _partition_screeners(
    screeners: dict[str, BaseStockScreener],
    history_bars: int,
) -> tuple[dict[str, BaseStockScreener], list[str]]:
    runnable: dict[str, BaseStockScreener] = {}
    unavailable: list[str] = []
    for name, screener in screeners.items():
        if history_bars < _required_bars_for_screener(name):
            unavailable.append(name)
        else:
            runnable[name] = screener
    return runnable, unavailable


def _run_screeners(
    *,
    symbol: str,
    stock_data: StockData,
    criteria: Optional[Dict],
    runnable: dict[str, BaseStockScreener],
    unavailable: list[str],
) -> _ScreenerExecution:
    # Screeners run one after another: they are CPU-bound Python, so threads
    # would only contend for the GIL. Scans parallelize across processes at
    # the symbol level instead (see app.infra.tasks.scan_compute_pool).
    results: dict[str, ScreenerResult] = {}
    hard_errors: list[str] = []
    insufficient: dict[str, str | None] = {}
    unavailable = list(unavailable)
    for name, screener in runnable.items():
        try:
            result = screener.scan_stock(symbol, stock_data, criteria)
        except Exception as exc:
            logger.error("Error running %s screener on %s: %s", name, symbol, exc)
            hard_errors.append(name)
            continue
        logger.debug(
            "%s - %s: score=%.1f, passes=%s, rating=%s",
            symbol,
            name,
            result.score,
            result.passes,
            result.rating,
        )
        if result.rating == "Insufficient Data":
            unavailable.append(name)
            details = result.details if isinstance(result.details, dict) else {}
            insufficient[name] = details.get("reason") or details.get("error")
        else:
            results[name] = result
    return _ScreenerExecution(
        results=results,
        unavailable=unavailable,
        hard_errors=hard_errors,
        insufficient=insufficient,
    )


class ScanOrchestrator:
    """
    Orchestrates multi-screener stock analysis.

    Coordinates:
    1. Getting screeners from registry
    2. Merging data requirements
    3. Fetching data once
    4. Running all screeners
    5. Combining results
    """

    def __init__(
        self,
        data_provider: StockDataProvider,
        registry: ScreenerRegistry,
        market_rs_reader: MarketRsReader | None = None,
        result_assembler: ScanResultAssembler | None = None,
    ):
        """Initialize orchestrator with injected dependencies.

        Args:
            data_provider: Port for fetching stock data
            registry: Registry of available screeners
        """
        self._data_provider = data_provider
        self._registry = registry
        self._market_rs_reader = market_rs_reader
        self._result_assembler = result_assembler or ScanResultAssembler()

    def get_merged_requirements(
        self,
        screener_names: List[str],
        criteria: Optional[Dict] = None,
    ) -> DataRequirements:
        """Merge data requirements once for a screener set (batch optimization)."""
        if not settings.setup_engine_enabled:
            screener_names = [n for n in screener_names if n != "setup_engine"]
        if not screener_names:
            return DataRequirements()

        screeners = self._registry.get_multiple(screener_names)
        return _requirements_with_opportunity_state_fields(
            DataRequirements.merge_all(
                [
                    screener.get_data_requirements(criteria)
                    for screener in screeners.values()
                ]
            )
        )

    def scan_stock_multi(
        self,
        symbol: str,
        screener_names: List[str],
        criteria: Optional[Dict] = None,
        composite_method: str = "weighted_average",
        pre_merged_requirements: Optional[DataRequirements] = None,
        pre_fetched_data: Optional[StockData] = None,
        market_rs_resolution: MarketRsResolution | None = None,
    ) -> Dict:
        """
        Run multiple screeners on a single stock.

        Args:
            symbol: Stock symbol
            screener_names: List of screener names to run
            criteria: Optional criteria/parameters for screeners
            composite_method: How to combine scores (weighted_average, maximum, minimum)
            pre_merged_requirements: Optional pre-merged requirements (batch optimization)
            pre_fetched_data: Optional pre-fetched stock data (batch optimization)
            market_rs_resolution: Optional publication pinned by the calling workflow

        Returns:
            Dict with combined results from all screeners
        """
        try:
            method_enum = _composite_method(composite_method)
            screener_names = self._enabled_screener_names(screener_names)
            if not screener_names:
                return self._all_screeners_disabled_result(symbol)

            try:
                screeners = self._registry.get_multiple(screener_names)
            except ValueError as exc:
                logger.error("Error getting screeners: %s", exc)
                return self._error_result(symbol, str(exc))

            stock_data = self._prepare_stock_data(
                symbol=symbol,
                screeners=screeners,
                criteria=criteria,
                pre_merged_requirements=pre_merged_requirements,
                pre_fetched_data=pre_fetched_data,
                market_rs_resolution=market_rs_resolution,
            )

            history_bars = _history_bar_count(stock_data)
            context_error = self._prepare_scan_context(
                symbol=symbol,
                stock_data=stock_data,
                composite_method=composite_method,
                history_bars=history_bars,
                screener_names=screener_names,
            )
            if context_error is not None:
                return context_error

            runnable, unavailable = _partition_screeners(screeners, history_bars)
            preparation_error = self._pre_execution_result(
                symbol=symbol,
                stock_data=stock_data,
                composite_method=composite_method,
                history_bars=history_bars,
                screener_names=screener_names,
                runnable=runnable,
            )
            if preparation_error is not None:
                return preparation_error

            execution = _run_screeners(
                symbol=symbol,
                stock_data=stock_data,
                criteria=criteria,
                runnable=runnable,
                unavailable=unavailable,
            )
            execution_error = self._post_execution_result(
                symbol=symbol,
                stock_data=stock_data,
                composite_method=composite_method,
                history_bars=history_bars,
                screener_names=screener_names,
                execution=execution,
            )
            if execution_error is not None:
                return execution_error

            return self._assemble_success_result(
                symbol=symbol,
                stock_data=stock_data,
                criteria=criteria,
                composite_method=composite_method,
                method_enum=method_enum,
                history_bars=history_bars,
                screener_names=screener_names,
                execution=execution,
            )

        except Exception as exc:
            logger.error("Error orchestrating scan for %s: %s", symbol, exc)
            return self._error_result(symbol, str(exc))

    @staticmethod
    def _enabled_screener_names(screener_names: List[str]) -> List[str]:
        if settings.setup_engine_enabled:
            return screener_names
        return [name for name in screener_names if name != "setup_engine"]

    @staticmethod
    def _all_screeners_disabled_result(symbol: str) -> Dict:
        return {
            "symbol": symbol,
            "composite_score": 0,
            "rating": "Error",
            "error": "All requested screeners are disabled",
            "current_price": None,
            "screeners_run": [],
        }

    def _prepare_stock_data(
        self,
        *,
        symbol: str,
        screeners: dict[str, BaseStockScreener],
        criteria: Optional[Dict],
        pre_merged_requirements: Optional[DataRequirements],
        pre_fetched_data: Optional[StockData],
        market_rs_resolution: MarketRsResolution | None,
    ) -> StockData:
        if pre_fetched_data is not None:
            stock_data = pre_fetched_data
            logger.debug("Using pre-fetched data for %s", symbol)
        else:
            requirements = self._scan_requirements(
                symbol=symbol,
                screeners=screeners,
                criteria=criteria,
                pre_merged_requirements=pre_merged_requirements,
            )
            stock_data = self._data_provider.prepare_data(symbol, requirements)
        self._apply_market_rs(stock_data, market_rs_resolution)
        return stock_data

    @staticmethod
    def _scan_requirements(
        *,
        symbol: str,
        screeners: dict[str, BaseStockScreener],
        criteria: Optional[Dict],
        pre_merged_requirements: Optional[DataRequirements],
    ) -> DataRequirements:
        if pre_merged_requirements is not None:
            logger.debug("Using pre-merged requirements for %s", symbol)
            return _requirements_with_opportunity_state_fields(pre_merged_requirements)
        requirements = _requirements_with_opportunity_state_fields(
            DataRequirements.merge_all(
                [
                    screener.get_data_requirements(criteria)
                    for screener in screeners.values()
                ]
            )
        )
        logger.info("Merged data requirements for %s: %s", symbol, requirements)
        return requirements

    def _apply_market_rs(
        self,
        stock_data: StockData,
        market_rs_resolution: MarketRsResolution | None,
    ) -> None:
        if stock_data.rs_source is not None:
            return
        resolution = market_rs_resolution
        if resolution is None and self._market_rs_reader is not None:
            resolution = self._market_rs_reader.get(
                market=str(stock_data.market or "US").strip().upper(),
                symbols=(stock_data.symbol.strip().upper(),),
                as_of_date=None,
                formula_version=None,
            )
        if resolution is not None:
            self._data_provider.apply_market_rs_resolution(
                {stock_data.symbol.strip().upper(): stock_data},
                resolution,
            )

    def _prepare_scan_context(
        self,
        *,
        symbol: str,
        stock_data: StockData,
        composite_method: str,
        history_bars: int,
        screener_names: List[str],
    ) -> Dict | None:
        if stock_data.precomputed_scan_context is not None or history_bars == 0:
            return None
        try:
            stock_data.precomputed_scan_context = _build_precomputed_scan_context(
                stock_data
            )
        except CanonicalStockRsUnavailable as exc:
            return self._insufficient_data_result(
                symbol,
                stock_data,
                composite_method=composite_method,
                history_bars=history_bars,
                applicable_screeners=[],
                unavailable_screeners=screener_names,
                reason=str(exc),
            )
        return None

    def _pre_execution_result(
        self,
        *,
        symbol: str,
        stock_data: StockData,
        composite_method: str,
        history_bars: int,
        screener_names: List[str],
        runnable: dict[str, BaseStockScreener],
    ) -> Dict | None:
        if history_bars >= LISTING_ONLY_MIN_BARS and runnable:
            return None
        return self._insufficient_data_result(
            symbol,
            stock_data,
            composite_method=composite_method,
            history_bars=history_bars,
            applicable_screeners=[],
            unavailable_screeners=screener_names,
            reason=stock_data.get_error_summary() or "Insufficient price history",
        )

    def _post_execution_result(
        self,
        *,
        symbol: str,
        stock_data: StockData,
        composite_method: str,
        history_bars: int,
        screener_names: List[str],
        execution: _ScreenerExecution,
    ) -> Dict | None:
        if execution.hard_errors:
            failed = ", ".join(sorted(execution.hard_errors))
            return self._error_result(symbol, f"Screener execution failed: {failed}")
        if execution.insufficient:
            return self._insufficient_data_result(
                symbol,
                stock_data,
                composite_method=composite_method,
                history_bars=history_bars,
                applicable_screeners=[
                    name for name in screener_names if name in execution.results
                ],
                unavailable_screeners=[
                    name for name in screener_names if name in execution.unavailable
                ],
                reason=_insufficient_screener_reason(execution.insufficient),
            )
        if execution.results:
            return None
        return self._insufficient_data_result(
            symbol,
            stock_data,
            composite_method=composite_method,
            history_bars=history_bars,
            applicable_screeners=[],
            unavailable_screeners=screener_names,
            reason=(
                stock_data.get_error_summary()
                or "Insufficient data for applicable screeners"
            ),
        )

    def _assemble_success_result(
        self,
        *,
        symbol: str,
        stock_data: StockData,
        criteria: Optional[Dict],
        composite_method: str,
        method_enum: CompositeMethod,
        history_bars: int,
        screener_names: List[str],
        execution: _ScreenerExecution,
    ) -> Dict:
        domain_outputs = {
            name: _to_domain_output(name, result)
            for name, result in execution.results.items()
        }
        composite_score = calculate_composite_score(domain_outputs, method_enum)
        ipo_result = execution.results.get("ipo")
        ipo_bonus = _compute_ipo_bonus(
            ipo_result.score if ipo_result is not None else None,
            history_bars,
        )
        scan_mode = "full"
        data_status = "complete"
        composite_reason = None
        if history_bars < FULL_SCAN_MIN_BARS:
            scan_mode = _scan_mode_for_history(history_bars)
            data_status = "insufficient_history"
            if ipo_bonus > 0:
                composite_score = min(100.0, composite_score + ipo_bonus)
                composite_reason = "ipo_uplift"

        rating = calculate_overall_rating(composite_score, domain_outputs)
        completeness = (
            stock_data.fundamentals.get("field_completeness_score")
            if stock_data.fundamentals
            else None
        )
        adjustment = apply_quality_policy(rating, completeness)
        result = self._result_assembler.assemble(
            ScanResultAssemblyRequest(
                symbol=symbol,
                stock_data=stock_data,
                screener_results=execution.results,
                composite_score=composite_score,
                overall_rating=adjustment.rating.value,
                composite_method=composite_method,
                applicable_screeners=tuple(
                    name for name in screener_names if name in execution.results
                ),
                unavailable_screeners=tuple(
                    name for name in screener_names if name in execution.unavailable
                ),
                history_bars=history_bars,
                scan_mode=scan_mode,
                data_status=data_status,
                is_scannable=True,
                ipo_bonus=ipo_bonus,
                composite_reason=composite_reason,
                quality_downgrade_reason=adjustment.reason,
                field_completeness_score=completeness,
                opportunity_parameters=build_setup_engine_parameters(
                    (criteria or {}).get("setup_engine_parameters")
                ),
            )
        )
        if data_status == "insufficient_history":
            for key, value in partial_history_metrics(stock_data).items():
                result.setdefault(key, value)
        return result

    def _error_result(self, symbol: str, error: str) -> Dict:
        """Return result for errors."""
        return {
            "symbol": symbol,
            "composite_score": 0,
            "rating": "Error",
            "error": f"Scan error: {error}",
            "current_price": None,
            "screeners_run": [],
            "result_status": "error",
            "data_status": "error",
            "is_scannable": False,
            "scan_mode": "listing_only",
            "history_bars": 0,
            "applicable_screeners": [],
            "unavailable_screeners": [],
            "composite_reason": None,
            "ipo_bonus": 0.0,
        }

    def _insufficient_data_result(
        self,
        symbol: str,
        stock_data: StockData,
        *,
        composite_method: str,
        history_bars: int,
        applicable_screeners: list[str],
        unavailable_screeners: list[str],
        reason: str,
    ) -> Dict:
        """Return result for insufficient data."""
        result = {
            "symbol": symbol,
            "composite_score": None,
            "rating": "Insufficient Data",
            "reason": reason,
            "current_price": stock_data.get_current_price(),
            "screeners_run": [],
            "composite_method": composite_method,
            "screeners_passed": 0,
            "screeners_total": 0,
            "result_status": "insufficient_history",
            "data_status": "insufficient_history",
            "is_scannable": False,
            "scan_mode": _scan_mode_for_history(history_bars),
            "history_bars": history_bars,
            "applicable_screeners": list(applicable_screeners),
            "unavailable_screeners": list(unavailable_screeners),
            "composite_reason": None,
            "ipo_bonus": 0.0,
            **market_rs_audit_fields(stock_data),
            "details": {
                "screeners": {},
                "data_errors": stock_data.fetch_errors
                if stock_data.fetch_errors
                else None,
                **market_rs_audit_fields(stock_data),
            },
        }
        if stock_data.fundamentals:
            if stock_data.fundamentals.get("market_cap") is not None:
                result["market_cap"] = stock_data.fundamentals["market_cap"]
            if stock_data.fundamentals.get("market_cap_usd") is not None:
                result["market_cap_usd"] = stock_data.fundamentals["market_cap_usd"]
            if stock_data.fundamentals.get("eps_rating") is not None:
                result["eps_rating"] = stock_data.fundamentals["eps_rating"]
            if stock_data.fundamentals.get("ipo_date"):
                result["ipo_date"] = stock_data.fundamentals["ipo_date"]
            if stock_data.fundamentals.get("sector"):
                result["gics_sector"] = stock_data.fundamentals["sector"]
            if stock_data.fundamentals.get("industry"):
                result["gics_industry"] = stock_data.fundamentals["industry"]
        result.update(partial_history_metrics(stock_data))
        result.update(
            build_data_limited_projection(result, stock_data, "insufficient_data")
        )
        return result
