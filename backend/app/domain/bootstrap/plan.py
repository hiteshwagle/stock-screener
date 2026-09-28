"""Pure Bootstrap workflow plan."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from app.domain.markets.catalog import get_market_catalog


class BootstrapQueueKind(str, Enum):
    DATA_FETCH = "data_fetch"
    MARKET_JOBS = "market_jobs"
    CELERY = "celery"


class BootstrapOperation(str, Enum):
    REFRESH_STOCK_UNIVERSE = "refresh_stock_universe"
    REFRESH_OFFICIAL_MARKET_UNIVERSE = "refresh_official_market_universe"
    LOAD_TRACKED_IBD_INDUSTRY_GROUPS = "load_tracked_ibd_industry_groups"
    SMART_REFRESH_CACHE = "smart_refresh_cache"
    WAIT_FOR_BOOTSTRAP_PRICE_WARMUP = "wait_for_bootstrap_price_warmup"
    REFRESH_ALL_FUNDAMENTALS = "refresh_all_fundamentals"
    CALCULATE_MARKET_RS_SNAPSHOT = "calculate_market_rs_snapshot"
    BOOTSTRAP_BALANCED_MARKET_RS = "bootstrap_balanced_market_rs"
    CALCULATE_DAILY_BREADTH_WITH_GAPFILL = "calculate_daily_breadth_with_gapfill"
    CALCULATE_MARKET_EXPOSURE = "calculate_market_exposure"
    CALCULATE_DAILY_GROUP_RANKINGS_WITH_GAPFILL = "calculate_daily_group_rankings_with_gapfill"
    CALCULATE_DAILY_GROUP_RANKINGS = "calculate_daily_group_rankings"
    BUILD_DAILY_SNAPSHOT = "build_daily_snapshot"
    ENSURE_GROUP_HISTORY = "ensure_group_history"


@dataclass(frozen=True)
class BootstrapStage:
    key: str
    operation: BootstrapOperation
    queue_kind: BootstrapQueueKind
    kwargs: dict[str, Any]


@dataclass(frozen=True)
class MarketBootstrapPlan:
    market: str
    stages: tuple[BootstrapStage, ...]


@dataclass(frozen=True)
class BootstrapPlan:
    primary_market: str
    enabled_markets: tuple[str, ...]
    market_plans: tuple[MarketBootstrapPlan, ...]


def _normalize_markets(
    primary_market: str, enabled_markets: Iterable[str]
) -> tuple[str, ...]:
    catalog = get_market_catalog()
    ordered: list[str] = []

    for raw_market in (primary_market, *tuple(enabled_markets)):
        market = catalog.get(raw_market).code
        if market not in ordered:
            ordered.append(market)

    return tuple(ordered)


def _stage(
    *,
    key: str,
    operation: BootstrapOperation,
    queue_kind: BootstrapQueueKind,
    market: str,
    **kwargs: Any,
) -> BootstrapStage:
    return BootstrapStage(
        key=key,
        operation=operation,
        queue_kind=queue_kind,
        kwargs={"market": market, "activity_lifecycle": "bootstrap", **kwargs},
    )


def _build_market_plan(
    market: str,
    *,
    activate_balanced_rs: bool = False,
) -> MarketBootstrapPlan:
    supports_group_rankings = (
        get_market_catalog().get(market).capabilities.group_rankings
    )
    supports_breadth = get_market_catalog().get(market).capabilities.breadth
    stages = [
        _stage(
            key="universe",
            operation=(
                BootstrapOperation.REFRESH_STOCK_UNIVERSE
                if market == "US"
                else BootstrapOperation.REFRESH_OFFICIAL_MARKET_UNIVERSE
            ),
            queue_kind=BootstrapQueueKind.DATA_FETCH,
            market=market,
        ),
    ]

    if market == "US":
        stages.append(
            _stage(
                key="industry_groups",
                operation=BootstrapOperation.LOAD_TRACKED_IBD_INDUSTRY_GROUPS,
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            )
        )

    stages.extend(
        [
            _stage(
                key="prices",
                operation=BootstrapOperation.SMART_REFRESH_CACHE,
                queue_kind=BootstrapQueueKind.DATA_FETCH,
                market=market,
                mode="bootstrap",
                ensure_group_history=supports_group_rankings,
            ),
            _stage(
                key="price_warmup",
                operation=BootstrapOperation.WAIT_FOR_BOOTSTRAP_PRICE_WARMUP,
                queue_kind=BootstrapQueueKind.CELERY,
                market=market,
            ),
            _stage(
                key="fundamentals",
                operation=BootstrapOperation.REFRESH_ALL_FUNDAMENTALS,
                queue_kind=BootstrapQueueKind.DATA_FETCH,
                market=market,
            ),
            _stage(
                key="market_rs",
                operation=(
                    BootstrapOperation.BOOTSTRAP_BALANCED_MARKET_RS
                    if activate_balanced_rs
                    else BootstrapOperation.CALCULATE_MARKET_RS_SNAPSHOT
                ),
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
            ),
            _stage(
                key="groups",
                operation=(
                    BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS
                    if supports_group_rankings
                    else BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS_WITH_GAPFILL
                ),
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
                execution_policy="refresh_guarded",
                **({"strict": True} if supports_group_rankings else {}),
            ),
            _stage(
                key="snapshot",
                operation=BootstrapOperation.BUILD_DAILY_SNAPSHOT,
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
                universe_name=f"market:{market}",
                publish_pointer_key=f"latest_published_market:{market}",
                bootstrap_cache_only_if_covered=True,
            ),
        ]
    )

    if supports_group_rankings:
        stages.append(
            _stage(
                key="group_history",
                operation=BootstrapOperation.ENSURE_GROUP_HISTORY,
                queue_kind=BootstrapQueueKind.MARKET_JOBS,
                market=market,
                strict=True,
            )
        )

    # Breadth and exposure run last: nothing downstream reads them, and a
    # breadth failure the task cannot catch (OOM kill, soft time limit) ends
    # the chain. Placed here it can no longer withhold the scanner.
    if supports_breadth:
        stages.extend(
            [
                _stage(
                    key="breadth",
                    operation=BootstrapOperation.CALCULATE_DAILY_BREADTH_WITH_GAPFILL,
                    queue_kind=BootstrapQueueKind.MARKET_JOBS,
                    market=market,
                    execution_policy="refresh_guarded",
                ),
                _stage(
                    key="exposure",
                    operation=BootstrapOperation.CALCULATE_MARKET_EXPOSURE,
                    queue_kind=BootstrapQueueKind.MARKET_JOBS,
                    market=market,
                ),
            ]
        )

    return MarketBootstrapPlan(market=market, stages=tuple(stages))


def build_bootstrap_plan(
    *,
    primary_market: str,
    enabled_markets: Iterable[str],
    balanced_activation_markets: Iterable[str] = (),
) -> BootstrapPlan:
    markets = _normalize_markets(primary_market, enabled_markets)
    activation_markets = {
        get_market_catalog().get(market).code for market in balanced_activation_markets
    }
    return BootstrapPlan(
        primary_market=markets[0],
        enabled_markets=markets,
        market_plans=tuple(
            _build_market_plan(
                market,
                activate_balanced_rs=market in activation_markets,
            )
            for market in markets
        ),
    )
