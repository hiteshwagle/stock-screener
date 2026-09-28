from __future__ import annotations

import pytest

from app.domain.bootstrap.plan import (
    BootstrapOperation,
    BootstrapQueueKind,
    build_bootstrap_plan,
)


def test_us_bootstrap_plan_includes_us_only_industry_group_seed() -> None:
    plan = build_bootstrap_plan(primary_market="US", enabled_markets=["US"])

    assert [stage.key for stage in plan.market_plans[0].stages] == [
        "universe",
        "industry_groups",
        "prices",
        "price_warmup",
        "fundamentals",
        "market_rs",
        "groups",
        "snapshot",
        "group_history",
        "breadth",
        "exposure",
    ]
    assert plan.market_plans[0].stages[1].queue_kind == BootstrapQueueKind.MARKET_JOBS
    assert plan.market_plans[0].stages[3].queue_kind == BootstrapQueueKind.CELERY
    assert (
        plan.market_plans[0].stages[-3].operation
        == BootstrapOperation.ENSURE_GROUP_HISTORY
    )
    assert plan.market_plans[0].stages[2].kwargs["ensure_group_history"] is True


def test_non_us_bootstrap_plan_uses_official_universe_without_industry_seed() -> None:
    plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["HK", "US"])
    hk_plan = plan.market_plans[0]

    assert hk_plan.market == "HK"
    assert [stage.operation for stage in hk_plan.stages] == [
        BootstrapOperation.REFRESH_OFFICIAL_MARKET_UNIVERSE,
        BootstrapOperation.SMART_REFRESH_CACHE,
        BootstrapOperation.WAIT_FOR_BOOTSTRAP_PRICE_WARMUP,
        BootstrapOperation.REFRESH_ALL_FUNDAMENTALS,
        BootstrapOperation.CALCULATE_MARKET_RS_SNAPSHOT,
        BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS,
        BootstrapOperation.BUILD_DAILY_SNAPSHOT,
        BootstrapOperation.ENSURE_GROUP_HISTORY,
        BootstrapOperation.CALCULATE_DAILY_BREADTH_WITH_GAPFILL,
        BootstrapOperation.CALCULATE_MARKET_EXPOSURE,
    ]
    snapshot_stage = next(stage for stage in hk_plan.stages if stage.key == "snapshot")
    assert snapshot_stage.kwargs == {
        "market": "HK",
        "universe_name": "market:HK",
        "publish_pointer_key": "latest_published_market:HK",
        "activity_lifecycle": "bootstrap",
        "bootstrap_cache_only_if_covered": True,
    }
    breadth_stage = next(stage for stage in hk_plan.stages if stage.key == "breadth")
    groups_stage = next(stage for stage in hk_plan.stages if stage.key == "groups")

    assert breadth_stage.kwargs["execution_policy"] == "refresh_guarded"
    assert groups_stage.kwargs["execution_policy"] == "refresh_guarded"
    assert groups_stage.kwargs["strict"] is True
    history_stage = next(stage for stage in hk_plan.stages if stage.key == "group_history")
    assert history_stage.kwargs["strict"] is True


def test_au_bootstrap_plan_refreshes_universe_before_prices_and_fundamentals() -> None:
    plan = build_bootstrap_plan(primary_market="AU", enabled_markets=["AU"])
    au_plan = plan.market_plans[0]

    assert au_plan.market == "AU"
    assert [stage.key for stage in au_plan.stages[:3]] == [
        "universe",
        "prices",
        "price_warmup",
    ]
    assert (
        au_plan.stages[0].operation
        == BootstrapOperation.REFRESH_OFFICIAL_MARKET_UNIVERSE
    )
    assert au_plan.stages[0].kwargs["market"] == "AU"


@pytest.mark.parametrize("market", ("AU", "SG", "MY"))
def test_breadth_disabled_bootstrap_omits_breadth_and_exposure(
    market: str,
) -> None:
    plan = build_bootstrap_plan(primary_market=market, enabled_markets=(market,))
    stage_keys = [stage.key for stage in plan.market_plans[0].stages]

    assert "breadth" not in stage_keys
    assert "exposure" not in stage_keys
    assert "snapshot" in stage_keys


def test_bootstrap_plan_deduplicates_primary_and_enabled_markets_in_order() -> None:
    plan = build_bootstrap_plan(primary_market="HK", enabled_markets=["US", "HK", "US"])

    assert [market_plan.market for market_plan in plan.market_plans] == ["HK", "US"]


def test_fresh_bootstrap_requires_balanced_activation_before_groups() -> None:
    plan = build_bootstrap_plan(
        primary_market="US",
        enabled_markets=("US",),
        balanced_activation_markets=("US",),
    )
    operations = [stage.operation for stage in plan.market_plans[0].stages]

    assert BootstrapOperation.BOOTSTRAP_BALANCED_MARKET_RS in operations
    assert BootstrapOperation.CALCULATE_MARKET_RS_SNAPSHOT not in operations
    assert operations.index(
        BootstrapOperation.BOOTSTRAP_BALANCED_MARKET_RS
    ) < operations.index(BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS)
    assert operations.index(
        BootstrapOperation.BOOTSTRAP_BALANCED_MARKET_RS
    ) < operations.index(BootstrapOperation.BUILD_DAILY_SNAPSHOT)


def test_nonempty_bootstrap_keeps_shadow_market_rs_stage() -> None:
    plan = build_bootstrap_plan(
        primary_market="US",
        enabled_markets=("US",),
        balanced_activation_markets=(),
    )

    operations = [stage.operation for stage in plan.market_plans[0].stages]
    assert BootstrapOperation.CALCULATE_MARKET_RS_SNAPSHOT in operations
    assert BootstrapOperation.BOOTSTRAP_BALANCED_MARKET_RS not in operations


def test_bootstrap_plan_activates_only_markets_still_pending() -> None:
    plan = build_bootstrap_plan(
        primary_market="US",
        enabled_markets=("US", "HK"),
        balanced_activation_markets=("HK",),
    )

    operations_by_market = {
        market_plan.market: [stage.operation for stage in market_plan.stages]
        for market_plan in plan.market_plans
    }
    assert BootstrapOperation.CALCULATE_MARKET_RS_SNAPSHOT in operations_by_market["US"]
    assert (
        BootstrapOperation.BOOTSTRAP_BALANCED_MARKET_RS
        not in operations_by_market["US"]
    )
    assert BootstrapOperation.BOOTSTRAP_BALANCED_MARKET_RS in operations_by_market["HK"]
