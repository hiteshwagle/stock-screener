"""Local-default bootstrap orchestration tasks."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from celery import chain

from ..celery_app import celery_app
from ..database import SessionLocal
from ..domain.bootstrap.plan import (
    BootstrapOperation,
    BootstrapQueueKind,
    MarketBootstrapPlan,
    build_bootstrap_plan,
)
from ..domain.markets.catalog import get_market_catalog
from ..domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from ..services.bootstrap_dispatch_lifecycle import (
    BootstrapDispatchLifecycle,
    BootstrapMarketCompletion,
    PersistedBootstrapDispatchStore,
)
from ..services.bootstrap_price_readiness import evaluate_bootstrap_price_warmup_wait
from ..services.bootstrap_queue_manifest_recorder import (
    BootstrapDispatchError,  # noqa: F401 - compatibility export
    BootstrapQueueManifestRecorder,
)
from ..services.bootstrap_run_manifest import (
    BootstrapRunManifestRepository,
    StaleBootstrapDispatch,
)
from ..services.market_activity_service import (
    mark_market_activity_completed,
    mark_market_activity_failed,
    mark_market_activity_progress,
)
from ..services.runtime_bootstrap_readiness import (
    evaluate_market_readiness as _evaluate_market_readiness,
)
from ..services.runtime_bootstrap_readiness import (
    readiness_failure as _readiness_failure,
)
from ..tasks.market_queues import (
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
    normalize_market,
)

logger = logging.getLogger(__name__)
WAIT_FOR_BOOTSTRAP_PRICE_WARMUP_TASK_NAME = (
    "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup"
)
BOOTSTRAP_PRICE_WARMUP_RETRY_COUNTDOWN_SECONDS = 30
BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES = 120


def _bootstrap_universe_name(market: str) -> str:
    return f"market:{market.upper()}"


def _queue_for_stage(stage) -> str:
    if stage.queue_kind == BootstrapQueueKind.DATA_FETCH:
        return data_fetch_queue_for_market(stage.kwargs["market"])
    if stage.queue_kind == BootstrapQueueKind.MARKET_JOBS:
        return market_jobs_queue_for_market(stage.kwargs["market"])
    if stage.queue_kind == BootstrapQueueKind.CELERY:
        return "celery"
    raise ValueError(f"Unsupported bootstrap queue kind: {stage.queue_kind}")


def _build_market_bootstrap_signatures(market_plan: MarketBootstrapPlan) -> list:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import (
        calculate_daily_breadth_with_gapfill,
        calculate_market_exposure,
    )
    from app.tasks.cache_tasks import smart_refresh_cache
    from app.tasks.daily_market_pipeline_tasks import dispatch_options_after_snapshot
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.group_history_tasks import ensure_group_history
    from app.tasks.group_rank_tasks import (
        calculate_daily_group_rankings,
        calculate_daily_group_rankings_with_gapfill,
    )
    from app.tasks.industry_tasks import load_tracked_ibd_industry_groups
    from app.tasks.market_rs_tasks import (
        bootstrap_balanced_market_rs,
        calculate_market_rs_snapshot,
    )
    from app.tasks.universe_tasks import (
        refresh_official_market_universe,
        refresh_stock_universe,
    )

    task_by_operation = {
        BootstrapOperation.REFRESH_STOCK_UNIVERSE: refresh_stock_universe,
        BootstrapOperation.REFRESH_OFFICIAL_MARKET_UNIVERSE: refresh_official_market_universe,
        BootstrapOperation.LOAD_TRACKED_IBD_INDUSTRY_GROUPS: load_tracked_ibd_industry_groups,
        BootstrapOperation.SMART_REFRESH_CACHE: smart_refresh_cache,
        BootstrapOperation.WAIT_FOR_BOOTSTRAP_PRICE_WARMUP: wait_for_bootstrap_price_warmup,
        BootstrapOperation.REFRESH_ALL_FUNDAMENTALS: refresh_all_fundamentals,
        BootstrapOperation.CALCULATE_MARKET_RS_SNAPSHOT: calculate_market_rs_snapshot,
        BootstrapOperation.BOOTSTRAP_BALANCED_MARKET_RS: bootstrap_balanced_market_rs,
        BootstrapOperation.CALCULATE_DAILY_BREADTH_WITH_GAPFILL: (
            calculate_daily_breadth_with_gapfill
        ),
        BootstrapOperation.CALCULATE_MARKET_EXPOSURE: calculate_market_exposure,
        BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS_WITH_GAPFILL: (
            calculate_daily_group_rankings_with_gapfill
        ),
        BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS: (
            calculate_daily_group_rankings
        ),
        BootstrapOperation.BUILD_DAILY_SNAPSHOT: build_daily_snapshot,
        BootstrapOperation.ENSURE_GROUP_HISTORY: ensure_group_history,
    }
    supports_options = (
        get_market_catalog().get(market_plan.market).capabilities.options_analytics
    )
    signatures = []
    for stage in market_plan.stages:
        signatures.append(
            task_by_operation[stage.operation]
            .si(**stage.kwargs)
            .set(queue=_queue_for_stage(stage))
        )
        # Same follow-on as the daily pipeline. Deliberately not a plan stage:
        # Options is asynchronous and must never gate bootstrap. It takes the
        # snapshot result (.s) to pick the run and never raises.
        if (
            supports_options
            and stage.operation == BootstrapOperation.BUILD_DAILY_SNAPSHOT
        ):
            signatures.append(
                dispatch_options_after_snapshot.s(market=market_plan.market).set(
                    queue="celery"
                )
            )
    return signatures


@celery_app.task(
    bind=True,
    name=WAIT_FOR_BOOTSTRAP_PRICE_WARMUP_TASK_NAME,
    queue="celery",
    soft_time_limit=300,
    max_retries=BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES,
)
def wait_for_bootstrap_price_warmup(
    self,
    market: str,
    activity_lifecycle: str | None = None,
) -> dict:
    from ..wiring.bootstrap import get_market_calendar_service, get_price_cache

    market_code = normalize_market(market)
    lifecycle = activity_lifecycle or "bootstrap"
    task_name = getattr(self, "name", WAIT_FOR_BOOTSTRAP_PRICE_WARMUP_TASK_NAME)
    task_id = getattr(getattr(self, "request", None), "id", None)
    db = SessionLocal()
    try:
        warmup_metadata = get_price_cache().get_warmup_metadata(market=market_code)
        as_of_date = get_market_calendar_service().last_completed_trading_day(
            market_code
        )
        retries = getattr(getattr(self, "request", None), "retries", 0) or 0
        decision = evaluate_bootstrap_price_warmup_wait(
            db,
            market=market_code,
            as_of_date=as_of_date,
            warmup_metadata=warmup_metadata,
            retries=retries,
            max_retries=BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES,
        )
        if decision.ready:
            mark_market_activity_completed(
                db,
                market=market_code,
                stage_key="prices",
                lifecycle=lifecycle,
                task_name=task_name,
                task_id=task_id,
                message="Price cache coverage ready",
            )
            return decision.ready_payload()

        if decision.exhausted:
            mark_market_activity_failed(
                db,
                market=market_code,
                stage_key="prices",
                lifecycle=lifecycle,
                task_name=task_name,
                task_id=task_id,
                message=f"Price cache warmup unavailable: {decision.failure_reason}",
            )
            raise RuntimeError(decision.failure_reason)

        mark_market_activity_progress(
            db,
            market=market_code,
            stage_key="prices",
            lifecycle=lifecycle,
            task_name=task_name,
            task_id=task_id,
            percent=decision.progress_percent,
            current=decision.current,
            total=decision.total,
            message=decision.retry_message,
        )
        raise self.retry(
            exc=RuntimeError(
                f"waiting_for_bootstrap_price_coverage:{market_code}: "
                f"{decision.failure_reason}"
            ),
            countdown=BOOTSTRAP_PRICE_WARMUP_RETRY_COUNTDOWN_SECONDS,
            max_retries=BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES,
        )
    finally:
        db.close()


def _apply_bootstrap_workflow(workflow, errback):
    return workflow.apply_async(link_error=errback)


def _queue_market_bootstrap_workflow(
    market_plan: MarketBootstrapPlan,
    *,
    completion_task,
    completion_kwargs: dict,
    errback_task,
    errback_kwargs: dict,
):
    completion = completion_task.si(**completion_kwargs).set(queue="celery")
    errback = errback_task.s(**errback_kwargs).set(queue="celery")
    return _apply_bootstrap_workflow(
        chain(*_build_market_bootstrap_signatures(market_plan), completion),
        errback,
    )


def _bootstrap_dispatch_store() -> PersistedBootstrapDispatchStore:
    return PersistedBootstrapDispatchStore(SessionLocal)


@dataclass(frozen=True)
class BalancedActivationDispatchState:
    fresh_install: bool
    pending_markets: tuple[str, ...]


def _balanced_activation_state_at_dispatch(
    enabled_markets: Iterable[str],
) -> BalancedActivationDispatchState:
    from ..services.bootstrap_readiness_service import BootstrapReadinessService

    db = SessionLocal()
    try:
        manifest = BootstrapRunManifestRepository().load(db)
        enabled = tuple(
            dict.fromkeys(normalize_market(market) for market in enabled_markets)
        )
        if manifest is not None:
            pending = tuple(
                market
                for market in manifest.pending_balanced_activation_markets
                if market in enabled
            )
            return BalancedActivationDispatchState(
                fresh_install=manifest.fresh_install,
                pending_markets=pending,
            )
        readiness = BootstrapReadinessService()
        pristine = readiness.is_pristine_installation(db)
        startup_seed_import = (
            False
            if pristine
            else readiness.is_pre_bootstrap_seed_import_installation(db)
        )
        fresh_install = pristine or startup_seed_import
        return BalancedActivationDispatchState(
            fresh_install=fresh_install,
            pending_markets=enabled if fresh_install else (),
        )
    finally:
        db.close()


def _bootstrap_dispatch_lifecycle() -> BootstrapDispatchLifecycle:
    from ..infra.db.repositories.market_rs_repo import MarketRsRunRepository

    return BootstrapDispatchLifecycle(
        repository=BootstrapRunManifestRepository(),
        formula_reader=MarketRsRunRepository(),
    )


def _is_current_bootstrap_dispatch(db, *, dispatch_id: str | None) -> bool:
    repository = BootstrapRunManifestRepository()
    if dispatch_id is None:
        current = repository.load(db)
        return bool(current is not None and current.dispatch_id is None)
    return repository.is_current_dispatch(
        db,
        dispatch_id=dispatch_id,
    )


def _finish_bootstrap_market(
    db,
    *,
    dispatch_id: str | None,
    completion: BootstrapMarketCompletion,
) -> bool:
    try:
        _bootstrap_dispatch_lifecycle().finish_market(
            db,
            dispatch_id=dispatch_id,
            completion=completion,
        )
    except StaleBootstrapDispatch:
        return False
    return True


def queue_local_runtime_bootstrap(
    *, primary_market: str, enabled_markets: Iterable[str]
) -> str:
    requested_markets = tuple(enabled_markets)
    activation_state = _balanced_activation_state_at_dispatch(
        (primary_market, *requested_markets)
    )
    plan = build_bootstrap_plan(
        primary_market=primary_market,
        enabled_markets=requested_markets,
        balanced_activation_markets=activation_state.pending_markets,
    )
    primary = plan.primary_market
    enabled = list(plan.enabled_markets)

    market_plans_by_code = {
        market_plan.market: market_plan for market_plan in plan.market_plans
    }
    primary_plan = market_plans_by_code[primary]
    manifest_recorder = BootstrapQueueManifestRecorder.create(
        primary_market=primary,
        enabled_markets=enabled,
        store=_bootstrap_dispatch_store(),
        fresh_install=activation_state.fresh_install,
        pending_balanced_activation_markets=activation_state.pending_markets,
    )
    manifest_recorder.record_queueing()

    try:
        primary_task = _queue_market_bootstrap_workflow(
            primary_plan,
            completion_task=complete_local_runtime_bootstrap,
            completion_kwargs={
                "primary_market": primary,
                "dispatch_id": manifest_recorder.dispatch_id,
                **(
                    {"expected_formula_version": BALANCED_RS_FORMULA_VERSION}
                    if primary in activation_state.pending_markets
                    else {}
                ),
            },
            errback_task=fail_local_runtime_bootstrap,
            errback_kwargs={
                "primary_market": primary,
                "dispatch_id": manifest_recorder.dispatch_id,
            },
        )
        manifest_recorder.record_dispatched_market(
            market=primary,
            task_id=primary_task.id,
        )

        for market_plan in plan.market_plans:
            if market_plan.market == primary:
                continue
            background_task = _queue_market_bootstrap_workflow(
                market_plan,
                completion_task=complete_background_market_bootstrap,
                completion_kwargs={
                    "market": market_plan.market,
                    "dispatch_id": manifest_recorder.dispatch_id,
                    **(
                        {"expected_formula_version": BALANCED_RS_FORMULA_VERSION}
                        if market_plan.market in activation_state.pending_markets
                        else {}
                    ),
                },
                errback_task=fail_background_market_bootstrap,
                errback_kwargs={
                    "market": market_plan.market,
                    "dispatch_id": manifest_recorder.dispatch_id,
                },
            )
            manifest_recorder.record_dispatched_market(
                market=market_plan.market,
                task_id=background_task.id,
            )

    except Exception as exc:
        manifest_recorder.record_dispatch_failed_safely()
        raise manifest_recorder.dispatch_error(exc) from exc

    try:
        manifest_recorder.record_queued()
    except Exception as exc:
        manifest_recorder.record_dispatch_failed_safely()
        raise manifest_recorder.dispatch_error(exc) from exc

    logger.info(
        "Queued local runtime bootstrap",
        extra={
            "primary_market": primary,
            "enabled_markets": enabled,
            "task_id": manifest_recorder.primary_task_id,
        },
    )
    return manifest_recorder.primary_task_id


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.complete_local_runtime_bootstrap",
    queue="celery",
)
def complete_local_runtime_bootstrap(
    primary_market: str,
    enabled_markets: Iterable[str] | None = None,
    expected_formula_version: str | None = None,
    dispatch_id: str | None = None,
) -> dict:
    from ..services.runtime_preferences_service import get_runtime_preferences

    del (
        enabled_markets
    )  # Legacy Celery payload compatibility; readiness is primary-only.
    primary_market = normalize_market(primary_market)
    db = SessionLocal()
    try:
        if not _is_current_bootstrap_dispatch(db, dispatch_id=dispatch_id):
            return {
                "status": "stale",
                "primary_market": primary_market,
                "market": primary_market,
            }
        prefs = get_runtime_preferences(db)
        completion = _evaluate_market_readiness(
            db,
            market=primary_market,
            bootstrap_started_at=prefs.bootstrap_started_at,
            expected_formula_version=expected_formula_version,
        )
        if not completion.ready:
            failure = completion.failure or _readiness_failure(None)
            finished = _finish_bootstrap_market(
                db,
                dispatch_id=dispatch_id,
                completion=BootstrapMarketCompletion.failed(
                    market=completion.market,
                    primary=True,
                    stage_key=failure.stage_key,
                    message=failure.activity_message,
                ),
            )
            if not finished:
                return {
                    "status": "stale",
                    "primary_market": primary_market,
                    "market": completion.market,
                }
            return {
                "status": "failed",
                "primary_market": primary_market,
                "market": completion.market,
                "reason": failure.result_reason,
            }
        finished = _finish_bootstrap_market(
            db,
            dispatch_id=dispatch_id,
            completion=BootstrapMarketCompletion.ready(
                market=completion.market,
                primary=True,
                expected_formula_version=expected_formula_version,
            ),
        )
        if not finished:
            return {
                "status": "stale",
                "primary_market": primary_market,
                "market": completion.market,
            }
    finally:
        db.close()

    logger.info(
        "Completed local runtime bootstrap",
        extra={
            "primary_market": primary_market,
        },
    )
    return {
        "status": "ready",
        "primary_market": primary_market,
        "market": completion.market,
    }


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.complete_background_market_bootstrap",
    queue="celery",
)
def complete_background_market_bootstrap(
    market: str,
    expected_formula_version: str | None = None,
    dispatch_id: str | None = None,
) -> dict:
    from ..services.runtime_preferences_service import get_runtime_preferences

    db = SessionLocal()
    try:
        normalized_market = normalize_market(market)
        if not _is_current_bootstrap_dispatch(db, dispatch_id=dispatch_id):
            return {"status": "stale", "market": normalized_market}
        prefs = get_runtime_preferences(db)
        completion = _evaluate_market_readiness(
            db,
            market=market,
            bootstrap_started_at=prefs.bootstrap_started_at,
            expected_formula_version=expected_formula_version,
        )
        if completion.ready:
            finished = _finish_bootstrap_market(
                db,
                dispatch_id=dispatch_id,
                completion=BootstrapMarketCompletion.ready(
                    market=completion.market,
                    primary=False,
                    expected_formula_version=expected_formula_version,
                ),
            )
            if not finished:
                return {"status": "stale", "market": completion.market}
            return {
                "status": "ready",
                "market": completion.market,
            }
        failure = completion.failure or _readiness_failure(None)
        finished = _finish_bootstrap_market(
            db,
            dispatch_id=dispatch_id,
            completion=BootstrapMarketCompletion.failed(
                market=completion.market,
                primary=False,
                stage_key=failure.stage_key,
                message=failure.activity_message,
            ),
        )
        if not finished:
            return {"status": "stale", "market": completion.market}
        return {
            "status": "failed",
            "market": completion.market,
            "reason": failure.result_reason,
        }
    finally:
        db.close()


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.fail_local_runtime_bootstrap",
    queue="celery",
)
def fail_local_runtime_bootstrap(
    *_celery_errback_args,
    primary_market: str,
    dispatch_id: str | None = None,
) -> dict:
    primary_market = normalize_market(primary_market)
    db = SessionLocal()
    try:
        if not _is_current_bootstrap_dispatch(db, dispatch_id=dispatch_id):
            return {
                "status": "stale",
                "primary_market": primary_market,
                "market": str(primary_market).upper(),
            }
        finished = _finish_bootstrap_market(
            db,
            dispatch_id=dispatch_id,
            completion=BootstrapMarketCompletion.failed(
                market=primary_market,
                primary=True,
                stage_key=None,
                message="Bootstrap failed",
            ),
        )
        if not finished:
            return {
                "status": "stale",
                "primary_market": primary_market,
                "market": str(primary_market).upper(),
            }
    finally:
        db.close()

    logger.warning(
        "Marked local runtime bootstrap failed",
        extra={
            "primary_market": primary_market,
            "callback_args": _celery_errback_args,
        },
    )
    return {
        "status": "failed",
        "primary_market": primary_market,
        "market": str(primary_market).upper(),
    }


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.fail_background_market_bootstrap",
    queue="celery",
)
def fail_background_market_bootstrap(
    *_celery_errback_args,
    market: str,
    dispatch_id: str | None = None,
) -> dict:
    market = normalize_market(market)
    db = SessionLocal()
    try:
        if not _is_current_bootstrap_dispatch(db, dispatch_id=dispatch_id):
            return {"status": "stale", "market": str(market).upper()}
        finished = _finish_bootstrap_market(
            db,
            dispatch_id=dispatch_id,
            completion=BootstrapMarketCompletion.failed(
                market=market,
                primary=False,
                stage_key=None,
                message="Bootstrap failed",
            ),
        )
        if not finished:
            return {"status": "stale", "market": str(market).upper()}
    finally:
        db.close()

    logger.warning(
        "Marked background market bootstrap failed",
        extra={
            "market": market,
            "callback_args": _celery_errback_args,
        },
    )
    return {
        "status": "failed",
        "market": str(market).upper(),
    }
