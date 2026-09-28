"""Bootstrap readiness evaluation for enabled Markets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping

from sqlalchemy.orm import Session

from ..domain.markets.catalog import get_market_catalog
from ..infra.db.models.feature_store import FeatureRun
from ..infra.db.models.relative_strength import MarketRsRun
from ..infra.db.repositories.market_rs_repo import (
    MarketRsFormulaNotConfigured,
    MarketRsRunRepository,
)
from ..models.app_settings import AppSetting
from ..models.industry import IBDGroupRank
from ..models.market_breadth import MarketBreadth
from ..models.market_exposure import MarketExposure
from ..models.scan_result import SCAN_TRIGGER_SOURCE_AUTO, Scan
from ..models.stock import StockFundamental, StockPrice
from ..models.stock_universe import StockUniverse

PRE_BOOTSTRAP_SEED_IMPORT_KEY = "runtime.bootstrap.pre_bootstrap_seed_import"
PRE_BOOTSTRAP_SEED_IMPORT_CATEGORY = "runtime_activity"
PRE_BOOTSTRAP_SEED_IMPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MarketBootstrapReadiness:
    market: str
    core_ready: bool
    scan_ready: bool
    rs_ready: bool = True

    @property
    def ready(self) -> bool:
        return self.core_ready and self.scan_ready and self.rs_ready


@dataclass(frozen=True)
class BootstrapReadiness:
    empty_system: bool
    market_results: dict[str, MarketBootstrapReadiness]

    @property
    def ready(self) -> bool:
        return all(result.ready for result in self.market_results.values())

    @property
    def missing_markets(self) -> list[str]:
        return [
            market
            for market, result in self.market_results.items()
            if not result.ready
        ]


def has_compatible_cot_publication(db: Session) -> bool:
    """A COT publication the deployed read path accepts (see ``CotQueryService``).

    An upgrade that bumps a registry, schema or calculation version leaves a
    pointer the read path rejects, so pointer existence alone is not enough.
    """
    from ..domain.cot.models import (
        COT_CALCULATION_VERSION,
        COT_REGISTRY_VERSION,
        COT_SCHEMA_VERSION,
    )
    from ..infra.db.models.cot import CotImportRun, CotPublicationPointer
    from ..infra.db.repositories.cot_repository import LATEST_PUBLICATION_KEY

    return (
        db.query(CotPublicationPointer.key)
        .join(CotImportRun, CotImportRun.id == CotPublicationPointer.run_id)
        .filter(
            CotPublicationPointer.key == LATEST_PUBLICATION_KEY,
            CotImportRun.registry_version == COT_REGISTRY_VERSION,
            CotImportRun.schema_version == COT_SCHEMA_VERSION,
            CotImportRun.calculation_version == COT_CALCULATION_VERSION,
        )
        .first()
        is not None
    )


class BootstrapReadinessService:
    def normalize_market(self, market: str) -> str:
        return get_market_catalog().get(market).code

    def is_empty_system(self, db: Session) -> bool:
        return not (
            self._has_active_universe_rows(db)
            or self._has_price_rows(db)
            or self._has_fundamental_rows(db)
        )

    def is_pristine_installation(self, db: Session) -> bool:
        persisted_queries = (
            db.query(StockUniverse.id),
            db.query(StockPrice.id),
            db.query(StockFundamental.id),
            db.query(MarketBreadth.id),
            db.query(MarketExposure.id),
            db.query(Scan.id),
            db.query(FeatureRun.id),
            db.query(IBDGroupRank.id),
            db.query(MarketRsRun.id),
        )
        return not any(
            query.limit(1).first() is not None for query in persisted_queries
        )

    def has_bootstrap_output_rows(self, db: Session) -> bool:
        output_queries = (
            db.query(MarketBreadth.id),
            db.query(MarketExposure.id),
            db.query(Scan.id),
            db.query(FeatureRun.id),
            db.query(IBDGroupRank.id),
            db.query(MarketRsRun.id),
        )
        return any(query.limit(1).first() is not None for query in output_queries)

    def mark_pre_bootstrap_seed_import(self, db: Session, *, source: str) -> bool:
        """Mark that a pristine startup task may import raw seed rows."""
        source = str(source).strip()
        if not source:
            return False
        if not self.is_pristine_installation(db):
            return False

        now = datetime.now(timezone.utc).isoformat()
        setting = (
            db.query(AppSetting)
            .filter(AppSetting.key == PRE_BOOTSTRAP_SEED_IMPORT_KEY)
            .one_or_none()
        )
        sources: set[str] = set()
        if setting is not None:
            payload = self._decode_pre_bootstrap_seed_payload(setting)
            if payload is not None:
                sources.update(
                    self._normalize_pre_bootstrap_seed_sources(
                        payload.get("sources"),
                    )
                )
        sources.add(str(source))
        encoded = json.dumps(
            {
                "schema_version": PRE_BOOTSTRAP_SEED_IMPORT_SCHEMA_VERSION,
                "sources": sorted(sources),
                "updated_at": now,
            },
            sort_keys=True,
        )
        if setting is None:
            db.add(
                AppSetting(
                    key=PRE_BOOTSTRAP_SEED_IMPORT_KEY,
                    value=encoded,
                    category=PRE_BOOTSTRAP_SEED_IMPORT_CATEGORY,
                    description=(
                        "Pristine startup task may have imported raw bootstrap seed rows."
                    ),
                )
            )
        else:
            setting.value = encoded
            setting.category = PRE_BOOTSTRAP_SEED_IMPORT_CATEGORY
            setting.description = (
                "Pristine startup task may have imported raw bootstrap seed rows."
            )
        return True

    def clear_pre_bootstrap_seed_import_marker(
        self,
        db: Session,
        *,
        source: str | None = None,
    ) -> bool:
        setting = self._load_pre_bootstrap_seed_import_setting(db)
        if setting is None:
            return False
        if source is None:
            db.delete(setting)
            return True

        source = str(source).strip()
        if not source:
            return False
        payload = self._decode_pre_bootstrap_seed_payload(setting)
        sources = (
            set(self._normalize_pre_bootstrap_seed_sources(payload.get("sources")))
            if payload is not None
            else set()
        )
        if source not in sources:
            return False
        sources.remove(source)
        if not sources:
            db.delete(setting)
            return True

        setting.value = json.dumps(
            {
                "schema_version": PRE_BOOTSTRAP_SEED_IMPORT_SCHEMA_VERSION,
                "sources": sorted(sources),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        )
        setting.category = PRE_BOOTSTRAP_SEED_IMPORT_CATEGORY
        setting.description = (
            "Pristine startup task may have imported raw bootstrap seed rows."
        )
        return True

    def has_pre_bootstrap_seed_import_marker(self, db: Session) -> bool:
        setting = self._load_pre_bootstrap_seed_import_setting(db)
        return self._has_valid_pre_bootstrap_seed_metadata(setting)

    def is_pre_bootstrap_seed_import_installation(self, db: Session) -> bool:
        """Return true only for raw rows imported by marked startup work."""
        return (
            self.has_pre_bootstrap_seed_import_marker(db)
            and not self.has_bootstrap_output_rows(db)
        )

    def _load_pre_bootstrap_seed_import_setting(
        self,
        db: Session,
    ) -> AppSetting | None:
        return (
            db.query(AppSetting)
            .filter(AppSetting.key == PRE_BOOTSTRAP_SEED_IMPORT_KEY)
            .one_or_none()
        )

    @staticmethod
    def _decode_pre_bootstrap_seed_payload(
        setting: AppSetting,
    ) -> dict | None:
        try:
            payload = json.loads(setting.value)
        except (TypeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _normalize_pre_bootstrap_seed_sources(raw_sources: object) -> tuple[str, ...]:
        if not isinstance(raw_sources, list):
            return ()
        return tuple(
            sorted(
                {
                    source.strip()
                    for source in raw_sources
                    if isinstance(source, str) and source.strip()
                }
            )
        )

    def _has_valid_pre_bootstrap_seed_metadata(
        self,
        setting: AppSetting | None,
    ) -> bool:
        if (
            setting is None
            or setting.category != PRE_BOOTSTRAP_SEED_IMPORT_CATEGORY
        ):
            return False
        payload = self._decode_pre_bootstrap_seed_payload(setting)
        if payload is None:
            return False
        if (
            payload.get("schema_version")
            != PRE_BOOTSTRAP_SEED_IMPORT_SCHEMA_VERSION
        ):
            return False
        if not self._normalize_pre_bootstrap_seed_sources(payload.get("sources")):
            return False
        updated_at_raw = payload.get("updated_at")
        if not isinstance(updated_at_raw, str):
            return False
        try:
            datetime.fromisoformat(updated_at_raw)
        except ValueError:
            return False
        return True

    def has_core_market_data(self, db: Session, market: str) -> bool:
        return (
            self._has_active_universe_rows(db, market)
            and self._has_price_rows(db, market)
            and self._has_fundamental_rows(db, market)
        )

    def has_completed_auto_scan(
        self,
        db: Session,
        market: str,
        *,
        bootstrap_started_at: datetime | None = None,
    ) -> bool:
        normalized_market = self.normalize_market(market)
        query = (
            db.query(Scan.id)
            .join(FeatureRun, FeatureRun.id == Scan.feature_run_id)
            .filter(
                Scan.universe_market == normalized_market,
                Scan.status == "completed",
                Scan.trigger_source == SCAN_TRIGGER_SOURCE_AUTO,
                FeatureRun.status == "published",
            )
        )
        if bootstrap_started_at is not None:
            query = query.filter(Scan.started_at >= bootstrap_started_at)
        return query.limit(1).first() is not None

    def evaluate(
        self,
        db: Session,
        *,
        enabled_markets: list[str],
        bootstrap_started_at: datetime | None = None,
        expected_formula_versions: Mapping[str, str] | None = None,
    ) -> BootstrapReadiness:
        normalized_markets = [
            self.normalize_market(market) for market in enabled_markets
        ]
        normalized_expectations = {
            self.normalize_market(market): formula_version
            for market, formula_version in (expected_formula_versions or {}).items()
        }
        return BootstrapReadiness(
            empty_system=self.is_empty_system(db),
            market_results={
                market: MarketBootstrapReadiness(
                    market=market,
                    core_ready=self.has_core_market_data(db, market),
                    scan_ready=self.has_completed_auto_scan(
                        db,
                        market,
                        bootstrap_started_at=bootstrap_started_at,
                    ),
                    rs_ready=self._has_expected_formula(
                        db,
                        market=market,
                        expected_formula_version=normalized_expectations.get(market),
                    ),
                )
                for market in normalized_markets
            },
        )

    def stage_status(self, db: Session, market: str) -> dict[str, str]:
        """Stored output of each derived stage the market supports.

        Separate from ``evaluate``: these stages never gate the scanner, so a
        failed breadth run shows here as ``missing`` instead of blocking.
        """
        market = self.normalize_market(market)
        capabilities = get_market_catalog().get(market).capabilities
        outputs = {}
        if capabilities.breadth:
            outputs.update(breadth=MarketBreadth, exposure=MarketExposure)
        if capabilities.group_rankings:
            outputs["groups"] = IBDGroupRank
        return {
            stage: (
                "ready"
                if db.query(model.id).filter(model.market == market).limit(1).first()
                else "missing"
            )
            for stage, model in outputs.items()
        }

    def feature_status(self, db: Session, *, enabled_markets: list[str]) -> dict[str, str]:
        """Publication state of optional features backed by external providers."""
        from ..config import settings
        from ..infra.db.models.options_analytics import OptionsAnalyticsPointer
        from ..infra.db.models.social_signals import (
            SocialSignalRunPointer,
            SocialSourceRegistry,
        )
        from ..use_cases.options_analytics import (
            OPTIONS_ANALYTICS_CALCULATION_VERSION,
        )

        def published(query) -> str:
            return "ready" if query.limit(1).first() is not None else "missing"

        options_on = settings.options_analytics_enabled and any(
            get_market_catalog().get(market).capabilities.options_analytics
            for market in enabled_markets
        )
        registry = db.get(SocialSourceRegistry, 1)
        social_on = (
            registry is not None
            and registry.mode == "live"
            and registry.provider != "disabled"
        )
        return {
            "cot": "ready" if has_compatible_cot_publication(db) else "missing",
            "options": (
                published(
                    # The read path keys publications by calculation version.
                    db.query(OptionsAnalyticsPointer).filter(
                        OptionsAnalyticsPointer.market == "US",
                        OptionsAnalyticsPointer.calculation_version
                        == OPTIONS_ANALYTICS_CALCULATION_VERSION,
                    )
                )
                if options_on
                else "disabled"
            ),
            "social": (
                published(db.query(SocialSignalRunPointer)) if social_on else "disabled"
            ),
        }

    def _has_expected_formula(
        self,
        db: Session,
        *,
        market: str,
        expected_formula_version: str | None,
    ) -> bool:
        if expected_formula_version is None:
            return True
        try:
            active_formula = MarketRsRunRepository().active_formula(
                db,
                market=market,
            )
        except MarketRsFormulaNotConfigured:
            return False
        return active_formula == expected_formula_version

    def _has_active_universe_rows(
        self,
        db: Session,
        market: str | None = None,
    ) -> bool:
        query = db.query(StockUniverse.id).filter(StockUniverse.is_active.is_(True))
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None

    def _has_price_rows(
        self,
        db: Session,
        market: str | None = None,
    ) -> bool:
        query = (
            db.query(StockPrice.id)
            .join(StockUniverse, StockUniverse.symbol == StockPrice.symbol)
            .filter(StockUniverse.is_active.is_(True))
        )
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None

    def _has_fundamental_rows(
        self,
        db: Session,
        market: str | None = None,
    ) -> bool:
        query = (
            db.query(StockFundamental.id)
            .join(StockUniverse, StockUniverse.symbol == StockFundamental.symbol)
            .filter(StockUniverse.is_active.is_(True))
        )
        if market is not None:
            query = query.filter(StockUniverse.market == self.normalize_market(market))
        return query.limit(1).first() is not None
