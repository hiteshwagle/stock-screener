"""Resolve metadata that is frozen into breadth contributor snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

from sqlalchemy.orm import Session

from app.domain.feature_store.run_metadata import feature_run_market
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.stock_universe import StockUniverse
from app.services.ibd_industry_service import IBDIndustryService

from .contributors import NO_GROUP_LABEL
from .types import BreadthContributorMetadata


_METADATA_KEYS = ("company_name", "ibd_industry_group")


def _text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _details_value(details: object, key: str) -> Any:
    if not isinstance(details, dict):
        return None
    if details.get(key) is not None:
        return details[key]
    extended = details.get("extended")
    return extended.get(key) if isinstance(extended, dict) else None


def _run_order(run: FeatureRun) -> tuple[int, datetime, int]:
    published = 1 if run.status == "published" else 0
    timestamp = (
        run.published_at
        or run.completed_at
        or run.updated_at
        or run.created_at
        or datetime.min
    )
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.replace(tzinfo=None)
    return published, timestamp, int(run.id or 0)


class BreadthContributorMetadataLoader:
    """Load current or exact-date metadata without look-ahead."""

    @staticmethod
    def current(
        db: Session,
        market: str,
        symbols: Sequence[str],
    ) -> Mapping[str, BreadthContributorMetadata]:
        normalized_market = str(market).strip().upper()
        normalized_symbols = tuple(sorted({str(item).strip().upper() for item in symbols}))
        rows = (
            db.query(StockUniverse.symbol, StockUniverse.name)
            .filter(
                StockUniverse.market == normalized_market,
                StockUniverse.symbol.in_(normalized_symbols),
            )
            .all()
            if normalized_symbols
            else []
        )
        names = {str(symbol).upper(): _text(name) for symbol, name in rows}
        memberships = IBDIndustryService.get_group_memberships(
            db,
            market=normalized_market,
        )
        groups = {
            str(symbol).upper(): group
            for group, group_symbols in memberships.items()
            for symbol in group_symbols
            if _text(group) is not None
        }
        return MappingProxyType(
            {
                symbol: BreadthContributorMetadata(
                    company_name=names.get(symbol),
                    ibd_industry_group=_text(groups.get(symbol)) or NO_GROUP_LABEL,
                )
                for symbol in normalized_symbols
            }
        )

    @staticmethod
    def historical(
        db: Session,
        market: str,
        symbols_by_date: Mapping[date, Sequence[str]],
    ) -> Mapping[date, Mapping[str, BreadthContributorMetadata]]:
        normalized_market = str(market).strip().upper()
        ordered_dates = tuple(sorted(symbols_by_date))
        runs = (
            db.query(FeatureRun)
            .filter(
                FeatureRun.as_of_date.in_(ordered_dates),
                FeatureRun.status.in_(("published", "completed")),
            )
            .all()
            if ordered_dates
            else []
        )
        selected_by_date: dict[date, FeatureRun] = {}
        for run in runs:
            if feature_run_market(run) != normalized_market:
                continue
            existing = selected_by_date.get(run.as_of_date)
            if existing is None or _run_order(run) > _run_order(existing):
                selected_by_date[run.as_of_date] = run

        run_ids = tuple(run.id for run in selected_by_date.values())
        # Project only the metadata keys (top level and ``extended``): a US
        # run's full details documents expand to ~2.3 GB in Python and
        # OOM-killed the 512M breadth worker. A missing key, or a document
        # that is not an object, extracts as NULL.
        details = StockFeatureDaily.details_json
        feature_rows = (
            db.query(
                StockFeatureDaily.run_id,
                StockFeatureDaily.symbol,
                *(details[key] for key in _METADATA_KEYS),
                *(details[("extended", key)] for key in _METADATA_KEYS),
            )
            .filter(StockFeatureDaily.run_id.in_(run_ids))
            .all()
            if run_ids
            else []
        )
        count = len(_METADATA_KEYS)
        details_by_run_symbol = {
            (run_id, str(symbol).upper()): {
                **{key: value for key, value in zip(_METADATA_KEYS, values[:count])},
                "extended": dict(zip(_METADATA_KEYS, values[count:])),
            }
            for run_id, symbol, *values in feature_rows
        }

        result: dict[date, Mapping[str, BreadthContributorMetadata]] = {}
        for calculation_date in ordered_dates:
            run = selected_by_date.get(calculation_date)
            values: dict[str, BreadthContributorMetadata] = {}
            for raw_symbol in symbols_by_date[calculation_date]:
                symbol = str(raw_symbol).strip().upper()
                details = (
                    details_by_run_symbol.get((run.id, symbol), {})
                    if run is not None
                    else {}
                )
                values[symbol] = BreadthContributorMetadata(
                    company_name=_text(_details_value(details, "company_name")),
                    ibd_industry_group=(
                        _text(_details_value(details, "ibd_industry_group"))
                        or NO_GROUP_LABEL
                    ),
                )
            result[calculation_date] = MappingProxyType(values)
        return MappingProxyType(result)
