"""Point-in-time reconstruction of Market universe membership."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from zoneinfo import ZoneInfo

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.domain.markets.catalog import MarketCatalog, get_market_catalog
from app.models.stock_universe import (
    StockUniverse,
    StockUniverseStatusEvent,
    UNIVERSE_EVENT_STATUS_CHANGED,
    UNIVERSE_STATUS_ACTIVE,
)
from app.services.market_calendar_service import MarketCalendarService


@dataclass(frozen=True)
class PointInTimeUniverseMember:
    symbol: str
    currency: str
    is_common_stock: bool = True


@dataclass(frozen=True)
class PointInTimeUniverse:
    market: str
    as_of_date: date
    symbols: tuple[str, ...]
    universe_hash: str
    members: tuple[PointInTimeUniverseMember, ...] = ()


class PointInTimeUniverseUnavailable(RuntimeError):
    """Raised when historical lifecycle evidence cannot reproduce membership."""


def hash_point_in_time_universe_symbols(symbols: tuple[str, ...]) -> str:
    payload = "".join(f"{symbol}\n" for symbol in symbols).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PointInTimeUniverseService:
    def __init__(
        self,
        *,
        market_calendar: MarketCalendarService | None = None,
        market_catalog: MarketCatalog | None = None,
    ) -> None:
        self._market_calendar = market_calendar or MarketCalendarService()
        self._market_catalog = market_catalog or get_market_catalog()

    def _snapshot(
        self,
        *,
        market: str,
        as_of_date: date,
        symbols: tuple[str, ...],
        members: tuple[PointInTimeUniverseMember, ...] = (),
    ) -> PointInTimeUniverse:
        return PointInTimeUniverse(
            market=market,
            as_of_date=as_of_date,
            symbols=symbols,
            universe_hash=hash_point_in_time_universe_symbols(symbols),
            members=members,
        )

    def earliest_membership_date(
        self,
        db: Session,
        *,
        market: str,
    ) -> date | None:
        """First market-local date whose historical universe can have members.

        ``resolve`` admits only rows first seen before the end of
        ``as_of_date`` in the market's timezone, so every earlier date
        resolves to an empty universe: it is unavailable, not a failure.
        Returns ``None`` when the market has no common-stock rows at all.
        """
        normalized = self._market_calendar.normalize_market(market)
        first_seen = (
            db.query(func.min(StockUniverse.first_seen_at))
            .filter(
                StockUniverse.market == normalized,
                StockUniverse.is_common_stock.is_(True),
            )
            .scalar()
        )
        if first_seen is None:
            return None
        if first_seen.tzinfo is None:
            first_seen = first_seen.replace(tzinfo=timezone.utc)
        market_timezone = ZoneInfo(
            self._market_catalog.get(normalized).display_timezone
        )
        return first_seen.astimezone(market_timezone).date()

    def resolve(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date,
    ) -> PointInTimeUniverse:
        normalized = self._market_calendar.normalize_market(market)
        if as_of_date == self._market_calendar.market_now(normalized).date():
            rows = tuple(
                db.query(StockUniverse)
                .filter(
                    StockUniverse.market == normalized,
                    StockUniverse.active_filter(),
                    StockUniverse.is_common_stock.is_(True),
                )
                .order_by(StockUniverse.symbol.asc())
                .all()
            )
            symbols = tuple(row.symbol for row in rows)
            return self._snapshot(
                market=normalized,
                as_of_date=as_of_date,
                symbols=symbols,
                members=tuple(
                    PointInTimeUniverseMember(
                        symbol=row.symbol,
                        currency=row.currency,
                        is_common_stock=row.is_common_stock,
                    )
                    for row in rows
                ),
            )

        market_timezone = ZoneInfo(
            self._market_catalog.get(normalized).display_timezone
        )
        cutoff = datetime.combine(
            as_of_date + timedelta(days=1),
            time.min,
            tzinfo=market_timezone,
        ).astimezone(timezone.utc)

        candidate_rows = tuple(
            db.query(StockUniverse)
            .filter(
                StockUniverse.market == normalized,
                StockUniverse.first_seen_at < cutoff,
                StockUniverse.is_common_stock.is_(True),
            )
            .order_by(StockUniverse.symbol.asc())
                .all()
        )
        if not candidate_rows:
            return self._snapshot(
                market=normalized,
                as_of_date=as_of_date,
                symbols=(),
            )
        candidates = tuple(row.symbol for row in candidate_rows)
        rows_by_symbol = {row.symbol: row for row in candidate_rows}

        events = (
            db.query(StockUniverseStatusEvent)
            .filter(
                StockUniverseStatusEvent.symbol.in_(candidates),
                StockUniverseStatusEvent.event_type
                == UNIVERSE_EVENT_STATUS_CHANGED,
                StockUniverseStatusEvent.created_at < cutoff,
            )
            .order_by(
                StockUniverseStatusEvent.symbol.asc(),
                StockUniverseStatusEvent.created_at.desc(),
                StockUniverseStatusEvent.id.desc(),
            )
            .all()
        )
        latest_by_symbol: dict[str, StockUniverseStatusEvent] = {}
        for event in events:
            latest_by_symbol.setdefault(event.symbol, event)

        missing = tuple(
            symbol for symbol in candidates if symbol not in latest_by_symbol
        )
        if missing:
            raise PointInTimeUniverseUnavailable(
                f"{normalized} historical universe for {as_of_date.isoformat()} "
                f"is missing lifecycle events for: {', '.join(missing)}"
            )

        symbols = tuple(
            symbol
            for symbol in candidates
            if latest_by_symbol[symbol].new_status == UNIVERSE_STATUS_ACTIVE
        )
        return self._snapshot(
            market=normalized,
            as_of_date=as_of_date,
            symbols=symbols,
            members=tuple(
                PointInTimeUniverseMember(
                    symbol=symbol,
                    currency=rows_by_symbol[symbol].currency,
                    is_common_stock=rows_by_symbol[symbol].is_common_stock,
                )
                for symbol in symbols
            ),
        )
