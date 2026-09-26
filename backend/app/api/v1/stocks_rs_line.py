"""RS-line chart-overlay endpoint (kept out of the already-large stocks.py)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ...analysis.patterns.rs_line import blue_dot_series, compute_rs_line
from ...database import get_db
from ...schemas.stock import RSLinePoint, RSLineResponse
from ...services.benchmark_cache_service import BenchmarkCacheService
from ...services.symbol_format import require_valid_symbol
from ...wiring.bootstrap import get_price_cache
from ._price_history import PERIOD_DAYS, resolve_symbol_market, window_cutoff

logger = logging.getLogger(__name__)
router = APIRouter()


def _load_rs_line(db: Session, symbol: str, period: str) -> RSLineResponse:
    """Compute the RS line + blue-dot dates for charting.

    New-high detection runs over the full cached window (a 252-day lookback needs
    more history than the displayed period); the output series is then trimmed to
    the requested ``period`` so it aligns with the candlestick x-axis.
    """
    days = PERIOD_DAYS.get(period)
    if days is None:
        raise HTTPException(status_code=422, detail=f"Unsupported period: {period}")

    market = resolve_symbol_market(db, symbol) or "US"
    cache_period = "5y" if period == "5y" else "2y"

    benchmark_service = BenchmarkCacheService()
    benchmark_symbol = benchmark_service.get_benchmark_symbol(market)

    stock_df = get_price_cache().get_cached_only(symbol.upper(), period=cache_period)
    benchmark_df = benchmark_service.get_benchmark_data(market=market, period=cache_period)

    empty = RSLineResponse(symbol=symbol.upper(), benchmark_symbol=benchmark_symbol, rs_line=[], blue_dots=[])
    if stock_df is None or len(stock_df) == 0 or benchmark_df is None or len(benchmark_df) == 0:
        logger.warning("RS line unavailable for %s (market=%s): missing cached data", symbol, market)
        return empty

    rs_line_full = compute_rs_line(stock_df["Close"], benchmark_df["Close"], normalize=True)
    blue_full = blue_dot_series(stock_df["Close"], benchmark_df["Close"])

    cutoff = window_cutoff(rs_line_full.index, days)
    rs_window = rs_line_full[rs_line_full.index >= cutoff].dropna()
    blue_window = blue_full[(blue_full.index >= cutoff) & blue_full]

    return RSLineResponse(
        symbol=symbol.upper(),
        benchmark_symbol=benchmark_symbol,
        rs_line=[RSLinePoint(time=ts.strftime("%Y-%m-%d"), value=round(float(v), 4)) for ts, v in rs_window.items()],
        blue_dots=[ts.strftime("%Y-%m-%d") for ts in blue_window.index],
    )


@router.get("/{symbol}/rs-line", response_model=RSLineResponse)
def get_rs_line(
    symbol: str = Depends(require_valid_symbol),
    period: str = "6mo",
    db: Session = Depends(get_db),
):
    """RS line (stock / market benchmark) plus 'blue dot' dates for chart overlay.

    A blue dot marks a date where the RS line made a new 252-day high while price
    did not — emerging leadership ahead of the breakout.
    """
    return _load_rs_line(db, symbol, period)
