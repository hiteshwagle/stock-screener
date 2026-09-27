"""Bulk read of daily features and explicitly identified IBD metadata."""

from sqlalchemy import JSON, and_, case, column, func, null, select, true

from app.domain.feature_store.run_metadata import feature_run_market
from app.infra.db.models.feature_store import (
    FeatureRun,
    FeatureRunPointer,
    FeatureRunUniverseSymbol,
    StockFeatureDaily,
)
from app.infra.db.portability import is_postgres
from app.models.industry import IBDIndustryGroup
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse

# The only details_json keys the Matrix reads. A US publication carries ~10k
# rows whose full documents total ~166 MB compressed; selecting the whole
# column OOM-killed the reader, so only these values leave the database.
MATRIX_DETAIL_KEYS = (
    "gics_sector",
    "price_change_1d",
    "perf_week",
    "perf_month",
    "rs_rating",
)


def _matrix_detail_columns(*, postgres):
    """Return (join target, columns) projecting ``MATRIX_DETAIL_KEYS``.

    Each column is typed ``JSON`` so it decodes to exactly the value the old
    ``details_json.get(key)`` produced (int stays int, a string stays a
    string); the payload builder's normalization is unchanged.

    ``details_json`` is ``json``, not ``jsonb``, so on PostgreSQL every
    ``->`` re-parses the whole stored document. ``json_to_record`` parses it
    once per row instead. It raises on a non-object, and SQLAlchemy stores a
    Python ``None`` as the JSON scalar ``null``, so anything but an object is
    mapped to SQL NULL first (``json_typeof`` only lexes the first token).
    The function returns no row for NULL, hence the ``LEFT ... ON true``
    join: such a feature row is still listed, with every detail missing, as
    the old ``isinstance(details, dict)`` fallback did. Other dialects (the
    SQLite unit tests) use per-key extraction.
    """
    if postgres:
        details = StockFeatureDaily.details_json
        object_details = case(
            (func.json_typeof(details) == "object", details),
            else_=null(),
        )
        record = (
            func.json_to_record(object_details)
            .table_valued(*(column(key, JSON) for key in MATRIX_DETAIL_KEYS))
            .render_derived(name="matrix_details", with_types=True)
        )
        return record, [record.c[key] for key in MATRIX_DETAIL_KEYS]
    return None, [StockFeatureDaily.details_json[key] for key in MATRIX_DETAIL_KEYS]


def matrix_rows_statement(*, run_id, market, postgres):
    """One publication's Matrix rows, without the full details documents."""
    detail_record, detail_columns = _matrix_detail_columns(postgres=postgres)
    statement = (
        select(
            StockFeatureDaily.symbol,
            StockFeatureDaily.as_of_date,
            FeatureRunUniverseSymbol.symbol.label("member_symbol"),
            StockUniverse.name,
            StockUniverse.sector,
            StockFundamental.market_cap_usd,
            StockFundamental.updated_at,
            IBDIndustryGroup.industry_group,
            IBDIndustryGroup.source,
            IBDIndustryGroup.confidence,
            IBDIndustryGroup.updated_at,
            *detail_columns,
        )
        .select_from(StockFeatureDaily)
        .outerjoin(
            FeatureRunUniverseSymbol,
            and_(
                FeatureRunUniverseSymbol.run_id == StockFeatureDaily.run_id,
                FeatureRunUniverseSymbol.symbol == StockFeatureDaily.symbol,
            ),
        )
        .outerjoin(
            StockUniverse,
            and_(
                StockUniverse.symbol == StockFeatureDaily.symbol,
                StockUniverse.market == market,
            ),
        )
        .outerjoin(StockFundamental, StockFundamental.symbol == StockFeatureDaily.symbol)
        .outerjoin(
            IBDIndustryGroup,
            and_(
                IBDIndustryGroup.symbol == StockFeatureDaily.symbol,
                IBDIndustryGroup.market == market,
            ),
        )
    )
    if detail_record is not None:
        statement = statement.outerjoin(detail_record, true())
    return statement.where(StockFeatureDaily.run_id == run_id)


class GroupMatrixRepository:
    def latest_published_run(self, db, *, market):
        pointer = db.get(FeatureRunPointer, f"latest_published_market:{market}")
        if pointer:
            run = db.get(FeatureRun, pointer.run_id)
            if run and run.status == "published" and feature_run_market(run) == market:
                return run
        runs = (
            db.query(FeatureRun)
            .filter(FeatureRun.status == "published")
            .order_by(FeatureRun.published_at.desc(), FeatureRun.id.desc())
        )
        return next((run for run in runs if feature_run_market(run) == market), None)

    def universe_count(self, db, *, run_id):
        return (
            db.query(func.count(FeatureRunUniverseSymbol.symbol))
            .filter(FeatureRunUniverseSymbol.run_id == run_id)
            .scalar()
        )

    def load_rows(self, db, *, run_id, market):
        rows = db.execute(
            matrix_rows_statement(
                run_id=run_id, market=market, postgres=is_postgres(db)
            )
        ).all()
        result = []
        for (
            symbol,
            as_of,
            member,
            name,
            sector,
            cap,
            cap_date,
            group,
            source,
            confidence,
            group_date,
            gics_sector,
            price_change_1d,
            perf_week,
            perf_month,
            rs_rating,
        ) in rows:
            if member is None:
                raise ValueError(
                    "Matrix feature row is not a member of its publication"
                )
            result.append(
                {
                    "symbol": symbol,
                    "feature_as_of_date": as_of,
                    "company_name": name,
                    "sector": gics_sector or sector,
                    "ibd_industry_group": group,
                    "classification_source": source,
                    "classification_confidence": confidence,
                    "classification_updated_at": group_date,
                    "market_cap_usd": cap,
                    "fundamentals_updated_at": cap_date,
                    "price_change_1d": price_change_1d,
                    "price_change_1w": perf_week,
                    "price_change_1m": perf_month,
                    "rs_rating": rs_rating,
                }
            )
        return result
