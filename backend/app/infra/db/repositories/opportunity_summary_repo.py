"""SQL aggregate readers for persisted opportunity-state projections."""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy import (
    ColumnElement,
    Select,
    String,
    cast,
    func,
    literal_column,
    select,
    text,
)
from sqlalchemy.orm import Session, sessionmaker

from app.domain.scanning.opportunity_state import ActionState
from app.domain.scanning.opportunity_summary import OpportunityStateSummary
from app.infra.db.models.feature_store import StockFeatureDaily
from app.infra.db.portability import json_text
from app.models.scan_result import ScanResult

logger = logging.getLogger(__name__)

# The two details_json keys the opportunity projection carries, in the order the
# index ``ix_sfd_run_action_state_survivor`` (migration ``20260926_0058``) stores
# them. ``action_state`` is filtered, so it has to sit directly after the
# ``run_id`` prefix to be reachable as an index condition; ``correction_survivor``
# is only ever tested for truth and goes last.
ACTION_STATE_KEY = "action_state"
CORRECTION_SURVIVOR_KEY = "correction_survivor"

# The two spellings of JSON ``true`` that ``->>`` can yield, one per backend.
# PostgreSQL keeps the boolean and renders it as text ``'true'``; SQLite stores it
# as the integer ``1``, so ``->>`` returns ``'1'``. The counted predicate has to
# match both, or identical data counts differently depending on the backend --
# see ``counted_opportunity_predicates``.
_SURVIVOR_TRUE_TEXT = "true"
_SURVIVOR_TRUE_SQLITE = "1"

# Migration 20260926_0058 carries the index the counted path depends on. It is
# asked for by name because a missing index is not an error anywhere: the query
# still returns the right answer, just slowly.
_COUNT_INDEX_NAME = "ix_sfd_run_action_state_survivor"

# The probe is matched against the relation the aggregation actually reads, not
# against the name alone: PostgreSQL allows the same index name in another
# schema, and a valid index there would otherwise answer for a missing or
# invalid one here -- selecting the slow counted path on the strength of an
# index it cannot use.
#
# ``to_regclass`` resolves the name exactly the way the runtime query does --
# first match in ``search_path`` -- and yields its OID, so the probe is tied to
# the *same* relation the aggregation will read. Matching every same-named table
# across ``current_schemas(false)`` would not: with the table in a later schema
# but a valid index in the first, the probe would answer "usable" for a relation
# the counted query cannot index.
#
# ``indisvalid AND indisready`` -- an interrupted ``CREATE INDEX CONCURRENTLY``
# leaves a same-named index that exists, refuses inserts, and cannot serve a
# sequential scan. Present is not the same as usable, so both flags are checked.
_INDEX_USABLE_SQL = text(
    """
    SELECT count(*)
    FROM pg_index i
    JOIN pg_class idx ON idx.oid = i.indexrelid
    WHERE idx.relname = :name
      AND i.indrelid = to_regclass(:table)
      AND i.indisvalid
      AND i.indisready
    """
)


def _count_index_is_usable(session: Session) -> bool:
    """Whether the index the counted path needs is present, usable, and here.

    The counted shape only pays off with this index. Its own measurement puts
    the same sixteen counts at 6.8 ms with the index and 56.7 s without, so a
    build that was never run, was rolled back, or failed halfway would turn the
    Daily Snapshot into a multi-minute scan rather than a slow query.

    A backend without these catalogs returns ``False``, which keeps the grouped
    fallback in charge -- correct on every backend, fastest on one.

    The probe runs on a **connection-level** savepoint, not
    ``session.begin_nested()``. That distinction is the whole point: the ORM
    version flushes the session unconditionally before opening the savepoint --
    measured, even with the project's ``autoflush=False``:

        sessionmaker(autoflush=False); session.add(pending); with session.begin_nested():
          -> before_flush fired, INSERT issued

    This repository shares one session with every writer repository, so a
    pending write that fails during that flush would surface here as an index
    probe failure. ``_recover_aborted_probe`` could then roll back the caller's
    unit of work and this method would carry on through the fallback -- quietly
    discarding staged work that had nothing to do with the index.

    ``session.connection().begin_nested()`` issues ``SAVEPOINT`` on the
    connection without touching the ORM's pending state. Verified against the
    same probe-failure path:

        before_flush fired   : no
        staged write retained: yes
        session still usable : yes
    """
    try:
        # ``begin_nested`` on the connection, not the session: no flush, so the
        # caller's pending writes are neither issued nor risked here.
        with session.connection().begin_nested():
            return bool(
                session.execute(
                    _INDEX_USABLE_SQL,
                    {
                        "name": _COUNT_INDEX_NAME,
                        "table": StockFeatureDaily.__tablename__,
                    },
                ).scalar()
            )
    except Exception as exc:  # noqa: BLE001 - an unreadable catalog means "not usable".
        _recover_aborted_probe(session)
        logger.warning(
            "Opportunity summary: index probe failed, using the grouped fallback (%s)",
            exc,
        )
        return False


def _recover_aborted_probe(session: Session) -> None:
    """Make sure the session can still serve the fallback query.

    ``begin_nested()`` unwinds its savepoint on the way out, but a dialect that
    cannot open one at all leaves the aborted transaction in place. If the
    session is still unusable, roll it back rather than hand it to the caller.
    """
    try:
        if session.is_active:
            return
        session.rollback()
    except Exception as exc:  # noqa: BLE001 - recovery must not mask the original failure.
        logger.warning("Opportunity summary: probe recovery failed (%s)", exc)


def survivor_predicate(details) -> ColumnElement:
    """The one survivor test, for any ``details_json``/``details`` column.

    Single definition on purpose. ``for_feature_run`` (counted) and
    ``for_scan``/``_aggregate`` (grouped) read the same projection through
    different shapes, and before this they decided the survivor flag with
    different SQL: the grouped path used ``as_boolean().is_(True)``, the counted
    path a text comparison. Identical data then classified differently depending
    on which path read it -- on PostgreSQL the cast also accepts ``'t'``,
    ``'yes'`` and ``'on'``, which the text comparison rejected, and it raises on
    a value like ``'2'`` while the text comparison does not.

    Cast-free by requirement: this expression is also the one migration
    ``20260926_0058`` indexes, and ``CREATE INDEX`` evaluates it against every
    row of every historical run. A cast that can raise would fail the startup
    migration on a single bad value.

    ``IN`` matching ``lower(...)`` needs both members because the two backends
    spell JSON ``true`` differently: PostgreSQL keeps the boolean and ``->>``
    renders the text ``'true'``; SQLite stores it as the integer ``1``, so its
    ``->>`` yields ``'1'``. ``'1'`` never occurs in the Postgres text form, so
    the extra member is inert there.

    Values outside the list -- ``'t'``, ``'yes'``, ``'on'``, ``'2'`` -- are not
    survivors. That is the deliberate reading: only the two canonical spellings
    of the flag count, and no input can make this raise.
    """
    return func.lower(json_text(details, (CORRECTION_SURVIVOR_KEY,))).in_(
        [_SURVIVOR_TRUE_TEXT, _SURVIVOR_TRUE_SQLITE]
    )


def counted_opportunity_predicates() -> tuple[ColumnElement, ColumnElement]:
    """The two predicates ``ix_sfd_run_action_state_survivor`` was built for.

    Returned as SQLAlchemy expressions for the *runtime* callers (the counted
    feature-run path and the drift guard) rather than only as SQL text, so a
    change here is what the migration is compared against -- one definition, two
    consumers, no second copy to forget.

    Both keys go through ``portability.json_text``, which inlines them as SQL
    literals. ``details["action_state"].as_string()`` would render
    ``->> %(details_json_1)s`` instead: psycopg2 interpolates that client-side,
    but a generic plan (and every server-side-binding driver) sees a parameter,
    cannot match it to the index's literal, and silently drops the index -- which
    on this table means a 154 s full scan instead of milliseconds.

    The survivor test is a cast-free ``lower(...) IN ('true', '1')`` rather than
    ``CAST(... AS BOOLEAN) IS true``. A boolean cast raises
    ``invalid input syntax for type boolean`` on non-boolean text, and inside the
    index that fails the startup migration (one bad value anywhere in the table)
    and every later insert/update carrying such a value. The ``IN`` list cannot
    raise for any input.

    Both list members are needed for the two backends to agree. PostgreSQL stores
    a JSON ``true`` and ``->>`` hands it back as the text ``'true'``; SQLite
    stores the same value as the integer ``1``, so its ``->>`` yields ``'1'``.
    Matching only ``'true'`` silently counts every survivor as a non-survivor on
    SQLite, where the grouped pivot used to read it as truthy -- so the survivor
    counts would differ by backend for identical data. ``'1'`` never occurs in
    the Postgres text form, so the extra member is inert there.
    """
    details = StockFeatureDaily.details_json
    return (
        cast(json_text(details, (ACTION_STATE_KEY,)), String),
        survivor_predicate(details),
    )


class SqlOpportunityStateSummaryRepository:
    """Reads the opportunity projection from the feature store, two ways.

    ``for_feature_run`` counts the projection with both keys in the ``WHERE``
    clause; ``for_scan``/``_aggregate`` group it. Both classify a row through
    the shared ``survivor_predicate``, so the shape differs but the answer
    cannot.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def for_scan(self, scan_id: str) -> OpportunityStateSummary:
        """Aggregate the opportunity projection for one legacy scan run."""
        return self._aggregate(
            model=ScanResult,
            details=ScanResult.details,
            predicate=ScanResult.scan_id == scan_id,
        )

    def for_feature_run(self, run_id: int) -> OpportunityStateSummary:
        """Aggregate the opportunity projection for one feature-store run.

        Counted rather than grouped: the feature-store table is large enough per
        run that the grouped shape's per-row document reads dominate the Daily
        Snapshot build. See ``_count_feature_run``.

        Falls back to the grouped shape when the index the counted path needs is
        absent or unusable. Both shapes return the same summary, so the caller
        only sees a latency difference -- and the counted shape without its index
        is the slow one.
        """
        if not _count_index_is_usable(self._session):
            logger.warning(
                "Opportunity summary: %s is missing or unusable; falling back to the "
                "grouped aggregate for run %s",
                _COUNT_INDEX_NAME,
                run_id,
            )
            return self._aggregate(
                model=StockFeatureDaily,
                details=StockFeatureDaily.details_json,
                predicate=StockFeatureDaily.run_id == int(run_id),
            )
        return self._count_feature_run(int(run_id))

    def _count_feature_run(self, run_id: int) -> OpportunityStateSummary:
        """Count the projection with both keys in the ``WHERE`` clause.

        The grouped pass (``_aggregate``) reads ``action_state`` and
        ``correction_survivor`` out of every row and re-parses each row's stored
        ``details_json`` to do it -- 10,079 rows at ~64 kB measured 19,357 ms on
        the production table. Neither ``Index Scan`` nor ``Bitmap Heap Scan`` is
        an index-only scan, so an expression index on the two *grouped* keys does
        not remove that work: the planner picked such an index up and the time
        did not move (19,768 ms).

        Counting per state instead of grouping moves both keys into ``WHERE``,
        where ``ix_sfd_run_action_state_survivor`` (migration ``20260926_0058``)
        serves them as index conditions and the stored document is read far less.
        Measured against a copy of the production table, with the index and its
        ``ANALYZE``: 6,281 / 6,162 / 8,667 / 9,646 / 12,537 ms disk-bound, and
        12.292 ms buffer-warm. In the same session the grouped pass stayed at
        19,176-19,465 ms.

        ``rows_total`` and ``survivor_count`` are served as index-only scans; the
        per-state counts fall back to a ``Bitmap Heap Scan`` when the state
        matches thousands of rows, so they still visit the heap. That residual
        cost is why this rewrite and the migration only make sense together --
        the grouped shape reads every document rather than a subset. Without the
        index this shape costs 154,462 ms, since each of the sixteen counts scans
        the run on its own.

        The keys are extracted through ``json_text`` rather than
        ``details["key"].as_string()``. That is not a stylistic choice: the
        subscript form renders the key as a *bind parameter*
        (``->> %(details_json_1)s``), which a generic plan cannot match to the
        index's inline literal, so the index would silently go unused and this
        shape would fall back to the 154 s full-scan case. ``json_text`` exists
        for exactly this and inlines the key.

        The unknown-state contract is preserved: an unrecognised ``action_state``
        lands in no bucket, as it did when the pivot skipped it, while still
        counting towards ``rows_total`` and, for a survivor, towards
        ``survivor_count``.
        """
        # Read through the shared definition, so the predicates the planner sees
        # and the ones the index was built from cannot drift apart.
        #
        # The survivor test stays in SQL, as it was upstream. Deciding it in
        # Python instead makes ``JSON_EXTRACT(...) IS 1`` collapse to
        # ``bool(...)`` on SQLite, where the string "false" and the integer 2
        # are both truthy -- so identical data would count differently
        # depending on the backend.
        action_state, survivor = counted_opportunity_predicates()
        predicate = StockFeatureDaily.run_id == run_id

        def counted(*conditions) -> Select:
            """``count(*)`` over the run, narrowed by *conditions*.

            Each count emits the same expression text ``_aggregate`` used, so
            the indexed predicate and the query predicate stay byte-identical
            (minus the table qualifier). ``test_opportunity_summary_index_drift``
            pins that against the migration.
            """
            return (
                select(func.count())
                .select_from(StockFeatureDaily)
                .where(predicate, *conditions)
                .scalar_subquery()
            )

        columns = [
            counted().label("rows_total"),
            counted(survivor).label("survivor_count"),
        ]
        for state in ActionState:
            state_predicate = action_state == state.value
            columns.append(counted(state_predicate).label(f"state_{state.value}"))
            columns.append(
                counted(state_predicate, survivor).label(f"survivor_{state.value}")
            )

        # One statement, one round trip: the sixteen counts are correlated
        # scalar subqueries, not sixteen separate queries.
        row = self._session.execute(select(*columns)).one()
        values = row._mapping

        action_state_counts: dict[ActionState, int] = defaultdict(int)
        survivor_action_state_counts: dict[ActionState, int] = defaultdict(int)
        for state in ActionState:
            action_state_counts[state] = values[f"state_{state.value}"]
            survivor_action_state_counts[state] = values[f"survivor_{state.value}"]

        return OpportunityStateSummary(
            rows_total=values["rows_total"],
            survivor_count=values["survivor_count"],
            action_state_counts=dict(action_state_counts),
            survivor_action_state_counts=dict(survivor_action_state_counts),
        )

    def _aggregate(self, *, model, details, predicate) -> OpportunityStateSummary:
        """Read ``action_state`` and ``correction_survivor`` once per row,
        group by the two, and pivot the buckets into a summary.

        Both columns are ``JSON`` rather than ``JSONB``, so each ``->>`` still
        parses the whole stored breakdown: two reads per row is the floor here
        without a schema change. The previous shape paid that per key *and*
        per action state, sixteen parses for the same answer.

        ``details`` is ``scan_results.details`` for a legacy scan run and
        ``stock_feature_daily.details_json`` for a feature-store run; the keys
        are the same in both.
        """
        action_state = details["action_state"].as_string()
        # The survivor test goes through the shared definition, so the grouped
        # path cannot classify a row differently from the counted one. It stays
        # in SQL -- deciding it in Python collapses ``JSON_EXTRACT(...) IS 1`` to
        # ``bool(...)`` on SQLite, where the string "false" and the integer 2 are
        # both truthy, so identical data would count differently by backend.
        survivor = survivor_predicate(details)
        buckets = (
            self._session.query(
                survivor.label("survivor"),
                action_state.label("action_state"),
                func.count().label("rows"),
            )
            .select_from(model)
            .filter(predicate)
            # Group by output position. Repeating the expressions would rely on
            # the driver inlining its bind parameters: psycopg2 interpolates on
            # the client, so the server sees identical text, but a driver that
            # binds server-side (asyncpg) emits $1 in the SELECT and $3 in the
            # GROUP BY, and PostgreSQL then rejects it as a grouping error.
            .group_by(literal_column("1"), literal_column("2"))
            .all()
        )

        rows_total = 0
        survivor_count = 0
        action_state_counts: dict[ActionState, int] = defaultdict(int)
        survivor_action_state_counts: dict[ActionState, int] = defaultdict(int)

        for is_survivor, raw_state, rows in buckets:
            rows_total += rows
            if is_survivor:
                survivor_count += rows
            # Unknown states stay out of every bucket; they are still counted
            # in rows_total so the totals never silently drift.
            try:
                state = ActionState(raw_state)
            except ValueError:
                continue
            action_state_counts[state] += rows
            if is_survivor:
                survivor_action_state_counts[state] += rows

        return OpportunityStateSummary(
            rows_total=rows_total,
            survivor_count=survivor_count,
            action_state_counts=dict(action_state_counts),
            survivor_action_state_counts=dict(survivor_action_state_counts),
        )


class SessionOpportunityStateSummaryReader:
    """Own a short-lived session for telemetry and other service consumers."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def for_scan(self, scan_id: str) -> OpportunityStateSummary:
        with self._session_factory() as session:
            return SqlOpportunityStateSummaryRepository(session).for_scan(scan_id)

    def for_feature_run(self, run_id: int) -> OpportunityStateSummary:
        with self._session_factory() as session:
            return SqlOpportunityStateSummaryRepository(session).for_feature_run(run_id)
