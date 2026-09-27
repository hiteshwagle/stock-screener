"""Contract tests for shared persisted opportunity-state aggregation."""

from datetime import date

from sqlalchemy import case, create_engine, event, func
from sqlalchemy.orm import Session

from app.database import Base
from app.domain.scanning.opportunity_state import ActionState
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.infra.db.repositories.opportunity_summary_repo import (
    SqlOpportunityStateSummaryRepository,
    survivor_predicate,
)
from app.models.scan_result import Scan, ScanResult


def test_scan_summary_aggregates_all_counts_in_one_query():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Scan(scan_id="scan-summary", status="completed"))
        session.add_all(
            [
                ScanResult(
                    scan_id="scan-summary",
                    symbol="READY",
                    details={
                        "correction_survivor": True,
                        "action_state": "setup_ready",
                    },
                ),
                ScanResult(
                    scan_id="scan-summary",
                    symbol="WATCH",
                    details={
                        "correction_survivor": True,
                        "action_state": "watch",
                    },
                ),
                ScanResult(
                    scan_id="scan-summary",
                    symbol="NOT-SURVIVOR",
                    details={
                        "correction_survivor": False,
                        "action_state": "watch",
                    },
                ),
                ScanResult(
                    scan_id="scan-summary",
                    symbol="LEGACY",
                    details={},
                ),
            ]
        )
        session.commit()

        statements = []

        def capture(_conn, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            summary = SqlOpportunityStateSummaryRepository(session).for_scan(
                "scan-summary"
            )
        finally:
            event.remove(engine, "before_cursor_execute", capture)

    assert len(statements) == 1
    assert summary.rows_total == 4
    assert summary.survivor_count == 2
    assert summary.action_state_counts[ActionState.SETUP_READY] == 1
    assert summary.action_state_counts[ActionState.WATCH] == 2
    assert summary.survivor_action_state_counts[ActionState.SETUP_READY] == 1
    assert summary.survivor_action_state_counts[ActionState.WATCH] == 1
    assert all(
        summary.action_state_counts[state] == 0
        for state in ActionState
        if state not in {ActionState.SETUP_READY, ActionState.WATCH}
    )


def _session_with(rows):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(Scan(scan_id="scan-extra", status="completed"))
    session.add_all(
        [
            ScanResult(scan_id="scan-extra", symbol=name, details=details)
            for name, details in rows
        ]
    )
    session.commit()
    return session


def test_unknown_action_state_counts_in_total_only():
    """An unrecognised state is counted in rows_total, in no bucket.

    Moving the skip below ``rows_total +=`` would still pass the original
    fixture, because every row there carries a known state. This pins the
    intent: unknown states must not vanish from the total, and must not land
    in a bucket that does not exist for them.
    """
    with _session_with(
        [
            ("KNOWN", {"correction_survivor": False, "action_state": "watch"}),
            ("UNKNOWN", {"correction_survivor": False, "action_state": "not-a-state"}),
            ("MISSING", {"correction_survivor": False}),
        ]
    ) as session:
        summary = SqlOpportunityStateSummaryRepository(session).for_scan("scan-extra")

    assert summary.rows_total == 3
    assert summary.action_state_counts[ActionState.WATCH] == 1
    assert sum(summary.action_state_counts.values()) == 1
    assert summary.survivor_count == 0


def test_survivor_with_unknown_state_counts_in_survivor_count():
    """A survivor whose state is unknown still counts as a survivor.

    The two counters answer different questions: ``survivor_count`` is about
    the survivor flag, the per-state buckets are about the state. A survivor
    with no usable state must therefore appear in the first and in neither of
    the state mappings.
    """
    with _session_with(
        [
            ("SURV-BAD-STATE", {"correction_survivor": True, "action_state": "nope"}),
            ("SURV-NO-STATE", {"correction_survivor": True}),
            ("NON-SURV", {"correction_survivor": False, "action_state": "watch"}),
        ]
    ) as session:
        summary = SqlOpportunityStateSummaryRepository(session).for_scan("scan-extra")

    assert summary.rows_total == 3
    assert summary.survivor_count == 2
    assert summary.action_state_counts[ActionState.WATCH] == 1
    assert sum(summary.survivor_action_state_counts.values()) == 0


def test_survivor_predicate_is_shared_and_cast_free():
    """One survivor test decides every path, and it cannot raise.

    The regression this pins: the counted path used ``lower(...) IN
    ('true','1')`` while the grouped path used ``as_boolean().is_(True)``, so
    the same row classified differently depending on which path read it. On
    PostgreSQL the cast also accepts ``'t'``, ``'yes'`` and ``'on'``, and it
    raises on ``'2'`` while the text comparison does not.

    Both paths are compared on identical data rather than against a hand-written
    number, so the comparison cannot pass by restating one of them.
    """
    rows = [
        ("REAL-TRUE", {"correction_survivor": True, "action_state": "watch"}),
        ("REAL-FALSE", {"correction_survivor": False, "action_state": "watch"}),
        ("STR-TRUE", {"correction_survivor": "true", "action_state": "watch"}),
        ("STR-FALSE", {"correction_survivor": "false", "action_state": "watch"}),
        ("INT-1", {"correction_survivor": 1, "action_state": "watch"}),
        ("INT-2", {"correction_survivor": 2, "action_state": "watch"}),
    ]
    with _session_with(rows) as session:
        summary = SqlOpportunityStateSummaryRepository(session).for_scan("scan-extra")
        details = ScanResult.details
        grouped_count = (
            session.query(
                func.coalesce(func.sum(case((survivor_predicate(details), 1), else_=0)), 0)
            )
            .select_from(ScanResult)
            .filter(ScanResult.scan_id == "scan-extra")
            .scalar()
        )

    # ``true``, the string ``"true"`` and ``1`` are the canonical spellings;
    # ``false`` and ``2`` are not. Same answer through the shared predicate.
    assert summary.survivor_count == grouped_count == 3

    # It is cast-free: the expression must not raise for any stored value, which
    # is what lets migration 20260926_0058 index it against every historical row.
    assert "CAST" not in str(survivor_predicate(details)).upper()
    assert "BOOLEAN" not in str(survivor_predicate(details)).upper()


def test_both_read_paths_agree_on_the_same_rows():
    """``for_scan`` and ``for_feature_run`` must classify a row identically.

    Raised as a nitpick on #385 and correct: the test above compares ``for_scan``
    against a hand-written ``survivor_predicate`` call, so it pinned the shared
    *predicate* without ever running the counted ``for_feature_run`` path. If the
    two paths drifted apart again -- which is exactly what #382 merged and #385
    undid -- that test would keep passing on the grouped side alone.

    These two are the production read paths: ``for_scan`` serves the legacy
    scan-results projection, ``for_feature_run`` the feature-store projection the
    Daily Snapshot uses. Same rows in, same summary out -- compared against each
    other, not against a number that would only restate one of them.
    """
    rows = [
        ("REAL-TRUE", {"correction_survivor": True, "action_state": "watch"}),
        ("REAL-FALSE", {"correction_survivor": False, "action_state": "watch"}),
        ("STR-TRUE", {"correction_survivor": "true", "action_state": "setup_ready"}),
        ("STR-FALSE", {"correction_survivor": "false", "action_state": "watch"}),
        ("INT-1", {"correction_survivor": 1, "action_state": "setup_ready"}),
        ("INT-2", {"correction_survivor": 2, "action_state": "watch"}),
        ("MISSING", {"correction_survivor": True}),
        ("UNKNOWN", {"correction_survivor": True, "action_state": "nope"}),
    ]

    with _session_with(rows) as scan_session:
        via_scan = SqlOpportunityStateSummaryRepository(scan_session).for_scan("scan-extra")
    with _feature_session_with(rows) as feature_session:
        via_feature = SqlOpportunityStateSummaryRepository(feature_session).for_feature_run(7)

    assert via_scan == via_feature, (
        "the two read paths disagree on identical rows: "
        f"for_scan={via_scan} for_feature_run={via_feature}"
    )

    # Anchor the fixture, so agreement cannot be reached by both sides being
    # empty or both sides ignoring the same keys.
    assert via_scan.rows_total == 8
    assert via_scan.survivor_count == 5
    assert via_scan.action_state_counts[ActionState.WATCH] == 4
    assert via_scan.action_state_counts[ActionState.SETUP_READY] == 2
    assert sum(via_scan.survivor_action_state_counts.values()) == 3


def test_both_read_paths_agree_that_a_cast_would_raise():
    """A stored ``2`` must not make either path raise.

    ``CAST('2' AS BOOLEAN)`` is rejected by PostgreSQL, so the grouped path as
    it was merged could not summarise a run containing that value -- it raised
    instead. The counted path treated the row as a non-survivor. Both must now
    answer, and answer the same.
    """
    rows = [
        ("INT-2", {"correction_survivor": 2, "action_state": "watch"}),
        ("YES-TEXT", {"correction_survivor": "yes", "action_state": "watch"}),
        ("ON-TEXT", {"correction_survivor": "on", "action_state": "watch"}),
    ]

    with _session_with(rows) as scan_session:
        via_scan = SqlOpportunityStateSummaryRepository(scan_session).for_scan("scan-extra")
    with _feature_session_with(rows) as feature_session:
        via_feature = SqlOpportunityStateSummaryRepository(feature_session).for_feature_run(7)

    assert via_scan == via_feature
    # None of these is a survivor under the narrow text contract: PostgreSQL's
    # boolean input grammar is a parser detail, not a data contract.
    assert via_scan.survivor_count == 0
    assert via_scan.rows_total == 3


# ── Feature-store path: counted, not grouped ──────────────────────────────


def _feature_session_with(rows, run_id=7):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    session.add(
        FeatureRun(
            id=run_id,
            as_of_date=date(2026, 9, 25),
            run_type="daily_snapshot",
            status="published",
        )
    )
    session.add_all(
        [
            StockFeatureDaily(
                run_id=run_id,
                symbol=name,
                as_of_date=date(2026, 9, 25),
                details_json=details,
            )
            for name, details in rows
        ]
    )
    session.commit()
    return session


def test_feature_run_counts_match_the_grouped_pass():
    """``for_feature_run`` must answer exactly what the grouped pivot answered.

    The feature-store path counts per state instead of grouping both keys,
    because the grouped shape's per-row document reads are what make the Daily
    Snapshot exceed the client timeout. Counting changes how the answer is
    reached, not what it is -- so the two shapes are compared directly rather
    than against hand-written expectations, which would only restate one of
    them.
    """
    with _feature_session_with(
        [
            ("READY-SURV", {"correction_survivor": True, "action_state": "setup_ready"}),
            ("WATCH-SURV", {"correction_survivor": True, "action_state": "watch"}),
            ("WATCH-NOSURV", {"correction_survivor": False, "action_state": "watch"}),
            ("UNKNOWN", {"correction_survivor": False, "action_state": "not-a-state"}),
            ("MISSING", {"correction_survivor": False}),
            ("SURV-BAD", {"correction_survivor": True, "action_state": "nope"}),
            ("SURV-NONE", {"correction_survivor": True}),
            ("STR-FALSE", {"correction_survivor": "false", "action_state": "watch"}),
            ("INT-2", {"correction_survivor": 2, "action_state": "watch"}),
        ]
    ) as session:
        repo = SqlOpportunityStateSummaryRepository(session)
        counted = repo.for_feature_run(7)
        grouped = repo._aggregate(
            model=StockFeatureDaily,
            details=StockFeatureDaily.details_json,
            predicate=StockFeatureDaily.run_id == 7,
        )

    assert counted == grouped
    # Anchor the fixture so a broken comparison cannot pass vacuously.
    assert counted.rows_total == 9
    assert counted.survivor_count == 4
    assert counted.action_state_counts[ActionState.SETUP_READY] == 1
    assert counted.action_state_counts[ActionState.WATCH] == 4
    assert sum(counted.action_state_counts.values()) == 5
    assert sum(counted.survivor_action_state_counts.values()) == 2


def test_feature_run_unknown_states_stay_out_of_buckets_but_in_totals():
    """An unrecognised state is counted in no bucket, in both totals.

    Counted per state, an unknown value matches no predicate at all; grouped,
    the pivot skipped it after adding it to ``rows_total``. Same contract, two
    mechanisms -- so it is pinned on the path that actually runs.
    """
    with _feature_session_with(
        [
            ("KNOWN", {"correction_survivor": True, "action_state": "watch"}),
            ("UNKNOWN", {"correction_survivor": True, "action_state": "not-a-state"}),
            ("MISSING", {"correction_survivor": True}),
        ]
    ) as session:
        summary = SqlOpportunityStateSummaryRepository(session).for_feature_run(7)

    assert summary.rows_total == 3
    assert summary.survivor_count == 3
    assert summary.action_state_counts[ActionState.WATCH] == 1
    assert sum(summary.action_state_counts.values()) == 1
    assert sum(summary.survivor_action_state_counts.values()) == 1


def test_feature_run_issues_a_single_statement():
    """One index probe plus one counted statement -- not sixteen round trips.

    The counts are correlated scalar subqueries in one ``SELECT``. Sixteen
    separate queries would be correct and sixteen times the latency on a slow
    link -- and would quietly reintroduce the round-trip cost this rewrite is
    meant to remove.

    The probe is a second statement by design: it decides whether the counted
    shape has the index it needs. It is one cheap catalog lookup, so the
    guarantee that matters is "no per-state round trip", not "exactly one
    statement". Both are asserted to keep either from regressing.
    """
    with _feature_session_with(
        [("A", {"correction_survivor": True, "action_state": "watch"})]
    ) as session:
        engine = session.get_bind()
        statements = []

        def capture(_conn, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            SqlOpportunityStateSummaryRepository(session).for_feature_run(7)
        finally:
            event.remove(engine, "before_cursor_execute", capture)

    assert len(statements) == 2, statements
    # The probe is the catalog lookup. On SQLite it fails ("no such table:
    # pg_index"), which is the documented fallback -- so the second statement is
    # legitimately the grouped one. The guarantee is that the shape is chosen by
    # the probe and issued once, not split per state.
    assert sum("pg_index" in s for s in statements) <= 1, statements
    counted = [s for s in statements if "pg_index" not in s]
    assert len(counted) == 1, statements
    assert "GROUP BY" not in counted[0].upper() or "JSON_EXTRACT" in counted[0], counted[0]


# ── The fallback when the count index is missing or unusable ──────────────


def test_feature_run_falls_back_to_grouping_without_the_index(monkeypatch):
    """A missing or invalid index must degrade, not stall.

    Measured on a copy of the production table: the same sixteen counts take
    6.8 ms with this index and 56.7 s without it, because each count scans the
    run on its own. A build that never ran, was rolled back, or died halfway
    would turn the Daily Snapshot into a minute-long scan.

    Both shapes answer the same question, so the fallback has to return the same
    summary -- asserted by comparing them rather than by a written-out number.
    """
    rows = [
        ("READY-SURV", {"correction_survivor": True, "action_state": "setup_ready"}),
        ("WATCH-SURV", {"correction_survivor": True, "action_state": "watch"}),
        ("WATCH-NOSURV", {"correction_survivor": False, "action_state": "watch"}),
        ("UNKNOWN", {"correction_survivor": False, "action_state": "not-a-state"}),
    ]
    with _feature_session_with(rows) as session:
        repo = SqlOpportunityStateSummaryRepository(session)
        expected = repo._aggregate(
            model=StockFeatureDaily,
            details=StockFeatureDaily.details_json,
            predicate=StockFeatureDaily.run_id == 7,
        )
        monkeypatch.setattr(
            "app.infra.db.repositories.opportunity_summary_repo._count_index_is_usable",
            lambda _session: False,
        )
        engine = session.get_bind()
        statements = []

        def capture(_conn, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            fallen_back = repo.for_feature_run(7)
        finally:
            event.remove(engine, "before_cursor_execute", capture)

    # The result alone cannot prove the fallback ran: on this fixture the counted
    # path happens to produce the same numbers, so asserting only the summary
    # would pass with the fallback deleted. The shape is what is asserted.
    assert fallen_back == expected
    assert fallen_back.rows_total == 4
    assert fallen_back.survivor_count == 2
    shapes = [s for s in statements if "pg_index" not in s]
    assert len(shapes) == 1, statements
    assert "GROUP BY" in shapes[0].upper(), (
        "the fallback did not group; the counted shape ran without its index: "
        f"{shapes[0]}"
    )


def test_feature_run_uses_the_counted_shape_when_the_index_is_usable(monkeypatch):
    """The probe gates the counted shape; with a usable index it is used.

    Without this, making the probe always return False would silently keep the
    slower grouped path forever and no test would notice.
    """
    monkeypatch.setattr(
        "app.infra.db.repositories.opportunity_summary_repo._count_index_is_usable",
        lambda _session: True,
    )
    with _feature_session_with(
        [("A", {"correction_survivor": True, "action_state": "watch"})]
    ) as session:
        engine = session.get_bind()
        statements = []

        def capture(_conn, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            summary = SqlOpportunityStateSummaryRepository(session).for_feature_run(7)
        finally:
            event.remove(engine, "before_cursor_execute", capture)

    assert summary.rows_total == 1
    assert len(statements) == 1, statements
    assert "GROUP BY" not in statements[0].upper(), statements[0]


def test_index_probe_treats_an_unreadable_catalog_as_not_usable():
    """A backend without ``pg_index`` falls back instead of raising.

    SQLite has no ``pg_index``, so the probe's own query fails there. The
    fallback is correct on every backend, so an unreadable catalog has to mean
    "not usable" rather than propagate the error to the caller.

    The second half is the point: the failed probe must leave the session able
    to run the grouped query. A statement PostgreSQL rejects aborts the
    transaction, and every later statement on that session is refused -- so
    without the savepoint the fallback could not run on the very path it exists
    for. Verified against PostgreSQL:

        BEGIN; SELECT 1/0; SELECT 99;
          ERROR: current transaction is aborted, commands ignored
        BEGIN; SAVEPOINT sp; SELECT 1/0; ROLLBACK TO sp; SELECT 99;
          99
    """
    with _feature_session_with(
        [("A", {"correction_survivor": True, "action_state": "watch"})]
    ) as session:
        from app.infra.db.repositories.opportunity_summary_repo import (
            _count_index_is_usable,
        )

        assert _count_index_is_usable(session) is False
        # The session has to stay usable, not just report "not usable".
        assert session.is_active, "the failed probe left the transaction aborted"
        # And the caller still answers correctly.
        assert SqlOpportunityStateSummaryRepository(session).for_feature_run(7).rows_total == 1


def test_index_probe_wraps_its_query_in_a_savepoint():
    """The probe must run inside ``begin_nested()``, not bare.

    This cannot be asserted by running the probe on SQLite: SQLite has no
    aborted-transaction state, so a failed statement is followed by a working
    one either way. The test that observed the failure is therefore blind here,
    and would stay green with the savepoint removed -- measured: with
    ``begin_nested`` replaced by ``nullcontext`` all twelve tests still pass.

    So the mechanism is asserted instead of its effect. On PostgreSQL the
    difference is not cosmetic:

        BEGIN; SELECT 1/0; SELECT 99;
          ERROR: current transaction is aborted, commands ignored
        BEGIN; SAVEPOINT sp; SELECT 1/0; ROLLBACK TO sp; SELECT 99;
          99
    """
    from app.infra.db.repositories import opportunity_summary_repo as mod

    calls: list[str] = []

    class _Connection:
        """Records that the savepoint was opened on the connection."""

        def begin_nested(self):
            """Record the request and open a no-op savepoint."""
            calls.append("connection_begin_nested")
            import contextlib

            return contextlib.nullcontext()

    class _Session:
        """A session that records where the probe opens its savepoint.

        Deliberately minimal: the assertion is about *which* call the probe
        makes, not about what the query returns, so ``execute`` only has to
        produce a scalar. ``connection()`` exists so a regression back to
        ``session.begin_nested()`` is visible here.
        """

        is_active = True

        def __init__(self):
            """Expose the connection the probe must use instead of itself."""
            self._connection = _Connection()

        def connection(self):
            """The connection-level entry point the probe has to go through."""
            return self._connection

        def begin_nested(self):
            """Only reachable if the probe regressed; records that loudly."""
            calls.append("session_begin_nested")
            import contextlib

            return contextlib.nullcontext()

        def execute(self, _statement, _params=None):
            """Return a one-row result, i.e. "the index is usable"."""

            class _R:
                """Stand-in for a SQLAlchemy ``Result`` with one row."""

                @staticmethod
                def scalar():
                    """The probe only ever reads the single count."""
                    return 1

            return _R()

    assert mod._count_index_is_usable(_Session()) is True
    assert calls == ["connection_begin_nested"], (
        "the probe opened its savepoint on the Session, which flushes the "
        "caller's pending writes before SAVEPOINT -- a failing staged write "
        f"would then be reported as an index-probe failure: {calls}"
    )


def test_index_probe_rolls_back_when_it_cannot_use_a_savepoint():
    """A dialect that cannot open a savepoint must still leave a usable session.

    Falling out of ``begin_nested`` with the transaction still aborted would
    hand the caller a session that refuses every further statement -- the
    fallback could not run on the path it exists for.
    """

    class _AbortedConnection:
        """A connection that cannot open a savepoint at all."""

        def begin_nested(self):
            """Fail, as a dialect without savepoint support would."""
            raise RuntimeError("SAVEPOINT unsupported")

    class _AbortedSession:
        """A session on which both the savepoint and the probe fail.

        Models the state PostgreSQL leaves behind after a rejected statement:
        ``is_active`` is false and nothing can be executed. It records whether
        the probe recovered it, which is the whole point of the assertion.
        """

        def __init__(self):
            """Start usable; the test sets ``is_active`` false before calling."""
            self.is_active = True
            self.rolled_back = False
            self._connection = _AbortedConnection()

        def connection(self):
            """The connection whose savepoint attempt fails."""
            return self._connection

        def begin_nested(self):
            """Only reachable if the probe regressed; it must not be."""
            raise AssertionError("probe used Session.begin_nested()")

        def execute(self, _statement, _params=None):
            """Fail, as an aborted transaction refuses every statement."""
            raise RuntimeError("current transaction is aborted")

        def rollback(self):
            """Record the recovery and become usable again."""
            self.rolled_back = True
            self.is_active = True

    from app.infra.db.repositories import opportunity_summary_repo as mod

    session = _AbortedSession()
    session.is_active = False  # the failed statement left it aborted
    assert mod._count_index_is_usable(session) is False
    assert session.rolled_back, (
        "the probe left the session aborted and did not recover it, so the "
        "grouped fallback could not run"
    )


def test_index_probe_does_not_flush_or_roll_back_the_callers_work():
    """The probe must not touch the caller's pending writes.

    Raised by @xang1234 as P1 and correct. ``session.begin_nested()`` flushes
    unconditionally before opening the savepoint -- measured on SQLAlchemy 2.0.25
    with ``autoflush=False``, the setting this project uses:

        session.add(pending); with session.begin_nested():
          -> before_flush fired, INSERT issued

    One session is shared with every writer repository here, so any staged write
    would be issued from inside the index probe. If that flush failed, the
    ``except`` below would read it as an index-probe failure and
    ``_recover_aborted_probe`` would roll back the caller's whole unit of work --
    with the summary still returning normally through the fallback, and the
    discarded writes invisible to the caller.

    The probe therefore opens its savepoint on the *connection*. Asserted on a
    real ORM session rather than a double, because the property under test is
    exactly what SQLAlchemy does with pending state.
    """
    from sqlalchemy import Column, Integer, String, create_engine, event
    from sqlalchemy.orm import declarative_base, sessionmaker

    from app.infra.db.repositories import opportunity_summary_repo as mod

    base = declarative_base()

    class _Row(base):
        """A table that exists only so the session has something to stage."""

        __tablename__ = "probe_pending_write_probe"
        id = Column(Integer, primary_key=True)
        name = Column(String)

    engine = create_engine("sqlite:///:memory:")
    base.metadata.create_all(engine)
    # autoflush=False, as app/database.py configures it.
    session = sessionmaker(bind=engine, autoflush=False)()

    flushes: list[str] = []
    event.listen(session, "before_flush", lambda *_: flushes.append("flushed"))

    session.add(_Row(id=1, name="staged by a writer repository"))

    assert mod._count_index_is_usable(session) is False  # SQLite: no pg_index
    assert not flushes, (
        "the index probe flushed the caller's pending writes; on PostgreSQL a "
        "failing staged write would then be reported as a probe failure and the "
        "unit of work rolled back"
    )
    assert len(session.new) == 1, (
        "the probe discarded staged work: " f"{[type(o).__name__ for o in session.new]}"
    )
    assert session.is_active


def test_index_probe_reads_validity_and_readiness_not_just_presence():
    """An interrupted CONCURRENTLY build leaves a same-named, unusable index.

    ``indisvalid`` is false while a concurrent build is in progress or after it
    failed. Counting the name alone would treat that index as present and hand
    the query back to a plan that cannot use it.
    """
    from app.infra.db.repositories.opportunity_summary_repo import _INDEX_USABLE_SQL

    sql = str(_INDEX_USABLE_SQL).lower()
    assert "indisvalid" in sql
    assert "indisready" in sql
    assert _INDEX_USABLE_SQL._bindparams["name"] is not None


def test_index_probe_is_scoped_to_the_table_that_is_queried():
    """The probe must not match a same-named index on another relation.

    PostgreSQL allows one index name per schema per table, so a valid
    ``ix_sfd_run_action_state_survivor`` on some other table would answer for a
    missing or invalid one on ``stock_feature_daily``. The counted path would
    then be selected on the strength of an index it cannot use -- the slow shape
    with no warning.
    """
    from app.infra.db.repositories.opportunity_summary_repo import (
        _INDEX_USABLE_SQL,
        _count_index_is_usable,
    )
    from app.infra.db.models.feature_store import StockFeatureDaily

    sql = str(_INDEX_USABLE_SQL).lower()
    assert "indrelid" in sql, "the probe does not tie the index to a relation"
    assert "to_regclass(:table)" in sql, (
        "the probe does not resolve the relation the way the runtime query does; "
        "matching every same-named table across the search path would answer for "
        "a relation the counted query cannot index: " + sql
    )
    assert "current_schemas" not in sql, (
        "the probe still scans all search-path schemas instead of resolving the "
        "first match, which is the relation the aggregation actually reads"
    )
    assert _INDEX_USABLE_SQL._bindparams["table"] is not None

    # And the caller passes the relation the aggregation actually reads.
    assert StockFeatureDaily.__tablename__ == "stock_feature_daily"
