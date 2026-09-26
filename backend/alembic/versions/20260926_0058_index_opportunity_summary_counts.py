"""Index the Daily Snapshot's opportunity-summary counts.

``opportunity_summary_repo._aggregate`` reads two ``details_json`` keys once per
row of a run and pivots them into per-state counts. ``20260925_0057`` covered
the snapshot's *filter* predicates; this one covers the projection.

The first fix attempt — an expression index on the two keys — does not work, and
the measurement matters more than the shape:

* ``stock_feature_daily`` carries 10,079 rows for a run, each with a ~64 kB
  ``details_json``. Aggregating by ``GROUP BY`` measured 19,357 ms.
* With an expression index on ``(run_id, survivor, action_state)`` the planner
  *did* pick it up — as a ``Bitmap Index Scan`` — and the time stayed 19,768 ms.
  Neither ``Index Scan`` nor ``Bitmap Heap Scan`` is an ``Index Only Scan``, so
  every tuple still goes back to the heap, and the heap re-parses the whole
  stored document per row.

PostgreSQL only chooses an index-only scan when all *table* expressions the
query needs are available from the index. For ``f(x)`` it sees that ``x`` (here:
``details_json``) is needed, does not find it, and concludes an index-only scan
is impossible — documented in "Index-Only Scans and Covering Indexes". An
expression index therefore helps a filter (``WHERE``, served as an index
condition) but not a projection (``GROUP BY`` key, read per row). That is why
``20260925_0057`` worked and this one has to be shaped differently.

So the read side moves to scalar counts: ``count(*)`` per state instead of
grouping the extracted values. The expression then sits in the ``WHERE`` clause,
reaches the index as an index condition, and no ``details_json`` is read at all.

The exact expression text is what makes the index usable, so both sides are
kept byte-identical (minus the table qualifier) and a drift guard compares them
as strings. PostgreSQL normalises redundant parentheses away -- measured, both
``CAST((details_json ->> 'x') AS BOOLEAN)`` and
``CAST(details_json ->> 'x' AS BOOLEAN)`` matched the index, at 0.227 ms and
0.092 ms for the same count -- but the repository compiles the unparenthesised
form and the DDL follows it rather than relying on that.

One trap that is *not* cosmetic: the key must be an inline SQL literal, not a
bind parameter. ``details["action_state"].as_string()`` renders
``->> %(details_json_1)s``, which a generic plan cannot match to the index's
literal, so the index would silently go unused. The repository reads both keys
through ``portability.json_text`` for that reason.

Column order is load-bearing. ``(run_id, action_state, survivor)`` serves both
``action_state = …`` and the survivor predicate; the reverse order
``(run_id, survivor, action_state)`` measured 19.7 s because the filtered column
is not the one directly after the equality prefix.

Measured on a copy of the production table (63,333 rows, 10,079 in the run):

    GROUP BY, no index              19,357 ms   (cold: every row re-parses)
    16 scalar counts, no index     154,462 ms   (16 independent passes)
    16 scalar counts + this index    6,281 – 12,537 ms

``EXPLAIN (ANALYZE, BUFFERS)`` for the final statement, with this index and its
``ANALYZE`` in place — the subplans that still reach the heap:

    Result                                       12.292 ms
      ->  Index Only Scan ix_sfd_run_* (rows_total)   1.189 ms   10 buffers
      ->  Index Scan      this index  (survivor)      0.054 ms   11 buffers
      ->  Bitmap Heap Scan this index (per state)     7.069 ms  826 buffers

``rows_total`` and the survivor count are served from the index. The per-state
counts are not: when ``action_state = …`` matches thousands of rows the planner
takes a ``Bitmap Index Scan`` plus a ``Bitmap Heap Scan`` instead of an
index-only scan, so those tuples still visit the heap. That is the residual cost,
and it is why the grouped pass and this index belong together — the grouped shape
reads every document instead of a subset.

Timings move with cache state and must not be compared across sessions. The
6,281–12,537 ms figures above were disk-bound (``read=`` in ``Buffers``); the
12.292 ms above is buffer-warm (``hit=``). Quoting either without its cache state
is what made an earlier projection of ≈4.2 s look contradicted by a measured
6–12 s: both were real, on different cache states.

``ANALYZE`` runs here on purpose. ``CREATE INDEX CONCURRENTLY`` leaves expression
statistics empty, and without them the planner rates the new index as unusable
and keeps the full scan — the failure ``20260925_0057`` hit in production, where
the index existed, validated, and went unused at 19,251 ms until ``ANALYZE``
brought the same query to 31 ms. Skipping it would ship an index that is built
and inert.

PostgreSQL-only, matching ``20260617_0021``, ``20260821_0028`` and
``20260925_0057``: SQLite does not parse these JSON operators. The build runs
``CONCURRENTLY`` inside an autocommit block so a large run does not block writes,
and ``_rebuild_invalid_indexes`` clears an invalid same-named index an
interrupted build may have left, which ``IF NOT EXISTS`` would otherwise treat
as done.

Deploy note: ``CONCURRENTLY`` still reads every row's ``details_json`` across
every run in the table, and the ``ANALYZE`` after it reads the table again. On a
large table the first container start after this revision can spend a while here,
and a compose healthcheck tighter than that build fails the rollout. Build the
index ahead of the deploy if that becomes the case:

    CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_sfd_run_action_state_survivor
        ON stock_feature_daily (
            run_id,
            (CAST(details_json ->> 'action_state' AS VARCHAR)),
            (lower(details_json ->> 'correction_survivor') IN ('true', '1'))
        );

The ``ANALYZE`` has to run in the same pass — an index built ahead of the deploy
still needs its statistics before the planner will use it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260926_0058"
down_revision = "20260925_0057"
branch_labels = None
depends_on = None

_TABLE = "stock_feature_daily"
_INDEX_NAME = "ix_sfd_run_action_state_survivor"

# The two keys the opportunity summary extracts, in index order. ``action_state``
# is filtered (it must sit directly after the ``run_id`` prefix to be reachable
# as an index condition); ``correction_survivor`` is only ever tested for truth,
# so it goes last.
_STATE_KEY = "action_state"
_SURVIVOR_KEY = "correction_survivor"

# The two spellings of JSON ``true`` that ``->>`` can yield, one per backend.
# PostgreSQL keeps the boolean and renders it as text ``'true'``; SQLite stores
# it as the integer ``1``, so ``->>`` returns ``'1'``. The predicate has to match
# both or the survivor counts differ by backend.
_SURVIVOR_TRUE_TEXT = "true"
_SURVIVOR_TRUE_SQLITE = "1"

_INVALID_INDEX_SQL = """
SELECT pg_catalog.format('%I.%I', n.nspname, idx.relname)
FROM pg_catalog.pg_index AS i
JOIN pg_catalog.pg_class AS idx ON idx.oid = i.indexrelid
JOIN pg_catalog.pg_namespace AS n ON n.oid = idx.relnamespace
WHERE i.indrelid = pg_catalog.to_regclass('stock_feature_daily')
  AND idx.relname = :index_name
  AND NOT i.indisvalid
"""


def _state_expr() -> str:
    """SQL for the indexed action-state value.

    Must stay byte-identical (minus the table qualifier) to what
    ``details["action_state"].as_string()`` compiles for the same key, or the
    planner declines the index and the count falls back to a full JSON scan.
    ``test_opportunity_summary_index_drift`` pins that linkage.
    """
    return f"CAST(details_json ->> '{_STATE_KEY}' AS VARCHAR)"


def _survivor_expr() -> str:
    """SQL for the indexed survivor flag. Same linkage as ``_state_expr()``.

    Cast-free on purpose. The earlier spelling
    ``CAST(details_json ->> 'correction_survivor' AS BOOLEAN) IS true`` raises
    ``invalid input syntax for type boolean`` on any non-boolean text (``"2"``,
    ``"maybe"``). Inside an index that is a foot-gun rather than a filter:

    * ``CREATE INDEX`` evaluates the expression for **every row of every
      historical run**, so one bad value fails the startup migration and blocks
      the deploy.
    * Once the index exists, **any insert or update** carrying such a value
      fails — a feature-run write breaks instead of a summary read.

    ``lower(...) IN ('true', '1')`` cannot raise for any input, and it is the
    predicate the repository compiles, which keeps the drift guard byte-exact.

    Both list members are needed for the two backends to agree. PostgreSQL stores
    a JSON ``true`` and ``->>`` hands it back as the text ``'true'``; SQLite
    stores the same value as the integer ``1``, so its ``->>`` yields ``'1'``.
    Matching only ``'true'`` -- which is what ``20260821_0028``'s index on this
    key does, and what this migration did at first -- silently counts every
    survivor as a non-survivor on SQLite, where the grouped pivot used to read it
    as truthy. ``'1'`` never occurs in the Postgres text form, so the extra
    member is inert there.
    """
    return (
        f"lower(details_json ->> '{_SURVIVOR_KEY}') "
        f"IN ('{_SURVIVOR_TRUE_TEXT}', '{_SURVIVOR_TRUE_SQLITE}')"
    )


def _rebuild_invalid_indexes() -> None:
    """Drop a same-named index a previous interrupted build left invalid.

    ``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction and is not
    atomic: a failure leaves an invalid index under the target name. The
    following ``IF NOT EXISTS`` would consider the name taken and skip the
    build, so the counts would keep full-scanning with no error anywhere.

    Deliberately local to this revision, as in ``20260925_0057``.
    """
    bind = op.get_bind()
    if op.get_context().as_sql:
        # Offline / ``--sql`` generation: there is no catalog to inspect, and
        # the emitted script is expected to contain only the CREATE statements.
        return
    invalid = bind.execute(
        sa.text(_INVALID_INDEX_SQL), {"index_name": _INDEX_NAME}
    ).scalar()
    if invalid:
        with op.get_context().autocommit_block():
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {invalid}")


def _create_indexes() -> None:
    # Recovery runs before the create: an invalid index left by an interrupted
    # CONCURRENTLY build would otherwise make ``IF NOT EXISTS`` skip the rebuild.
    _rebuild_invalid_indexes()
    with op.get_context().autocommit_block():
        op.execute(
            f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_INDEX_NAME} "
            f"ON {_TABLE} (run_id, ({_state_expr()}), ({_survivor_expr()}))"
        )


def _analyze_table() -> None:
    """Refresh the expression statistics the new index needs to be usable.

    ``CREATE INDEX CONCURRENTLY`` does not create them, and without statistics
    the planner rates the expression index as unusable and keeps the sequential
    plan. ``20260925_0057`` shipped an index that existed, validated, and went
    unused at 19,251 ms for exactly this reason.
    """
    op.execute(f"ANALYZE {_TABLE}")


def _drop_indexes() -> None:
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_INDEX_NAME}")


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _create_indexes()
        _analyze_table()


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _drop_indexes()
