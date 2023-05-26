"""Contains the Postgres implementation of the Database interface."""
from __future__ import annotations

import collections
import concurrent
import concurrent.futures
import math
import os
import textwrap
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

import psycopg
import psycopg.rows

from postbound.db import db
from postbound.qal import qal, base, clauses, transform, formatter
from postbound.optimizer import jointree
from postbound.optimizer.physops import operators as physops
from postbound.optimizer.planmeta import hints as planmeta
from postbound.util import collections as collection_utils, logging, misc as utils, typing as type_utils

HintBlock = collections.namedtuple("HintBlock", ["preparatory_statements", "hints", "query"])


class PostgresInterface(db.Database):
    """Database implementation for PostgreSQL backends."""

    def __init__(self, connect_string: str, system_name: str = "Postgres", *, cache_enabled: bool = True) -> None:
        self.connect_string = connect_string
        self._connection = psycopg.connect(connect_string, application_name="PostBOUND",
                                           row_factory=psycopg.rows.tuple_row)
        self._connection.autocommit = True
        self._cursor = self._connection.cursor()

        self._db_schema = PostgresSchemaInterface(self)
        self._db_stats = PostgresStatisticsInterface(self)

        super().__init__(system_name, cache_enabled=cache_enabled)

    def schema(self) -> db.DatabaseSchema:
        return self._db_schema

    def statistics(self, emulated: bool | None = None, cache_enabled: Optional[bool] = None) -> db.DatabaseStatistics:
        if emulated is not None:
            self._db_stats.emulated = emulated
        if cache_enabled is not None:
            self._db_stats.cache_enabled = cache_enabled
        return self._db_stats

    def hinting(self) -> db.HintService:
        return PostgresHintService()

    def execute_query(self, query: qal.SqlQuery | str, *, cache_enabled: Optional[bool] = None) -> Any:
        cache_enabled = cache_enabled or (cache_enabled is None and self._cache_enabled)
        query = self._prepare_query_execution(query)

        if cache_enabled and query in self._query_cache:
            query_result = self._query_cache[query]
        else:
            try:
                self._cursor.execute(query)
                query_result = self._cursor.fetchall()
            except (psycopg.InternalError, psycopg.OperationalError) as e:
                msg = "\n".join([f"At {utils.current_timestamp()}", "For query:", str(query), "Message:", str(e)])
                raise db.DatabaseServerError(msg, e)
            except psycopg.Error as e:
                msg = "\n".join([f"At {utils.current_timestamp()}", "For query:", str(query), "Message:", str(e)])
                raise db.DatabaseUserError(msg, e)
            if cache_enabled:
                self._query_cache[query] = query_result

        # simplify the query result as much as possible: [(42, 24)] becomes (42, 24) and [(1,), (2,)] becomes [1, 2]
        # [(42, 24), (4.2, 2.4)] is left as-is
        if not query_result:
            return []
        result_structure = query_result[0]  # what do the result tuples look like?
        if len(result_structure) == 1:  # do we have just one column?
            query_result = [row[0] for row in query_result]  # if it is just one column, unwrap it
        return query_result if len(query_result) > 1 else query_result[0]  # if it is just one row, unwrap it

    def optimizer(self) -> db.OptimizerInterface:
        return PostgresOptimizer(self)

    def database_name(self) -> str:
        self._cursor.execute("SELECT CURRENT_DATABASE();")
        db_name = self._cursor.fetchone()[0]
        return db_name

    def database_system_version(self) -> utils.Version:
        self._cursor.execute("SELECT VERSION();")
        pg_ver = self._cursor.fetchone()[0]
        # version looks like "PostgreSQL 14.6 on x86_64-pc-linux-gnu, compiled by gcc (...)
        return utils.Version(pg_ver.split(" ")[1])

    def describe(self) -> dict:
        base_info = {
            "system_name": self.database_system_name(),
            "system_version": self.database_system_version(),
            "database": self.database_name(),
            "statistics_settings": {
                "emulated": self._db_stats.emulated,
                "cache_enabled": self._db_stats.cache_enabled
            }
        }
        self._cursor.execute("SELECT name, setting FROM pg_settings")
        system_settings = self._cursor.fetchall()
        base_info["system_settings"] = dict(system_settings)
        return base_info

    def reset_connection(self) -> None:
        self._connection.cancel()
        self._cursor.close()
        self._connection.close()
        self._connection = psycopg.connect(self.connect_string)
        self._cursor = self._connection.cursor()

    def cursor(self) -> db.Cursor:
        return self._cursor

    def close(self) -> None:
        self._cursor.close()
        self._connection.close()

    def prewarm_tables(self, tables: Optional[base.TableReference | Iterable[base.TableReference]] = None,
                       *more_tables: base.TableReference) -> None:
        tables = list(collection_utils.enlist(tables)) + list(more_tables)
        if not tables:
            return
        tables = set(tab.full_name for tab in tables)  # eliminate duplicates if tables are selected multiple times
        prewarm_invocations = [f"pg_prewarm('{tab}')" for tab in tables]
        prewarm_text = ", ".join(prewarm_invocations)
        prewarm_query = f"SELECT {prewarm_text}"
        self._cursor.execute(prewarm_query)

    @type_utils.module_local
    def _prepare_query_execution(self, query: qal.SqlQuery | str, *, drop_explain: bool = False) -> str:
        """Provides the query in a unified format, taking care of preparatory statements as necessary.

        `drop_explain` can be used to remove any EXPLAIN clauses from the query. Note that all actions that require
        the "semantics" of the query to be known (e.g. EXPLAIN modifications or query hints) and are therefore only
        executed for instances of the qal queries.
        """
        if not isinstance(query, qal.SqlQuery):
            return query

        if drop_explain:
            query = transform.drop_clause(query, clauses.Explain)
        if query.hints and query.hints.preparatory_statements:
            self._cursor.execute(query.hints.preparatory_statements)
            query = transform.drop_hints(query, preparatory_statements_only=True)
        return self.hinting().format_query(query)

    @type_utils.module_local
    def _obtain_query_plan(self, query: str) -> dict:
        """Provides the query plan (without ANALYZE) for the given query."""
        if not query.upper().startswith("EXPLAIN (FORMAT JSON)"):
            query = "EXPLAIN (FORMAT JSON) " + query
        self._cursor.execute(query)
        return self._cursor.fetchone()[0]


class PostgresSchemaInterface(db.DatabaseSchema):
    """Schema-specific parts of the general Postgres interface."""

    def __int__(self, postgres_db: PostgresInterface) -> None:
        super().__init__(postgres_db)

    def lookup_column(self, column: base.ColumnReference | str,
                      candidate_tables: list[base.TableReference]) -> base.TableReference:
        column = column.name if isinstance(column, base.ColumnReference) else column
        for table in candidate_tables:
            table_columns = self._fetch_columns(table)
            if column in table_columns:
                return table
        candidate_tables = [table.full_name for table in candidate_tables]
        raise ValueError(f"Column '{column}' not found in candidate tables {candidate_tables}")

    def is_primary_key(self, column: base.ColumnReference) -> bool:
        if not column.table:
            raise base.UnboundColumnError(column)
        index_map = self._fetch_indexes(column.table)
        return index_map.get(column.name, False)

    def has_secondary_index(self, column: base.ColumnReference) -> bool:
        if not column.table:
            raise base.UnboundColumnError(column)
        index_map = self._fetch_indexes(column.table)

        # The index map contains an entry for each attribute that actually has an index. The value is True, if the
        # attribute (which is known to be indexed), is even the Primary Key
        # Our method should return False in two cases: 1) the attribute is not indexed at all; and 2) the attribute
        # actually is the Primary key. Therefore, by assuming it is the PK in case of absence, we get the correct
        # value.
        return not index_map.get(column.name, True)

    def datatype(self, column: base.ColumnReference) -> str:
        if not column.table:
            raise base.UnboundColumnError(column)
        query_template = textwrap.dedent("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = {tab} AND column_name = {col}""".format(tab=column.table.full_name, col=column.name))
        self._db.cursor().execute(query_template)
        result_set = self._db.cursor().fetchone()
        return result_set[0]

    def _fetch_columns(self, table: base.TableReference) -> list[str]:
        """Retrieves all physical columns for a given table from the PG metadata catalogs."""
        query_template = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        self._db.cursor().execute(query_template, (table.full_name,))
        result_set = self._db.cursor().fetchall()
        return [col[0] for col in result_set]

    def _fetch_indexes(self, table: base.TableReference) -> dict[str, bool]:
        """Retrieves all index structures for a given table based on the PG metadata catalogs.

        The resulting dictionary will contain one entry per column if there is any index on that column.
        If the index is the primary key index, the column name will be mapped to `True`, otherwise to `False`.
        Columns without any index structure will not appear in the dictionary at all.
        """
        # query adapted from https://wiki.postgresql.org/wiki/Retrieve_primary_key_columns

        index_query = textwrap.dedent(f"""
            SELECT attr.attname, idx.indisprimary
            FROM pg_index idx
                JOIN pg_attribute attr
                ON idx.indrelid = attr.attrelid AND attr.attnum = ANY(idx.indkey)
            WHERE idx.indrelid = '{table.full_name}'::regclass
        """)
        self._db.cursor().execute(index_query)
        result_set = self._db.cursor().fetchall()
        index_map = dict(result_set)
        return index_map


# Postgres stores its array datatypes in a more general array-type structure (anyarray).
# However, to extract the individual entries from such an array, the need to be casted to a typed array structure.
# This dictionary contains the necessary casts for the actual column types.
# For example, suppose a column contains integer values. If this column is aggregated into an anyarray entry, the
# appropriate converter for this array is int[]. In other words DTypeArrayConverters["integer"] = "int[]"
_DTypeArrayConverters = {
    "integer": "int[]",
    "text": "text[]",
    "character varying": "text[]"
}


class PostgresStatisticsInterface(db.DatabaseStatistics):
    """Statistics-specific parts of the Postgres interface."""

    def __init__(self, postgres_db: PostgresInterface) -> None:
        super().__init__(postgres_db)

    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> Optional[int]:
        count_query = f"SELECT reltuples FROM pg_class WHERE oid = '{table.full_name}'::regclass"
        self._db.cursor().execute(count_query)
        count = self._db.cursor().fetchone()[0]
        return count

    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> Optional[int]:
        dist_query = "SELECT n_distinct FROM pg_stats WHERE tablename = %s and attname = %s"
        self._db.cursor().execute(dist_query, (column.table.full_name, column.name))
        dist_values = self._db.cursor().fetchone()[0]

        # interpreting the n_distinct column is difficult, since different value ranges indicate different things
        # (see https://www.postgresql.org/docs/current/view-pg-stats.html)
        # If the value is >= 0, it represents the actual (approximated) number of distinct non-zero values in the
        # column.
        # If the value is < 0, it represents 'the negative of the number of distinct values divided by the number of
        # rows'. Therefore, we have to correct the number of distinct values manually in this case.
        if dist_values >= 0:
            return dist_values

        # correct negative values
        n_rows = self._retrieve_total_rows_from_stats(column.table)
        return -1 * n_rows * dist_values

    def _retrieve_min_max_values_from_stats(self, column: base.ColumnReference) -> Optional[tuple[Any, Any]]:
        # Postgres does not keep track of min/max values, so we need to determine them manually
        if not self.enable_emulation_fallback:
            raise db.UnsupportedDatabaseFeatureError(self._db, "min/max value statistics")
        return self._calculate_min_max_values(column, cache_enabled=True)

    def _retrieve_most_common_values_from_stats(self, column: base.ColumnReference,
                                                k: int) -> Sequence[tuple[Any, int]]:
        # Postgres stores the Most common values in a column of type anyarray (since in this column, many MCVs from
        # many different tables and data types are present). However, this type is not very convenient to work on.
        # Therefore, we first need to convert the anyarray to an array of the actual attribute type.

        # determine the attributes data type to figure out how it should be converted
        attribute_query = "SELECT data_type FROM information_schema.columns WHERE table_name = %s AND column_name = %s"
        self._db.cursor().execute(attribute_query, (column.table.full_name, column.name))
        attribute_dtype = self._db.cursor().fetchone()[0]
        attribute_converter = _DTypeArrayConverters[attribute_dtype]

        # now, load the most frequent values. Since the frequencies are expressed as a fraction of the total number of
        # rows, we need to multiply this number again to obtain the true number of occurrences
        mcv_query = textwrap.dedent("""
                SELECT UNNEST(most_common_vals::text::{conv}),
                    UNNEST(most_common_freqs) * (SELECT reltuples FROM pg_class WHERE oid = '{tab}'::regclass)
                FROM pg_stats
                WHERE tablename = %s AND attname = %s""".format(conv=attribute_converter, tab=column.table.full_name))
        self._db.cursor().execute(mcv_query, (column.table.full_name, column.name))
        return self._db.cursor().fetchall()[:k]


@dataclass
class HintParts:
    """Captures the different kinds of Postgres-hints to collect them more easily."""
    settings: list[str]
    hints: list[str]

    @staticmethod
    def empty() -> HintParts:
        """An empty hint parts object, i.e. no hints have been specified, yet."""
        return HintParts([], [])

    def merge_with(self, other: HintParts) -> HintParts:
        """Combines the hints that are contained in this hint parts object with all hints in the other object.

        This construct new hint parts and leaves the current object unmodified.
        """
        merged_settings = self.settings + [setting for setting in other.settings if setting not in self.settings]
        merged_hints = self.hints + [hint for hint in other.hints if hint not in self.hints]
        return HintParts(merged_settings, merged_hints)


def _is_hash_join(join_tree_node: jointree.IntermediateJoinNode,
                  operator_assignment: Optional[physops.PhysicalOperatorAssignment]) -> bool:
    """Checks, whether the given node should be executed as a hash join.

    Fails gracefully for base table and unspecified operator assignments (by returning False).
    """
    if not operator_assignment:
        return False
    selected_operators: Optional[physops.JoinOperatorAssignment] = operator_assignment[join_tree_node.tables()]
    if not selected_operators:
        return False
    return selected_operators.operator == physops.JoinOperators.HashJoin


def _generate_leading_hint_content(join_tree_node: jointree.AbstractJoinTreeNode,
                                   operator_assignment: Optional[physops.PhysicalOperatorAssignment] = None) -> str:
    """Builds part of the Leading hint to enforce join order and join direction for the given join node."""
    if isinstance(join_tree_node, jointree.BaseTableNode):
        return join_tree_node.table.identifier()
    if not isinstance(join_tree_node, jointree.IntermediateJoinNode):
        raise ValueError(f"Unknown join tree node: {join_tree_node}")

    # for Postgres, the inner relation of a Hash join is the one that gets the hash table and the outer relation is
    # the one being probed. For all other joins, the inner/outer relation actually is the inner/outer relation
    # Therefore, we want to have the smaller relation as the inner relation for hash joins and the other way around
    # for all other joins

    has_directional_information = isinstance(join_tree_node.annotation, physops.DirectionalJoinOperatorAssignment)
    if has_directional_information:
        annotation: physops.DirectionalJoinOperatorAssignment = join_tree_node.annotation
        inner_tables, outer_tables = annotation.inner, annotation.outer
        inner_child = (join_tree_node.left_child if join_tree_node.left_child.tables() == inner_tables
                       else join_tree_node.right_child)
        outer_child = (join_tree_node.left_child if inner_child == join_tree_node.right_child
                       else join_tree_node.right_child)
        inner_child, outer_child = ((outer_child, inner_child) if annotation.operator == physops.JoinOperators.HashJoin
                                    else (inner_child, outer_child))
    else:
        left, right = join_tree_node.left_child, join_tree_node.right_child
        left_bound = left.upper_bound if left.upper_bound and not math.isnan(left.upper_bound) else -math.inf
        right_bound = right.upper_bound if right.upper_bound and not math.isnan(right.upper_bound) else math.inf

        if _is_hash_join(join_tree_node, operator_assignment):
            inner_child, outer_child = (left, right) if right_bound > left_bound else (right, left)
        elif left_bound > right_bound:
            inner_child, outer_child = right, left
        else:
            inner_child, outer_child = left, right

    inner_hint = _generate_leading_hint_content(inner_child, operator_assignment)
    outer_hint = _generate_leading_hint_content(outer_child, operator_assignment)
    return f"({outer_hint} {inner_hint})"


def _generate_pg_join_order_hint(query: qal.SqlQuery,
                                 join_order: jointree.LogicalJoinTree | jointree.PhysicalQueryPlan,
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment] = None
                                 ) -> tuple[qal.SqlQuery, Optional[HintParts]]:
    """Generates the Leading hint to enforce join order and join direction for the given query.

    This function needs access to the operator assignment in addition to the join tree, because the actual join
    directions in the leading hint depend on the selected join operators.

    More specifically, the join tree assumes that the left join partner of a join node acts as the outer relation
    whereas the right partner acts as the inner relation. For hash joins this means that the inner relation should be
    probed whereas the hash table is created for the outer relation. However, Postgres denotes the directions
    exactly the other way around. Therefore, the direction has to be swapped for hash joins.
    """
    if len(join_order) < 2:
        return query, None
    leading_hint = _generate_leading_hint_content(join_order.root, operator_assignment)
    leading_hint = f"Leading({leading_hint})"
    hints = HintParts([], [leading_hint])
    return query, hints


PostgresOptimizerSettings = {
    physops.JoinOperators.NestedLoopJoin: "enable_nestloop",
    physops.JoinOperators.HashJoin: "enable_hashjoin",
    physops.JoinOperators.SortMergeJoin: "enable_mergejoin",
    physops.ScanOperators.SequentialScan: "enable_seqscan",
    physops.ScanOperators.IndexScan: "enable_indexscan",
    physops.ScanOperators.IndexOnlyScan: "enable_indexonlyscan",
    physops.ScanOperators.BitmapScan: "enable_bitmapscan"
}
"""Denotes all (session-global) optimizer settings that modify the allowed physical operators."""

# based on PG_HINT_PLAN extension (https://github.com/ossc-db/pg_hint_plan)
# see https://github.com/ossc-db/pg_hint_plan#hints-list for details
PostgresOptimizerHints = {
    physops.JoinOperators.NestedLoopJoin: "NestLoop",
    physops.JoinOperators.HashJoin: "HashJoin",
    physops.JoinOperators.SortMergeJoin: "MergeJoin",
    physops.ScanOperators.SequentialScan: "SeqScan",
    physops.ScanOperators.IndexScan: "IndexOnlyScan",
    physops.ScanOperators.IndexOnlyScan: "IndexOnlyScan",
    physops.ScanOperators.BitmapScan: "BitmapScan"
}
"""Denotes all physical operators that can be enforced for individual parts of a query.

These settings overwrite the session-global optimizer settings.
"""


def _generate_join_key(tables: Iterable[base.TableReference]) -> str:
    """Builds a PG_HINT_PLAN-compatible identifier for the join consisting of the given tables."""
    return " ".join(tab.identifier() for tab in tables)


def _generate_pg_operator_hints(physical_operators: physops.PhysicalOperatorAssignment) -> HintParts:
    """Generates the hints and preparatory statements to enforce the selected optimization in Postgres."""
    settings = []
    for operator, enabled in physical_operators.global_settings.items():
        setting = "on" if enabled else "off"
        operator_key = PostgresOptimizerSettings[operator]
        settings.append(f"SET {operator_key} = '{setting}';")

    hints = []
    for table, scan_assignment in physical_operators.scan_operators.items():
        table_key = table.identifier()
        scan_assignment = PostgresOptimizerHints[scan_assignment.operator]
        hints.append(f"{scan_assignment}({table_key})")

    if hints:
        hints.append("")
    for join, join_assignment in physical_operators.join_operators.items():
        join_key = _generate_join_key(join)
        join_assignment = PostgresOptimizerHints[join_assignment.operator]
        hints.append(f"{join_assignment}({join_key})")

    if not settings and not hints:
        return HintParts.empty()

    return HintParts(settings, hints)


def _escape_setting(setting) -> str:
    """Transforms the setting variable into a string that can be used in an SQL query."""
    if isinstance(setting, float) or isinstance(setting, int):
        return str(setting)
    elif isinstance(setting, bool):
        return "TRUE" if setting else "FALSE"
    return f"'{setting}'"


def _generate_pg_parameter_hints(plan_parameters: planmeta.PlanParameterization) -> HintParts:
    """Produces the cardinality and parallelization hints for Postgres."""
    hints, settings = [], []
    for join, cardinality_hint in plan_parameters.cardinality_hints.items():
        if len(join) < 2:
            # pg_hint_plan can only generate cardinality hints for joins
            continue
        join_key = _generate_join_key(join)
        hints.append(f"Rows({join_key} #{cardinality_hint})")

    for join, num_workers in plan_parameters.parallel_worker_hints.items():
        if len(join) != 1:
            # pg_hint_plan can only generate parallelization hints for single tables
            continue
        table: base.TableReference = collection_utils.simplify(join)
        hints.append(f"Parallel({table.identifier()} {num_workers} hard)")

    for operator, setting in plan_parameters.system_specific_settings.items():
        setting = _escape_setting(setting)
        settings.append(f"SET {operator} = {setting};")

    return HintParts(settings, hints)


def _generate_hint_block(parts: HintParts) -> Optional[clauses.Hint]:
    """Constructs the hint block for the given hint parts"""
    settings, hints = parts.settings, parts.hints
    if not settings and not hints:
        return None
    settings_block = "\n".join(settings)
    hints_block = "\n".join(["/*+"] + ["  " + hint for hint in hints] + ["*/"]) if hints else ""
    return clauses.Hint(settings_block, hints_block)


def _apply_hint_block_to_query(query: qal.SqlQuery, hint_block: Optional[clauses.Hint]) -> qal.SqlQuery:
    """Generates a new query with the given hint block."""
    return transform.add_clause(query, hint_block) if hint_block else query


PostgresJoinHints = {physops.JoinOperators.NestedLoopJoin, physops.JoinOperators.IndexNestedLoopJoin,
                     physops.JoinOperators.HashJoin, physops.JoinOperators.SortMergeJoin}
PostgresScanHints = {physops.ScanOperators.SequentialScan, physops.ScanOperators.IndexScan,
                     physops.ScanOperators.IndexOnlyScan, physops.ScanOperators.BitmapScan}
PostgresPlanHints = {planmeta.HintType.CardinalityHint, planmeta.HintType.ParallelizationHint,
                     planmeta.HintType.JoinOrderHint, planmeta.HintType.JoinDirectionHint}


class PostgresHintService(db.HintService):

    def generate_hints(self, query: qal.SqlQuery,
                       join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan] = None,
                       physical_operators: Optional[physops.PhysicalOperatorAssignment] = None,
                       plan_parameters: Optional[planmeta.PlanParameterization] = None) -> qal.SqlQuery:
        adapted_query = query
        hint_parts = None

        if join_order:
            adapted_query, hint_parts = _generate_pg_join_order_hint(adapted_query, join_order, physical_operators)

        hint_parts = hint_parts if hint_parts else HintParts.empty()
        if physical_operators:
            operator_hints = _generate_pg_operator_hints(physical_operators)
            hint_parts = hint_parts.merge_with(operator_hints)

        if plan_parameters:
            plan_hints = _generate_pg_parameter_hints(plan_parameters)
            hint_parts = hint_parts.merge_with(plan_hints)

        hint_block = _generate_hint_block(hint_parts)
        adapted_query = _apply_hint_block_to_query(adapted_query, hint_block)
        return adapted_query

    def format_query(self, query: qal.SqlQuery) -> str:
        return formatter.format_quick(query)

    def supports_hint(self, hint: physops.PhysicalOperator | planmeta.HintType) -> bool:
        return hint in PostgresJoinHints | PostgresScanHints | PostgresPlanHints


# noinspection PyProtectedMember
class PostgresOptimizer(db.OptimizerInterface):
    def __init__(self, postgres_instance: PostgresInterface) -> None:
        self._pg_instance = postgres_instance

    def query_plan(self, query: qal.SqlQuery | str) -> db.QueryExecutionPlan:
        query = self._pg_instance._prepare_query_execution(query, drop_explain=True)
        raw_query_plan = self._pg_instance._obtain_query_plan(query)
        query_plan = PostgresExplainPlan(raw_query_plan)
        return query_plan.as_query_execution_plan()

    def cardinality_estimate(self, query: qal.SqlQuery | str) -> int:
        query = self._pg_instance._prepare_query_execution(query, drop_explain=True)
        query_plan = self._pg_instance._obtain_query_plan(query)
        estimate = query_plan[0]["Plan"]["Plan Rows"]
        return estimate

    def cost_estimate(self, query: qal.SqlQuery | str) -> float:
        query = self._pg_instance._prepare_query_execution(query, drop_explain=True)
        query_plan = self._pg_instance._obtain_query_plan(query)
        estimate = query_plan[0]["Plan"]["Total Cost"]
        return estimate


def connect(*, name: str = "postgres", connect_string: str | None = None,
            config_file: str | None = ".psycopg_connection", cache_enabled: bool = True,
            private: bool = False) -> PostgresInterface:
    """Convenience function to seamlessly connect to a Postgres instance.

    This function obtains a connect-string to the database according to the following rules:

    1. if the connect-string is supplied directly via the `connect_string` parameter, this is used
    2. if the connect-string is not supplied, it is read from the file indicated by `config_file`
    3. if the `config_file` does not exist, an error is raised

    The Postgres instance can be supplied a name via the `name` parameter if multiple connections to different
    Postgres instances should be maintained simultaneously. Otherwise, the parameter defaults to `postgres`.

    Caching behaviour of the Postgres instance can be controlled via the `cache_enabled` parameter.

    After a connection to the Postgres instance has been obtained, it is registered automatically by the current
    `DatabasePool` instance, unless `private` is set to `True`.
    """
    db_pool = db.DatabasePool.get_instance()
    if config_file and not connect_string:
        if not os.path.exists(config_file):
            raise ValueError("Config file was given, but does not exist: " + config_file)
        with open(config_file, "r") as f:
            connect_string = f.readline().strip()
    elif not connect_string:
        raise ValueError("Connect string or config file are required to connect to Postgres")

    postgres_db = PostgresInterface(connect_string, system_name=name, cache_enabled=cache_enabled)
    if not private:
        db_pool.register_database(name, postgres_db)
    return postgres_db


def _parallel_query_initializer(connect_string: str, local_data: threading.local, verbose: bool = False) -> None:
    """Internal function for the `ParallelQueryExecutor` to setup worker connections."""
    log = logging.make_logger(verbose)
    tid = threading.get_ident()
    connection = psycopg.connect(connect_string, application_name=f"PostBOUND parallel worker ID {tid}")
    connection.autocommit = True
    local_data.connection = connection
    log(f"[worker id={tid}, ts={logging.timestamp()}] Connected")


def _parallel_query_worker(query: str | qal.SqlQuery, local_data: threading.local, verbose: bool = False) -> Any:
    """Internal function for the `ParallelQueryExecutor` to run individual queries."""
    log = logging.make_logger(verbose)
    connection: psycopg.connection.Connection = local_data.connection
    connection.rollback()
    cursor = connection.cursor()

    log(f"[worker id={threading.get_ident()}, ts={logging.timestamp()}] Now executing query {query}")
    cursor.execute(str(query))
    log(f"[worker id={threading.get_ident()}, ts={logging.timestamp()}] Executed query {query}")

    result_set = cursor.fetchall()
    cursor.close()
    while (isinstance(result_set, list) or isinstance(result_set, tuple)) and len(result_set) == 1:
        result_set = result_set[0]

    return query, result_set


class ParallelQueryExecutor:
    """The ParallelQueryExecutor provides mechanisms to conveniently execute queries in parallel.

    The parallel execution happens by maintaining a number of worker threads that execute the incoming queries.
    The number of input queries can exceed the worker pool size, potentially by a large margin. If that is the case,
    input queries will be buffered until a worker is available.

    This parallel executor has nothing to do with the Database interface and acts entirely independently and
    Postgres-specific.
    """

    def __init__(self, connect_string: str, n_threads: int = None, *, verbose: bool = False) -> None:
        self._n_threads = n_threads if n_threads is not None and n_threads > 0 else os.cpu_count()
        self._connect_string = connect_string
        self._verbose = verbose

        self._thread_data = threading.local()
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self._n_threads,
                                                                  initializer=_parallel_query_initializer,
                                                                  initargs=(self._connect_string, self._thread_data,))
        self._tasks: list[concurrent.futures.Future] = []
        self._results = []

    def queue_query(self, query: qal.SqlQuery | str) -> None:
        """Adds a new query to the queue, to be executed as soon as possible."""
        future = self._thread_pool.submit(_parallel_query_worker, query, self._thread_data, self._verbose)
        self._tasks.append(future)

    def drain_queue(self, timeout: float = None) -> None:
        """Blocks, until all queries currently queued have terminated."""
        for future in concurrent.futures.as_completed(self._tasks, timeout=timeout):
            self._results.append(future.result())

    def result_set(self) -> dict[str | qal.SqlQuery, Any]:
        """Provides the results of all queries that have terminated already, mapping query -> result set"""
        return dict(self._results)

    def close(self) -> None:
        """Terminates all worker threads. The executor is essentially useless afterwards."""
        self._thread_pool.shutdown()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        running_workers = [future for future in self._tasks if future.running()]
        completed_workers = [future for future in self._tasks if future.done()]

        return (f"Concurrent query pool of {self._n_threads} workers, {len(self._tasks)} tasks "
                f"(run={len(running_workers)} fin={len(completed_workers)})")


PostgresExplainJoinNodes = {"Nested Loop": physops.JoinOperators.NestedLoopJoin,
                            "Hash Join": physops.JoinOperators.HashJoin,
                            "Merge Join": physops.JoinOperators.SortMergeJoin}
PostgresExplainScanNodes = {"Seq Scan": physops.ScanOperators.SequentialScan,
                            "Index Scan": physops.ScanOperators.IndexScan,
                            "Index Only Scan": physops.ScanOperators.IndexOnlyScan,
                            "Bitmap Heap Scan": physops.ScanOperators.BitmapScan}


class PostgresExplainNode:
    def __init__(self, explain_data: dict) -> None:
        self.node_type = explain_data.get("Node Type", None)

        self.cost = explain_data.get("Total Cost", math.nan)
        self.cardinality_estimate = explain_data.get("Plan Rows", math.nan)
        self.execution_time = explain_data.get("Actual Total Time", math.nan)
        self.true_cardinality = explain_data.get("Actual Rows", math.nan)
        self.loops = explain_data.get("Actual Loops", 1)

        self.relation_name = explain_data.get("Relation Name", None)
        self.relation_alias = explain_data.get("Alias", None)
        self.index_name = explain_data.get("Index Name", None)

        self.filter_condition = explain_data.get("Filter", None)
        self.index_condition = explain_data.get("Index Cond", None)
        self.join_filter = explain_data.get("Join Filter", None)
        self.hash_condition = explain_data.get("Hash Cond", None)
        self.recheck_condition = explain_data.get("Recheck Cond", None)

        self.parent_relationship = explain_data.get("Parent Relationship", None)
        self.parallel_workers = explain_data.get("Workers Launched", math.nan)

        self.children = [PostgresExplainNode(child) for child in explain_data.get("Plans", [])]

    def as_query_execution_plan(self) -> db.QueryExecutionPlan:
        if self.children and len(self.children) > 2:
            raise ValueError("Cannot transform parent node > 2 children")
        elif self.children and len(self.children) == 1:
            child_nodes = [self.children[0].as_query_execution_plan()]
            inner_child = None
        elif self.children:
            first_child, second_child = self.children
            child_nodes = [first_child.as_query_execution_plan(), second_child.as_query_execution_plan()]
            inner_child = child_nodes[0] if first_child.parent_relationship == "Inner" else child_nodes[1]
        else:
            child_nodes = None
            inner_child = None

        table = self._parse_table()
        is_scan = self.node_type in PostgresExplainScanNodes
        is_join = self.node_type in PostgresExplainJoinNodes
        par_workers = self.parallel_workers + 1  # in Postgres the control worker also processes input
        true_card = self.true_cardinality * self.loops

        if is_scan:
            operator = PostgresExplainScanNodes.get(self.node_type, None)
        elif is_join:
            operator = PostgresExplainJoinNodes.get(self.node_type, None)
        else:
            operator = None

        return db.QueryExecutionPlan(self.node_type, is_join=is_join, is_scan=is_scan, table=table,
                                     children=child_nodes, parallel_workers=par_workers,
                                     cost=self.cost, estimated_cardinality=self.cardinality_estimate,
                                     true_cardinality=true_card, execution_time=self.execution_time,
                                     physical_operator=operator, inner_child=inner_child)

    def _parse_table(self) -> Optional[base.TableReference]:
        if not self.relation_name:
            return None
        alias = self.relation_alias if self.relation_alias is not None else ""
        return base.TableReference(self.relation_name, alias)


class PostgresExplainPlan:
    def __init__(self, explain_data: dict) -> None:
        explain_data = explain_data[0] if isinstance(explain_data, list) else explain_data
        self.planning_time = explain_data.get("Planning Time", math.nan)
        self.execution_time = explain_data.get("Execution Time", math.nan)
        self.query_plan = PostgresExplainNode(explain_data["Plan"])

    def as_query_execution_plan(self) -> db.QueryExecutionPlan:
        return self.query_plan.as_query_execution_plan()
