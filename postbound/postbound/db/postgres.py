"""Contains the Postgres implementation of the Database interface.

In many ways the Postgres implementation can be thought of as the reference or blueprint implementation of the database
interface. This is due to two main reasons: first up, Postgres' capabilities follow a traditional architecture and its
features cover most of the general aspects of query optimization (i.e. supported operators, join orders and statistics).
Secondly, and on a more pragmatic note Potsgres was the first database system that was supported by PostBOUND and therefore
a lot of the original Postgres interfaces eventually evolved into the more abstract database-independent interfaces.
"""
from __future__ import annotations

import collections
import concurrent
import concurrent.futures
import math
import multiprocessing as mp
import os
import textwrap
import threading
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from multiprocessing import connection as mp_conn
from typing import Any, Optional

import psycopg
import psycopg.rows

from postbound.db import db
from postbound.qal import qal, base, expressions, clauses, transform, formatter
from postbound.optimizer import jointree, physops, planparams
from postbound.util import collections as collection_utils, dicts as dict_utils, logging
from postbound.util import misc as utils, typing as type_utils

# TODO: find a nice way to support index nested-loop join hints.
# Probably inspired by the join order/join direction handling?

HintBlock = collections.namedtuple("HintBlock", ["preparatory_statements", "hints", "query"])
"""Type alias to capture relevant info in a hint block under construction."""


@dataclass
class _GeQOState:
    """Captures the current configuration of the GeQO optimizer for a Postgres instance.

    Attributes
    ----------
    enabled : bool
        Whether the GeQO optimizer is enabled. If it is disabled, the values of all other settings can be ignored.
    threshold : int
        The minimum number of tables that need to be joined in order for the GeQO optimizer to be activated. All queries with
        less joined tables are optimized using a traditional dynamic programming-based optimizer.
    """
    enabled: bool
    threshold: int

    def triggers_geqo(self, query: qal.SqlQuery) -> bool:
        """Checks, whether a specific query would be optimized using GeQO.

        Parameters
        ----------
        query : qal.SqlQuery
            The query to check. Notice that eventual preparatory statements that might modify the GeQO configuration are
            ignored.

        Returns
        -------
        bool
            Whether GeQO would be used to optimize the query

        Warnings
        --------
        This check does not consider eventual side effects of the query that modify the optimizer configuration. For example,
        consider a query that would be optimized using GeQO given the current configuration. If this query contained a
        preparatory statement that disabled the GeQO optimizer, that statement would not influence the check result.
        """
        return self.enabled and len(query.tables()) >= self.threshold


def _query_contains_geqo_sensible_settings(query: qal.SqlQuery) -> bool:
    """Checks, whether a specific query contains any information that would be overwritten by GeQO.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to check

    Returns
    -------
    bool
        Whether the query is subject to unwanted GeQO modifications
    """
    return query.hints is not None and bool(query.hints.query_hints)


def _modifies_geqo_config(query: qal.SqlQuery) -> bool:
    """Heuristic to check whether a specific query changes the current GeQO config of a Postgres system.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to check

    Returns
    -------
    bool
        Whether the query modifies the GeQO config. Notice that this is a heuristic check - there could be false positives as
        well as false negatives!
    """
    if not query.hints or not query.hints.preparatory_statements:
        return False
    return "geqo" in query.hints.preparatory_statements.lower()


def _simplify_result_set(result_set: list[tuple[Any]]) -> Any:
    """Implementation of the result set simplification logic outlined in `Database.execute_query`.

    Parameters
    ----------
    result_set : list[tuple[Any]]
        Result set to simplify: each entry in the list corresponds to one row in the result set and each component of the
        tuples corresponds to one column in the result set

    Returns
    -------
    Any
        The simplified result set: if the result set consists just of a single row, this row is unwrapped from the list. If the
        result set contains just a single column, this is unwrapped from the tuple. Both simplifications are also combined,
        such that a result set of a single row of a single column is turned into the single value.
    """
    # simplify the query result as much as possible: [(42, 24)] becomes (42, 24) and [(1,), (2,)] becomes [1, 2]
    # [(42, 24), (4.2, 2.4)] is left as-is
    if not result_set:
        return []

    result_structure = result_set[0]  # what do the result tuples look like?
    if len(result_structure) == 1:  # do we have just one column?
        result_set = [row[0] for row in result_set]  # if it is just one column, unwrap it

    if len(result_set) == 1:  # if it is just one row, unwrap it
        return result_set[0]
    return result_set


class PostgresInterface(db.Database):
    """Database implementation for PostgreSQL backends.

    Parameters
    ----------
    connect_string : str
        Connection string for `psycopg` to establish a connection to the Postgres server
    system_name : str, optional
        Description of the specific Postgres server, by default ``"Postgres"``
    cache_enabled : bool, optional
        Whether to enable caching of database queries, by default ``True``
    """

    def __init__(self, connect_string: str, system_name: str = "Postgres", *, cache_enabled: bool = True) -> None:
        self.connect_string = connect_string
        self._connection = psycopg.connect(connect_string, application_name="PostBOUND",
                                           row_factory=psycopg.rows.tuple_row)
        self._connection.autocommit = True
        self._cursor = self._connection.cursor()

        self._db_stats = PostgresStatisticsInterface(self)
        self._db_schema = PostgresSchemaInterface(self)

        self._current_geqo_state = self._obtain_geqo_state()

        super().__init__(system_name, cache_enabled=cache_enabled)

    def schema(self) -> PostgresSchemaInterface:
        return self._db_schema

    def statistics(self) -> PostgresStatisticsInterface:
        return self._db_stats

    def hinting(self) -> PostgresHintService:
        return PostgresHintService()

    def execute_query(self, query: qal.SqlQuery | str, *, cache_enabled: Optional[bool] = None) -> Any:
        cache_enabled = cache_enabled or (cache_enabled is None and self._cache_enabled)
        query = self._prepare_query_execution(query)

        if cache_enabled and query in self._query_cache:
            query_result = self._query_cache[query]
        else:
            try:
                self._cursor.execute(query)
                query_result = self._cursor.fetchall() if self._cursor.rowcount >= 0 else None
                self._restore_geqo_state()
            except (psycopg.InternalError, psycopg.OperationalError) as e:
                msg = "\n".join([f"At {utils.current_timestamp()}", "For query:", str(query), "Message:", str(e)])
                raise db.DatabaseServerError(msg, e)
            except psycopg.Error as e:
                msg = "\n".join([f"At {utils.current_timestamp()}", "For query:", str(query), "Message:", str(e)])
                raise db.DatabaseUserError(msg, e)
            if cache_enabled:
                self._inflate_query_cache()
                self._query_cache[query] = query_result

        return _simplify_result_set(query_result)

    def optimizer(self) -> PostgresOptimizer:
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
        """Prepares the Postgres buffer pool with tuples from the given tables.

        Parameters
        ----------
        tables : Optional[base.TableReference  |  Iterable[base.TableReference]], optional
            The tables that should be placed into the buffer pool
        *more_tables : base.TableReference
            More tables that should be placed into the buffer pool, enabling a more convenient usage of this method.
            See examples for details on the usage.

        Examples
        --------
        >>> pg.prewarm_tables([table1, table2])
        >>> pg.prewarm_tables(table1, table2)
        """
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
        """Handles necessary setup logic that enable an arbitrary query to be executed by the database system.

        This setup process involves formatting the query to accomodate deviations from standard SQL by the database
        system, as well as executing preparatory statements of the query's `Hint` clause.

        `drop_explain` can be used to remove any EXPLAIN clauses from the query. Note that all actions that require
        the "semantics" of the query to be known (e.g. EXPLAIN modifications or query hints) and are therefore only
        executed for instances of the qal queries.

        Parameters
        ----------
        query : qal.SqlQuery | str
            The query to prepare. Only queries from the query abstraction layer can be prepared.
        drop_explain : bool, optional
            Whether any `Explain` clauses on the query should be ignored. This is intended for cases where the callee
            has its own handling of ``EXPLAIN`` blocks. Defaults to ``False``.

        Returns
        -------
        str
            A unified version of the query that is ready for execution
        """
        if not isinstance(query, qal.SqlQuery):
            return query

        requires_geqo_deactivation = _query_contains_geqo_sensible_settings(query) and not _modifies_geqo_config(query)
        if requires_geqo_deactivation and self._current_geqo_state.triggers_geqo(query):
            self._cursor.execute("SET enable_geqo = 'off';")

        if drop_explain:
            query = transform.drop_clause(query, clauses.Explain)
        if query.hints and query.hints.preparatory_statements:
            self._cursor.execute(query.hints.preparatory_statements)
            query = transform.drop_hints(query, preparatory_statements_only=True)
        return self.hinting().format_query(query)

    @type_utils.module_local
    def _obtain_query_plan(self, query: str) -> dict:
        """Provides the query plan that would be used for executing a specific query.

        Parameters
        ----------
        query : str
            The query to plan. It does not have to be an ``EXPLAIN`` query already, this will be added if necessary.

        Returns
        -------
        dict
            The raw ``EXPLAIN`` data.
        """
        if not query.upper().startswith("EXPLAIN (FORMAT JSON)"):
            query = "EXPLAIN (FORMAT JSON) " + query
        self._cursor.execute(query)
        return self._cursor.fetchone()[0]

    def _obtain_geqo_state(self) -> _GeQOState:
        """Fetches the current GeQO configuration from the database.

        Returns
        -------
        _GeQOState
            The relevant GeQO config
        """
        self._cursor.execute("SELECT name, setting FROM pg_settings "
                             "WHERE name = 'geqo' OR name = 'geqo_threshold' ORDER BY name;")
        geqo_enabled: bool = False
        geqo_threshold: int = 0
        for name, value in self._cursor.fetchall():
            if name == "geqo":
                geqo_enabled = value
            elif name == "geqo_threshold":
                geqo_threshold = value
            else:
                raise RuntimeError("Malformed GeQO query. This is a programming error, it's not your fault!")
        return _GeQOState(geqo_enabled, geqo_threshold)

    @type_utils.module_local
    def _restore_geqo_state(self) -> None:
        """Resets the GeQO configuration of the database system to the known `_current_geqo_state`."""
        geqo_enabled = "on" if self._current_geqo_state.enabled else "off"
        self._cursor.execute(f"SET geqo = '{geqo_enabled}';")
        self._cursor.execute(f"SET geqo_threshold = {self._current_geqo_state.threshold};")


class PostgresSchemaInterface(db.DatabaseSchema):
    """Database schema implementation for Postgres systems.

    Parameters
    ----------
    postgres_db : PostgresInterface
        The database for which schema information should be retrieved
    """

    def __int__(self, postgres_db: PostgresInterface) -> None:
        super().__init__(postgres_db)

    def tables(self) -> set[base.TableReference]:
        query_template = textwrap.dedent("""
                                         SELECT table_name
                                         FROM information_schema.tables
                                         WHERE table_catalog = %s AND table_schema = 'public'""")
        self._db.cursor().execute(query_template, (self._db.database_name(),))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        return set(base.TableReference(row[0]) for row in result_set)

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
        if column.table.virtual:
            raise base.VirtualTableError(column.table)
        index_map = self._fetch_indexes(column.table)
        return index_map.get(column.name, False)

    def has_secondary_index(self, column: base.ColumnReference) -> bool:
        if not column.table:
            raise base.UnboundColumnError(column)
        if column.table.virtual:
            raise base.VirtualTableError(column.table)
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
        if column.table.virtual:
            raise base.VirtualTableError(column.table)
        query_template = textwrap.dedent("""
            SELECT data_type FROM information_schema.columns
            WHERE table_name = {tab} AND column_name = {col}""".format(tab=column.table.full_name, col=column.name))
        self._db.cursor().execute(query_template)
        result_set = self._db.cursor().fetchone()
        return result_set[0]

    def _fetch_columns(self, table: base.TableReference) -> list[str]:
        """Retrieves all physical columns for a given table from the PG metadata catalogs.

        Parameters
        ----------
        table : base.TableReference
            The table whose columns should be loaded

        Returns
        -------
        list[str]
            The names of all columns

        Raises
        ------
        postbound.qal.base.VirtualTableError
            If the table is a virtual table (e.g. subquery or CTE)
        """
        if table.virtual:
            raise base.VirtualTableError(table)
        query_template = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        self._db.cursor().execute(query_template, (table.full_name,))
        result_set = self._db.cursor().fetchall()
        return [col[0] for col in result_set]

    def _fetch_indexes(self, table: base.TableReference) -> dict[str, bool]:
        """Retrieves all index structures for a given table based on the PG metadata catalogs.

        Parameters
        ----------
        table : base.TableReference
            The table whose indexes should be loaded

        Returns
        -------
        dict
            Contains a key for each column that has an index. The column keys map to booleans that indicate whether
            the corresponding index is a primary key index. Columns without any index do not appear in the dictionary.

        Raises
        ------
        postbound.qal.base.VirtualTableError
            If the table is a virtual table (e.g. subquery or CTE)
        """
        if table.virtual:
            raise base.VirtualTableError(table)
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
    """Statistics implementation for Postgres systems.

    Parameters
    ----------
    postgres_db : PostgresInterface
        The database instance for which the statistics should be retrieved
    emulated : bool, optional
        Whether the statistics interface should operate in emulation mode. To enable reproducibility, this is ``True``
        by default
    enable_emulation_fallback : bool, optional
        Whether emulation should be used for unsupported statistics when running in native mode, by default True
    cache_enabled : Optional[bool], optional
        Whether emulated statistics queries should be subject to caching, by default True. Set to ``None`` to use the
        caching behavior of the `db`
    """

    def __init__(self, postgres_db: PostgresInterface, *, emulated: bool = True,
                 enable_emulation_fallback: bool = True, cache_enabled: Optional[bool] = True) -> None:
        super().__init__(postgres_db, emulated=emulated, enable_emulation_fallback=enable_emulation_fallback,
                         cache_enabled=cache_enabled)

    def update_statistics(self, columns: Optional[base.ColumnReference | Iterable[base.ColumnReference]] = None, *,
                          tables: Optional[base.TableReference | Iterable[base.TableReference]] = None,
                          perfect_mcv: bool = False) -> None:
        """Instructs the Postgres server to update statistics for specific columns.

        Notice that is one of the methods of the database interface that explicitly mutates the state of the database system.

        Parameters
        ----------
        columns : Optional[base.ColumnReference  |  Iterable[base.ColumnReference]], optional
            The columns for which statistics should be updated. If no columns are given, columns are inferred based on the
            `tables` and all detected columns are used.
        tables : Optional[base.TableReference  |  Iterable[base.TableReference]], optional
            The table for which statistics should be updated. If `columns` are given, this parameter is completely ignored. If
            no columns and no tables are given, all tables in the current database are used.
        perfect_mcv : bool, optional
            Whether the database system should attempt to create perfect statistics. Perfect statistics means that for each of
            the columns MCV lists are created such that each distinct value is contained within the list. For large and diverse
            columns, this might lots of compute time as well as storage space. Notice, that the database system still has the
            ultimate decision on whether to generate MCV lists in the first place. Postgres also imposes a hard limit on the
            maximum allowed length of MCV lists and histogram widths.
        """
        if not columns and not tables:
            tables = self._db.schema().tables()
        if not columns and tables:
            tables = collection_utils.enlist(tables)
            columns = collection_utils.set_union(self._db.schema().columns(tab) for tab in tables)

        assert columns is not None
        columns: Iterable[base.ColumnReference] = collection_utils.enlist(columns)
        columns_map: dict[base.TableReference, list[str]] = dict_utils.generate_multi((col.table, col.name)
                                                                                      for col in columns)

        if perfect_mcv:
            for column in columns:
                n_distinct = self.distinct_values(column, emulated=False, cache_enabled=True)
                stats_target_query = textwrap.dedent(f"""ALTER TABLE {column.table} """
                                                     f"""ALTER COLUMN {column.name} """
                                                     f"""SET STATISTICS {n_distinct};""")
                # This query might issue a warning if the requested stats target is larger than the allowed maximum value
                # However, Postgres simply uses the maximum value in this case. To permit different maximum values in different
                # Postgres versions, we accept the warning and do not use a hard-coded maximum value with snapping logic
                # ourselves.
                self._db.cursor().execute(stats_target_query)

        columns_str = {table: ", ".join(col for col in columns) for table, columns in columns_map.items()}
        tables_and_columns = ", ".join(f"{table.full_name}({cols})" for table, cols in columns_str.items())

        query_template = f"ANALYZE {tables_and_columns}"
        self._db.cursor().execute(query_template)

    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> Optional[int]:
        count_query = f"SELECT reltuples FROM pg_class WHERE oid = '{table.full_name}'::regclass"
        self._db.cursor().execute(count_query)
        result_set = self._db.cursor().fetchone()
        if not result_set:
            return None
        count = result_set[0]
        return count

    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> Optional[int]:
        dist_query = "SELECT n_distinct FROM pg_stats WHERE tablename = %s and attname = %s"
        self._db.cursor().execute(dist_query, (column.table.full_name, column.name))
        result_set = self._db.cursor().fetchone()
        if not result_set:
            return None
        dist_values = result_set[0]

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
    """Models the different kinds of optimizer hints that are supported by Postgres.

    HintParts are designed to conveniently collect all kinds of hints in order to prepare the generation of a proper
    `Hint` clause.

    See Also
    --------
    postbound.qal.clauses.Hint
    """
    settings: list[str]
    """Settings are global to the current database connection and influence the selection of operators for all queries.

    Typical examples include ``SET enable_nestloop = 'off'``, which disables the usage of nested loop joins for all
    queries.
    """

    hints: list[str]
    """Hints are supplied by the *pg_hint_plan* extension and influence optimizer decisions on a per-query basis.

    Typical examples include the selection of a specific join order as well as the assignment of join operators to
    individual joins.

    References
    ----------

    .. pg_hint_plan extension: https://github.com/ossc-db/pg_hint_plan
    """

    @staticmethod
    def empty() -> HintParts:
        """Creates a new hint parts object without any contents.

        Returns
        -------
        HintParts
            A fresh plain hint parts object
        """
        return HintParts([], [])

    def merge_with(self, other: HintParts) -> HintParts:
        """Combines the hints that are contained in this hint parts object with all hints in the other object.

        This constructs new hint parts and leaves the current objects unmodified.

        Parameters
        ----------
        other : HintParts
            The additional hints to incorporate

        Returns
        -------
        HintParts
            A new hint parts object that contains the hints from both source objects
        """
        merged_settings = self.settings + [setting for setting in other.settings if setting not in self.settings]
        merged_hints = self.hints + [hint for hint in other.hints if hint not in self.hints]
        return HintParts(merged_settings, merged_hints)


def _is_hash_join(join_tree_node: jointree.IntermediateJoinNode,
                  operator_assignment: Optional[physops.PhysicalOperatorAssignment]) -> bool:
    """Checks, whether a specific join should be executed as a hash join.

    Parameters
    ----------
    join_tree_node : jointree.IntermediateJoinNode
        The join to check. Can contain a `PhysicalJoinMetadata` annotation that describes the join operator. This
        metadata is checked if no dedicated `operator_assignment` is provided.
    operator_assignment : Optional[physops.PhysicalOperatorAssignment]
        The operator assignment. If such an assignment is available, it will take precedence over all eventual
        operator selections that are part of the `join_tree_node` annotation.

    Returns
    -------
    bool
        Whether the join described by the given node should be executed as a hash join. For base tables or unspecified
        operator assignments this is ``False``.

    Notes
    -----
    This is method is necessary because the pg_hint_plan extension that we use for enforcing the join order does not
    only enforce the join order, but also the join direction (i.e. inner and outer relations) at the same time. This
    means that whenever we are concerned with the order in which relations should be joined, we need to decide which
    of these relations should be the inner and outer relation. While for some joins this does not matter too much
    (e.g. nested-loop join or sort-merge join), it becomes quite important for others (index nested-loop join and
    hash join). Sadly, the role of the inner and outer relations is not consistent accross all join types. Whereas for
    most joins the inner relations is being "probed" in some sense (e.g. index nested-loop join), for hash joins it is
    exactly the other way around, i.e. the inner relation is put into a hash table and the outer relation is probed
    against it. As a consequence, we need to swap the roles of the input relations for these joins. The
    `_is_hash_join` method implements the logic for figuring out whether the is necessary for a specific join.
    """
    if operator_assignment is not None:
        selected_operator: Optional[physops.JoinOperatorAssignment] = operator_assignment[join_tree_node.tables()]
        if selected_operator is not None:
            return selected_operator.operator == physops.JoinOperators.HashJoin

        # We do not need to check the global hash join setting b/c it is overwritten by the per-join setting
        # if we would not have a per-join setting, but a global setting the decision is the same in all cases:
        # If hash joins are disabled, the result would be False
        # If hash joins are enabled, this only means that the query optimizer is free to choose the operator, but not
        # forced to. Therefore, we have to infer that the join is not a hash join and the result is once again False
        # The only special case is if the hash join is the only globally enabled join operator and there is no per-join
        # assignment. In this case, the join has to be a hash join. This condition is checked below
        global_join_ops = {op for op in operator_assignment.get_globally_enabled_operators()
                           if isinstance(op, physops.JoinOperators)}
        return len(global_join_ops) == 1 and physops.JoinOperators.HashJoin in global_join_ops

    if not isinstance(join_tree_node.annotation, jointree.PhysicalJoinMetadata):
        return False

    return (join_tree_node.annotation.operator is not None
            and join_tree_node.annotation.operator.operator == physops.JoinOperators.HashJoin)


def _generate_leading_hint_content(join_tree_node: jointree.AbstractJoinTreeNode,
                                   operator_assignment: Optional[physops.PhysicalOperatorAssignment] = None) -> str:
    """Builds a substring of the ``Leading`` hint to enforce the join order for a specific part of the join tree.

    Parameters
    ----------
    join_tree_node : jointree.AbstractJoinTreeNode
        Subtree of the join order for which the hint should be generated
    operator_assignment : Optional[physops.PhysicalOperatorAssignment], optional
        Operators that allow a smarter assignment of join directions. This is necessary due to pecularities of the
        ``Leading`` hint. See reference below.

    Returns
    -------
    str
        Part of the join hint that corresponds to the join of the current subtree, e.g. for the join ``R ⋈ S``, the
        substring would be ``"R S"``.

    Raises
    ------
    ValueError
        If the `join_tree_node` is neither a base table node, nor an intermediate join node. This error should never
        ever be raised. If it is, it means that the join tree representation was updated to include other types of
        nodes (whatever these should be), and the implementation of this method was not updated accordingly. This is
        a severe bug in the method!

    See Also
    --------
    _is_hash_join

    Notes
    -----
    To assign directions to the join partners, the following rules are used (in the given order):

    1. If the join contains directional information (i.e. is annotated by a `DirectionalJoinOperatorAssignment`),
       this assignment is used
    2. If the join has cardinality estimates available on both input relations, the join direction is chosen such that
       the outer relation is the one with the smaller cardinality estimate. This does not really make a difference for
       plain nested-loop joins or sort-merge joins, but can lead to significant speedup for index nested-loop joins or
       hash joins.
    3. Otherwise, the left child node becomes the outer relation and the right child becomes the inner relation

    Notice that we use the terminology of inner and outer relation in the conventional way here (and not how Postgres
    applies these terms). Most importantly this means that for a hash join we consider the outer relation to be the
    one that is put into a hash table and the inner relation to be the one that is used for probing. The necessity to
    swap this meaning when applying this strategy for Postgres is merely an implementation detail.

    References
    ----------

    .. pg_hint_plan Leading hint: https://github.com/ossc-db/pg_hint_plan/blob/master/docs/hint_list.md
    """
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
        inner_tables = annotation.inner
        inner_child = (join_tree_node.left_child if join_tree_node.left_child.tables() == inner_tables
                       else join_tree_node.right_child)
        outer_child = (join_tree_node.left_child if inner_child == join_tree_node.right_child
                       else join_tree_node.right_child)
        inner_child, outer_child = ((outer_child, inner_child) if annotation.operator == physops.JoinOperators.HashJoin
                                    else (inner_child, outer_child))
    else:
        left, right = join_tree_node.left_child, join_tree_node.right_child
        has_left_bound = math.isfinite(left.cardinality)
        has_right_bound = math.isfinite(right.cardinality)

        if not has_left_bound or not has_right_bound:
            inner_child, outer_child = right, left
        # At this point we know that both child nodes have upper bounds
        elif _is_hash_join(join_tree_node, operator_assignment):
            # Apply the same Hash join direction correction as above
            inner_child, outer_child = (left, right) if left.cardinality < right.cardinality else (right, left)
        else:
            # Otherwise have the smaller relation be the outer one
            inner_child, outer_child = (right, left) if right.cardinality < left.cardinality else (left, right)

    inner_hint = _generate_leading_hint_content(inner_child, operator_assignment)
    outer_hint = _generate_leading_hint_content(outer_child, operator_assignment)
    return f"({outer_hint} {inner_hint})"


def _generate_pg_join_order_hint(query: qal.SqlQuery,
                                 join_order: jointree.LogicalJoinTree | jointree.PhysicalQueryPlan,
                                 operator_assignment: Optional[physops.PhysicalOperatorAssignment] = None
                                 ) -> tuple[qal.SqlQuery, Optional[HintParts]]:
    """Builds the entire ``Leading`` hint to enforce the join order for a specific query.

    Using a ``Leading`` hint, it is possible to generate the an arbitrarily nested join order for an input query.
    However, at the same time this hint also enforces the join direction (i.e. inner or outer relation) of the join
    partners. Due to some pecularities of the interpretation of inner and outer relation by Postgres, this method
    also needs to access the operator assignment in addition to the join tree. The directions in the hint depend on the
    selected join operators. See `_generate_leading_hint_content` for details. This method delegates most of the heavy
    lifting to it.

    Parameters
    ----------
    query : qal.SqlQuery
        The query for which the hint should be generated. Notice that the hint will not be incorporated at this stage
        already. Strictly speaking, this parameter is not even necessary for the hint generation part. However, it is
        still included to retain the ability to quickly switch to a query transformation-based join order enforcement
        at a later point in time.
    join_order : jointree.LogicalJoinTree | jointree.PhysicalQueryPlan
        The desired join order
    operator_assignment : Optional[physops.PhysicalOperatorAssignment], optional
        The operators that should be used to perform the actual joins. This is necessary to generate the correct join
        order (as outlined above)

    Returns
    -------
    tuple[qal.SqlQuery, Optional[HintParts]]
        A potentially transformed version of the input query along with the necessary hints to enforce the join order.
        All future hint generation steps should use the transformed query as input rather than the original one, even
        though right now no transformations are being performed and the returned query simply matches the input query.

    See Also
    --------
    _generate_leading_hint_content

    References
    ----------

    .. pg_hint_plan Leading hint: https://github.com/ossc-db/pg_hint_plan/blob/master/docs/hint_list.md
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
"""All (session-global) optimizer settings that modify the allowed physical operators."""

PostgresOptimizerHints = {
    physops.JoinOperators.NestedLoopJoin: "NestLoop",
    physops.JoinOperators.HashJoin: "HashJoin",
    physops.JoinOperators.SortMergeJoin: "MergeJoin",
    physops.ScanOperators.SequentialScan: "SeqScan",
    physops.ScanOperators.IndexScan: "IndexOnlyScan",
    physops.ScanOperators.IndexOnlyScan: "IndexOnlyScan",
    physops.ScanOperators.BitmapScan: "BitmapScan"
}
"""All physical operators that can be enforced for individual parts of a query.

These settings operate on a per-relation basis and overwrite the session-global optimizer settings. They are based on
the *pg_hint_plan* Postgres extension.

References
----------

.. pg_hint_plan hints: https://github.com/ossc-db/pg_hint_plan/blob/master/docs/hint_list.md
"""


def _generate_join_key(tables: Iterable[base.TableReference]) -> str:
    """Produces an identifier for the given join that is compatible with the *pg_hint_plan* operator hint syntax.

    Parameters
    ----------
    tables : Iterable[base.TableReference]
        The join in question, consisting exactly of the given tables.

    Returns
    -------
    str
        A *pg_hint_plan* compatible identifier that can be used to enforce operator hints for the join.
    """
    return " ".join(tab.identifier() for tab in tables)


def _generate_pg_operator_hints(physical_operators: physops.PhysicalOperatorAssignment) -> HintParts:
    """Builds the necessary operator-level hints and global settings to enforce a specific operator assignment.

    The resulting hints object will consist of pg_hint_plan hints for all operators that affect individual joins or
    scans and standard postgres settings that influence the selection of operators for the entire query. Notice that
    the per-operator hints overwrite global settings, e.g. one can disabled nested-loop joins globally, but than assign
    a nested-loop join for a specific join again. This will disable nested-loop joins for all joins in the query,
    except for the one join in question. For that join, a nested-loop join will be forced.

    Parameters
    ----------
    physical_operators : physops.PhysicalOperatorAssignment
        The operator settings in question

    Returns
    -------
    HintParts
        A Postgres and pg_hint_plan compatible encoding of the operator assignment
    """
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
        hints.append("")  # insert empty hint to force empty line between scan and join operators
    for join, join_assignment in physical_operators.join_operators.items():
        join_key = _generate_join_key(join)
        join_assignment = PostgresOptimizerHints[join_assignment.operator]
        hints.append(f"{join_assignment}({join_key})")

    if not settings and not hints:
        return HintParts.empty()

    return HintParts(settings, hints)


def _escape_setting(setting: Any) -> str:
    """Transforms the value of a setting variable such that it is a valid text for Postgres.

    Most importantly, this involves using the correct quoting for string values and the correct transformation for
    boolean values.

    Parameters
    ----------
    setting : Any
        The value to escape

    Returns
    -------
    str
        The escaped, string version of the value

    Warnings
    --------
    This is not a secure escape routine and does not protect against common security vulnerabilities found in the real
    world. PostBOUND is not intended for production-level usage and it is incredibly easy for a malicious actor to
    take over the system. It is a research tool that is only intended for use in a secure and trusted environment. The
    most important point for this function is that it does not handle nested quoting, so breaking out of the setting is
    very straightforward.
    """
    if isinstance(setting, float) or isinstance(setting, int):
        return str(setting)
    elif isinstance(setting, bool):
        return "TRUE" if setting else "FALSE"
    return f"'{setting}'"


def _generate_pg_parameter_hints(plan_parameters: planparams.PlanParameterization) -> HintParts:
    """Builds the necessary operator-level hints and global settings to communicate plan parameters to the optimizer.

    The resulting hints object will consist of pg_hint_plan hints that enforce custom cardinality estimates for
    specific joins, hints that enforce a parallel scan for a given base table and preparatory statements to accomodate
    all user-specific settings. Notice that due to limitations of the pg_hint_plan extension, cardinality hints
    currently only work for intermediate results and not for base tables. Conversely, it is not possible to supply
    parallelization hints for anything other than base tables, i.e. it is not possible to enforce the parallel
    execution of a join.


    Parameters
    ----------
    plan_parameters : planparams.PlanParameterization
        The parameters in question

    Returns
    -------
    HintParts
        A Postgres and pg_hint_plan compatible encoding of the operator assignment

    Warns
    -----
    Emits warnings if either 1) cardinality hints are supplied for base tables (currently pg_hint_plan can only
    communicate cardinality hints for intermediate joins), or 2) parallel worker hints are supplied for intermediate
    joins (currently pg_hint_plan can only communicate parallelization hints for parallel scans)
    """
    hints, settings = [], []
    for join, cardinality_hint in plan_parameters.cardinality_hints.items():
        if len(join) < 2:
            # pg_hint_plan can only generate cardinality hints for joins
            warnings.warn(f"Ignoring cardinality hint for base table {join}")
            continue
        join_key = _generate_join_key(join)
        hints.append(f"Rows({join_key} #{cardinality_hint})")

    for join, num_workers in plan_parameters.parallel_worker_hints.items():
        if len(join) != 1:
            # pg_hint_plan can only generate parallelization hints for single tables
            warnings.warn(f"Ignoring parallel workers hint for join {join}")
            continue
        table: base.TableReference = collection_utils.simplify(join)
        hints.append(f"Parallel({table.identifier()} {num_workers} hard)")

    for operator, setting in plan_parameters.system_specific_settings.items():
        setting = _escape_setting(setting)
        settings.append(f"SET {operator} = {setting};")

    return HintParts(settings, hints)


def _generate_hint_block(parts: HintParts) -> Optional[clauses.Hint]:
    """Transforms a collection of hints into a proper hint clause.

    Parameters
    ----------
    parts : HintParts
        The hints to combine

    Returns
    -------
    Optional[clauses.Hint]
        A syntactically correct hint clause tailored for Postgres and pg_hint_plan. If neither settings nor hints
        are contained in the `parts`, ``None`` is returned instead.
    """
    settings, hints = parts.settings, parts.hints
    if not settings and not hints:
        return None
    settings_block = "\n".join(settings)
    hints_block = "\n".join(["/*+"] + ["  " + hint for hint in hints] + ["*/"]) if hints else ""
    return clauses.Hint(settings_block, hints_block)


def _apply_hint_block_to_query(query: qal.SqlQuery, hint_block: Optional[clauses.Hint]) -> qal.SqlQuery:
    """Ensures that a hint block is added to a query.

    Since the query abstraction layer consists of immutable data objects, a new query has to be created. This method's
    only purpose is to catch situations where the hint block is empty.

    Parameters
    ----------
    query : qal.SqlQuery
        The query to apply the hint block to
    hint_block : Optional[clauses.Hint]
        The hint block to apply. If this is ``None``, no modifications are performed.

    Returns
    -------
    qal.SqlQuery
        The input query with the hint block applied
    """
    return transform.add_clause(query, hint_block) if hint_block else query


PostgresJoinHints = {physops.JoinOperators.NestedLoopJoin, physops.JoinOperators.HashJoin,
                     physops.JoinOperators.SortMergeJoin}
"""All join operators that are supported by Postgres."""

PostgresScanHints = {physops.ScanOperators.SequentialScan, physops.ScanOperators.IndexScan,
                     physops.ScanOperators.IndexOnlyScan, physops.ScanOperators.BitmapScan}
"""All scan operators that are supported by Postgres."""

PostgresPlanHints = {planparams.HintType.CardinalityHint, planparams.HintType.ParallelizationHint,
                     planparams.HintType.JoinOrderHint, planparams.HintType.JoinSubqueryHint,
                     planparams.HintType.JoinDirectionHint, planparams.HintType.OperatorHint}
"""All non-operator hints supported by Postgres, that can be used to enforce additional optimizer behaviour."""


class _PostgresCastExpression(expressions.CastExpression):
    """A specialized cast expression to handle the custom syntax for ``CAST`` statements used by Postgres.

    Parameters
    ----------
    original_cast : expressions.CastExpression
        The actual cast expression. The new cast expression acts as a decorator around the original expression.
    """

    def __init__(self, original_cast: expressions.CastExpression) -> None:
        super().__init__(original_cast.casted_expression, original_cast.target_type)

    def __str__(self) -> str:
        return f"{self.casted_expression}::{self.target_type}"


class PostgresLimitClause(clauses.Limit):
    """A specialized ``LIMIT`` clause implementation to handle Postgres custom syntax for limits / offsets

    Parameters
    ----------
    original_clause : clauses.Limit
        The actual ``LIMIT`` clause. The new limit clause acts as a decorator around the original clause.
    """

    def __init__(self, original_clause: clauses.Limit) -> None:
        super().__init__(limit=original_clause.limit, offset=original_clause.offset)

    def __str__(self) -> str:
        if self.limit and self.offset:
            return f"LIMIT {self.limit} OFFSET {self.offset}"
        elif self.limit:
            return f"LIMIT {self.limit}"
        elif self.offset:
            return f"OFFSET {self.offset}"
        else:
            return ""


def _replace_postgres_cast_expressions(expression: expressions.SqlExpression) -> expressions.SqlExpression:
    """Wraps a given expression by a `_PostgresCastExpression` if necessary.

    This is the replacment method required by the `replace_expressions` transformation. It wraps all `CastExpression`
    instances by a `_PostgresCastExpression` and leaves all other expressions intact.

    Parameters
    ----------
    expression : expressions.SqlExpression
        The expression to check

    Returns
    -------
    expressions.SqlExpression
        A potentially wrapped version of the original expression

    See Also
    --------
    postbound.qal.transform.replace_expressions
    """
    return _PostgresCastExpression(expression) if isinstance(expression, expressions.CastExpression) else expression


class PostgresHintService(db.HintService):
    """Postgres-specific implementation of the hinting capabilities.

    Most importantly, this service implements a mapping from the abstract optimization descisions (join order + operators) to
    their counterparts in the pg_hint_plan extension and integrates Postgres' few deviations from standard SQL syntax (``CAST``
    expressions and ``LIMIT`` clauses).

    Notice that by delegating the adaptation of Postgres' native optimizer to the pg_hint_plan extension, a couple of
    undesired side-effects have to be accepted:

    1. forcing a join order also involves forcing a specific join direction. Our implementation applies a couple of heuristics
       to mitigate a bad impact on performance
    2. the extension only instruments the dynamic programming-based optimizer. If the ``geqo_threshold`` is reached and the
       genetic optimizer takes over, no modifications are applied. Therefore, it is best to disable GeQO while working with
       Postgres. At the same time, this means that certain scenarios like custom cardinality estimation for the genetic
       optimizer cannot currently be tested

    See Also
    --------
    _generate_pg_join_order_hint

    References
    ----------

    .. pg_hint_plan extension: https://github.com/ossc-db/pg_hint_plan
    .. Postgres query planning configuration: https://www.postgresql.org/docs/current/runtime-config-query.html
    """

    def generate_hints(self, query: qal.SqlQuery,
                       join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan] = None,
                       physical_operators: Optional[physops.PhysicalOperatorAssignment] = None,
                       plan_parameters: Optional[planparams.PlanParameterization] = None) -> qal.SqlQuery:
        adapted_query = query
        hint_parts = None

        if join_order:
            adapted_query, hint_parts = _generate_pg_join_order_hint(adapted_query, join_order, physical_operators)

        hint_parts = hint_parts if hint_parts else HintParts.empty()

        physical_operators = (join_order.physical_operators()
                              if not physical_operators and isinstance(join_order, jointree.PhysicalQueryPlan)
                              else physical_operators)
        plan_parameters = (join_order.plan_parameters()
                           if not plan_parameters and isinstance(join_order, jointree.PhysicalQueryPlan)
                           else plan_parameters)

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
        query = transform.replace_expressions(query, _replace_postgres_cast_expressions)
        if query.limit_clause:
            query = transform.replace_clause(query, PostgresLimitClause(query.limit_clause))
        return formatter.format_quick(query)

    def supports_hint(self, hint: physops.PhysicalOperator | planparams.HintType) -> bool:
        return hint in PostgresJoinHints | PostgresScanHints | PostgresPlanHints


class PostgresOptimizer(db.OptimizerInterface):
    """Optimizer introspection for Postgres.

    Parameters
    ----------
    postgres_instance : PostgresInterface
        The database whose optimizer should be introspected
    """

    def __init__(self, postgres_instance: PostgresInterface) -> None:
        self._pg_instance = postgres_instance

    def query_plan(self, query: qal.SqlQuery | str) -> db.QueryExecutionPlan:
        if isinstance(query, qal.SqlQuery):
            query = self._pg_instance._prepare_query_execution(query, drop_explain=True)
        raw_query_plan = self._pg_instance._obtain_query_plan(query)
        query_plan = PostgresExplainPlan(raw_query_plan)
        self._pg_instance._restore_geqo_state()
        return query_plan.as_query_execution_plan()

    def analyze_plan(self, query: qal.SqlQuery) -> db.QueryExecutionPlan:
        query = self._pg_instance._prepare_query_execution(transform.as_explain_analyze(query))
        self._pg_instance.cursor().execute(query)
        raw_query_plan = self._pg_instance.cursor().fetchone()[0]
        query_plan = PostgresExplainPlan(raw_query_plan)
        self._pg_instance._restore_geqo_state()
        return query_plan.as_query_execution_plan()

    def cardinality_estimate(self, query: qal.SqlQuery | str) -> int:
        if isinstance(query, qal.SqlQuery):
            query = self._pg_instance._prepare_query_execution(query, drop_explain=True)
        query_plan = self._pg_instance._obtain_query_plan(query)
        estimate = query_plan[0]["Plan"]["Plan Rows"]
        self._pg_instance._restore_geqo_state()
        return estimate

    def cost_estimate(self, query: qal.SqlQuery | str) -> float:
        if isinstance(query, qal.SqlQuery):
            query = self._pg_instance._prepare_query_execution(query, drop_explain=True)
        query_plan = self._pg_instance._obtain_query_plan(query)
        estimate = query_plan[0]["Plan"]["Total Cost"]
        self._pg_instance._restore_geqo_state()
        return estimate


def connect(*, name: str = "postgres", connect_string: str | None = None,
            config_file: str | None = ".psycopg_connection", cache_enabled: bool = True,
            refresh: bool = False, private: bool = False) -> PostgresInterface:
    """Convenience function to seamlessly connect to a Postgres instance.

    This function obtains a connect-string to the database according to the following rules:

    1. if the connect-string is supplied directly via the `connect_string` parameter, this is used
    2. if the connect-string is not supplied, it is read from the file indicated by `config_file`. This file has to be located
       in the current working directory, or the file name has to describe the path to that file.
    3. if the `config_file` does not exist, an error is raised

    After a connection to the Postgres instance has been obtained, it is registered automatically on the current
    `DatabasePool` instance. This can be changed via the `private` parameter.

    Parameters
    ----------
    name : str, optional
        A name to identify the current connection if multiple connections to different Postgres instances should be maintained.
        This is used to register the instance on the `DatabasePool`. Defaults to ``"postgres"``.
    connect_string : str | None, optional
        A Psycopg-compatible connect string for the database. Supplying this parameter overwrites any other connection
        data
    config_file : str | None, optional
        A file containing a Psycopg-compatible connect string for the database. This is the default and preferred method of
        connecting to a Postgres database. Defaults to *.psycopg_connection*
    cache_enabled : bool, optional
        Controls the default caching behaviour of the Postgres instance. Caching is enabled by default.
    refresh : bool, optional
        If true, a new connection to the database will always be established, even if a connection to the same database is
        already pooled. The registration key will be suffixed to prevent collisions. By default, the current connection is
        re-used. If that is the case, no further information (e.g. config strings) is read and only the `name` is accessed.
    private : bool, optional
        If true, skips registration of the new instance on the `DatabasePool`. Registration is performed by default.

    Returns
    -------
    PostgresInterface
        The Postgres database object

    Raises
    ------
    ValueError
        If neither a config file nor a connect string was given, or if the connect file should be used but does not exist

    References
    ----------

    .. Psyopg v3: https://www.psycopg.org/psycopg3/ This is used internally by the Postgres interface to interact with the
       database
    """
    db_pool = db.DatabasePool.get_instance()
    if name in db_pool and not refresh:
        return db_pool.get_instance(name)

    if config_file and not connect_string:
        if not os.path.exists(config_file):
            wdir = os.getcwd()
            raise ValueError(f"Failed to obtain a database connection. Tried to read the config file '{config_file}' from "
                             f"your current working directory, but the file was not found. Your working directory is {wdir}. "
                             "Please either supply the connect string directly to the connect() method, or ensure that the "
                             "config file exists.")
        with open(config_file, "r") as f:
            connect_string = f.readline().strip()
    elif not connect_string:
        raise ValueError("Failed to obtain a database connection. Please either supply the connect string directly to the "
                         "connect() method, or put a configuration file in your working directory. See the documentation of "
                         "the connect() method for more details.")

    postgres_db = PostgresInterface(connect_string, system_name=name, cache_enabled=cache_enabled)
    if not private:
        orig_name = name
        instance_idx = 2
        while name in db_pool:
            name = f"{orig_name} - {instance_idx}"
            instance_idx += 1
        db_pool.register_database(name, postgres_db)
    return postgres_db


def _parallel_query_initializer(connect_string: str, local_data: threading.local, verbose: bool = False) -> None:
    """Internal function for the `ParallelQueryExecutor` to setup worker connections.

    Parameters
    ----------
    connect_string : str
        Connection info to establish a network connection to the Postgres instance. Delegates to Psycopg
    local_data : threading.local
        Data object to store the opened connection
    verbose : bool, optional
        Whether to print logging information, by default ``False``

    References
    ----------

    .. Psyopg v3: https://www.psycopg.org/psycopg3/ This is used internally by the Postgres interface to interact with the
       database
    """
    log = logging.make_logger(verbose)
    tid = threading.get_ident()
    connection = psycopg.connect(connect_string, application_name=f"PostBOUND parallel worker ID {tid}")
    connection.autocommit = True
    local_data.connection = connection
    log(f"[worker id={tid}, ts={logging.timestamp()}] Connected")


def _parallel_query_worker(query: str | qal.SqlQuery, local_data: threading.local,
                           verbose: bool = False) -> tuple[qal.SqlQuery | str, Any]:
    """Internal function for the `ParallelQueryExecutor` to run individual queries.

    Parameters
    ----------
    query : str | qal.SqlQuery
        The query to execute. The parallel executor does not make use of caching whatsoever, so no additional parameters are
        required.
    local_data : threading.local
        Data object that contains the database connection to use. This should have been initialized by
        `_parallel_query_initializer`
    verbose : bool, optional
        Whether to print logging information, by default ``False``

    Returns
    -------
    tuple[qal.SqlQuery | str, Any]
        A tuple of the original query and the (simplified) result set. See `Database.execute_query` for an outline of the
        simplification process. This method applies the same rules. The query is also provided to distinguish the different
        result sets that arrive in parallel.
    """
    log = logging.make_logger(verbose)
    connection: psycopg.connection.Connection = local_data.connection
    connection.rollback()
    cursor = connection.cursor()

    log(f"[worker id={threading.get_ident()}, ts={logging.timestamp()}] Now executing query {query}")
    cursor.execute(str(query))
    log(f"[worker id={threading.get_ident()}, ts={logging.timestamp()}] Executed query {query}")

    result_set = cursor.fetchall()
    cursor.close()

    return query, _simplify_result_set(result_set)


class ParallelQueryExecutor:
    """The ParallelQueryExecutor provides mechanisms to conveniently execute queries in parallel.

    The parallel execution happens by maintaining a number of worker threads that execute the incoming queries.
    The number of input queries can exceed the worker pool size, potentially by a large margin. If that is the case,
    input queries will be buffered until a worker is available.

    This parallel executor has nothing to do with the Database interface and acts entirely independently and
    Postgres-specific.

    Parameters
    ----------
    connect_string : str
        Connection info to establish a network connection to the Postgres instance. Delegates to Psycopg
    n_threads : Optional[int], optional
        The maximum number of parallel workers to use. If this is not specified, uses ``os.cpu_count()`` many workers.
    verbose : bool, optional
        Whether to print logging information during the query execution, by default ``False``

    See Also
    --------
    postbound.db.db.Database
    PostgresInterface

    References
    ----------

    .. Psyopg v3: https://www.psycopg.org/psycopg3/ This is used internally by the Postgres interface to interact with the
       database
    """

    def __init__(self, connect_string: str, n_threads: Optional[int] = None, *, verbose: bool = False) -> None:
        self._n_threads = n_threads if n_threads is not None and n_threads > 0 else os.cpu_count()
        self._connect_string = connect_string
        self._verbose = verbose

        self._thread_data = threading.local()
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self._n_threads,
                                                                  initializer=_parallel_query_initializer,
                                                                  initargs=(self._connect_string, self._thread_data,))
        self._tasks: list[concurrent.futures.Future] = []
        self._results: list[Any] = []

    def queue_query(self, query: qal.SqlQuery | str) -> None:
        """Adds a new query to the queue, to be executed as soon as possible.

        Parameters
        ----------
        query : qal.SqlQuery | str
            The query to execute
        """
        future = self._thread_pool.submit(_parallel_query_worker, query, self._thread_data, self._verbose)
        self._tasks.append(future)

    def drain_queue(self, timeout: Optional[float] = None) -> None:
        """Blocks, until all queries currently queued have terminated.

        Parameters
        ----------
        timeout : Optional[float], optional
            The number of seconds to wait until the calculation is aborted. Defaults to ``None``, which indicates no timeout,
            i.e. wait forever.

        Raises
        ------
        TimeoutError or concurrent.futures.TimeoutError
            If queries took longer than the given `timeout` to execute
        """
        for future in concurrent.futures.as_completed(self._tasks, timeout=timeout):
            self._results.append(future.result())

    def result_set(self) -> dict[str | qal.SqlQuery, Any]:
        """Provides the results of all queries that have terminated already, mapping query -> result set

        Returns
        -------
        dict[str | qal.SqlQuery, Any]
            The query results. The result set is simplified according to similar rules as `Database.execute_query`
        """
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


def _timeout_query_worker(query: qal.SqlQuery | str, pg_instance: PostgresInterface,
                          query_result_sender: mp_conn.Connection) -> None:
    """Internal function to the `TimeoutQueryExecutor` to run individual queries.

    Query results are sent via the pipe, not as a return value.

    Parameters
    ----------
    query : qal.SqlQuery | str
        Query to execute
    pg_instance : PostgresInterface
        Database connection to execute the query on
    query_result_sender : mp_conn.Connection
        Pipe connection to send the query result
    """
    result = pg_instance.execute_query(query)
    query_result_sender.send(result)


class TimeoutQueryExecutor:
    """The TimeoutQueryExecutor provides a mechanism to execute queries with a timeout attached.

    If the query takes longer than the designated timeout, its execution is cancelled. The query execution itself is delegated
    to the `PostgresInterface`, so all its rules still apply. At the same time, using the timeout executor service can
    invalidate some of the state that is exposed by the database interface (see *Warnings* below). Therefore, the relevant
    variables should be refreshed once the timeout executor was used.

    In addition to calling the `execute_query` method directly, the executor also implements ``__call__`` for more convenient
    access. Both methods accept the same parameters.

    Parameters
    ----------
    postgres_instance : Optional[PostgresInterface], optional
        Database to execute the queries. If omitted, this is inferred from the `DatabasePool`.

    Warnings
    --------
    When a query gets cancelled due to the timeout being reached, the current cursor as well as database connection might be
    refreshed. Any direct references to these instances should no longer be used.
    """
    def __init__(self, postgres_instance: Optional[PostgresInterface] = None) -> None:
        self._pg_instance = (postgres_instance if postgres_instance is not None
                             else db.DatabasePool.get_instance().current_database())

    def execute_query(self, query: qal.SqlQuery | str, timeout: float) -> Any:
        """Runs a query on the database connection, cancelling if it takes longer than a specific timeout.

        Parameters
        ----------
        query : qal.SqlQuery | str
            Query to execute
        timeout : float
            Maximum query execution time in seconds.

        Returns
        -------
        Any
            The query result if it terminated timely. Rules from `PostgresInterface.execute_query` apply.

        Raises
        ------
        TimeoutError
            If the query execution was not finished after `timeout` seconds.

        See Also
        --------
        PostgresInterface.execute_query
        PostgresInterface.reset_connection
        """
        query_result_sender, query_result_receiver = mp.Pipe(False)
        query_execution_worker = mp.Process(target=_timeout_query_worker, args=(query, self._pg_instance, query_result_sender))

        query_execution_worker.start()
        query_execution_worker.join(timeout)
        timed_out = query_execution_worker.is_alive()

        if timed_out:
            query_execution_worker.terminate()
            query_execution_worker.join()
            self._pg_instance.reset_connection()
            query_result = None
        else:
            query_result = query_result_receiver.recv()

        query_execution_worker.close()
        query_result_sender.close()
        query_result_receiver.close()

        if timed_out:
            raise TimeoutError(query)
        else:
            return query_result

    def __call__(self, query: qal.SqlQuery | str, timeout: float) -> Any:
        return self.execute_query(query, timeout)


PostgresExplainJoinNodes = {"Nested Loop": physops.JoinOperators.NestedLoopJoin,
                            "Hash Join": physops.JoinOperators.HashJoin,
                            "Merge Join": physops.JoinOperators.SortMergeJoin}
"""A mapping from Postgres EXPLAIN node names to the corresponding join operators."""

PostgresExplainScanNodes = {"Seq Scan": physops.ScanOperators.SequentialScan,
                            "Index Scan": physops.ScanOperators.IndexScan,
                            "Index Only Scan": physops.ScanOperators.IndexOnlyScan,
                            "Bitmap Heap Scan": physops.ScanOperators.BitmapScan}
"""A mapping from Postgres EXPLAIN node names to the corresponding scan operators."""


class PostgresExplainNode:
    """Simplified model of a plan node as provided by Postgres' ``EXPLAIN`` output in JSON format.

    Generally speaking, a node stores all the information about the plan node that we currently care about. This is mostly
    focused on optimizer statistics, along with some additional data. Explain nodes form a hierarchichal structure with each
    node containing an arbitrary number of child nodes. Notice that this model is very loose in the sense that no constraints
    are enforced and no sanity checking is performed. For example, this means that nodes can contain more than two children
    even though this can never happen in a real ``EXPLAIN`` plan. Similarly, the correspondence between filter predicates and
    the node typse (e.g. join filter for a join node) is not checked.

    All relevant data from the explain node is exposed as attributes on the objects. Even though these are mutable, they should
    be thought of as read-only data objects.

    Parameters
    ----------
    explain_data : dict
        The JSON data of the current explain node. This is parsed and prepared as part of the ``__init__`` method.

    Attributes
    ----------
    node_type : str | None, default None
        The node type. This should never be empty or ``None``, even though it is technically allowed.
    cost : float, default NaN
        The optimizer's cost estimation for this node. This includes the cost of all child nodes as well. This should normally
        not be ``NaN``, even though it is technically allowed.
    cardinality_estimate : float, default NaN
        The optimizer's estimation of the number of tuples that will be *produced* by this operator. This should normally not
        be ``NaN``, even though it is technically allowed.
    execution_time : float, default NaN
        For ``EXPLAIN ANALYZE`` plans, this is the actual total execution time of the node in seconds. For pure ``EXPLAIN``
        plans, this is ``NaN``
    true_cardinality : float, default NaN
        For ``EXPLAIN ANALYZE`` plans, this is the average of the number of tuples that were actually produced for each loop of
        the node. For pure ``EXPLAIN`` plans, this is ``NaN``
    loops : int, default 1
        For ``EXPLAIN ANALYZE`` plans, this is the number of times the operator was invoked. The number of invocations can mean
        a number of different things: for parallel operators, this normally matches the number of parallel workers. For scans,
        this matches the number of times a new tuple was requested (e.g. for an index nested-loop join the number of loops of
        the index scan part indicates how many times the index was probed).
    relation_name : str | None, default None
        The name of the relation/table that is processed by this node. This should be defined on scan nodes, but could also
        be present on other nodes.
    relation_alias : str | None, default None
        The alias of the relation/table under which the relation was accessed in th equery plan. See `relation_name`.
    index_name : str | None, default None
        The name of the index that was probed. This should be defined on index scans and index-only scans, but could also be
        present on other nodes.
    filter_condition : str | None, default None
        A post-processing filter that is applied to all rows emitted by this operator. This is most important for scan
        operations with an attached filter predicate, but can also be present on some joins.
    index_condition : str | None, default None
        The condition that is used to locate the matching tuples in an index scan or index-only scan
    join_filter : str | None, default None
        The condition that is used to determine matching tuples in a join
    hash_condition : str | None, default None
        The condition that is used to determine matching tuples in a hash join
    recheck_condition : str | None, default None
        For lossy bitmap scans or bitmap scans based on lossy indexes, this is post-processing check for whether the produced
        tuples actually match the filter condition
    parent_relationship : str | None, default None
        Describes the role that this node plays in relation to its parent. Common values are ``"inner"`` which denotes that
        this is the inner child of a join and ``"outer"`` which denotes the opposite.
    parallel_workers : str | None, default NaN
        For parallel operators in ``EXPLAIN ANALYZE`` plans, this is the actual number of worker processes that were started.
        Notice that in total there is one additional worker. This process takes care of spawning the other workers and
        managing them, but can also take part in the input processing.
    children : list[PostgresExplainNode]
        All child / input nodes for the current node
    """
    def __init__(self, explain_data: dict) -> None:
        self.node_type = explain_data.get("Node Type", None)

        self.cost = explain_data.get("Total Cost", math.nan)
        self.cardinality_estimate = explain_data.get("Plan Rows", math.nan)
        self.execution_time = explain_data.get("Actual Total Time", math.nan) / 1000
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

    def is_scan(self) -> bool:
        """Checks, whether the current node corresponds to a scan node.

        For Bitmap index scans, which are multi-level scan operators, this is true for the heap scan part that takes care of
        actually reading the tuples according to the bitmap provided by the bitmap index scan operators.

        Returns
        -------
        bool
            Whether the node is a scan node
        """
        return self.node_type in PostgresExplainScanNodes

    def is_join(self) -> bool:
        """Checks, whether the current node corresponds to a join node.

        Returns
        -------
        bool
            Whether the node is a join node
        """
        return self.node_type in PostgresExplainJoinNodes

    def is_analyze(self) -> bool:
        """Checks, whether this ``EXPLAIN`` plan is an ``EXPLAIN ANALYZE`` plan or a pure ``EXPLAIN`` plan.

        The analyze variant does not only obtain the plan, but actually executes it. This enables the comparison of the
        optimizer's estimates to the actual values. If a plan is an ``EXPLAIN ANALYZE`` plan, some attributes of this node
        receive actual values. These include `execution_time`, `true_cardinality`, `loops` and `parallel_workers`.


        Returns
        -------
        bool
            Whether the node represents part of an ``EXPLAIN ANALYZE`` plan
        """
        return not math.isnan(self.execution_time) or not math.isnan(self.true_cardinality)

    def filter_conditions(self) -> dict[str, str]:
        """Collects all filter conditions that are defined on this node

        Returns
        -------
        dict[str, str]
            A dictionary mapping the type of filter condition (e.g. index condition or join filter) to the actual filter value.
        """
        conditions: dict[str, str] = {}
        if self.filter_condition is not None:
            conditions["Filter"] = self.filter_condition
        if self.index_condition is not None:
            conditions["Index Cond"] = self.index_condition
        if self.join_filter is not None:
            conditions["Join Filter"] = self.join_filter
        if self.hash_condition is not None:
            conditions["Hash Cond"] = self.hash_condition
        if self.recheck_condition is not None:
            conditions["Recheck Cond"] = self.recheck_condition
        return conditions

    def inner_outer_children(self) -> Sequence[PostgresExplainNode]:
        """Provides the children of this node in a sequence of inner, outer if applicable.

        For all nodes where this structure is not meaningful (e.g. intermediate nodes that operate on a single relation or
        scan nodes), the child nodes are returned as-is (e.g. as a list of a single child or an empty list).

        Returns
        -------
        Sequence[PostgresExplainNode]
            The children of the current node in a unified format
        """
        if len(self.children) < 2:
            return self.children
        assert len(self.children) == 2

        first_child, second_child = self.children
        inner_child = first_child if first_child.parent_relationship == "Inner" else second_child
        outer_child = first_child if second_child == inner_child else second_child
        return (inner_child, outer_child)

    def parse_table(self) -> Optional[base.TableReference]:
        """Provides the table that is processed by this node.

        Returns
        -------
        Optional[base.TableReference]
            The table being scanned. For non-scan nodes, or nodes where no table can be inferred, ``None`` will be returned.
        """
        if not self.relation_name:
            return None
        alias = self.relation_alias if self.relation_alias is not None else ""
        return base.TableReference(self.relation_name, alias)

    def as_query_execution_plan(self) -> db.QueryExecutionPlan:
        """Transforms the postgres-specific plan to a standardized `QueryExecutionPlan` instance.

        Notice that this transformation is lossy since not all information from the Postgres plan can be represented in query
        execution plan instances. Furthermore, this transformation can be problematic for complicated queries that use
        special Postgres features. Most importantly, for queries involving subqueries, special node types and parent
        relationships can be contained in the plan, that cannot be represented by other parts of PostBOUND. If this method
        and the resulting query execution plans should be used on complex workloads, it is advisable to check the plans twice
        before continuing.

        Returns
        -------
        db.QueryExecutionPlan
            The equivalent query execution plan for this node

        Raises
        ------
        ValueError
            If the node contains more than two children.
        """
        if self.children and len(self.children) > 2:
            raise ValueError("Cannot transform parent node > 2 children")
        elif self.children and len(self.children) == 1:
            child_nodes = [self.children[0].as_query_execution_plan()]
            inner_child = None
        elif self.children:
            first_child, second_child = self.children
            first_plan, second_plan = first_child.as_query_execution_plan(), second_child.as_query_execution_plan()
            child_nodes = first_plan, second_plan
            inner_child = (first_plan if first_child.parent_relationship == "Inner" and self.node_type != "Hash Join"
                           else second_plan)
        else:
            child_nodes = None
            inner_child = None

        table = self.parse_table()
        is_scan = self.is_scan()
        is_join = self.is_join()
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

    def inspect(self, *, _indentation: int = 0) -> str:
        """Provides a pretty string representation of the ``EXPLAIN`` sub-plan that can be printed.

        Parameters
        ----------
        _indentation : int, optional
            This parameter is internal to the method and ensures that the correct indentation is used for the child nodes
            of the plan. When inspecting the root node, this value is set to its default value of `0`.

        Returns
        -------
        str
            A string representation of the ``EXPLAIN`` sub-plan.
        """
        padding = " " * _indentation
        prefix = f"{padding}<- " if padding else ""
        own_inspection = [prefix + str(self)]
        child_inspections = [child.inspect(_indentation=_indentation+2) for child in self.inner_outer_children()]
        return "\n".join(own_inspection + child_inspections)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        analyze_content = (f" (actual time={self.execution_time}s rows={self.true_cardinality} loops={self.loops})"
                           if self.is_analyze() else "")
        explain_content = f"(cost={self.cost} rows={self.cardinality_estimate})"
        conditions = " ".join(f"{condition}: {value}" for condition, value in self.filter_conditions().items())
        conditions = " " + conditions if conditions else ""
        scan_info = f" on {self.parse_table().identifier()}" if self.is_scan() else ""
        return self.node_type + scan_info + explain_content + analyze_content + conditions


class PostgresExplainPlan:
    """Models an entire ``EXPLAIN`` plan produced by Postgres

    In contrast to `PostgresExplainNode`, this includes additional parameters (planning time and execution time) for the entire
    plan, rather than just portions of it.

    This class supports all methods that are specified on the general `db.QueryExecutionPlan` and returns the correct data for
    its actual plan.

    Parameters
    ----------
    explain_data : dict
        The JSON data of the entire explain plan. This is parsed and prepared as part of the ``__init__`` method.


    Attributes
    ----------
    planning_time : float
        The time in seconds that the optimizer spent to build the plan
    execution_time : float
        The time in seconds the query execution engine needed to calculate the result set of the query. This does not account
        for network time to transmit the result set.
    query_plan : PostgresExplainNode
        The actual plan
    """
    def __init__(self, explain_data: dict) -> None:
        explain_data = explain_data[0] if isinstance(explain_data, list) else explain_data
        self.planning_time: float = explain_data.get("Planning Time", math.nan) / 1000
        self.execution_time: float = explain_data.get("Execution Time", math.nan) / 1000
        self.query_plan = PostgresExplainNode(explain_data["Plan"])
        self._normalized_plan = self.query_plan.as_query_execution_plan()

    def is_analyze(self) -> bool:
        """Checks, whether this ``EXPLAIN`` plan is an ``EXPLAIN ANALYZE`` plan or a pure ``EXPLAIN`` plan.

        The analyze variant does not only obtain the plan, but actually executes it. This enables the comparison of the
        optimizer's estimates to the actual values. If a plan is an ``EXPLAIN ANALYZE`` plan, some attributes of this node
        receive actual values. These include `execution_time`, `true_cardinality`, `loops` and `parallel_workers`.


        Returns
        -------
        bool
            Whether the plan represents an ``EXPLAIN ANALYZE`` plan
        """
        return self.query_plan.is_analyze()

    def as_query_execution_plan(self) -> db.QueryExecutionPlan:
        """Provides the actual explain plan as a normalized query execution plan instance

        For notes on pecularities of this method, take a look at the *See Also* section

        Returns
        -------
        db.QueryExecutionPlan
            The query execution plan

        See Also
        --------
        PostgresExplainNode.as_query_execution_plan
        """
        return self._normalized_plan

    def inspect(self) -> str:
        """Provides a pretty string representation of the actual plan.

        Returns
        -------
        str
            A string representation of the plan

        See Also
        --------
        PostgresExplainNode.inspect
        """
        return self.query_plan.inspect()

    def __getattribute__(self, name: str) -> Any:
        # All methods that are not defined on the Postgres plan delegate to the default DB plan
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            normalized_plan = object.__getattribute__(self, "_normalized_plan")
            return normalized_plan.__getattribute__(name)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_analyze():
            prefix = f"EXPLAIN ANALYZE (plan time={self.planning_time}, exec time={self.execution_time})"
        else:
            prefix = "EXPLAIN"

        return f"{prefix} root: {self.query_plan}"
