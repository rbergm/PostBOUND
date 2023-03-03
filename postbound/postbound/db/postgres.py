"""Contains the Postgres implementation of the Database interface."""

from __future__ import annotations

import collections
import concurrent
import concurrent.futures
import os
import textwrap
import threading
from typing import Any

import psycopg
import psycopg.rows

from postbound.db import db
from postbound.qal import qal, base, transform
from postbound.util import logging, misc as utils

HintBlock = collections.namedtuple("HintBlock", ["preparatory_statements", "hints", "query"])


class PostgresInterface(db.Database):
    """Database implementation for PostgreSQL backends."""

    def __init__(self, connect_string: str, name: str = "postgres", *, cache_enabled: bool = True) -> None:
        super().__init__(name, cache_enabled=cache_enabled)
        self._connect_string = connect_string
        self._connection = psycopg.connect(connect_string, application_name="PostBOUND",
                                           row_factory=psycopg.rows.tuple_row)
        self._connection.autocommit = True
        self._cursor = self._connection.cursor()

        self._db_schema = PostgresSchemaInterface(self)
        self._db_stats = PostgresStatisticsInterface(self)

    def schema(self) -> db.DatabaseSchema:
        return self._db_schema

    def statistics(self, emulated: bool | None = None, cache_enabled: bool | None = None) -> db.DatabaseStatistics:
        if emulated is not None:
            self._db_stats.emulated = emulated
        if cache_enabled is not None:
            self._db_stats.cache_enabled = cache_enabled
        return self._db_stats

    def execute_query(self, query: qal.SqlQuery | str, *, cache_enabled: bool | None = None) -> Any:
        cache_enabled = cache_enabled or (cache_enabled is None and self._cache_enabled)

        if isinstance(query, qal.SqlQuery):
            if query.hints and query.hints.preparatory_statements:
                self._cursor.execute(query.hints.preparatory_statements)
            query = transform.drop_hints(query, preparatory_statements_only=True)
            query = str(query)

        if cache_enabled and query in self._query_cache:
            query_result = self._query_cache[query]
        else:
            self._cursor.execute(query)
            query_result = self._cursor.fetchall()
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

    def cardinality_estimate(self, query: qal.SqlQuery | str) -> int:
        query = str(query)
        if not query.upper().startswith("EXPLAIN (FORMAT JSON)"):
            query = "EXPLAIN (FORMAT JSON) " + query
        self._cursor.execute(query)
        query_plan = self._cursor.fetchone()[0]
        estimate = query_plan[0]["Plan"]["Plan Rows"]
        return estimate

    def postgres_version(self) -> utils.Version:
        """Provides the version of the Postgres instance currently connected to."""
        self._cursor.execute("SELECT VERSION();")
        pg_ver = self._cursor.fetchone()[0]
        # version looks like "PostgreSQL 14.6 on x86_64-pc-linux-gnu, compiled by gcc (...)
        return utils.Version(pg_ver.split(" ")[1])

    def reset_connection(self) -> None:
        self._cursor.close()
        self._connection.rollback()
        self._cursor = self._connection.cursor()

    def cursor(self) -> db.Cursor:
        return self._cursor

    def close(self) -> None:
        self._cursor.close()
        self._connection.close()


class PostgresSchemaInterface(db.DatabaseSchema):
    """Schema-specific parts of the general Postgres interface."""

    def __int__(self, postgres_db: PostgresInterface) -> None:
        super().__init__(postgres_db)

    def lookup_column(self, column: base.ColumnReference,
                      candidate_tables: list[base.TableReference]) -> base.TableReference:
        for table in candidate_tables:
            table_columns = self._fetch_columns(table)
            if column.name in table_columns:
                return table
        candidate_tables = [table.full_name for table in candidate_tables]
        raise ValueError(f"Column '{column.name}' not found in candidate tables {candidate_tables}")

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
                        WHERE table_name = {tab} AND column_name = {col}""")
        datatype_query = query_template.format(tab=column.table.full_name, col=column.name)
        self._db.cursor().execute(datatype_query)
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
                                        WHERE idx.indrelid = '{table.full_name}'::regclass""")
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

    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> int:
        count_query = f"SELECT reltuples FROM pg_class WHERE oid = '{table.full_name}'::regclass"
        self._db.cursor().execute(count_query)
        count = self._db.cursor().fetchone()[0]
        return count

    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> int:
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

    def _retrieve_min_max_values_from_stats(self, column: base.ColumnReference) -> tuple:
        # Postgres does not keep track of min/max values, so we need to determine them manually
        if not self.enable_emulation_fallback:
            raise db.UnsupportedDatabaseFeatureError(self._db, "min/max value statistics")
        return self._calculate_min_max_values(column, cache_enabled=True)

    def _retrieve_most_common_values_from_stats(self, column: base.ColumnReference, k: int) -> list:
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

    postgres_db = PostgresInterface(connect_string, name=name, cache_enabled=cache_enabled)
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


def _parallel_query_worker(query: str, local_data: threading.local, verbose: bool = False) -> Any:
    """Internal function for the `ParallelQueryExecutor` to run individual queries."""
    log = logging.make_logger(verbose)
    connection: psycopg.connection.Connection = local_data.connection
    connection.rollback()
    cursor = connection.cursor()

    log(f"[worker id={threading.get_ident()}, ts={logging.timestamp()}] Now executing query {query}")
    cursor.execute(query)
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
        query = str(query)
        future = self._thread_pool.submit(_parallel_query_worker, query, self._thread_data, self._verbose)
        self._tasks.append(future)

    def drain_queue(self, timeout: float = None) -> None:
        """Blocks, until all queries currently queued have terminated."""
        for future in concurrent.futures.as_completed(self._tasks, timeout=timeout):
            self._results.append(future.result())

    def result_set(self) -> dict[str, Any]:
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
