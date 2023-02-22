
import atexit
import concurrent
import concurrent.futures
import json
import os
import re
import textwrap
import threading
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any

import psycopg2

from transform import util


# see https://regex101.com/r/HdKzQg/1
TableReferencePattern = re.compile(r"(?P<full_name>\S+) (AS (?P<alias>\S+))?")


class TableRef:
    @staticmethod
    def virtual(alias: str) -> "TableRef":
        return TableRef(None, alias, True)

    @staticmethod
    def parse(raw_table_ref: str) -> "TableRef":
        match = TableReferencePattern.match(raw_table_ref)
        if not match:
            raise ValueError(f"Could not parse table reference for '{raw_table_ref}'")
        full_name, alias = match.group("full_name", "alias")
        alias = "" if not alias else alias
        return TableRef(full_name, alias)

    def __init__(self, full_name: str, alias: str = "", virtual: bool = False):
        self.full_name = full_name
        self.alias = alias
        self.is_virtual = virtual

    def has_attr(self, attr_name) -> bool:
        if not isinstance(attr_name, str):
            warnings.warn("Treating non-string attribute as false: " + str(attr_name))
            return False

        table_qualifier = self.alias + "."
        return attr_name.startswith(table_qualifier)

    def bind_attribute(self, attr_name) -> str:
        return f"{self.alias}.{attr_name}"

    def qualifier(self) -> str:
        return self.alias if self.alias else self.full_name

    def to_mosp(self):
        if self.is_virtual:
            raise ValueError("Can not convert virtual tables")
        return {"value": self.full_name, "name": self.alias}

    def __lt__(self, __other: object) -> bool:
        if not isinstance(__other, type(self)):
            return NotImplemented(f"'<' not implemented between '{type(self)}' and '{type(__other)}'")
        other_tab: TableRef = __other
        return ((other_tab.full_name == self.full_name and self.alias < other_tab.alias)
                or self.full_name < other_tab.full_name)

    def __hash__(self) -> int:
        return hash((self.full_name, self.alias))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.full_name == other.full_name and self.alias == other.alias

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_virtual:
            return f"{self.alias} (virtual)"
        return f"{self.full_name} AS {self.alias}"


UnboundTable = TableRef("", "", virtual=True)


@dataclass
class AttributeRef:
    @staticmethod
    def parse(attribute_data: str, *, alias_map: Dict[str, TableRef]) -> "AttributeRef":
        try:
            table, attribute = attribute_data.split(".")
        except ValueError:
            table, attribute = "", attribute_data
        parsed_table = alias_map[table] if table else UnboundTable
        return AttributeRef(parsed_table, attribute)

    table: TableRef
    attribute: str

    def __hash__(self) -> int:
        return hash((self.table, self.attribute))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AttributeRef):
            return False
        return self.table == other.table and self.attribute == other.attribute

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.table.alias}.{self.attribute}"


_DTypeArrayConverters = {
    "integer": "int[]",
    "text": "text[]",
    "character varying": "text[]"
}

_dbschema_instance: "DBSchema" = None


class DBSchema:
    """Provides access to a database and its schema.

    All instances should be accessed via the `get_instance` method.
    """

    @staticmethod
    def get_instance(psycopg_connect: str = "", *, postgres_config_file: str = ".psycopg_connection",
                     renew: bool = False):
        """Provides access to a unified DB schema instance.

        This is especially usefull to create a database connection on the fly without having to carry
        configuration info everywhere. "Unified" in this context means that all clients access the very same schema
        instance and connection to the same database.

        Instances can be created via one of two ways: directly specifying the `psycopg_connect` string will use it
        to open a connection to the database. If this setting is omitted, the `postgres_config_file` will be read.
        This file is expected to contain a single-line string suitable for a psycopg connection as well.

        The `renew` flag can be set to force the creation of a new DBSchema instance. This is mostly intended to
        connect to multiple databases at the same time.

        Notice, that usage of a DB schema should happen sequentially since the underlying data structures are not
        thread-safe.
        """
        global _dbschema_instance
        if _dbschema_instance and not renew:
            return _dbschema_instance

        if not psycopg_connect:
            if not os.path.exists(postgres_config_file):
                warnings.warn("No .psycopg_connection file found, trying empty connect string as last resort. This "
                              "is likely not intentional.")
                psycopg_connect = ""
            else:
                with open(postgres_config_file, "r") as conn_file:
                    psycopg_connect = conn_file.readline().strip()
        conn = psycopg2.connect(psycopg_connect, application_name="PostBOUND")
        conn.autocommit = True
        new_instance = DBSchema(conn.cursor(), connection=conn, connect_string=psycopg_connect)

        if _dbschema_instance and renew:
            # retain the query cache so we don't loose previously cached queries from other databases
            new_instance.query_cache = _dbschema_instance.query_cache

        _dbschema_instance = new_instance

        # return the new instance instead of _dbschema so existing db schema instances are not overwritten (b/c they
        # received the new instance as well before)
        return new_instance

    def __init__(self, cursor: "psycopg2.cursor", *, connection: "psycopg2.connection" = None,
                 connect_string: str = ""):
        self.cursor = cursor
        self.connection = connection
        self.connect_string = connect_string

        self.index_map = {}
        self.estimates_cache = {}
        self.mcvs_cache = {}
        self.mcvs_online_cache = {}
        self.tuple_count_cache = {}
        # don't forget reset_caches when adding new caches!!

        # load query cache
        # if we can reload previous results, we need to be a bit careful with the file handling:
        # The query file must be opened for writing in every case in order to store the cache when the script
        # terminates. However, if we also must read previous content. This cannot take place in r+ mode, since this
        # would append the new data while keeping the previous data intact, leaving two JSON objects in the file
        # (which is illegal). Therefore, the file is first opened for reading, the cache is inflated and the file is
        # closed again.
        if os.path.isfile(".dbschema_query_cache.json"):
            with open(".dbschema_query_cache.json", "r") as query_cache_file:
                try:
                    self.query_cache = json.load(query_cache_file)
                except json.JSONDecodeError:
                    self.query_cache = {}
        else:
            self.query_cache = {}
        atexit.register(self._save_query_cache)

    def count_tuples(self, table: TableRef, *, cache_enabled=True) -> int:
        """Retrieves the current number of tuples in the table by executing a COUNT(*) query."""
        count_query = "SELECT COUNT(*) FROM {}".format(table.full_name)
        if cache_enabled and count_query in self.query_cache:
            return self.query_cache[count_query]

        if table.is_virtual:
            raise ValueError("Cannot count tuples of virtual table")

        self.cursor.execute(count_query)
        count = self.cursor.fetchone()[0]

        if cache_enabled:
            self.query_cache[count_query] = count
        return count

    def count_distinct_values(self, attribute: AttributeRef, *, cache_enabled=True) -> int:
        """Retrieves the current number of distinct values for a specific attribute, executing a COUNT() query."""
        count_query = f"SELECT DISTINCT COUNT({attribute.attribute} FROM {attribute.table.full_name}"
        if cache_enabled and count_query in self.query_cache:
            return self.query_cache[count_query]

        self.cursor.execute(count_query)
        count = self.cursor.fetchone()[0]

        if cache_enabled:
            self.query_cache[count_query] = count
        return count

    def lookup_attribute(self, attribute_name: str, candidate_tables: List[TableRef]):
        for table in [tab for tab in candidate_tables if not tab.is_virtual]:
            columns = self._fetch_columns(table.full_name)
            if attribute_name in columns:
                return table
        raise KeyError(f"Attribute not found: {attribute_name} in candidates {candidate_tables}")

    def execute_query(self, query: str, *, cache_enabled=True, analyze_mode: bool = False, explain_mode: bool = False,
                      **kwargs):
        if cache_enabled and query in self.query_cache:
            return self.query_cache[query]

        if explain_mode and not query.lower().startswith("explain (format json)"):
            query = f"EXPLAIN (FORMAT JSON) {query}"
        elif analyze_mode and not query.lower().startswith("explain (analyze, format json)"):
            query = f"EXPLAIN (ANALYZE, FORMAT JSON) {query}"

        pg_setttings = self._prepare_pg_settings(**kwargs)
        if pg_setttings:
            self.cursor.execute(pg_setttings)
        self.cursor.execute(query)
        result = util.simplify(self.cursor.fetchall())

        if cache_enabled:
            self.query_cache[query] = result
        return result

    def is_primary_key(self, attribute: AttributeRef) -> bool:
        """Checks if the given attribute is a primary key on its table."""
        if attribute.table not in self.index_map:
            self._inflate_index_map_for_table(attribute.table)
        index_map_for_table = self.index_map[attribute.table]
        return index_map_for_table.get(attribute.attribute, False)

    def has_secondary_idx_on(self, attribute: AttributeRef) -> bool:
        """Checks, whether the schema has a secondary index (e.g. Foreign key) specified on the given attribute."""
        if attribute.table not in self.index_map:
            self._inflate_index_map_for_table(attribute.table)
        index_map_for_table = self.index_map[attribute.table]

        # The index map contains an entry for each attribute that actually has an index. The value is True, if the
        # attribute (which is known to be indexed), is even the Primary Key
        # Our method should return False in two cases: 1) the attribute is not indexed at all; and 2) the attribute
        # actually is the Primary key. Therefore, by assuming it is the PK in case of absence, we get the correct
        # value.
        return not index_map_for_table.get(attribute.attribute, True)

    def pg_estimate(self, query: str, *, cache_enabled: bool = True):
        """Retrieves the number of result tuples estimated by the PG query optimizer for the given query."""
        if cache_enabled and query in self.estimates_cache:
            return self.estimates_cache[query]
        if not query.lower().startswith("explain (format json)"):
            explain_query = "explain (format json) " + query
        else:
            explain_query = query
        self.cursor.execute(explain_query)
        explain_result = self.cursor.fetchone()[0]
        estimate = explain_result[0]["Plan"]["Plan Rows"]

        if cache_enabled:
            self.estimates_cache[query] = estimate

        return estimate

    def load_most_common_values(self, attribute: AttributeRef, *, k: int = None, cache_enabled=True) -> list:
        """Retrieves the MCV-list from the pg_stats view.

        The list is returned as an ordered sequence of (value, count) pairs (starting with the most common value).

        Optionally, the list can be cut after `k` values. E.g. setting `k = 1` just returns the most common value. If
        less than `k` values are present, the entire list is returned.
        """
        if cache_enabled and attribute in self.mcvs_cache:
            mcvs = self.mcvs_cache[attribute]
            return mcvs[:k] if k else mcvs

        mcvs = self._load_mcvs(attribute)

        if cache_enabled:
            self.mcvs_cache[attribute] = mcvs
        return mcvs[:k] if k else mcvs

    def calculate_most_common_values(self, attribute: AttributeRef, *, k: int = 10,
                                     cache_enabled: bool = True) -> list:
        """
        In contrast to `load_most_common_values`, this function does not query the `pg_stats` view, but calculates the
        common values live from the actual data. This also means that `k` always has to be set to a value.
        Other than that, both functions work exactly the same.

        This process will probably take way longer than querying the stats view, but is guaranteed to always be
        exact and to always return a value.

        If any number of attributes occur with equal frequency, their order is defined by the order of the values
        themselves.
        """

        query_template = textwrap.dedent(f"""
                                         SELECT {attribute}, COUNT(*)
                                         FROM {attribute.table}
                                         GROUP BY {attribute}
                                         ORDER BY count DESC, {attribute}
                                         LIMIT {k}""")

        if cache_enabled and query_template in self.query_cache:
            mcvs = self.query_cache[query_template]
            return mcvs

        self.cursor.execute(query_template)
        mcvs = self.cursor.fetchall()

        if cache_enabled:
            self.query_cache[query_template] = mcvs
        return mcvs

    def load_tuple_count(self, table: TableRef, *, cache_enabled: bool = True) -> int:
        """Retrieves the total number of tuples from Postgres statistics, rather than executing a count query."""
        if table.is_virtual:
            raise ValueError("Cannot count tuples of virtual table")

        if cache_enabled and table in self.tuple_count_cache:
            return self.tuple_count_cache[table]

        count_query = f"SELECT reltuples FROM pg_class WHERE oid = '{table.full_name}'::regclass"
        self.cursor.execute(count_query)
        stats_count = self.cursor.fetchone()[0]

        if cache_enabled:
            self.tuple_count_cache[table] = stats_count

        return stats_count

    def load_distinct_value_count(self, attribute: AttributeRef, *, cache_enabled: bool = True) -> int:
        return NotImplemented

    def pg_version(self) -> util.Version:
        self.cursor.execute("SELECT VERSION();")
        pg_ver = self.cursor.fetchone()[0]
        return util.Version(pg_ver.split(" ")[1])

    def reset_caches(self):
        """Removes all cached content."""
        self.index_map = {}
        self.estimates_cache = {}
        self.mcvs_cache = {}
        self.mcvs_online_cache = {}
        self.tuple_count_cache = {}

    def reset_cursor(self):
        """Obtains a new database cursor while resetting the current connection. Requires a connection to be set."""
        if self.connection:
            self.cursor.close()
            self.connection.reset()
            self.cursor = self.connection.cursor()
        else:
            warnings.warn("Cannot reset cursor - schema instance has no connection specified.")

    def _prepare_pg_settings(self, **kwargs) -> str:
        settings = []
        for setting, value in kwargs.items():
            formatted = f"SET {setting} = "
            formatted += f"'{value}'" if isinstance(value, str) else f"{value}"
            settings.append(formatted)
        return "; ".join(settings)

    def _fetch_columns(self, table_name):
        base_query = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        self.cursor.execute(base_query, (table_name,))
        result_set = self.cursor.fetchall()
        return [col[0] for col in result_set]

    def _inflate_index_map_for_table(self, table: TableRef):
        # query adapted from https://wiki.postgresql.org/wiki/Retrieve_primary_key_columns
        index_query = textwrap.dedent(f"""
                                      SELECT a.attname, i.indisprimary
                                      FROM pg_index i
                                      JOIN pg_attribute a
                                      ON i.indrelid = a.attrelid
                                        AND a.attnum = any(i.indkey)
                                      WHERE i.indrelid = '{table.full_name}'::regclass
                                      """)
        self.cursor.execute(index_query)
        index_map = dict(self.cursor.fetchall())
        self.index_map[table] = index_map

    def _load_mcvs(self, attribute: AttributeRef) -> list:
        # Postgres stores the Most common values in a column of type anyarray (since in this column, many MCVS from
        # many different tables and data types are present). However, this type is not very convenient to work on.
        # Therefore, we first need to convert the anyarray to an array of the actual attribute type.

        # determine the attributes data type to figure out how it should be converted
        self.cursor.execute("SELECT data_type FROM information_schema.columns "
                            "WHERE table_name = %s AND column_name = %s",
                            (attribute.table.full_name, attribute.attribute))
        attribute_dtype = self.cursor.fetchone()[0]
        attribute_converter = _DTypeArrayConverters[attribute_dtype]

        # now, load the most frequent values
        self.cursor.execute(f"SELECT UNNEST(most_common_vals::text::{attribute_converter}), UNNEST(most_common_freqs) "
                            "FROM pg_stats "
                            "WHERE tablename = %s AND attname = %s",
                            (attribute.table.full_name, attribute.attribute))
        return self.cursor.fetchall()

    def _save_query_cache(self):
        with open(".dbschema_query_cache.json", "w") as query_cache_file:
            json.dump(self.query_cache, query_cache_file)

    def __repr__(self) -> str:
        return f"DBSchema (conn = {self.connection})"

    def __str__(self) -> str:
        return f"DBSchema (conn = {self.connection})"


def _parallel_query_initializer(connect_string: str, local_data: threading.local, verbose: bool = False) -> None:
    id = threading.get_ident()
    connection = psycopg2.connect(connect_string, application_name=f"PostBOUND parallel worker ID {id}")
    connection.autocommit = True
    local_data.connection = connection


def _parallel_query_worker(query: str, local_data: threading.local, verbose: bool = False) -> Any:
    log = util.make_logger(verbose)
    connection: psycopg2.connection = local_data.connection
    connection.rollback()
    cursor = connection.cursor()

    log(f"[worker id={threading.get_ident()}, ts={util.timestamp()}] Now executing query {query}")
    cursor.execute(query)
    log(f"[worker id={threading.get_ident()}, ts={util.timestamp()}] Executed query {query}")

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
    """
    def __init__(self, connect_string: str, n_threads: int = None, *, verbose: bool = False) -> None:
        self._n_threads = n_threads if n_threads is not None and n_threads > 0 else os.cpu_count()
        self._connect_string = connect_string
        self._verbose = verbose

        self._thread_data = threading.local()
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self._n_threads,
                                                                  initializer=_parallel_query_initializer,
                                                                  initargs=(self._connect_string, self._thread_data,))
        self._tasks: List[concurrent.futures.Future] = []
        self._results = []

    def queue_query(self, query: str) -> None:
        """Adds a new query to the queue, to be executed as soon as possible."""
        future = self._thread_pool.submit(_parallel_query_worker, query, self._thread_data, self._verbose)
        self._tasks.append(future)

    def drain_queue(self, timeout: float = None) -> None:
        """Blocks, until all queries currently queued have terminated."""
        for future in concurrent.futures.as_completed(self._tasks, timeout=timeout):
            self._results.append(future.result())

    def result_set(self) -> Dict[str, Any]:
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
