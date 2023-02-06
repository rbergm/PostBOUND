from __future__ import annotations

import abc
import atexit
import json
import os
import warnings
from typing import Any

from postbound.qal import base, qal
from postbound.util import dicts as dict_utils


class Database(abc.ABC):
    """A `Database` is PostBOUND's logical abstraction of physical database management systems.

    It provides high-level access to internal functionality provided by such systems. More specifically, each
    `Database` instance supports the following functionality:

    - executing arbitrary SQL queries
    - retrieving schema information, most importantly about primary keys and foreign keys
    - accessing statistical information, such as most common values or the number of rows in a table

    Notice, that all this information is by design read-only and functionality to write queries is intentionally not
    implemented (although one could issue `INSERT`/`UPDATE`/`DELETE` queries via the query execution functionality).

    This restriction to read-only information enables the caching of query results to provide them without running a
    query over and over again. This is achieved by storing the results of past queries in a special JSON file, which is
    read upon creation of the `Database` instance. If this behavior is not desired, it can simply be turned off via the
    `cache_enabled` property.
    """

    def __init__(self, name: str, *, cache_enabled: bool = True) -> None:
        self.name = name

        self._cache_enabled = cache_enabled
        self._query_cache = {}
        if self._cache_enabled:
            self.__inflate_query_cache()

    @abc.abstractmethod
    def schema(self) -> DatabaseSchema:
        """Provides access to the underlying schema information of the database."""
        raise NotImplementedError

    @abc.abstractmethod
    def statistics(self, emulated: bool | None = None) -> DatabaseStatistics:
        """Provides access to different tables and columns of the database."""
        raise NotImplementedError

    @abc.abstractmethod
    def execute_query(self, query: qal.SqlQuery | str, *, cache_enabled: bool | None = None) -> Any:
        """Executes the given query and returns the associated result set.

        The precise behaviour of this method depends on whether caching is enabled or not. If it is, the query will
        only be executed against the live database system, if it is not in the cache. Otherwise, the result will simply
        be retrieved. Caching can be enabled/disabled for just this one query via the `cache_enabled` switch. If this
        is not specified, caching depends on the `cache_enabled` property.

        This method tries to simplify the return value of the query for more convenient use. More specifically, if the
        query returns just a single row, this row is returned directly. Furthermore, if the query returns just a single
        column, that column is placed directly in to the encompassing list. Both simplifications will also be combined,
        such that a result set of a single row of a single value will be returned as that single value directly. In all
        other cases, the result will be a list consisting of the different result tuples.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cardinality_estimate(self, query: qal.SqlQuery | str) -> int:
        """Queries the DBMS query optimizer for its cardinality estimate, instead of executing the query."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset_connection(self) -> None:
        """Obtains a new network connection for the database. Useful for debugging purposes."""
        raise NotImplementedError

    def reset_cache(self) -> None:
        """Removes all results from the query cache. Useful for debugging purposes."""
        self._query_cache = {}

    @abc.abstractmethod
    def cursor(self) -> Any:
        raise NotImplementedError

    def _get_cache_enabled(self) -> bool:
        """Getter for the `cache_enabled` property."""
        return self._cache_enabled

    def _set_cache_enabled(self, enabled: bool) -> None:
        """Setter for the `cache_enabled` property. Inflates the query cache if necessary."""
        if enabled and not self._query_cache:
            self.__inflate_query_cache()
        self._cache_enabled = enabled

    cache_enabled = property(_get_cache_enabled, _set_cache_enabled)
    """Controls, whether the results of executed queries should be cached to prevent future re-execution."""

    def __inflate_query_cache(self) -> None:
        """Tries to read the query cache for this database."""
        query_cache_name = self.__query_cache_name()
        if os.path.isfile(query_cache_name):
            with open(query_cache_name, "r") as cache_file:
                try:
                    self._query_cache = json.load(cache_file)
                except json.JSONDecodeError as e:
                    warnings.warn("Could not read query cache: " + str(e))
                    self._query_cache = {}
        else:
            warnings.warn(f"Could not read query cache: File {query_cache_name} does not exist")
            self._query_cache = {}
        atexit.register(self.__store_query_cache)

    def __store_query_cache(self):
        """Stores the query cache into a JSON file."""
        query_cache_name = self.__query_cache_name()
        with open(query_cache_name, "w") as cache_file:
            json.dump(self._query_cache, cache_file)

    def __query_cache_name(self):
        """Provides a normalized file name for the query cache."""
        return f".query_cache_{self.name}.json"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.name


class DatabaseSchema(abc.ABC):
    """A database schema contains structural information about the database.

    For PostBOUND, this information is mostly limited to index structures and primary keys of different columns.
    """

    def __init__(self, db: Database):
        self._db = db

    @abc.abstractmethod
    def lookup_column(self, column: base.ColumnReference,
                      candidate_tables: list[base.TableReference]) -> base.TableReference:
        """Searches the `candidate_tables` for the given `column`.

        The first table that contains a column with a similar name will be returned. If no candidate table contains the
        column, a `ValueError` is raised."""
        raise NotImplementedError

    @abc.abstractmethod
    def is_primary_key(self, column: base.ColumnReference) -> bool:
        """Checks, whether the given `column` is the primary key for its associated table.

        If the `column` is not bound to any table, an `UnboundColumnError` will be raised.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def has_secondary_index(self, column: base.ColumnReference) -> bool:
        """Checks, whether the given `column` has a secondary index on its associated table.

        Primary key indexes will fail the test. If the `column` is not bound to any table, an `UnboundColumnError` will
        be raised.
        """
        raise NotImplementedError

    def has_index(self, column: base.ColumnReference) -> bool:
        """Checks, whether the given `column` has any index on its associated table.

        If the `column` is not bound to any table, an `UnboundColumnError` will be raised.
        """
        return self.is_primary_key(column) or self.has_secondary_index(column)


class DatabaseStatistics(abc.ABC):
    """Database statistics provide aggregated information about specific tables and columns of the database.

    This information can be retrieved in two forms: the internal statistics catalogs of the DBMS can be queried
    directly, or the information can be calculated based on the current contents of the live database.

    Since statistics are mostly database specific, the internal catalogs can provide information in different formats
    or granularity, or they do not provide the required information at all. Therefore, calculating on the live data
    is safer, albeit slower. To indicate how the statistics should be obtained, the `emulated` attribute exists.
    If set to `True`, all statistical information will be retrieved based on the live data. Conversely,
    `emulated = False` indicates that the internal metadata catalogs should be queried (and it is up to the client to
    do something useful with that information).
    """

    def __init__(self, db: Database):
        self.emulated = True
        self._db = db

    def total_rows(self, table: base.TableReference, *, emulated: bool | None = None,
                   cache_enabled: bool | None = None) -> int:
        """Provides (an estimate of) the total number of rows in a table."""
        if (emulated is not None and not emulated) or self.emulated:
            return self._calculate_total_rows(table, cache_enabled=cache_enabled)
        else:
            return self._retrieve_total_rows_from_stats(table)

    def distinct_values(self, column: base.ColumnReference, *, emulated: bool | None = None,
                        cache_enabled: bool | None = None) -> int:
        """Provides (an estimate of) the total number of different column values of a specific column.

        If the `column` is not bound to any table, an `UnboundColumnError` will be raised.
        """
        if not column.table:
            raise base.UnboundColumnError(column)
        if (emulated is not None and not emulated) or self.emulated:
            return self._calculate_distinct_values(column, cache_enabled=cache_enabled)
        else:
            return self._retrieve_distinct_values_from_stats(column)

    def most_common_values(self, column: base.ColumnReference, *, k: int = 10, emulated: bool | None = None,
                           cache_enabled: bool | None = None) -> list:
        """Provides (an estimate of) the total number of occurrences of the `k` most frequent values of a column.

         By default, `k = 10`. In `emulated` mode, the result will be an ordered sequence of `(value, frequency)`
         pairs, such that the first value has the highest frequency.

         If the `column` is not bound to any table, an `UnboundColumnError` will be raised.
         """
        if not column.table:
            raise base.UnboundColumnError(column)
        if (emulated is not None and not emulated) or self.emulated:
            return self._calculate_most_common_values(column, k, cache_enabled=cache_enabled)
        else:
            return self._retrieve_most_common_values_from_stats(column, k)

    def _calculate_total_rows(self, table: base.TableReference, *, cache_enabled: bool | None = None) -> int:
        query_template = "SELECT COUNT(*) FROM {tab}"
        count_query = query_template.format(tab=table.full_name)
        return self._db.execute_query(count_query, cache_enabled=cache_enabled)

    def _calculate_distinct_values(self, column: base.ColumnReference, *, cache_enabled: bool | None = None) -> int:
        query_template = "SELECT COUNT(DISTINCT {col}) FROM {tab}"
        count_query = query_template.format(col=column.name, tab=column.table.full_name)
        return self._db.execute_query(count_query, cache_enabled=cache_enabled)

    def _calculate_most_common_values(self, column: base.ColumnReference, k: int, *,
                                      cache_enabled: bool | None = None) -> list:
        query_template = "SELECT {col}, COUNT(*) AS n FROM {tab} GROUP BY {col} ORDER BY n DESC, {col} LIMIT {k}"
        count_query = query_template.format(col=column.name, tab=column.table.full_name, k=k)
        return self._db.execute_query(count_query, cache_enabled=cache_enabled)

    @abc.abstractmethod
    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> int:
        """Queries the DBMS-internal metadata for the number of rows in a table."""
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> int:
        """Queries the DBMS-internal metadata for the number of distinct values of the column.

        If the `column` is not bound to any table, an `UnboundColumnError` will be raised.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_most_common_values_from_stats(self, column: base.ColumnReference, k: int) -> list:
        """Queries the DBMS-internal metadata for the `k` most common values of the `column`.

        If the `column` is not bound to any table, an `UnboundColumnError` will be raised.
        """
        raise NotImplementedError


_DB_POOL: DatabasePool | None = None


class DatabasePool:
    """The `DatabasePool` allows different parts of the code base to easily obtain access to a database.

    This is achieved by maintaining one global pool of database connections which is shared by the entire system.
    New database instances can be registered and retrieved via unique keys. As long as there is just a single database
    instance, it can be accessed via the `current_database` method.
    """

    @staticmethod
    def get_instance():
        """Provides access to the database pool, creating a new pool instance if necessary."""
        global _DB_POOL
        if _DB_POOL is None:
            _DB_POOL = DatabasePool()
        return _DB_POOL

    def __init__(self):
        self._pool = {}

    def current_database(self) -> Database:
        """Provides the database that is currently stored in the pool, provided there is just one."""
        return dict_utils.value(self._pool)

    def register_database(self, key: str, db: Database) -> None:
        """Stores a new database to be accessed via `key`."""
        self._pool[key] = db

    def retrieve_database(self, key: str) -> Database:
        """Retrieves the database stored under `key`."""
        return self._pool[key]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"DatabasePool {self._pool}"
