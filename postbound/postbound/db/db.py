"""This module provides PostBOUNDs basic interaction with databases.

More specifically, this includes an interface to interact with databases, schema information and statistics as well
as a utility to easily obtain database connections.

Take a look at the central `Database` class for more details. All concrete database systems need to implement this
interface.
"""
from __future__ import annotations

import abc
import atexit
import json
import math
import os
import textwrap
import typing
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Optional

from postbound.qal import base, qal
from postbound.optimizer import jointree
from postbound.optimizer.physops import operators as physops
from postbound.optimizer.planmeta import hints as planmeta
from postbound.util import collections as collection_utils, dicts as dict_utils, misc


class Cursor(typing.Protocol):
    """Interface for database cursors that adhere to the Python Database API specification.

    This is not a complete representation and only focuses on the parts of the specification that are important for
    PostBOUND right now. In the future, additional methods might get added.

    This type is only intended to denote the expected return type of certain methods, the cursors themselves are
    supplied by the respective database integrations. There should be no need to implement one manually and all cursors
    should be compatible with this interface by default (since they are DB API 2.0 cursor objects).

    See PEP 249 for details (https://peps.python.org/pep-0249/)
    """

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self, operation: str, parameters: Optional[dict | Sequence] = None) -> Optional[Cursor]:
        raise NotImplementedError

    @abc.abstractmethod
    def fetchone(self) -> Optional[tuple]:
        raise NotImplementedError

    @abc.abstractmethod
    def fetchall(self) -> Optional[list[tuple]]:
        raise NotImplementedError


class Connection(typing.Protocol):
    """Interface for database connections that adhere to the Python Database API specification.

    This is not a complete representation and only focuses on the parts of the specification that are important for
    PostBOUND right now. In the future, additional methods might get added.

    This type is only intended to denote the expected return type of certain methods, the connections themselves are
    supplied by the respective database integrations. There should be no need to implement one manually and all
    connections should be compatible with this interface by default (since they are DB API 2.0 connection objects).

    See PEP 249 for details (https://peps.python.org/pep-0249/)
    """

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def cursor(self) -> Cursor:
        raise NotImplementedError


class HintService(abc.ABC):
    """Provides tools that enable queries to be executed on a specific database system, based on fixed query plans.

    Hints are PostBOUNDs way to enforce that decisions made in the optimization pipeline are respected by the native
    query optimizer once the query is executed in an actual database system.
    """

    @abc.abstractmethod
    def generate_hints(self, query: qal.SqlQuery,
                       join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan] = None,
                       physical_operators: Optional[physops.PhysicalOperatorAssignment] = None,
                       plan_parameters: Optional[planmeta.PlanParameterization] = None) -> qal.SqlQuery:
        """Generates the appropriate hints to enforce the optimized execution of the given query.

        All hints are placed in a `Hint` clause on the query. In addition, if the query needs to be transformed in some
        way, this also happens here.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def format_query(self, query: qal.SqlQuery) -> str:
        """Transforms the query into a database-specific string, mostly to incorporate deviations from standard SQL."""
        raise NotImplementedError

    @abc.abstractmethod
    def supports_hint(self, hint: physops.PhysicalOperator | planmeta.HintType) -> bool:
        """Checks, whether the database system is capable of using the specified hint or operator."""
        raise NotImplementedError


class Database(abc.ABC):
    """A `Database` is PostBOUND's logical abstraction of physical database management systems.

    It provides high-level access to internal functionality provided by such systems. More specifically, each
    `Database` instance supports the following functionality:

    - executing arbitrary SQL queries
    - retrieving schema information, most importantly about primary keys and foreign keys
    - accessing statistical information, such as most common values or the number of rows in a table
    - query formatting and generation of query hints to enforce optimizer decisions (join orders, operators, etc.)

    Notice, that all this information is by design read-only and functionality to write queries is intentionally not
    implemented (although one could issue `INSERT`/`UPDATE`/`DELETE` queries via the query execution functionality).

    This restriction to read-only information enables the caching of query results to provide them without running a
    query over and over again. This is achieved by storing the results of past queries in a special JSON file, which is
    read upon creation of the `Database` instance. If this behavior is not desired, it can simply be turned off via the
    `cache_enabled` property.

    Each database management system will need to implement this basic interface to enable PostBOUND to access the
    necessary information.
    """

    def __init__(self, system_name: str, *, cache_enabled: bool = True) -> None:
        self.system_name = system_name

        self._cache_enabled = cache_enabled
        self._query_cache = {}
        if self._cache_enabled:
            self.__inflate_query_cache()
        atexit.register(self.close)

    @abc.abstractmethod
    def schema(self) -> DatabaseSchema:
        """Provides access to the underlying schema information of the database."""
        raise NotImplementedError

    @abc.abstractmethod
    def statistics(self, emulated: bool | None = None, cache_enabled: Optional[bool] = None) -> DatabaseStatistics:
        """Provides access to different tables and columns of the database."""
        raise NotImplementedError

    @abc.abstractmethod
    def hinting(self) -> HintService:
        """Provides access to the hint generation facilities for the current database system."""
        raise NotImplementedError

    @abc.abstractmethod
    def optimizer(self) -> OptimizerInterface:
        """Provides access to optimizer-related functionality of the database system.

        This method can raise an `UnsupportedDatabaseFeatureError` if it does not provide external access to the
        optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute_query(self, query: qal.SqlQuery | str, *, cache_enabled: Optional[bool] = None) -> Any:
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
    def database_name(self) -> str:
        """Provides the name of the (physical) database that the database interface is connected to."""
        raise NotImplementedError

    def database_system_name(self) -> str:
        """Provides the name of the database management system that this interface is connected to."""
        return self.system_name

    @abc.abstractmethod
    def database_system_version(self) -> misc.Version:
        """Returns the release version of the database management system that this interface is connected to."""
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the current database connection as well as the system settings."""
        raise NotImplementedError

    @abc.abstractmethod
    def reset_connection(self) -> None:
        """Obtains a new network connection for the database. Useful for debugging purposes."""
        raise NotImplementedError

    def reset_cache(self) -> None:
        """Removes all results from the query cache. Useful for debugging purposes."""
        self._query_cache = {}

    @abc.abstractmethod
    def cursor(self) -> Cursor:
        """Provides a cursor to execute queries and iterate over result sets manually.

        The specific type of cursor being returned depends on the concrete database implementation. However, the cursor
        object should always implement the interface described in the Python DB API specification 2.0 (PEP 249).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Shuts down all currently open connections to the database."""
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
        atexit.register(self.__store_query_cache, query_cache_name)

    def __store_query_cache(self, query_cache_name: str):
        """Stores the query cache into a JSON file."""
        with open(query_cache_name, "w") as cache_file:
            json.dump(self._query_cache, cache_file)

    def __query_cache_name(self):
        """Provides a normalized file name for the query cache."""
        identifier = "_".join([self.database_system_name(),
                               self.database_system_version().formatted(prefix="v", separator="_"),
                               self.database_name()])
        return f".query_cache_{identifier}.json"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.database_name()} @ {self.database_system_name()} ({self.database_system_version()})"


class DatabaseSchema(abc.ABC):
    """A database schema contains structural information about the database.

    For PostBOUND, this information is mostly limited to index structures and primary keys of different columns.
    """

    def __init__(self, db: Database):
        self._db = db

    @abc.abstractmethod
    def lookup_column(self, column: base.ColumnReference | str,
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

    @abc.abstractmethod
    def datatype(self, column: base.ColumnReference) -> str:
        """Retrieves the (physical) data type of the given `column`.

        The provided type can be a standardized SQL-type, but it can be a type specific to the concrete database
        system just as well.

        If the `column` is not bound to any table, an `UnboundColumnError` will be raised.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Database schema of {self._db}"


class DatabaseStatistics(abc.ABC):
    """Database statistics provide aggregated information about specific tables and columns of the database.

    This information can be retrieved in two forms: the internal statistics catalogs of the DBMS can be queried
    directly, or the information can be calculated based on the current contents of the live database.

    Since statistics are mostly database specific, the internal catalogs can provide information in different formats
    or granularity, or they do not provide the required information at all. Therefore, calculating on the live data
    is safer, albeit slower (caching of database queries can somewhat mitigate this effect). To indicate how the
    statistics should be obtained, the `emulated` attribute exists. If set to `True`, all statistical information will
    be retrieved based on the live data. Conversely, `emulated = False` indicates that the internal metadata catalogs
    should be queried (and it is up to the client to do something useful with that information).

    If the fallback to emulated statistics is not desired, the `enable_emulation_fallback` attribute can be set to
    `False`. In this case, each time the database should provide a statistic it does not support, an
    `UnsupportedDatabaseFeatureError` will be raised. However, this setting is overwritten by the `emulated` property.

    Since the live computation of emulated statistics can be costly, the statistics interface has its own
    `cache_enabled` attribute. It can be set to `None` to use the default caching behaviour of the database system.
    However, if this attribute is set to `True` or `False` directly, caching will be used accordingly for all
    compute-intensive statistics operations. The default is to use the database setting.
    """

    def __init__(self, db: Database):
        self.emulated: Optional[bool] = True
        self.enable_emulation_fallback = True
        self.cache_enabled: Optional[bool] = None
        self._db = db

    def total_rows(self, table: base.TableReference, *, emulated: Optional[bool] = None,
                   cache_enabled: Optional[bool] = None) -> Optional[int]:
        """Provides (an estimate of) the total number of rows in a table.

        If the `table` is virtual, a `VirtualTableError` will be raised.
        """
        if table.virtual:
            raise base.VirtualTableError(table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_total_rows(table,
                                              cache_enabled=self._determine_caching_behaviour(cache_enabled))
        else:
            return self._retrieve_total_rows_from_stats(table)

    def distinct_values(self, column: base.ColumnReference, *, emulated: Optional[bool] = None,
                        cache_enabled: Optional[bool] = None) -> Optional[int]:
        """Provides (an estimate of) the total number of different column values of a specific column.

        If the `column` is not bound to any table, an `UnboundColumnError` will be raised. Likewise, virtual tables
        will raise a `VirtualTableError`.
        """
        if not column.table:
            raise base.UnboundColumnError(column)
        elif column.table.virtual:
            raise base.VirtualTableError(column.table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_distinct_values(column,
                                                   cache_enabled=self._determine_caching_behaviour(cache_enabled))
        else:
            return self._retrieve_distinct_values_from_stats(column)

    def min_max(self, column: base.ColumnReference, *, emulated: Optional[bool] = None,
                cache_enabled: Optional[bool] = None) -> Optional[tuple[Any, Any]]:
        """Provides (an estimate of) the minimum and maximum values in a column.

        If the `column` is not bound to any table, an `UnboundColumnError` will be raised. Likewise, virtual tables
        will raise a `VirtualTableError`.
        """
        if not column.table:
            raise base.UnboundColumnError(column)
        elif column.table.virtual:
            raise base.VirtualTableError(column.table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_min_max_values(column,
                                                  cache_enabled=self._determine_caching_behaviour(cache_enabled))
        else:
            return self._retrieve_min_max_values_from_stats(column)

    def most_common_values(self, column: base.ColumnReference, *, k: int = 10, emulated: Optional[bool] = None,
                           cache_enabled: Optional[bool] = None) -> Sequence[tuple[Any, int]]:
        """Provides (an estimate of) the total number of occurrences of the `k` most frequent values of a column.

         By default, `k = 10`. In `emulated` mode, the result will be an ordered sequence of `(value, frequency)`
         pairs, such that the first value has the highest frequency.

         If the `column` is not bound to any table, an `UnboundColumnError` will be raised. Likewise, virtual tables
        will raise a `VirtualTableError`.
         """
        if not column.table:
            raise base.UnboundColumnError(column)
        elif column.table.virtual:
            raise base.VirtualTableError(column.table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_most_common_values(column, k,
                                                      cache_enabled=self._determine_caching_behaviour(cache_enabled))
        else:
            return self._retrieve_most_common_values_from_stats(column, k)

    def _calculate_total_rows(self, table: base.TableReference, *,
                              cache_enabled: Optional[bool] = None) -> Optional[int]:
        """Retrieves the total number of rows of a table by issuing a COUNT(*) query against the live database.

        The table is assumed to be non-virtual.
        """
        query_template = "SELECT COUNT(*) FROM {tab}".format(tab=table.full_name)
        return self._db.execute_query(query_template, cache_enabled=self._determine_caching_behaviour(cache_enabled))

    def _calculate_distinct_values(self, column: base.ColumnReference, *,
                                   cache_enabled: Optional[bool] = None) -> Optional[int]:
        """Retrieves the number of distinct column values by issuing a COUNT(*) / GROUP BY query over that column
        against the live database.

        The column is assumed to be bound to a (non-virtual) table.
        """
        query_template = "SELECT COUNT(DISTINCT {col}) FROM {tab}".format(col=column.name, tab=column.table.full_name)
        return self._db.execute_query(query_template, cache_enabled=self._determine_caching_behaviour(cache_enabled))

    def _calculate_min_max_values(self, column: base.ColumnReference, *,
                                  cache_enabled: Optional[bool] = None) -> Optional[tuple[Any, Any]]:
        """Retrieves the minimum/maximum values in a column by issuing an aggregation query for that column against the
        live database.

        The column is assumed to be bound to a (non-virtual) table.
        """
        query_template = "SELECT MIN({col}), MAX({col}) FROM {tab}".format(col=column.name, tab=column.table.full_name)
        return self._db.execute_query(query_template, cache_enabled=self._determine_caching_behaviour(cache_enabled))

    def _calculate_most_common_values(self, column: base.ColumnReference, k: int, *,
                                      cache_enabled: Optional[bool] = None) -> Sequence[tuple[Any, int]]:
        """Retrieves the `k` most frequent values of a column along with their frequencies by issuing a COUNT(*) /
        GROUP BY query over that column against the live database.

        The column is assumed to be bound to a (non-virtual) table.
        """
        query_template = textwrap.dedent("""
            SELECT {col}, COUNT(*) AS n
            FROM {tab}
            GROUP BY {col}
            ORDER BY n DESC, {col}
            LIMIT {k}""".format(col=column.name, tab=column.table.full_name, k=k))
        return self._db.execute_query(query_template, cache_enabled=self._determine_caching_behaviour(cache_enabled))

    @abc.abstractmethod
    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> Optional[int]:
        """Queries the DBMS-internal metadata for the number of rows in a table.

        The table is assumed to be non-virtual.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> Optional[int]:
        """Queries the DBMS-internal metadata for the number of distinct values of the column.

        The column is assumed to be bound to a (non-virtual) table.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_min_max_values_from_stats(self, column: base.ColumnReference) -> Optional[tuple[Any, Any]]:
        """Queries the DBMS-internal metadata for the minimum / maximum value in a column.

        The column is assumed to be bound to a (non-virtual) table.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_most_common_values_from_stats(self, column: base.ColumnReference,
                                                k: int) -> Sequence[tuple[Any, int]]:
        """Queries the DBMS-internal metadata for the `k` most common values of the `column`.

        The column is assumed to be bound to a (non-virtual) table.
        """
        raise NotImplementedError

    def _determine_caching_behaviour(self, local_cache_enabled: Optional[bool]) -> bool:
        return self.cache_enabled if local_cache_enabled is None else local_cache_enabled

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Database statistics of {self._db}"


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


class UnsupportedDatabaseFeatureError(RuntimeError):
    """Indicates that some requested feature is not supported by the database.

    For example, PostgreSQL (at least up to version 15) does not capture minimum or maximum column values in its
    system statistics. Therefore, forcing the DBS to retrieve such information from its metadata could result in this
    error.
    """

    def __init__(self, database: Database, feature: str) -> None:
        super().__init__(f"Database {database.system_name} does not support feature {feature}")
        self.database = database
        self.feature = feature


class DatabaseServerError(RuntimeError):
    """Indicates an error caused by the database server occured while executing a database operation.

    The error was not due to a mistake in the user input (such as an SQL syntax error or access privilege violation),
    but an implementation issue instead (such as out of memory during query execution).
    """

    def __init__(self, message: str = "", context: Optional[object] = None) -> None:
        super().__init__(message)
        self.ctx = context


class DatabaseUserError(RuntimeError):
    """Indicates that a database operation failed due to an error on the user's end.

    The error could be due to an SQL syntax error, access privilege violation, etc.
    """

    def __init__(self, message: str = "", context: Optional[object] = None) -> None:
        super().__init__(message)
        self.ctx = context


class QueryExecutionPlan:
    """Abstraction for the smallest common denominator of query plan features.

    This can be used when information from a query plan is required to implement other functionality. A plan simply
    consists of a number of nodes which contain information specific to the actual node type.
    """

    def __init__(self, node_type: str, is_join: bool, is_scan: bool, *, table: Optional[base.TableReference] = None,
                 children: Optional[Iterable[QueryExecutionPlan, QueryExecutionPlan]] = None,
                 parallel_workers: float = math.nan, cost: float = math.nan,
                 estimated_cardinality: float = math.nan, true_cardinality: float = math.nan,
                 execution_time: float = math.nan, physical_operator: Optional[physops.PhysicalOperator] = None,
                 inner_child: Optional[QueryExecutionPlan] = None) -> None:
        self.node_type = node_type
        self.physical_operator = physical_operator
        self.is_join = is_join
        self.is_scan = is_scan
        if is_scan and not isinstance(physical_operator, physops.ScanOperators):
            warnings.warn("Supplied operator is scan operator but node is created as non-scan node")
        if is_join and not isinstance(physical_operator, physops.JoinOperators):
            warnings.warn("Supplied operator is join operator but node is created as non-join node")

        self.parallel_workers = parallel_workers
        self.children: Sequence[QueryExecutionPlan] = tuple(children) if children else ()
        self.inner_child = inner_child
        if self.inner_child and len(self.children) == 2:
            first_child, second_child = self.children
            self.outer_child = first_child if self.inner_child == second_child else second_child
        else:
            self.outer_child = None

        self.table = table

        self.cost = cost
        self.estimated_cardinality = estimated_cardinality
        self.true_cardinality = true_cardinality
        self.execution_time = execution_time

    def is_analyze(self) -> bool:
        """Checks, whether the current node also contains information about actual execution properties."""
        return not math.isnan(self.true_cardinality) or not math.isnan(self.execution_time)

    def tables(self) -> frozenset[base.TableReference]:
        """Provides all tables that are referenced in and below the current plan node."""
        own_table = [self.table] if self.table else []
        return frozenset(own_table + collection_utils.flatten(child.tables() for child in self.children))

    def is_base_join(self) -> bool:
        return self.is_join and all(child.is_scan_branch() for child in self.children)

    def is_bushy_join(self) -> bool:
        return self.is_join and all(child.is_join_branch() for child in self.children)

    def is_scan_branch(self) -> bool:
        return self.is_scan or (len(self.children) == 1 and self.children[0].is_scan_branch())

    def is_join_branch(self) -> bool:
        return self.is_join or (len(self.children) == 1 and self.children[0].is_join_branch())

    def fetch_base_table(self) -> Optional[base.TableReference]:
        if not self.is_scan_branch():
            raise ValueError("No unique base table for non-scan branches!")
        if self.table:
            return self.table
        return self.children[0].fetch_base_table()

    def total_processed_rows(self) -> float:
        if not self.is_analyze():
            return math.nan
        return self.true_cardinality + sum(child.total_processed_rows() for child in self.children)

    def scan_nodes(self) -> frozenset[QueryExecutionPlan]:
        own_node = [self] if self.is_scan else []
        child_scans = collection_utils.flatten(child.scan_nodes() for child in self.children)
        return frozenset(own_node + child_scans)

    def join_nodes(self) -> frozenset[QueryExecutionPlan]:
        own_node = [self] if self.is_join else []
        child_joins = collection_utils.flatten(child.join_nodes() for child in self.children)
        return frozenset(own_node + child_joins)

    def plan_depth(self) -> int:
        return 1 + max((child.plan_depth() for child in self.children), default=0)

    def simplify(self) -> Optional[QueryExecutionPlan]:
        if not self.is_join and not self.is_scan:
            if len(self.children) != 1:
                return None
            return self.children[0].simplify()

        simplified_children = [child.simplify() for child in self.children] if not self.is_scan else []
        simplified_inner = self.inner_child.simplify() if not self.is_scan and self.inner_child else None
        return QueryExecutionPlan(self.node_type, self.is_join, self.is_scan,
                                  table=self.table,
                                  children=simplified_children,
                                  parallel_workers=self.parallel_workers,
                                  cost=self.cost,
                                  estimated_cardinality=self.estimated_cardinality,
                                  true_cardinality=self.true_cardinality,
                                  execution_time=self.execution_time,
                                  physical_operator=self.physical_operator,
                                  inner_child=simplified_inner)

    def inspect(self, *, skip_intermediates: bool = False, _current_indentation: int = 0):
        padding = " " * _current_indentation
        prefix = f"{padding}<- " if padding else ""
        own_inspection = [prefix + str(self)]
        child_inspections = [
            child.inspect(skip_intermediates=skip_intermediates, _current_indentation=_current_indentation + 2)
            for child in self.children]

        if not skip_intermediates or self.is_join or self.is_scan or not _current_indentation:
            return "\n".join(own_inspection + child_inspections)
        else:
            return "\n".join(child_inspections)

    def __hash__(self) -> int:
        return hash((self.node_type, self.table, self.true_cardinality, tuple(self.children)))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.node_type == other.node_type and self.table == other.table
                and self.children == other.children and
                (self.true_cardinality == other.true_cardinality
                 or (math.isnan(self.true_cardinality) and math.isnan(other.true_cardinality)))
                )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_analyze():
            analyze_str = f" (execution time={self.execution_time} true cardinality={self.true_cardinality})"
        else:
            analyze_str = ""

        table_str = f" :: {self.table}" if self.table else ""
        plan_str = f" (cost={self.cost} estimated cardinality={self.estimated_cardinality})"
        return "".join((self.node_type, table_str, plan_str, analyze_str))


class OptimizerInterface(abc.ABC):
    """This interface provides high-level access to internal optimizer-related data for the database system."""

    @abc.abstractmethod
    def query_plan(self, query: qal.SqlQuery | str) -> QueryExecutionPlan:
        """Obtains the query execution plan for the given query."""
        raise NotImplementedError

    @abc.abstractmethod
    def analyze_plan(self, query: qal.SqlQuery | str) -> QueryExecutionPlan:
        """Executes the given query and provides the query execution plan supplemented with runtime information."""
        raise NotImplementedError

    @abc.abstractmethod
    def cardinality_estimate(self, query: qal.SqlQuery | str) -> int:
        """Queries the DBMS query optimizer for its cardinality estimate, instead of executing the query."""
        raise NotImplementedError

    @abc.abstractmethod
    def cost_estimate(self, query: qal.SqlQuery | str) -> float:
        """Queries the DBMS query optimizer for the estimated cost of executing the query."""
        raise NotImplementedError
