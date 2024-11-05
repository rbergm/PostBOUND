"""This module provides PostBOUNDs basic interaction with databases.

More specifically, this includes

- an interface to interact with databases (the `Database` interface)
- an interface to retrieve schema information (the `DatabaseSchema` interface)
- an interface to obtain different table-level and column-level statistics (the `DatabaseStatistics` interface)
- an interface to modify queries such that optimization decisions are respected during the actual query execution (the
  `HintService` interface)
- a simple model to represent query plans of the database systems (the `QueryExecutionPlan` class)
- an interface to access information of the native optimizer of the database system (the `OptimizerInterface` class)
- a utility to easily obtain database connections (the `DatabasePool` singleton class).

Take a look at the central `Database` class for more details. All concrete database systems need to implement this
interface.
"""
from __future__ import annotations

import abc
import atexit
import collections
import json
import math
import numbers
import os
import textwrap
import typing
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, Optional

from postbound.qal import base, parser, qal
from postbound.optimizer import jointree, physops, planparams
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


@typing.runtime_checkable
class PrewarmingSupport(typing.Protocol):
    """Some databases might support adding specific tables to their shared buffer.

    If so, they should implement this protocol to allow other parts of the framework to exploit this feature.
    """

    @abc.abstractmethod
    def prewarm_tables(self, tables: Optional[base.TableReference | Iterable[base.TableReference]] = None,
                       *more_tables: base.TableReference, exclude_table_pages: bool = False,
                       include_primary_index: bool = True, include_secondary_indexes: bool = True) -> None:
        """Prepares the database buffer pool with tuples from specific tables.

        Parameters
        ----------
        tables : Optional[base.TableReference  |  Iterable[base.TableReference]], optional
            The tables that should be placed into the buffer pool
        *more_tables : base.TableReference
            More tables that should be placed into the buffer pool, enabling a more convenient usage of this method.
            See examples for details on the usage.
        exclude_table_pages : bool, optional
            Whether the table data (i.e. pages containing the actual tuples) should *not* be prewarmed. This is off by default,
            meaning that prewarming is applied to the data pages. This can be toggled on to only prewarm index pages (see
            `include_primary_index` and `include_secondary_index`).
        include_primary_index : bool, optional
            Whether the pages of the primary key index should also be prewarmed. Enabled by default.
        include_secondary_indexes : bool, optional
            Whether the pages for secondary indexes should also be prewarmed. Enabled by default.

        Notes
        -----
        If the database should prewarm more table pages than can be contained in the shared buffer, the actual contents of the
        pool are not specified. All prewarming tasks might happen sequentially, in which case the first prewarmed relations
        will typically be evicted and only the last relations (tables or indexes) are retained in the shared buffer. The
        precise order in which the prewarming tasks are executed is not specified and depends on the actual relations.

        Examples
        --------
        >>> database.prewarm_tables([table1, table2])
        >>> database.prewarm_tables(table1, table2)
        """
        raise NotImplementedError


class QueryCacheWarning(UserWarning):
    """Warning to indicate that the query result cache was not found."""
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class Database(abc.ABC):
    """A `Database` is PostBOUND's logical abstraction of physical database management systems.

    It provides high-level access to internal functionality provided by such systems. More specifically, each
    `Database` instance supports the following functionality:

    - executing arbitrary SQL queries
    - retrieving schema information, most importantly about primary keys and foreign keys
    - accessing statistical information, such as most common values or the number of rows in a table
    - query formatting and generation of query hints to enforce optimizer decisions (join orders, operators, etc.)
    - introspection of the query optimizer to retrieve query execution plans, cost estimates, etc.

    Notice, that all this information is by design read-only and functionality to write queries is intentionally not
    implemented (although one could issue `INSERT`/`UPDATE`/`DELETE` queries via the query execution functionality).

    This restriction to read-only information enables the caching of query results to provide them without running a
    query over and over again. This is achieved by storing the results of past queries in a special JSON file, which is
    read upon creation of the `Database` instance. If this behavior is not desired, it can simply be turned off
    globally via the `cache_enabled` property, or on a per-method-call basis by setting the corresponding parameter. If
    no such parameter is available, the specific method does not make use of the caching mechanic.

    Each database management system will need to implement this basic interface to enable PostBOUND to access the
    necessary information.

    Parameters
    ----------
    system_name : str
        The name of the database system for which the connection is established. This is only really important to
        distinguish different instances of the interface in a convenient manner.
    cache_enabled : bool, optional
        Whether complex queries that are executed against the database system should be cached. This is especially
        usefull to emulate certain statistics that are not maintained by the specific database system (see
        `DatabaseStatistics` for details). If this is ``False``, the query cache will not be loaded as well.
        Defaults to ``True``.

    Notes
    -----
    When the `__init__` method is called, the connection to the specific database system has to be established already,
    i.e. calling any of the public methods should provide a valid result. This is particularly important, because this
    method takes care of the cache initialization. This initialization in turn relies on identifying the correct
    cache file, which in turn depends on the system name, system version and database name of the connection.
    """

    def __init__(self, system_name: str, *, cache_enabled: bool = True) -> None:
        self.system_name = system_name

        self._cache_enabled = cache_enabled
        self._query_cache: dict[str, list[tuple]] = {}
        if self._cache_enabled:
            self._inflate_query_cache()
        atexit.register(self.close)

    @abc.abstractmethod
    def schema(self) -> DatabaseSchema:
        """Provides access to the underlying schema information of the database.

        Returns
        -------
        DatabaseSchema
            An object implementing the schema interface for the actual database system. This should normally be
            completely stateless.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def statistics(self) -> DatabaseStatistics:
        """Provides access to the current statistics of the database.

        Implementing generalized statistics for a framework that supports multiple different physical database systems
        is much more complicated than it might seem at first. Therefore, different modes for the statistics
        provisioning exist. These modes can be changed by setting the properties of the interface. See the
        documentation of `DatabaseStatistics` for more details.

        Repeated calls to this method are guaranteed to provide the same object. Therefore, changes to the statistics
        interface configuration are guaranteed to be persisted accross multiple accesses to the statistics system.

        Returns
        -------
        DatabaseStatistics
            The statistics interface. Repeated calls to this method are guaranteed to provide the same object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def hinting(self) -> HintService:
        """Provides access to the hint generation facilities for the current database system.

        Returns
        -------
        HintService
            The hinting service. This should normally be completely stateless.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def optimizer(self) -> OptimizerInterface:
        """Provides access to optimizer-related functionality of the database system.

        Returns
        -------
        OptimizerInterface
            The optimizer interface. This should normally be completely stateless.

        Raises
        ------
        UnsupportedDatabaseFeatureError
            If the database system does not provide any sort of external access to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute_query(self, query: qal.SqlQuery | str, *, cache_enabled: Optional[bool] = None, raw: bool = False) -> Any:
        """Executes the given query and returns the associated result set.

        Parameters
        ----------
        query : qal.SqlQuery | str
            The query to execute. If it contains a `Hint` with `preparatory_statements`, these will be executed
            beforehand. Notice that such statements are never subject to caching.
        cache_enabled : Optional[bool], optional
            Controls the caching behavior for just this one query. The default value of ``None`` indicates that the
            "global" configuration of the database system should be used. Setting this parameter to a boolean value
            forces or deactivates caching for the specific query for the specific execution no matter what the "global"
            configuration is.
        raw : bool, optional
            Whether the result set should be returned as-is. By default, the result set is simplified. Raw mode skips this
            step.

        Returns
        -------
        Any
            Result set of the input query. This is a list of equal-length tuples in the most general case. Each
            component of the tuple corresponds to a specific column of the result set and each tuple corresponds to a
            row in the result set. However, many queries do not provide a 2-dimensional result set (e.g. ``COUNT(*)``
            queries). In such cases, the nested structure of the result set makes it quite cumbersome to use.
            Therefore, this method tries to simplify the return value of the query for more convenient use (if `raw` mode is
            disabled). More specifically, if the query returns just a single row, this row is returned directly as a tuple.
            Furthermore, if the query returns just a single column, the values of that column are returned directly in
            a list. Both simplifications will also be combined, such that a result set of a single row of a single
            value will be returned as that single value directly. In all other cases, the result will be a list
            consisting of the different result tuples.

        Notes
        -----
        This method is mainly intended to execute read-only SQL queries. In fact, the only types of SQL queries that
        can be modelled by the query abstraction layer are precisely such read-only queries. However, if one really
        needs to execute mutating queries, they can be issued as plain text. Just remember that this behavior is
        heavily discouraged!

        The precise behavior of this method depends on whether caching is enabled or not. If it is, the query will
        only be executed against the live database system, if it is not in the cache. Otherwise, the result will simply
        be retrieved. Caching can be enabled/disabled for just this one query via the `cache_enabled` switch. If this
        is not specified, caching depends on the `cache_enabled` property.

        If caching should be used for this method, but is disabled at a database-level, the current cache will still
        be read and persisted. This ensures that all cached queries are properly saved and none of the previous cache
        content is lost.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def database_name(self) -> str:
        """Provides the name of the (physical) database that the database interface is connected to.

        Returns
        -------
        str
            The database name, e.g. ``"imdb"`` or ``"tpc-h"``
        """
        raise NotImplementedError

    def database_system_name(self) -> str:
        """Provides the name of the database management system that this interface is connected to.

        Returns
        -------
        str
            The database system name, e.g. ``"PostgreSQL"``
        """
        return self.system_name

    @abc.abstractmethod
    def database_system_version(self) -> misc.Version:
        """Returns the release version of the database management system that this interface is connected to.

        Returns
        -------
        misc.Version
            The version
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the current database connection as well as its system settings.

        This description is intended to transparently document which customizations have been applied, thereby giving
        an idea of how the default query execution might have been affected. It can be JSON-serialized and will be
        included by most of the output of the utilities in the `runner` module of the `experiments` package.

        Returns
        -------
        dict
            The actual description
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset_connection(self) -> None:
        """Obtains a new network connection for the database. Useful for debugging purposes or in case of crashes.

        Notice that resetting the connection can have unintended side-effects if other methods rely on the cursor
        object. After resetting, the former cursor object will probably no longer be valid. Therefore, this method
        should be used with caution.

        See Also
        --------
        Database.cursor
        """
        raise NotImplementedError

    def reset_cache(self) -> None:
        """Removes all results from the query cache. Useful for debugging purposes."""
        self._query_cache = {}

    @abc.abstractmethod
    def cursor(self) -> Cursor:
        """Provides a cursor to execute queries and iterate over result sets manually.

        Returns
        -------
        Cursor
            A cursor compatible with the Python DB API specification 2.0 (PEP 249). The specific cursor type depends on
            the concrete database implementation however.

        References
        ----------

        .. Python DB API specification 2.0 (PEP 249): https://peps.python.org/pep-0249/
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        """Shuts down all currently open connections to the database."""
        raise NotImplementedError

    def _get_cache_enabled(self) -> bool:
        """Getter for the `cache_enabled` property.

        Returns
        -------
        bool
            Whether caching is currently enabled
        """
        return self._cache_enabled

    def _set_cache_enabled(self, enabled: bool) -> None:
        """Setter for the `cache_enabled` property. Inflates the query cache if necessary.

        If the cache should be enabled now, but no cached data exists, the cache will be inflated from disk.

        Parameters
        ----------
        enabled : bool
            Whether caching should be enabled
        """
        if enabled and not self._query_cache:
            self._inflate_query_cache()
        self._cache_enabled = enabled

    cache_enabled = property(_get_cache_enabled, _set_cache_enabled)
    """Controls, whether the results of executed queries should be cached to prevent future re-execution.

    If caching should be enabled later on and no cached data exists, the cache will be inflated from disk.
    """

    def _inflate_query_cache(self) -> None:
        """Tries to read the query cache for this database.

        This reads a JSON file that contains all cached queries and their result sets. It should not be edited
        manually.
        """
        if self._query_cache:
            return
        query_cache_name = self._query_cache_name()
        if os.path.isfile(query_cache_name):
            with open(query_cache_name, "r") as cache_file:
                try:
                    self._query_cache = json.load(cache_file)
                except json.JSONDecodeError as e:
                    warnings.warn("Could not read query cache: " + str(e), category=QueryCacheWarning)
                    self._query_cache = {}
        else:
            warnings.warn(f"Could not read query cache: File {query_cache_name} does not exist", category=QueryCacheWarning)
            self._query_cache = {}
        atexit.register(self._store_query_cache, query_cache_name)

    def _store_query_cache(self, query_cache_name: str) -> None:
        """Stores the query cache into a JSON file.

        Parameters
        ----------
        query_cache_name : str
            The path where to write the file to. If it exists, it will be overwritten.
        """
        with open(query_cache_name, "w") as cache_file:
            json.dump(self._query_cache, cache_file)

    def _query_cache_name(self) -> str:
        """Provides a normalized file name for the query cache.

        Returns
        -------
        str
            The cache file name. It consists of the database system name, system version and the name of the database
        """
        identifier = "_".join([self.database_system_name(),
                               self.database_system_version().formatted(prefix="v", separator="_"),
                               self.database_name()])
        return f".query_cache_{identifier}.json"

    def __hash__(self) -> int:
        return hash(self._query_cache_name())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._query_cache_name() == other._query_cache_name()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.database_name()} @ {self.database_system_name()} ({self.database_system_version()})"


class DatabaseSchema(abc.ABC):
    """This interface provides access to different information about the logical structure of a database.

    In contrast to database statistics, schema information is much more standardized. PostBOUND therefore only takes on
    the role of a mediator to delegate requests to different parts of the schema to the approapriate - and sometimes
    system specific - metadata catalogs of the database systems. For each kind of schema information a dedicated query
    method exists. Take a look at these methods to understand the functionality provided by the database schema
    interface.

    Parameters
    ----------
    db : Database
        The database for which the schema information should be read. This is required to obtain cursors that request
        the desired data.
    """

    def __init__(self, db: Database):
        self._db = db

    def tables(self) -> set[base.TableReference]:
        """Fetches all user-defined tables that are contained in the current database.

        Returns
        -------
        set[base.TableReference]
            All tables in the current schema, including materialized views, etc.
        """
        query_template = "SELECT table_name FROM information_schema.tables WHERE table_catalog = %s"
        self._db.cursor().execute(query_template, (self._db.database_name(),))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        return set(base.TableReference(row[0]) for row in result_set)

    def columns(self, table: base.TableReference | str) -> set[base.ColumnReference]:
        """Fetches all columns of the given table.

        Parameters
        ----------
        table : base.TableReference | str
            A table in the current schema

        Returns
        -------
        set[base.ColumnReference]
            All columns for the given table. Will be empty if the table is not found or does not contain any columns.

        Raises
        ------
        postbound.qal.base.VirtualTableError
            If the given table is virtual (e.g. subquery or CTE)
        """
        table = table if isinstance(table, base.TableReference) else base.TableReference(table)
        if table.virtual:
            raise base.VirtualTableError(table)
        query_template = textwrap.dedent("""
                                         SELECT column_name
                                         FROM information_schema.columns
                                         WHERE table_catalog = %s AND table_name = %s
                                         """)
        db_name = self._db.database_name()
        self._db.cursor().execute(query_template, (db_name, table.full_name))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        return set(base.ColumnReference(row[0], table) for row in result_set)

    def is_view(self, table: base.TableReference | str) -> bool:
        """Checks, whether a specific table is actually is a view.

        Parameters
        ----------
        table : base.TableReference | str
            The table to check. May not be a virtual table.

        Returns
        -------
        bool
            Whether the table is a view

        Raises
        ------
        ValueError
            If the table was not found in the current database
        """
        if isinstance(table, base.TableReference) and table.virtual:
            raise base.VirtualTableError(table)
        table = table if isinstance(table, str) else table.full_name
        db_name = self._db.database_name()
        query_template = textwrap.dedent("""
                                         SELECT table_type
                                         FROM information_schema.tables
                                         WHERE table_catalog = %s AND table_name = %s
                                         """)
        self._db.cursor().execute(query_template, (db_name, table))
        result_set = self._db.cursor().fetchall()
        assert result_set is not None
        if not result_set:
            raise ValueError(f"Table '{table}' not found in database '{db_name}'")
        table_type = result_set[0][0]
        return table_type == "VIEW"

    @abc.abstractmethod
    def lookup_column(self, column: base.ColumnReference | str,
                      candidate_tables: Iterable[base.TableReference]) -> base.TableReference:
        """Searches for a table that owns the given column.

        Parameters
        ----------
        column : base.ColumnReference | str
            The column that is being looked up
        candidate_tables : Iterable[base.TableReference]
            Tables that could possibly own the given column

        Returns
        -------
        base.TableReference
            The first of the `candidate_tables` that has a column of similar name.

        Raises
        ------
        ValueError
            If none of the candidate tables has a column of the given name.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_primary_key(self, column: base.ColumnReference) -> bool:
        """Checks, whether a column is the primary key for its associated table.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check

        Returns
        -------
        bool
            Whether the column is the primary key of its table. If it is part of a compound primary key, this is
            ``False``.

        Raises
        ------
        postbound.qal.base.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.base.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def has_secondary_index(self, column: base.ColumnReference) -> bool:
        """Checks, whether a secondary index is available for a specific column.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check

        Returns
        -------
        bool
            Whether a secondary index of any kind was created for the column. Compound indexes and primary key indexes
            fail this test.

        Raises
        ------
        postbound.qal.base.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.base.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        raise NotImplementedError

    def has_index(self, column: base.ColumnReference) -> bool:
        """Checks, whether there is any index structure available on a column

        Parameters
        ----------
        column : base.ColumnReference
            The column to check

        Returns
        -------
        bool
            Whether any kind of index (primary, or secondary) is available for the column. Only compound indexes will
            fail this test.

        Raises
        ------
        postbound.qal.base.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.base.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        return self.is_primary_key(column) or self.has_secondary_index(column)

    @abc.abstractmethod
    def datatype(self, column: base.ColumnReference) -> str:
        """Retrieves the (physical) data type of a column.

        The provided type can be a standardized SQL-type, but it can be a type specific to the concrete database
        system just as well. It is up to the user to figure this out and to react accordingly.

        Parameters
        ----------
        column : base.ColumnReference
            The colum to check

        Returns
        -------
        str
            The datatype. Will never be empty.

        Raises
        ------
        postbound.qal.base.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.base.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Database schema of {self._db}"


class DatabaseStatistics(abc.ABC):
    """The statistics interface provides unified access to table-level and column-level statistics.

    There are two main challenges when implementing a generalized statistics interface for different database systems.
    The first one is the non-deterministic creation and maintenance of statistics by most database systems. This means
    that creating two identical databases on the same database system on the same machine might still yield different
    statistical values. This is because database systems oftentimes create statistics from random samples of column
    values to speed up computation. However, such variability hurts our efforts to enable reproducible experiments
    since different performances metrics might not be due to differences in the optimization algorithms but due to bad
    luck when creating the statistics (whether it is a good sign if an algorithm is that fragile to deviations in
    statistics is another question). The second main challenge is that different database systems maintain different
    statistics. Even though many statistics are considered quite "basic" by the research community, not all systems
    developers deemed all statistics necessary for their optimizer. Once again, this can severly hinder the application
    of an optimization algorithm if it relies on a basic statistic that just happens to not be available on the desired
    target database system.

    To address both of these issues, the statistics interface operates in two different modes: in *native* mode it
    simply delegates all requests to statistical information to the corresponding catalogs of the database systems.
    Alternatively, the statistics interface can create the illusion of a normalized and standardized statistics
    catalogue. This so-called *emulated* mode does not rely on the statistics catalogs and issues equivalent SQL
    queries instead. For example, if a statistic on the number of distinct values of a column is requested, this
    emulated by running a ``SELECT COUNT(DISTINCT column) FROM table`` query.

    The current mode can be customized using the boolean `emualted` property. If the statistics interface operates in
    native mode (i.e. based on the actual statistics catalog) and the user requests a statistic that is not available
    in the selected database system, the behavior depends on another attribute: `enable_emulation_fallback`. If this
    boolean attribute is ``True``, an emulated statistic will be calculated instead. Otherwise, an
    `UnsupportedDatabaseFeatureError` is raised.

    Since the live computation of emulated statistics can be costly, the statistics interface has its own
    `cache_enabled` attribute. It can be set to `None` to use the default caching behavior of the database system.
    However, if this attribute is set to `True` or `False` directly, caching will be used accordingly for all
    compute-intensive statistics operations (and only such operations). Once again, this only works because PostBOUND
    assumes the database to be immutable.

    Parameters
    ----------
    db : Database
        The database for which the schema information should be read. This is required to hook into the database cache
        and to obtain the cursors to actuall execute queries.
    emulated : bool, optional
        Whether the statistics interface should operate in emulation mode. To enable reproducibility, this is ``True``
        by default
    enable_emulation_fallback : bool, optional
        Whether emulation should be used for unsupported statistics when running in native mode, by default True
    cache_enabled : Optional[bool], optional
        Whether emulated statistics queries should be subject to caching, by default True. Set to ``None`` to use the
        caching behavior of the `db`

    See Also
    --------
    postbound.postbound.OptimizationPipeline : The basic optimization process applied by PostBOUND
    """

    def __init__(self, db: Database, *, emulated: bool = True, enable_emulation_fallback: bool = True,
                 cache_enabled: Optional[bool] = True) -> None:
        self.emulated = emulated
        self.enable_emulation_fallback = enable_emulation_fallback
        self.cache_enabled = cache_enabled
        self._db = db

    def total_rows(self, table: base.TableReference, *, emulated: Optional[bool] = None,
                   cache_enabled: Optional[bool] = None) -> Optional[int]:
        """Provides (an estimate of) the total number of rows in a table.

        Parameters
        ----------
        table : base.TableReference
            The table to check
        emulated : Optional[bool], optional
            Whether to force emulation mode for this single call. Defaults to ``None`` which indicates that the
            emulation setting of the statistics interface should be used.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to ``None`` which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Optional[int]
            The total number of rows in the table. If no such statistic exists, but the database system in principle
            maintains the statistic, ``None`` is returned. For example, this situation can occur if the database system
            only maintains a row count if the table has at least a certain size and the table in question did not reach
            that size yet.

        Raises
        ------
        base.VirtualTableError
            If the given table is virtual (e.g. subquery or CTE)
        """
        if table.virtual:
            raise base.VirtualTableError(table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_total_rows(table,
                                              cache_enabled=self._determine_caching_behavior(cache_enabled))
        else:
            return self._retrieve_total_rows_from_stats(table)

    def distinct_values(self, column: base.ColumnReference, *, emulated: Optional[bool] = None,
                        cache_enabled: Optional[bool] = None) -> Optional[int]:
        """Provides (an estimate of) the total number of different column values of a specific column.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check
        emulated : Optional[bool], optional
            Whether to force emulation mode for this single call. Defaults to ``None`` which indicates that the
            emulation setting of the statistics interface should be used.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to ``None`` which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Optional[int]
            The number of distinct values in the column. If no such statistic exists, but the database system in
            principle maintains the statistic, ``None`` is returned. For example, this situation can occur if the
            database system only maintains a distinct value count if the column values are distributed in a
            sufficiently diverse way.

        Raises
        ------
        postbound.qal.base.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.base.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        if not column.table:
            raise base.UnboundColumnError(column)
        elif column.table.virtual:
            raise base.VirtualTableError(column.table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_distinct_values(column,
                                                   cache_enabled=self._determine_caching_behavior(cache_enabled))
        else:
            return self._retrieve_distinct_values_from_stats(column)

    def min_max(self, column: base.ColumnReference, *, emulated: Optional[bool] = None,
                cache_enabled: Optional[bool] = None) -> Optional[tuple[Any, Any]]:
        """Provides (an estimate of) the minimum and maximum values in a column.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check
        emulated : Optional[bool], optional
            Whether to force emulation mode for this single call. Defaults to ``None`` which indicates that the
            emulation setting of the statistics interface should be used.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to ``None`` which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Optional[tuple[Any, Any]]
            A tuple of minimum and maximum value. If no such statistic exists, but the database system in principle
            maintains the statistic, ``None`` is returned. For example, this situation can occur if thec database
            system only maintains the min/max value if they are sufficiently far apart.

        Raises
        ------
        postbound.qal.base.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.base.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        if not column.table:
            raise base.UnboundColumnError(column)
        elif column.table.virtual:
            raise base.VirtualTableError(column.table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_min_max_values(column,
                                                  cache_enabled=self._determine_caching_behavior(cache_enabled))
        else:
            return self._retrieve_min_max_values_from_stats(column)

    def most_common_values(self, column: base.ColumnReference, *, k: int = 10, emulated: Optional[bool] = None,
                           cache_enabled: Optional[bool] = None) -> Sequence[tuple[Any, int]]:
        """Provides (an estimate of) the total number of occurrences of the `k` most frequent values of a column.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check
        k : int, optional
            The maximum number of most common values to return. Defaults to 10. If there are less values available, all
            of the available values will be returned.
        emulated : Optional[bool], optional
            Whether to force emulation mode for this single call. Defaults to ``None`` which indicates that the
            emulation setting of the statistics interface should be used.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to ``None`` which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Sequence[tuple[Any, int]]
            The most common values in pairs of (value, frequency), starting with the highest frequency. Notice that
            this sequence can be empty if no values are available. This can happen if the database system in principle
            maintains this statistic but does considers the value distribution to uniform to make the maintenance
            worthwhile. Likewise, if less common values exist than the requested `k` value, only the available values
            will be returned (and the sequence will be shorter than `k` in that case).

        Raises
        ------
        postbound.qal.base.UnboundColumnError
            If the column is not associated with any table
        postbound.qal.base.VirtualTableError
            If the table associated with the column is a virtual table (e.g. subquery or CTE)
        """
        if not column.table:
            raise base.UnboundColumnError(column)
        elif column.table.virtual:
            raise base.VirtualTableError(column.table)
        if emulated or (emulated is None and self.emulated):
            return self._calculate_most_common_values(column, k,
                                                      cache_enabled=self._determine_caching_behavior(cache_enabled))
        else:
            return self._retrieve_most_common_values_from_stats(column, k)

    def _calculate_total_rows(self, table: base.TableReference, *, cache_enabled: Optional[bool] = None) -> int:
        """Retrieves the total number of rows of a table by issuing a ``COUNT(*)`` query against the live database.

        The table is assumed to be non-virtual.

        Parameters
        ----------
        table : base.TableReference
            The table to check
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to ``None`` which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        int
            The total number of rows in the table.
        """
        query_template = "SELECT COUNT(*) FROM {tab}".format(tab=table.full_name)
        return self._db.execute_query(query_template, cache_enabled=self._determine_caching_behavior(cache_enabled))

    def _calculate_distinct_values(self, column: base.ColumnReference, *, cache_enabled: Optional[bool] = None) -> int:
        """Retrieves the number of distinct column values by issuing a ``COUNT(*)`` / ``GROUP BY`` query over that
        column against the live database.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to ``None`` which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        int
            The number of distinct values in the column
        """
        query_template = "SELECT COUNT(DISTINCT {col}) FROM {tab}".format(col=column.name, tab=column.table.full_name)
        return self._db.execute_query(query_template, cache_enabled=self._determine_caching_behavior(cache_enabled))

    def _calculate_min_max_values(self, column: base.ColumnReference, *,
                                  cache_enabled: Optional[bool] = None) -> tuple[Any, Any]:
        """Retrieves the minimum/maximum values in a column by issuing an aggregation query for that column against the
        live database.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to ``None`` which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        tuple[Any, Any]
            A tuple of ``(min val, max val)``
        """
        query_template = "SELECT MIN({col}), MAX({col}) FROM {tab}".format(col=column.name, tab=column.table.full_name)
        return self._db.execute_query(query_template, cache_enabled=self._determine_caching_behavior(cache_enabled))

    def _calculate_most_common_values(self, column: base.ColumnReference, k: int, *,
                                      cache_enabled: Optional[bool] = None) -> Sequence[tuple[Any, int]]:
        """Retrieves the `k` most frequent values of a column along with their frequencies by issuing a query over that
        column against the live database.

        The actual query combines a ``COUNT(*)`` aggregation, with a grouping over the column values, followed by a
        count-based ordering and limit.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check
        k : int
            The number of most frequent values to retrieve. If less values are available (because there are not as much
            distinct values in the column), the frequencies of all values is returned.
        cache_enabled : Optional[bool], optional
            Whether to enable result caching in emulation mode. Defaults to ``None`` which indicates that the caching
            setting of the statistics interface should be used.

        Returns
        -------
        Sequence[tuple[Any, int]]
            The most common values in ``(value, frequency)`` pairs, ordered by largest frequency first. Can be smaller
            than the requested `k` value if the column contains less distinct values.
        """
        query_template = textwrap.dedent("""
            SELECT {col}, COUNT(*) AS n
            FROM {tab}
            GROUP BY {col}
            ORDER BY n DESC, {col}
            LIMIT {k}""".format(col=column.name, tab=column.table.full_name, k=k))
        return self._db.execute_query(query_template, cache_enabled=self._determine_caching_behavior(cache_enabled))

    @abc.abstractmethod
    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> Optional[int]:
        """Queries the DBMS-internal metadata for the number of rows in a table.

        The table is assumed to be non-virtual.

        Parameters
        ----------
        table : base.TableReference
            The table to check

        Returns
        -------
        Optional[int]
            The total number of rows in the table. If no such statistic exists, but the database system in principle
            maintains the statistic, ``None`` is returned. For example, this situation can occur if the database system
            only maintains a row count if the table has at least a certain size and the table in question did not reach
            that size yet.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> Optional[int]:
        """Queries the DBMS-internal metadata for the number of distinct values of the column.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check

        Returns
        -------
        Optional[int]
            The number of distinct values in the column. If no such statistic exists, but the database system in
            principle maintains the statistic, ``None`` is returned. For example, this situation can occur if the
            database system only maintains a distinct value count if the column values are distributed in a
            sufficiently diverse way.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_min_max_values_from_stats(self, column: base.ColumnReference) -> Optional[tuple[Any, Any]]:
        """Queries the DBMS-internal metadata for the minimum / maximum value in a column.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check

        Returns
        -------
        Optional[tuple[Any, Any]]
            A tuple of minimum and maximum value. If no such statistic exists, but the database system in principle
            maintains the statistic, ``None`` is returned. For example, this situation can occur if thec database
            system only maintains the min/max value if they are sufficiently far apart.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_most_common_values_from_stats(self, column: base.ColumnReference,
                                                k: int) -> Sequence[tuple[Any, int]]:
        """Queries the DBMS-internal metadata for the `k` most common values of the `column`.

        The column is assumed to be bound to a (non-virtual) table.

        Parameters
        ----------
        column : base.ColumnReference
            The column to check
        k : int, optional
            The maximum number of most common values to return. Defaults to 10. If there are less values available, all
            of the available values will be returned.

        Returns
        -------
        Sequence[tuple[Any, int]]
            The most common values in pairs of (value, frequency), starting with the highest frequency. Notice that
            this sequence can be empty if no values are available. This can happen if the database system in principle
            maintains this statistic but does considers the value distribution to uniform to make the maintenance
            worthwhile. Likewise, if less common values exist than the requested `k` value, only the available values
            will be returned (and the sequence will be shorter than `k` in that case).
        """
        raise NotImplementedError

    def _determine_caching_behavior(self, local_cache_enabled: Optional[bool]) -> Optional[bool]:
        """Utility to quickly figure out which caching behavior to use.

        This method is intended to be called by the top-level methods that provide statistics and enable a selective
        caching which overwrites the caching behavior of the statistics interface.

        Parameters
        ----------
        local_cache_enabled : Optional[bool]
            The caching setting selected by the callee / user.

        Returns
        -------
        Optional[bool]
            Whether caching should be enabled or the determined by the actual database interface.
        """
        return self.cache_enabled if local_cache_enabled is None else local_cache_enabled

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Database statistics of {self._db}"


class HintWarning(UserWarning):
    """Custom warning category for hinting-related problems."""
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class HintService(abc.ABC):
    """Provides the necessary tools to generate system-specific query instances based on optimizer decisions.

    Hints are PostBOUNDs way to enforce that decisions made in the optimization pipeline are respected by the native
    query optimizer once the query is executed in an actual database system. The general documentation provides much
    more information about why this is necessary and how PostBOUND approaches query optimization and query generation.

    Each database system has to implement this interface to be usable as part of an optimization pipeline.

    See Also
    --------
    OptimizationPipeline.optimize_query : For a general introduction into the query optimization process
    """

    @abc.abstractmethod
    def generate_hints(self, query: qal.SqlQuery,
                       join_order: Optional[jointree.LogicalJoinTree | jointree.PhysicalQueryPlan] = None,
                       physical_operators: Optional[physops.PhysicalOperatorAssignment] = None,
                       plan_parameters: Optional[planparams.PlanParameterization] = None) -> qal.SqlQuery:
        """Transforms the input query such that the given optimization decisions are respected during query execution.

        In the most common case this involves building a `Hint` clause that encodes the optimization decisions in a
        system-specific way. However, depending on the concrete database system, this might also involve a
        restructuring of certain parts of the query, e.g. the usage of specific join statements, the introduction of
        non-standard SQL statements, or a reordering of the ``FROM`` clause.

        Notice that all optimization information is optional. If individual parameters are set to ``None``, nothing
        has been enforced by PostBOUND's optimization process and the native optimizer of the database system should
        "fill the gaps".

        Parameters
        ----------
        query : qal.SqlQuery
            The query that should be transformed
        join_order : Optional[jointree.LogicalJoinTree  |  jointree.PhysicalQueryPlan], optional
            The sequence in which individual joins should be executed. If this is a `PhysicalQueryPlan` and all other
            optimization parameters are ``None``, this essentially enforces the given query plan.
        physical_operators : Optional[physops.PhysicalOperatorAssignment], optional
            The physical operators that should be used for the query execution. In addition to selecting specific
            operators for specific joins or scans, this can also include disabling certain operators for the entire
            query. If a `join_order` is also given and includes physical operators, these should be ignored and only
            the `physical_operators` should be used.
        plan_parameters : Optional[planparams.PlanParameterization], optional
            Additional parameters and metadata for the native optimizer of the database system. Probably the most
            important use-case of these parameters is the supply of cardinality estimates for different joins and
            scans. For example, these can be combined with a join order to influence the physical operators that the
            native optimizer chooses. Another scenario is to only supply such cardinality estimates and leave the
            `join_order` and `physical_operators` completely empty, which essentially simulates a different cardinality
            estimation algorithm for the query. Notice however, that in this scenario cardinality estimates for all
            possible intermediate results of the query have to be supplied. Otherwise, the native optimizer once
            again "fills the gaps" and uses its own estimates for the remaining intermediate results that it explores
            during plan enumeration. This would probably effectively break the estimation algorithm.

        Returns
        -------
        qal.SqlQuery
            The transformed query. It contains all necessary information to enforce the optimization decisions as best
            as possible. Notice that whether the native optimizer of the database system is obliged to respect the
            optimization decisions depends on the specific system. For example, for MySQL hints are really just hints
            and the optimizer is only encouraged to use specific operators but not forced to do so.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def format_query(self, query: qal.SqlQuery) -> str:
        """Transforms the query into a database-specific string, mostly to incorporate deviations from standard SQL.

        This method is necessary because the query abstraction layer is focused on modelling and unifying different
        parts of an SQL query. However, some database systems (cough .. MySQL .. cough) deviate from standard SQL
        syntax and express different parts of a query different. The most prominent example are older versions of
        MySQL that used double quotes for string values rather than the SQL standard single quotes. Therefore, the
        `format_query` method takes an abstract representation of an SQL query as input and turns it into a string
        representation that accounts for all such deviations.

        Parameters
        ----------
        query : qal.SqlQuery
            The query that should be adapted for the database system

        Returns
        -------
        str
            An equivalent notation of the query that incorporates system-specific deviations from standard SQL.
            Notice that this query possibly can no longer be parsed by the query abstraction layer. It is a one-way
            process.

        See Also
        --------
        postbound.qal : the query abstraction layer provided by PostBOUND
        """
        raise NotImplementedError

    @abc.abstractmethod
    def supports_hint(self, hint: physops.PhysicalOperator | planparams.HintType) -> bool:
        """Checks, whether the database system is capable of using the specified hint or operator

        Parameters
        ----------
        hint : physops.PhysicalOperator | planparams.HintType
            The hint/feature to check

        Returns
        -------
        bool
            Indicates whether the feature is supported by the specific database system.
        """
        raise NotImplementedError


class QueryExecutionPlan:
    """Heavily simplified system-independent model of physical query execution plans.

    A plan is a tree structure of `QueryExecutionPlan` objects. Each plan is a node in the tree and can have additional
    child nodes. These represent sub-operators that provide the input for the current plan.

    Since the information contained in physical query execution plans varies heavily between different database
    systems, this class models the smallest common denominator between all such plans. It focuses focuses on joins,
    scans and intermediate nodes and does not restrict the specific node types. Generally speaking, this model enforces
    very little constraints on the query plans. It is up to the users of the query plans to accomodate for deviations
    from the expected data and to the producers of query plan instances to be as strict as possible with the produced
    plans.

    In addition to the unified data storage in the different attributes, a number of introspection methods are also
    defined.

    All parameters to the `__init__` method are available as attributes on the new object. However, they should be
    considered read-only, even though this is not enforced.

    Comparisons between two query plans and hashing of a plan is based on the node type, the children and the table of each
    node. Other attributes are not considered for these operations such as cost estimates.


    Parameters
    ----------
    node_type : str
        The name of the operator/node
    is_join : bool
        Whether the operator represents a join. This is usually but not always mutually exclusive to the `is_scan`
        attribute.
    is_scan : bool
        Whether the operator represents a scan. This is usually but not always mutually exclusive to the `is_join`
        attribute.
    table : Optional[base.TableReference], optional
        For scan operators this can denote the table being scanned. Defaults to ``None`` if not applicable.
    children : Optional[Iterable[QueryExecutionPlan]], optional
        The sub-operators that provide input for the current operator. Can be ``None`` leaf nodes (most probably scans)
    parallel_workers : float, optional
        The total number of parallel workers that are used by the current operator. Defaults to ``math.nan`` if unknown
        or not applicable (e.g. if the operator does not support parallelization).
    cost : float, optional
        The cost of the operator, according to some cost model. This can be the cost as estimated by the native query
        optimizer of the database system. Alternatively, this can also represent costs from a custom cost model or even
        the real cost of the operator, depending on the context. Defaults to ``math.nan`` if unknown or not important
        in the current context.
    estimated_cardinality : float, optional
        The estimated number of rows that are produced by the operator. This estimate can be provided by the database
        system, or by some other estimation algorithm depending on context. Defaults to ``math.nan`` if unknown or not
        important in the current context.
    true_cardinality : float, optional
        The actual number of rows that are produced by the operator. This number is usually obtained by running the
        query and counting the output tuples. Notice that this number might only be an approximation of the true true
        cardinality (although usually a pretty precise one at that). Defaults to ``math.nan`` if the query execution
        plan is really only a plan and therefore does not contain any execution knowledge, or if the cardinality is
        simply not relevant in the current context.
    execution_time : float, optional
        The time in seconds it took to execute the operator. For intermediate nodes, this includes the execution time
        of all child operators as well. Usually, this denotes the wall time, but can be a more precise (e.g.
        tick-based) or coarse measurement, depending on the context. Defaults to  ``math.nan`` if the query execution
        plan is really only a plan and therefore does not contain any execution knowledge, or if the execution time is
        simply not relevant in the current context.
    cached_pages : float, optional
        The number of pages that were processed by this node and could be retrieved from cache (including child nodes).
        Defaults to ``math.nan`` if this is unknown or not applicable (e.g. for non-analyze plans). Notice that *cache* in this
        context means a database system-level cache (also called buffer pool, etc.), not the OS cache or a hardware cache.
    scanned_pages : float, optional
        The number of pages that were processed by this node and had to be read from disk and could not be retrieved from
        cache (including child nodes). Default to ``math.nan`` if this unknown or not applicable (e.g. for non-analyze plans).
        See `cached_pages` for more details.
    physical_operator : Optional[physops.PhysicalOperator], optional
        The physical operator that corresponds to the current node if such an operator exists. Defaults to ``None``
        if no such operator exists or it could not be derived by the producer of the query plan.
    inner_child : Optional[QueryExecutionPlan], optional
        For intermediate operators that contain an inner and an outer relation, this denotes the inner relation. This
        assumes that the operator as exactly two children and allows to infer the outer children. Defaults to ``None``
        if the concept of inner/outer children is not applicable to the database system, not relevant to the current
        context, or if the relation could not be inferred.
    subplan_input : Optional[QueryExecutionPlan], optional
        Input node that was computed as part of a nested query. Such nodes will typically be used within the filter predicates
        of the node.
    subplan_root : bool, optional
        Whether the current node is the root of a subplan that computes a nested query. See `subplan_input` for more details.
    subplan_name : str, optional
        For subplan roots, this is the name of the subplan being exported. For nodes with subplan input, this is the name of
        the subplan being imported.

    Notes
    -----
    In case of parallel execution, all measures should be thought of "meaningful totals", i.e. the cardinality
    numbers are the total number of tuples produced by all workers. The execution time should denote the wall time, it
    took to execute the entire operator (which just happened to include parallel processing), **not** an average of the
    worker execution time or some other measure.
    """
    def __init__(self, node_type: str, is_join: bool, is_scan: bool, *, table: Optional[base.TableReference] = None,
                 children: Optional[Iterable[QueryExecutionPlan]] = None,
                 parallel_workers: float = math.nan, cost: float = math.nan,
                 estimated_cardinality: float = math.nan, true_cardinality: float = math.nan,
                 execution_time: float = math.nan,
                 cached_pages: float = math.nan, scanned_pages: float = math.nan,
                 physical_operator: Optional[physops.PhysicalOperator] = None,
                 inner_child: Optional[QueryExecutionPlan] = None,
                 subplan_input: Optional[QueryExecutionPlan] = None, is_subplan_root: bool = False,
                 subplan_name: str = "") -> None:
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
        self.outer_child: Optional[QueryExecutionPlan] = None
        if self.inner_child and len(self.children) == 2:
            first_child, second_child = self.children
            self.outer_child = first_child if self.inner_child == second_child else second_child

        self.table = table

        self.cost = cost
        self.estimated_cardinality = estimated_cardinality
        self.true_cardinality = true_cardinality
        self.execution_time = execution_time

        self.cached_pages = cached_pages
        self.scanned_pages = scanned_pages

        self.subplan_input = subplan_input
        if self.subplan_input:
            self.subplan_input.is_subplan_root = True
        self.is_subplan_root = is_subplan_root
        self.subplan_name = subplan_name

    @property
    def total_accessed_pages(self) -> float:
        """The total number of pages that where processed in this node, as well as all child nodes.

        This includes pages that were fetched from the DB cache, as well as pages that had to be read from disk.

        Returns
        -------
        float
            The number of pages. Can be ``NaN`` if this number cannot be inferred, e.g. for non-analyze plans.
        """
        return self.cached_pages + self.scanned_pages

    @property
    def cache_hit_ratio(self) -> float:
        """The share of pages that could be fetched from cache compared to the total number of processed pages.

        Returns
        -------
        float
            The hit ratio. Can be ``NaN`` if the ratio cannot be inferred, e.g. for non-analyze plans.
        """
        return self.cached_pages / self.total_accessed_pages

    def is_analyze(self) -> bool:
        """Checks, whether the plan contains runtime information.

        If that is the case, the plan most likely contains additional information about the execution time of the
        operator as well as the actual cardinality that was produced.

        Returns
        -------
        bool
            Whether the plan corresponds to an ANALYZE plan.
        """
        return not math.isnan(self.true_cardinality) or not math.isnan(self.execution_time)

    def tables(self) -> frozenset[base.TableReference]:
        """Collects all tables that are referenced by this operator as well as all child operators.

        Most likely this corresponds to all tables that were scanned below the current operator.

        Returns
        -------
        frozenset[base.TableReference]
            The tables
        """
        own_table = [self.table] if self.table else []
        return frozenset(own_table + collection_utils.flatten(child.tables() for child in self.children))

    def is_base_join(self) -> bool:
        """Checks, whether this operator is a join and only contains base table children.

        This does not necessarily mean that the children will already be scan operators, but it means that this
        operator does not consume intermediate relations that have been joined themselves. For example, a hash join
        can have one child that is scanned directly, and another child that is an intermediate hash node. That hash
        node takes care of creating the hash table for its children, which in turn is a scan of a base table.

        Returns
        -------
        bool
            Whether this operator corresponds to a join of base table relations
        """
        return self.is_join and all(child.is_scan_branch() for child in self.children)

    def is_bushy_join(self) -> bool:
        """Checks, whether this operator is a join of two intermediate tables.

        An intermediate table is a table that is itself composed of a join of base tables. The children can be
        arbitrary other operators, but each of the children will at some point have a join operator as a child node.

        Returns
        -------
        bool
            Whether this operator correponds to a join of intermediate relations
        """
        return self.is_join and all(child.is_join_branch() for child in self.children)

    def is_scan_branch(self) -> bool:
        """Checks, whether this branch of the query plan leads to a scan eventually.

        Most importantly, a *scan branch* does not contain any joins and is itself not a join. Typically, it will end
        with a scan leaf and only contain intermediate nodes (such as a Hash node in PostgreSQL) afterwards.

        Returns
        -------
        bool
            Whether this node does not contain any joins below.
        """
        return self.is_scan or (len(self.children) == 1 and self.children[0].is_scan_branch())

    def is_join_branch(self) -> bool:
        """Checks, whether this branch of the query plan leads to a join eventually.

        In contrast to a scan branch, a join branch is guaranteed to contain at least one more child (potentially
        transitively) that is a join node. Alternatively, the current node might be a join node itself.

        Returns
        -------
        bool
            Whether this node does contain at least one join children (or is itself a join)
        """
        return self.is_join or (len(self.children) == 1 and self.children[0].is_join_branch())

    def fetch_base_table(self) -> Optional[base.TableReference]:
        """Provides the base table that is associated with this scan branch.

        This method basically traverses the current branch of the query execution plan until a node with an associated
        `table` is found.

        Returns
        -------
        Optional[base.TableReference]
            The associated table of the highest child node that has a valid `table` attribute. As a special case this
            might be the table of this very plan node. If none of the child nodes contain a valid table returns
            ``None``.

        Raises
        ------
        ValueError
            _description_

        See Also
        --------
        is_scan_branch
        """
        if not self.is_scan_branch():
            raise ValueError("No unique base table for non-scan branches!")
        if self.table:
            return self.table
        return self.children[0].fetch_base_table()

    def total_processed_rows(self) -> float:
        """Counts the sum of all rows that have been processed at each node below and including this plan node.

        This basically calculates the value of the *C_out* cost model for the current branch. Calling this method on
        the root node of the query plan provides the *C_out* value of the entire plan.

        Returns
        -------
        float
            The *C_out* value

        Notes
        -----
        *C_out* is defined as

        .. math ::
            C_{out}(T) = \\begin{cases}
                |T|, & \\text{if T is a single relation} \\\\
                |T| + C_{out}(T_1) + C_{out}(T_2), & \\text{if } T = T_1 \\bowtie T_2
            \\end{cases}

        Notice that there are variations of the *C_out* value that do not include any cost for single relations and
        only measure the cost of joins.
        """
        if not self.is_analyze():
            return math.nan
        return self.true_cardinality + sum(child.total_processed_rows() for child in self.children)

    def qerror(self) -> float:
        """Calculates the q-error of the current node.

        Returns
        -------
        float
            The q-error

        Notes
        -----
        The q-error can only be calculate for analyze nodes. We use the following formula:

        .. math ::
            qerror(e, a) = \\frac{max(e, a) + 1}{min(e, a) + 1}

        where *e* is the estimated cardinality of the node and *a* is the actual cardinality. Notice that we add 1 to both the
        numerator as well as the denominator to prevent infinite errors for nodes that do not process any rows (e.g. due to
        pruning).
        """
        if math.isnan(self.true_cardinality):
            return math.nan
        # we add 1 to our q-error to prevent probelms with 0 cardinalities
        return ((max(self.estimated_cardinality, self.true_cardinality) + 1)
                / (min(self.estimated_cardinality, self.true_cardinality) + 1))

    def scan_nodes(self) -> frozenset[QueryExecutionPlan]:
        """Provides all scan nodes under and including this node.

        Returns
        -------
        frozenset[QueryExecutionPlan]
            The scan nodes
        """
        own_node = [self] if self.is_scan else []
        child_scans = collection_utils.flatten(child.scan_nodes() for child in self.children)
        return frozenset(own_node + child_scans)

    def join_nodes(self) -> frozenset[QueryExecutionPlan]:
        """Provides all join nodes under and including this node.

        Returns
        -------
        frozenset[QueryExecutionPlan]
            The join nodes
        """
        own_node = [self] if self.is_join else []
        child_joins = collection_utils.flatten(child.join_nodes() for child in self.children)
        return frozenset(own_node + child_joins)

    def iternodes(self) -> Iterable[QueryExecutionPlan]:
        """Provides all nodes in the plan, in breadth-first order.

        The current node is also included in the output.

        Returns
        -------
        Iterable[QueryExecutionPlan]
            The nodes.
        """
        nodes = [self]
        for child_node in self.children:
            nodes.extend(child_node.iternodes())
        return nodes

    def plan_depth(self) -> int:
        """Calculates the maximum path length from this node to a leaf node.

        Since the current node is included in the calculation, the minimum value is 1 (if this node already is a leaf
        node).

        Returns
        -------
        int
            The path length
        """
        return 1 + max((child.plan_depth() for child in self.children), default=0)

    def find_first_node(self, predicate: Callable[[QueryExecutionPlan], bool], *,
                        traversal: Literal["left", "right", "inner", "outer"] = "left") -> Optional[QueryExecutionPlan]:
        """Recursively searches for the first node that matches a specific predicate.

        Parameters
        ----------
        predicate : Callable[[QueryExecutionPlan], bool]
            The predicate to check. The predicate is called on each node in the tree and should return ``True`` if the node
            matches the desired search criteria.
        traversal : Literal["left", "right", "inner", "outer"], optional
            The traversal strategy to use. It indicates which child node should be checked first if the `predicate` is not
            satisfied by the current node. Defaults to ``"left"`` which means that the left child is checked first.

        Returns
        -------
        Optional[QueryExecutionPlan]
            The first node that matches the predicate. If no such node exists, ``None`` is returned.
        """
        if predicate(self):
            return self
        if not self.children:
            return None

        if len(self.children) == 1:
            return self.children[0].find_first_node(predicate, traversal=traversal)
        if len(self.children) != 2:
            raise ValueError("Cannot traverse plan nodes with more than two children")

        if traversal == "inner" or traversal == "outer":
            first_child_to_check = self.inner_child if traversal == "inner" else self.outer_child
            second_child_to_check = self.outer_child if traversal == "inner" else self.inner_child
        else:
            first_child_to_check = self.children[0] if traversal == "left" else self.children[1]
            second_child_to_check = self.children[1] if traversal == "left" else self.children[0]

        first_result = first_child_to_check.find_first_node(predicate, traversal=traversal)
        if first_result is not None:
            return first_result
        return second_child_to_check.find_first_node(predicate, traversal=traversal)

    def find_all_nodes(self, predicate: Callable[[QueryExecutionPlan], bool]) -> Sequence[QueryExecutionPlan]:
        """Recursively searches for all nodes that match a specific predicate.

        Parameters
        ----------
        predicate : Callable[[QueryExecutionPlan], bool]
            The predicate to check. The predicate is called on each node in the tree and should return ``True`` if the node
            matches the desired search criteria.

        Returns
        -------
        Sequence[QueryExecutionPlan]
            All nodes that match the predicate. If no such node exists, an empty sequence is returned. Matches are returned in
            depth-first order, i.e. nodes higher up in the plan are returned before their matching child nodes.
        """
        def _handler(node: QueryExecutionPlan, predicate: Callable[[QueryExecutionPlan], bool]) -> list[QueryExecutionPlan]:
            matches = [node] if predicate(node) else []
            for child in node.children:
                matches.extend(_handler(child, predicate))
            return matches
        return _handler(self, predicate)

    def simplify(self) -> Optional[QueryExecutionPlan]:
        """Provides a query execution plan that is stripped of all non-join and non-scan nodes.

        Notice that this operation can break certain assumptions of mathematical relation between parent and child
        nodes, e.g. plan costs might no longer add up correctly.

        Returns
        -------
        Optional[QueryExecutionPlan]
            The simplified plan. If this method is called on a non-scan and non-join node that does not have any more
            children, ``None`` is returned. In all other cases, the final join or the only scan node will be returned.
        """
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

    def inspect(self, *, fields: Optional[Iterable[str]] = None, skip_intermediates: bool = False,
                _current_indentation: int = 0) -> str:
        """Provides a nice hierarchical string representation of the plan structure.

        The representation typically spans multiple lines and uses indentation to separate parent nodes from their
        children.

        Parameters
        ----------
        fields : Optional[Iterable[str]], optional
            The attributes of the nodes that should be included in the output. Can be set to any number of the available
            attributes. If no fields are given, a default configuration inspired by Postgres' **EXPLAIN ANALYZE** output is
            used.
        skip_intermediates : bool, optional
            Whether non-scan and non-join nodes should be excluded from the representation. Defaults to ``False``.
        _current_indentation : int, optional
            Internal parameter to the `_inspect` function. Should not be modified by the user. Denotes how deeply
            recursed we are in the plan tree. This enables the correct calculation of the current indentation level.
            Defaults to 0 for the root node.

        Returns
        -------
        str
            A string representation of the query plan
        """
        # TODO: include subplan_input in the inspection
        padding = " " * _current_indentation
        prefix = f"{padding}<- " if padding else ""

        if not fields:
            own_inspection = [prefix + self._explain_text()]
        else:
            attr_values = {attr: getattr(self, attr) for attr in fields}
            pretty_attrs = {attr: round(val, 3) if isinstance(val, numbers.Number) else val
                            for attr, val in attr_values.items()}
            attr_str = " ".join(f"{attr}={val}" for attr, val in pretty_attrs.items())
            own_inspection = [prefix + f"{self.node_type} ({attr_str})"]

        child_inspections = [
            child.inspect(fields=fields, skip_intermediates=skip_intermediates, _current_indentation=_current_indentation + 2)
            for child in self.children]
        if self.subplan_input:
            subplan_name = self.subplan_input.subplan_name if self.subplan_input.subplan_name else ""
            child_inspections.append(f"{padding}SubPlan: {subplan_name}")
            child_inspections.append(self.subplan_input.inspect(fields=fields, skip_intermediates=skip_intermediates,
                                                                _current_indentation=_current_indentation + 2))

        if not skip_intermediates or self.is_join or self.is_scan or not _current_indentation:
            return "\n".join(own_inspection + child_inspections)
        else:
            return "\n".join(child_inspections)

    def plan_summary(self) -> dict[str, object]:
        """Generates a short summary about some node statistics.

        Currently, the following information is reported:

        - The *C_out* value of the subtree
        - The maximum and average q-error of all nodes in the subtree (including the current node)
        - A usage count of all physical operators used in the subtree (including the current node)

        Returns
        -------
        dict[str, object]
            The node summary
        """
        all_nodes = list(self.iternodes())
        summary: dict[str, object] = {}
        summary["estimated_card"] = round(self.estimated_cardinality, 3)
        summary["actual_card"] = round(self.true_cardinality, 3)
        summary["estimated_cost"] = round(self.cost, 3)
        summary["c_out"] = self.total_processed_rows()
        summary["max_qerror"] = round(max(node.qerror() for node in all_nodes), 3)
        summary["avg_qerror"] = round(sum(node.qerror() for node in all_nodes) / len(all_nodes), 3)
        summary["phys_ops"] = collections.Counter(child.node_type for child in all_nodes)
        return summary

    def _explain_text(self) -> str:
        """Generates an EXPLAIN-like text representation of the current node.

        Returns
        -------
        str
            A textual description of the node
        """
        if self.is_analyze():
            exec_time = round(self.execution_time, 3)
            true_card = round(self.true_cardinality, 3)
            analyze_str = f" (execution time={exec_time} true cardinality={true_card})"
        else:
            analyze_str = ""

        if self.table and (not self.is_subplan_root and self.subplan_name):
            table_str = f" :: {self.table}, {self.subplan_name}"
        elif self.table:
            table_str = f" :: {self.table}" if self.table else ""
        elif not self.is_subplan_root and self.subplan_name:
            table_str = f" :: {self.subplan_name}"
        else:
            table_str = ""
        cost = round(self.cost, 3)
        estimated_card = round(self.estimated_cardinality, 3)
        plan_str = f" (cost={cost} estimated cardinality={estimated_card})"
        return "".join((self.node_type, table_str, plan_str, analyze_str))

    def __json__(self) -> dict:
        return vars(self)

    def __hash__(self) -> int:
        return hash((self.node_type, self.table, tuple(self.children)))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.node_type == other.node_type and self.table == other.table
                and self.children == other.children)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        normalized_node_type = self.node_type.replace(" ", "")
        if self.table:
            return f"{normalized_node_type}({self.table.identifier()})"
        child_texts = ", ".join(str(child) for child in self.children)
        return f"{normalized_node_type}({child_texts})"


def read_query_plan_json(json_data: dict) -> QueryExecutionPlan:
    """Reconstructs a query execution plan from its JSON representation.

    If the JSON data is somehow malformed, arbitrary errors can be raised.

    Parameters
    ----------
    json_data : dict
        The JSON data

    Returns
    -------
    QueryExecutionPlan
        The corresponding plan
    """
    json_data = dict(json_data)

    json_data["children"] = [read_query_plan_json(child_data) for child_data in json_data["children"]]
    json_data["inner_child"] = (read_query_plan_json(json_data["inner_child"])
                                if json_data["inner_child"] is not None else None)
    del json_data["outer_child"]

    table = json_data["table"]
    if table is not None:
        json_data["table"] = parser.JsonParser().load_table(json_data["table"])

    operator: str = json_data["physical_operator"]
    if operator is not None:
        json_data["physical_operator"] = physops.read_operator_json(operator)

    return QueryExecutionPlan(**json_data)


class OptimizerInterface(abc.ABC):
    """Provides high-level access to internal optimizer-related data for the database system.

    Each funtionality is available through a dedicated method. Notice that not all database systems necessarily
    support all of this functions.
    """
    @abc.abstractmethod
    def query_plan(self, query: qal.SqlQuery | str) -> QueryExecutionPlan:
        """Obtains the query execution plan for a specific query.

        This respects all hints that potentially influence the optimization process.

        Parameters
        ----------
        query : qal.SqlQuery | str
            The input query

        Returns
        -------
        QueryExecutionPlan
            The corresponding execution plan. This will never be an ``ANALYZE`` plan, but contain as much meaningful
            information as can be derived for the specific database system (e.g. regarding cardinality and cost
            estimates)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def analyze_plan(self, query: qal.SqlQuery) -> QueryExecutionPlan:
        """Executes a specific query and provides the query execution plan supplemented with runtime information.

        This respects all hints that potentially influence the optimization process.

        Parameters
        ----------
        query : qal.SqlQuery
            The input query

        Returns
        -------
        QueryExecutionPlan
            The corresponding execution plan. This plan will be an ``ANALYZE`` plan and contain all information that
            can be derived for the specific database system (e.g. cardinality estimates as well as true cardinality
            counts)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cardinality_estimate(self, query: qal.SqlQuery | str) -> int:
        """Queries the DBMS query optimizer for its cardinality estimate, instead of executing the query.

        The cardinality estimate will correspond to the estimate for the final node. Therefore, running this method
        with aggregate queries is not particularly meaningful.

        Parameters
        ----------
        query : qal.SqlQuery | str
            The input query

        Returns
        -------
        int
            The cardinality estimate of the native optimizer for the database system.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cost_estimate(self, query: qal.SqlQuery | str) -> float:
        """Queries the DBMS query optimizer for the estimated cost of executing the query.

        The cost estimate will correspond to the estimate for the final node. Typically, this cost includes the cost
        of all sub-operators as well.

        Parameters
        ----------
        query : qal.SqlQuery | str
            The input query

        Returns
        -------
        float
            The cost estimate of the native optimizer for the database system.
        """
        raise NotImplementedError


_DB_POOL: DatabasePool | None = None
"""Private variable that captures the current singleton instance of the `DatabasePool`."""


class DatabasePool:
    """The database pool allows different parts of the code base to easily obtain access to a database.

    This is achieved by maintaining one global pool of database connections which is shared by the entire system.
    New database instances can be registered and retrieved via unique keys. As long as there is just a single database
    instance, it can be accessed via the `current_database` method.

    The database pool implementation follows the singleton pattern. Use the static `get_instance` method to retrieve
    the database pool instance. All other functionality is provided based on that pool instance.

    References
    ----------

    .. Singleton pattern: https://en.wikipedia.org/wiki/Singleton_pattern
    """

    @staticmethod
    def get_instance() -> DatabasePool:
        """Provides access to the singleton database pool, creating a new pool instance if necessary.

        Returns
        -------
        DatabasePool
            The current pool instance
        """
        global _DB_POOL
        if _DB_POOL is None:
            _DB_POOL = DatabasePool()
        return _DB_POOL

    def __init__(self):
        self._pool: dict[str, Database] = {}

    def current_database(self) -> Database:
        """Provides the database that is currently stored in the pool, provided there is just one.

        Returns
        -------
        Database
            The only database in the pool

        Raises
        ------
        ValueError
            If there are multiple database instances registered in the pool
        """
        return dict_utils.value(self._pool)

    def register_database(self, key: str, db: Database) -> None:
        """Stores a new database in the pool.

        This method is typically called by the connect methods of the respective database system implementations.

        Parameters
        ----------
        key : str
            A unique identifier under which the database can be retrieved
        db : Database
            The database to store
        """
        self._pool[key] = db

    def retrieve_database(self, key: str) -> Database:
        """Provides the database that is registered under a specific key.

        Parameters
        ----------
        key : str
            The key that was previously used to register the database

        Returns
        -------
        Database
            The corresponding database

        Raises
        ------
        KeyError
            If no database was registered under the given key.
        """
        return self._pool[key]

    def empty(self) -> bool:
        """Checks, whether the database pool is currently emtpy (i.e. no database are registered).

        Returns
        -------
        bool
            *True* if the pool is empty.
        """
        return len(self._pool) == 0

    def clear(self) -> None:
        """Removes all currently registered databases from the pool."""
        self._pool.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._pool

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"DatabasePool {self._pool}"


class UnsupportedDatabaseFeatureError(RuntimeError):
    """Indicates that some requested feature is not supported by the database.

    For example, PostgreSQL (at least up to version 15) does not capture minimum or maximum column values in its
    system statistics. Therefore, forcing the DBS to retrieve such information from its metadata could result in this
    error.

    Parameters
    ----------
    database : Database
        The database that was requested to provide the problematic feature
    feature : str
        A textual description for the requested feature
    """

    def __init__(self, database: Database, feature: str) -> None:
        super().__init__(f"Database {database.system_name} does not support feature {feature}")
        self.database = database
        self.feature = feature


class DatabaseServerError(RuntimeError):
    """Indicates an error caused by the database server occured while executing a database operation.

    The error was **not** due to a mistake in the user input (such as an SQL syntax error or access privilege
    violation), but an implementation issue instead (such as out of memory during query execution).

    Parameters
    ----------
    message : str, optional
        A textual description of the error, e.g. *out of memory*. Can be left empty by default.
    context : Optional[object], optional
        Additional context information for when the error occurred, e.g. the query that caused the error. Mainly
        intended for debugging purposes.
    """

    def __init__(self, message: str = "", context: Optional[object] = None) -> None:
        super().__init__(message)
        self.ctx = context


class DatabaseUserError(RuntimeError):
    """Indicates that a database operation failed due to an error on the user's end.

    The error could be due to an SQL syntax error, access privilege violation, etc.

    Parameters
    ----------
    message : str, optional
        A textual description of the error, e.g. *no such table*. Can be left empty by default.
    context : Optional[object], optional
        Additional context information for when the error occurred, e.g. the query that caused the error. Mainly
        intended for debugging purposes.
    """

    def __init__(self, message: str = "", context: Optional[object] = None) -> None:
        super().__init__(message)
        self.ctx = context
