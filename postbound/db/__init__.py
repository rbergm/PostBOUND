"""The `db` module provides tools to interact with various database instances.

Generally speaking, the interactions are bidirectional: on the one hand, common database concepts like retrieving statistical
information, introspecting the logical schema or obtaining physical query execution plans are enabled through abstract
interfaces and can be used by the optimizer modules. On the other hand, the `db` modules provides tools to enforce optimization
decisions made as part of an optimization pipeline or other modules when executing the query on the actual database system.

Recall that PostBOUND does not interact with the query optimizer directly and instead relies on system-specific hints or other
special properties of the target database system to influence the optimizer behaviour. Therefore, the optimization process
usually terminates with transforming the original input query to a logically equivalent query that at the same time contains
the necessary modifications for optimization.

The central entrypoint to all database interaction is the abstract `Database` class. This class is inherited by all supported
database systems (currently PostgreSQL and MySQL). Each `Database` instace provides some basic functionality on its own (such
as the ability to execute queries), but delegates most of the work to specific and tailored interfaces. For example, the
`DatabaseSchema` models all access to the logical schema of a database and the `OptimizerInterface` encapsulates the
functionality to retrieve cost estimates or phyiscal query execution plans. All of these interfaces are once again abstract and
implemented according to the specifics of the actual database system.

Take a look at the individual interfaces for further information about their functionality and intended usage.

This module provide direct access to the Postgres interface along with a shortcut method to retrieve the current database
(aptly called `current_database`). In the background, this method delegates to the `DatabasePool`.
If you want to use the MySQL interface, make sure to install PostBOUND with MySQL support enabled and import `mysql` from
the `db` package.
"""

from __future__ import annotations

from . import _duckdb as duckdb
from . import postgres
from ._db import (
    Connection,
    Cursor,
    Database,
    DatabasePool,
    DatabaseSchema,
    DatabaseServerError,
    DatabaseStatistics,
    DatabaseUserError,
    HintService,
    HintWarning,
    OptimizerInterface,
    PrewarmingSupport,
    QueryCacheWarning,
    TimeoutSupport,
    UnsupportedDatabaseFeatureError,
)

__all__ = [
    "postgres",
    "duckdb",
    "Cursor",
    "Connection",
    "PrewarmingSupport",
    "TimeoutSupport",
    "QueryCacheWarning",
    "Database",
    "DatabaseSchema",
    "DatabaseStatistics",
    "HintWarning",
    "HintService",
    "OptimizerInterface",
    "DatabasePool",
    "UnsupportedDatabaseFeatureError",
    "DatabaseServerError",
    "DatabaseUserError",
]


def current_database() -> Database:
    """Provides the current database from the `DatabasePool`.

    Returns
    -------
    Database
        The current database instance. If there is not exactly one database in the pool, a `ValueError` is raised.

    See Also
    --------
    DatabasePool.current_database
    """
    return DatabasePool.current_database()
