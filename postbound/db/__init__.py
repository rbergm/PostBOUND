"""The `db` module provides tools to interact with various database instances.

Generally speaking, the interactions are bidirectional: on the one hand, common database concepts like retrieving
statistical information, introspecting the logical schema or obtaining physical query execution plans are enabled
through abstract interfaces. On the other hand, the `db` modules provides tools to enforce optimization decisions made
as part of an optimization pipeline or other modules when executing the query on the actual database system.

Recall that PostBOUND does not interact with the query optimizer directly and instead relies on system-specific hints
or other special properties of the target database system to influence the optimizer behaviour. Therefore, the
optimization process usually terminates with transforming the original input query to a logically equivalent query that
at the same time contains the necessary modifications for optimization.

The central entrypoint to all database interaction is the abstract `Database` class. This class is inherited by all
supported database systems (currently PostgreSQL and MySQL). Each `Database` instace provides some basic functionality
on its own (such as the ability to execute queries), but delegates most of the work to specific and tailored
interfaces. For example, the `DatabaseSchema` models all access to the logical schema of a database and the
`OptimizerInterface` encapsulates the functionality to retrieve cost estimates or phyiscal query execution plans. All
of these interfaces are once again abstract and implemented according to the specifics of the actual database system.

Take a look at the individual interfaces for further information about their functionality and intended usage.
"""

from ._db import (
    Cursor,
    Connection,
    PrewarmingSupport,
    QueryCacheWarning,
    Database,
    DatabaseSchema,
    DatabaseStatistics,
    HintWarning,
    HintService,
    QueryExecutionPlan,
    read_query_plan_json,
    OptimizerInterface,
    DatabasePool,
    UnsupportedDatabaseFeatureError,
    DatabaseServerError,
    DatabaseUserError
)
from . import postgres

__all__ = [
    "Cursor",
    "Connection",
    "PrewarmingSupport",
    "QueryCacheWarning",
    "Database",
    "DatabaseSchema",
    "DatabaseStatistics",
    "HintWarning",
    "HintService",
    "QueryExecutionPlan",
    "read_query_plan_json",
    "OptimizerInterface",
    "DatabasePool",
    "UnsupportedDatabaseFeatureError",
    "DatabaseServerError",
    "DatabaseUserError",
    "postgres"
]
