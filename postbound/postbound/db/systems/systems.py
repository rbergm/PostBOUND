from __future__ import annotations

import abc
import typing
from typing import Generic, Type

from postbound.db import db, postgres as pg_db
from postbound.db.hints import provider as hint_provider, postgres_provider
from postbound.qal import qal, formatter, transform
from postbound.optimizer.physops import operators as physops

DatabaseType = typing.TypeVar("DatabaseType", bound=db.Database)


class DatabaseSystem(abc.ABC, Generic[DatabaseType]):
    """A `DatabaseSystem` is designed as a "one-stop-shop" to supply optimized queries to a database.

    This expands upon the `Database` interface which focuses on the interaction with the database with two important
    methods: `query_adaptor` provides the appropriate conversion to ensure that an optimized query is executed as
    intended by the PostBOUND plan. `format_query` transforms a query object into a string that can be executed by
    the database system. This second method is necessary to work with deviations from standard SQL (e.g. single vs.
    double-quoted string values in MySQL).

    All instances of this class are expected to have a constructor that takes exactly one argument: the `Database`
    that they should connect to.
    """

    def __init__(self, database: DatabaseType):
        self.database = database

    def interface(self) -> DatabaseType:
        """Provides access to the actual database connection."""
        return self.database

    @abc.abstractmethod
    def query_adaptor(self) -> hint_provider.HintProvider:
        """Provides access to the hint generation and query preparation service for this database system.

        This service is intended to ensure that the optimized join order and operator selection are actually applied
        to the query by generating appropriate hints and transforming the query in other system-specific ways.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def format_query(self, query: qal.SqlQuery) -> str:
        """Transforms the query into a databases-specific string.

        This is mostly done to incorporate deviations from standard SQL notation.

        Notice that the query transformation does not include hints since these have to be evaluated in a
        cursor-specific way. This is acceptable since the individual hint blocks are expected to be system-specific
        anyway.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def supports_hint(self, hint: physops.PhysicalOperator | physops.HintType) -> bool:
        """Checks, whether the database system is capable of using the specified operator."""
        raise NotImplementedError


_DB_SYS_REGISTRY: DatabaseSystem | None = None


class DatabaseSystemRegistry:
    @staticmethod
    def load_system_for(database: db.Database) -> DatabaseSystem:
        registry = DatabaseSystemRegistry._get_instance()
        target_system = registry.entries[type(database)]
        return object.__new__(target_system, database)

    @staticmethod
    def register_system(database: Type[db.Database], system: Type[DatabaseSystem]) -> None:
        registry = DatabaseSystemRegistry._get_instance()
        registry.entries[database] = system

    @staticmethod
    def _get_instance() -> DatabaseSystemRegistry:
        global _DB_SYS_REGISTRY
        if not _DB_SYS_REGISTRY:
            _DB_SYS_REGISTRY = DatabaseSystemRegistry()
        return _DB_SYS_REGISTRY

    def __init__(self) -> None:
        self.entries: dict[Type[db.Database], Type[DatabaseSystem]] = {}


PG_OPERATORS = {physops.JoinOperators.HashJoin, physops.JoinOperators.NestedLoopJoin,
                physops.JoinOperators.SortMergeJoin,
                physops.ScanOperators.SequentialScan, physops.ScanOperators.IndexScan,
                physops.ScanOperators.IndexOnlyScan}


class Postgres(DatabaseSystem[pg_db.PostgresInterface]):
    """Postgres implementation"""

    def __init__(self, postgres_db: pg_db.PostgresInterface):
        super().__init__(postgres_db)

    def query_adaptor(self) -> hint_provider.HintProvider:
        return postgres_provider.PostgresHintProvider()

    def format_query(self, query: qal.SqlQuery) -> str:
        return formatter.format_quick(transform.drop_hints(query))

    def supports_hint(self, hint: physops.PhysicalOperator) -> bool:
        return hint in PG_OPERATORS


DatabaseSystemRegistry.register_system(pg_db.PostgresInterface, Postgres)


class MySql(DatabaseSystem):
    pass
