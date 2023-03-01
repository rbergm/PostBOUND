from __future__ import annotations

import abc

from postbound.db import db, postgres as pg_db
from postbound.db.hints import provider as hint_provider, postgres_provider
from postbound.qal import qal, formatter, transform


class DatabaseSystem(abc.ABC):

    @abc.abstractmethod
    def interface(self) -> db.Database:
        raise NotImplementedError

    @abc.abstractmethod
    def query_adaptor(self) -> hint_provider.HintProvider:
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


class Postgres(DatabaseSystem):

    def __init__(self, postgres_db: pg_db.PostgresInterface):
        self.postgres_db = postgres_db

    def interface(self) -> db.Database:
        return self.postgres_db

    def query_adaptor(self) -> hint_provider.HintProvider:
        return postgres_provider.PostgresHintProvider()

    def format_query(self, query: qal.SqlQuery) -> str:
        return formatter.format_quick(transform.drop_hints(query))


class MySql(DatabaseSystem):
    pass
