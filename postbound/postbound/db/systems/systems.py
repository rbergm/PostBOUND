from __future__ import annotations

import abc

from postbound.db import db, postgres as pg_db
from postbound.db.hints import provider as hint_provider
from postbound.qal import qal, format


class DatabaseSystem(abc.ABC):

    @abc.abstractmethod
    def interface(self) -> db.Database:
        raise NotImplementedError

    @abc.abstractmethod
    def query_adaptor(self) -> hint_provider.HintProvider:
        raise NotImplementedError

    @abc.abstractmethod
    def format_query(self, query: qal.SqlQuery) -> str:
        raise NotImplementedError


class Postgres(DatabaseSystem):

    def __init__(self, postgres_db: pg_db.PostgresInterface):
        self.postgres_db = postgres_db

    def interface(self) -> db.Database:
        return self.postgres_db

    def query_adaptor(self) -> hint_provider.HintProvider:
        return hint_provider.PostgresHintProvider()

    def format_query(self, query: qal.SqlQuery) -> str:
        return format.format_quick(query)


class MySql(DatabaseSystem):
    pass
