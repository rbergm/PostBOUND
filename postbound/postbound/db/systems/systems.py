from __future__ import annotations

import abc

from postbound.db import db
from postbound.db.hints import provider as hint_provider
from postbound.qal import qal


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
    pass


class MySql(DatabaseSystem):
    pass
