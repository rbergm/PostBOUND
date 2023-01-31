"""`qal` is the query abstraction layer.

It contains classes and methods to conveniently work with different parts of SQL queries.
"""

import abc
from dataclasses import dataclass
from collections.abc import Iterable
from typing import List

import mo_sql_parsing as mosp

from postbound.qal import base, predicates as preds


class SqlQuery(abc.ABC):
    def __init__(self, mosp_data: dict) -> None:
        self._mosp_data = mosp_data

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        raise NotImplementedError()

    def is_explicit(self) -> bool:
        return not self.is_implicit()

    @abc.abstractmethod
    def tables(self) -> Iterable[base.TableReference]:
        """Provides all tables that are referenced in the query."""
        raise NotImplementedError

    @abc.abstractmethod
    def predicates(self) -> preds.QueryPredicates:
        """Provides all predicates in this query."""
        raise NotImplementedError


class ImplicitSqlQuery(SqlQuery):
    def __init__(self, mosp_data: dict) -> None:
        super().__init__(mosp_data)

    def is_implicit(self) -> bool:
        return True


class ExplicitSqlQuery(SqlQuery):
    def __init__(self, mosp_data: dict) -> None:
        super().__init__(mosp_data)

    def is_implicit(self) -> bool:
        return False


@dataclass
class BaseProjection:
    expression: base.SqlExpression
    target_name: str


class QueryProjection:
    def parts(self) -> List[BaseProjection]:
        pass


class QueryFormatError(RuntimeError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)



def __is_implicit_query(mosp_data: dict) -> bool:
    pass


def __is_explicit_query(mosp_data: dict) -> bool:
    pass


def parse_query(raw_query: str) -> SqlQuery:
    mosp_data = mosp.parse(raw_query)
    if __is_implicit_query(mosp_data):
        return ImplicitSqlQuery(mosp_data)
    elif __is_explicit_query(mosp_data):
        return ExplicitSqlQuery(mosp_data)
    else:
        raise QueryFormatError("Query must be either implicit or explicit, not a mixture of both: " + raw_query)

def parse_table(table: str) -> base.TableReference:
    pass
