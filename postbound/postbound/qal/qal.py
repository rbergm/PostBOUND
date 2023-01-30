import abc
import numbers
import typing
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Set, Tuple, Union

import mo_sql_parsing as mosp

from postbound.util import errors, dicts as dict_utils

_T = typing.TypeVar("_T")


class SqlExpression(abc.ABC):
    def __repr__(self) -> str:
        return str(self)


class StaticValueExpression(SqlExpression, typing.Generic[_T]):
    def __init__(self, value: _T) -> None:
        self._value = value

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return f"{self._value}" if isinstance(self._value, numbers.Number) else f"'{self._value}'"


class CastExpression(SqlExpression):
    def __init__(self, expression: SqlExpression, target_type: str) -> None:
        self._casted_expression = expression
        self._target_type = target_type

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return f"{self._casted_expression}::{self._target_type}"


class MathematicalExpression(SqlExpression):
    def __init__(self, operator: str, first_argument: SqlExpression, second_argument: SqlExpression = None) -> None:
        self._operator = operator
        self._first_arg = first_argument
        self._second_arg = second_argument

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return f"{self._first_arg} {self._operator} {self._second_arg}"


class ColumnExpression(SqlExpression):
    def __init__(self, column: "ColumnReference") -> None:
        self._col = column

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return str(self._col)


@dataclass
class TableReference:
    """A table reference represents a database table.

    It can either be a physical table, a CTE, or an entirely virtual query created via subqueries. Note that a table
    reference is indeed just a reference and not a 1:1 "representation" since each table can be sourced multiple times
    in a query. Therefore, in addition to the table name, each instance can optionally also contain an alias to
    distinguish between different references to the same table.
    """
    full_name: str
    alias: str = ""

    def is_virtual(self) -> bool:
        return not self.full_name


@dataclass
class ColumnReference:
    name: str
    table: Union[TableReference, None] = None
    attached_expression: Union[SqlExpression, None] = None


class SqlQuery(abc.ABC):
    def __init__(self, mosp_data: dict) -> None:
        self._mosp_data = mosp_data

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        raise NotImplementedError()

    def is_explicit(self) -> bool:
        return not self.is_implicit()

    @abc.abstractmethod
    def tables(self) -> Iterable[TableReference]:
        """Provides all tables that are referenced in the query."""
        raise NotImplementedError

    @abc.abstractmethod
    def predicates(self) -> "QueryPredicates":
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
class BaseColumnProjection:
    column: ColumnReference
    target_name: str


class QueryProjection:
    pass


__ReflexiveOps = ["=", "!=", "<>"]
__MospCompoundOperations = {"and", "or", "not"}


class NoJoinPredicateError(errors.StateError):
    def __init__(self, msg: str = ""):
        super().__init__(msg)


class NoFilterPredicateError(errors.StateError):
    def __init__(self, msg: str = ""):
        super().__init__(msg)


class AbstractPredicate(abc.ABC):
    def __init__(self, mosp_data: dict) -> None:
        self._mosp_data = mosp_data

    @abc.abstractmethod
    def is_compound(self) -> bool:
        """Checks, whether this predicate is a conjunction/disjunction/negation of base predicates."""
        raise NotImplementedError

    def is_base(self) -> bool:
        """Checks, whether this predicate is a base predicate i.e. not a conjunction/disjunction/negation."""
        return not self.is_compound()

    @abc.abstractmethod
    def is_join(self) -> bool:
        """Checks, whether this predicate describes a join between two tables."""
        raise NotImplementedError

    def is_filter(self) -> bool:
        """Checks, whether this predicate is a filter on a base table rather than a join of base tables."""
        return not self.is_join()

    @abc.abstractmethod
    def collect_columns(self) -> Set[ColumnReference]:
        """Provides all columns that are referenced by this predicate."""
        raise NotImplementedError

    def collect_tables(self) -> Set[ColumnReference]:
        """Provides all tables that are accessed by this predicate."""
        return {attribute.table for attribute in self.collect_columns()}

    def contains_table(self, table: TableReference) -> bool:
        """Checks, whether this predicate filters or joins a column of the given table."""
        return any(table == tab for tab in self.collect_tables())

    def joins_table(self, table: TableReference) -> bool:
        """Checks, whether this predicate describes a join and one of the join partners is the given table."""
        return self.is_join() and self.contains_table(table)

    def columns_of(self, table: TableReference) -> Set[ColumnReference]:
        """Retrieves all columns of the given table that are referenced by this predicate."""
        return {attr for attr in self.collect_columns() if attr.table == table}

    @abc.abstractmethod
    def join_partners_of(self, table: TableReference) -> Set[ColumnReference]:
        """Retrieves all columns that are joined with the given table.

        If this predicate is not a join, an error will be raised.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def join_partners(self) -> Set[Tuple[ColumnReference, ColumnReference]]:
        """Provides all pairs of columns that are joined within this predicate.

        If this predicate is not a join, an error will be raised.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def base_predicates(self) -> Iterable["AbstractPredicate"]:
        """Provides all base predicates that form this predicate.

        This is most usefull to iterate over all leaves of a compound predicate, for base predicates it simply returns
        the predicate itself.
        """
        raise NotImplementedError

    def _assert_join_predicate(self) -> None:
        """Raises a `NoJoinPredicateError` if `self` is not a join."""
        if not self.is_join():
            raise NoJoinPredicateError

    def _assert_filter_predicate(self) -> None:
        """Raises a `NoFilterPredicateError` if `self` is not a filter."""
        if not self.is_filter():
            raise NoFilterPredicateError

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, __o: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class BasePredicate(AbstractPredicate, abc.ABC):
    def __init__(self, mosp_data: dict) -> None:
        super().__init__(mosp_data)

        self._operation = dict_utils.key(mosp_data)


class BinaryPredicate(BasePredicate):
    pass


class UnaryPredicate(BasePredicate):
    pass


class CompoundPredicate:
    pass


class QueryPredicates:
    def get(self) -> AbstractPredicate:
        pass

    def filters(self) -> Iterable[AbstractPredicate]:
        pass

    def joins(self) -> Iterable[AbstractPredicate]:
        pass


class QueryFormatError(RuntimeError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def parse_table(table: str) -> TableReference:
    pass


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
