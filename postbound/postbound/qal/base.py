"""Fundamental types for the query abstraction layer."""

import abc
import numbers
import typing
from dataclasses import dataclass
from typing import Union

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
