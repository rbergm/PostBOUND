from __future__ import annotations

import abc
import enum
import numbers
import typing
from typing import Iterable

from postbound.qal import base, qal

_T = typing.TypeVar("_T")


class MathematicalSqlOperators(enum.Enum):
    Add = "+"
    Subtract = "-"
    Multiply = "*"
    Divide = "/"
    Modulo = "%"
    Negate = "-"


class LogicalSqlOperators(enum.Enum):
    Equal = "="
    NotEqual = "<>"
    Less = "<"
    LessEqual = "<="
    Greater = ">"
    GreaterEqual = ">="
    Like = "LIKE"
    NotLike = "NOT LIKE"
    ILike = "ILIKE"
    NotILike = "NOT ILIKE"
    In = "IN"
    Exists = "IS NULL"
    Missing = "IS NOT NULL"
    Between = "BETWEEN"


class LogicalSqlCompoundOperators(enum.Enum):
    And = "AND"
    Or = "OR"
    Not = "NOT"


SqlOperator = typing.Type[MathematicalSqlOperators | LogicalSqlOperators | LogicalSqlCompoundOperators]


class SqlExpression(abc.ABC):

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced by this expression."""
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides all columns that are referenced by this predicate.

        If a column is referenced multiple times, it is also returned multiple times.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterchildren(self) -> Iterable[SqlExpression]:
        """Provides unified access to all child expressions of the concrete expression type.

        For "leaf" expressions such as static values, the iterable will not contain any elements. Otherwise, all
        _direct_ children will be returned. For example, a mathematical expression could return both the left, as well
        as the right operand.
        """
        raise NotImplementedError

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are accessed by this expression."""
        return {column.table for column in self.columns() if column.is_bound()}

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class StaticValueExpression(SqlExpression, typing.Generic[_T]):
    def __init__(self, value: _T) -> None:
        self.value = value

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.value == other.value

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return f"{self.value}" if isinstance(self.value, numbers.Number) else f"'{self.value}'"


class CastExpression(SqlExpression):
    def __init__(self, expression: SqlExpression, target_type: str) -> None:
        self.casted_expression = expression
        self.target_type = target_type

    def columns(self) -> set[base.ColumnReference]:
        return self.casted_expression.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.casted_expression.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.casted_expression]

    def __hash__(self) -> int:
        return hash(tuple([self.casted_expression, self.target_type]))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.casted_expression == other.casted_expression
                and self.target_type == other.target_type)

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return f"{self.casted_expression}::{self.target_type}"


class MathematicalExpression(SqlExpression):
    def __init__(self, operator: SqlOperator, first_argument: SqlExpression,
                 second_argument: SqlExpression | list[SqlExpression] | None = None) -> None:
        self.operator = operator
        self.first_arg = first_argument
        self.second_arg = second_argument

        if isinstance(self.second_arg, list) and len(self.second_arg) == 1:
            self.second_arg = self.second_arg[0]

    def columns(self) -> set[base.ColumnReference]:
        all_columns = set(self.first_arg.columns())
        if isinstance(self.second_arg, list):
            for expression in self.second_arg:
                all_columns |= expression.columns()
        elif isinstance(self.second_arg, SqlExpression):
            all_columns |= self.second_arg.columns()
        return all_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return list(self.first_arg.itercolumns()) + self.second_arg.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.first_arg, self.second_arg]

    def __hash__(self) -> int:
        return hash(tuple([self.operator, self.first_arg, self.second_arg]))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.operator == other.operator
                and self.first_arg == other.first_arg
                and self.second_arg == other.second_arg)

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        operator_str = self.operator.value
        if self.operator == MathematicalSqlOperators.Negate:
            return f"{operator_str}{self.first_arg}"
        if isinstance(self.second_arg, list):
            all_args = [self.first_arg] + self.second_arg
            return operator_str.join(str(arg) for arg in all_args)
        return f"{self.first_arg} {operator_str} {self.second_arg}"


class ColumnExpression(SqlExpression):
    def __init__(self, column: base.ColumnReference) -> None:
        self.column = column

    def columns(self) -> set[base.ColumnReference]:
        return {self.column}

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return [self.column]

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def __hash__(self) -> int:
        return hash(self.column)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.column == other.column

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return str(self.column)


class FunctionExpression(SqlExpression):
    def __init__(self, function: str, arguments: list[SqlExpression] | None = None, *, distinct: bool = False) -> None:
        self.function = function
        self.arguments = [] if arguments is None else arguments
        self.distinct = distinct

    def columns(self) -> set[base.ColumnReference]:
        all_columns = set()
        for arg in self.arguments:
            all_columns |= arg.columns()
        return all_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        all_columns = []
        for arg in self.arguments:
            all_columns.extend(arg.itercolumns())
        return all_columns

    def iterchildren(self) -> Iterable[SqlExpression]:
        return list(self.arguments)

    def __hash__(self) -> int:
        return hash(tuple([self.function, self.distinct] + self.arguments))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.function == other.function
                and self.arguments == other.arguments
                and self.distinct == other.distinct)

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        function_str = self.function.upper()
        args_str = ", ".join(str(arg) for arg in self.arguments)
        distinct_str = "DISTINCT " if self.distinct else ""
        return f"{function_str}({distinct_str}{args_str})"


class SubqueryExpression(SqlExpression):
    def __init__(self, subquery: qal.SqlQuery) -> None:
        self.query = subquery

    def columns(self) -> set[base.ColumnReference]:
        return self.query.predicates().root().columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.query.predicates().root().itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def __hash__(self) -> int:
        return hash(self.query)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.query == other.query

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return f"({self.query})"


class StarExpression(SqlExpression):
    def columns(self) -> set[base.ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def __hash__(self) -> int:
        return hash("*")

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "*"
