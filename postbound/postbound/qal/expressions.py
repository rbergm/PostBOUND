"""Models all supported SQL expressions. Predicates and clauses are located in separate modules.

See the package description for more details on how these concepts are related. The`SqlExpression` provides a
high-level introduction into the structure of different expressions.
"""
from __future__ import annotations

import abc
import enum
import numbers
import typing
from typing import Iterable, Union

from postbound.qal import base, qal

T = typing.TypeVar("T")
"""Typed expressions use this generic type variable."""


class MathematicalSqlOperators(enum.Enum):
    """The supported mathematical operators."""
    Add = "+"
    Subtract = "-"
    Multiply = "*"
    Divide = "/"
    Modulo = "%"
    Negate = "-"


class LogicalSqlOperators(enum.Enum):
    """The supported unary and binary operators."""
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
    """The supported compound operators."""
    And = "AND"
    Or = "OR"
    Not = "NOT"


SqlOperator = Union[MathematicalSqlOperators, LogicalSqlOperators, LogicalSqlCompoundOperators]
"""Captures all different kinds of operators in one type."""


class SqlExpression(abc.ABC):
    """Base class for all expressions.

    Expressions can be inserted in many different places in a SQL query. For example, a SELECT clause produces columns
    such as in `SELECT R.a FROM R`, but it can also modify the column values slightly, such as in
    `SELECT R.a + 42 FROM R`. To account for all  these different situations, the `SqlExpression` is intended to form
    a hierarchical trees and chains of expressions.

    For example, a complicated expressions such as `my_udf(R.a::interval + 42)` consisting of a user-defined function,
    a value cast and a mathematical operation is represented the following way:
    `FunctionExpression(MathematicalExpression(CastExpression(ColumnExpression), StaticValueExpression))`. The methods
    provided by all expression instances enable a more convenient use and access to the expression hierarchies.

    The different kinds of expressions are represented using different subclasses of the `SqlExpression` interface.
    """

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


class StaticValueExpression(SqlExpression, typing.Generic[T]):
    """An expression that wraps a literal/static value."""

    def __init__(self, value: T) -> None:
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
    """An expression that casts the type of another nested expression.

    Note that PostBOUND itself does not know about the semantics of the actual types or casts. Eventual errors due to
    illegal casts are only caught at runtime by the actual database system.
    """

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
    """
    A mathematical expression computes a result value based on an arbitrary expression, an operator and potentially
    a number of additional expressions/arguments.
    """

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
    """A column expression wraps the reference to a column."""

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
    """The function expression indicates a call to an arbitrary function.

    The actual function might be one of the standard SQL functions, an aggregation function or a user-defined one.
    PostBOUND does not make any difference here.
    """

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
    """A subquery expression wraps an arbitrary subquery."""

    def __init__(self, subquery: qal.SqlQuery) -> None:
        self.query = subquery

    def columns(self) -> set[base.ColumnReference]:
        predicates = self.query.predicates()
        return predicates.root().columns() if predicates else set()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        predicates = self.query.predicates()
        return predicates.root().itercolumns() if predicates else []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def tables(self) -> set[base.TableReference]:
        return self.query.tables()

    def __hash__(self) -> int:
        return hash(self.query)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.query == other.query

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        query_str = str(self.query).removesuffix(";")
        return f"({query_str})"


class StarExpression(SqlExpression):
    """A special expression used in SELECT clauses to select all columns."""

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
