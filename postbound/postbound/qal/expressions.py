"""Models all supported SQL expressions. Predicates and clauses are located in separate modules.

See the package description for more details on how these concepts are related. The`SqlExpression` provides a
high-level introduction into the structure of different expressions.
"""
from __future__ import annotations

import abc
import enum
import numbers
import typing
from collections.abc import Iterable, Sequence
from typing import Union, Optional

from postbound.qal import base, qal
from postbound.util import collections as collection_utils

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

    def __init__(self, hash_val: int):
        self._hash_val = hash_val

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

    def __hash__(self) -> int:
        return self._hash_val

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
        self._value = value
        super().__init__(hash(value))

    @property
    def value(self) -> T:
        """Get the value."""
        return self._value

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.value == other.value

    def __str__(self) -> str:
        return f"{self.value}" if isinstance(self.value, numbers.Number) else f"'{self.value}'"


class CastExpression(SqlExpression):
    """An expression that casts the type of another nested expression.

    Note that PostBOUND itself does not know about the semantics of the actual types or casts. Eventual errors due to
    illegal casts are only caught at runtime by the actual database system.
    """

    def __init__(self, expression: SqlExpression, target_type: str) -> None:
        if not expression or not target_type:
            raise ValueError("Expression and target type are required")
        self._casted_expression = expression
        self._target_type = target_type

        hash_val = hash((self._casted_expression, self._target_type))
        super().__init__(hash_val)

    @property
    def casted_expression(self) -> SqlExpression:
        """Get the expression that is being casted."""
        return self._casted_expression

    @property
    def target_type(self) -> str:
        """Get the type to which to cast to."""
        return self._target_type

    def columns(self) -> set[base.ColumnReference]:
        return self.casted_expression.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.casted_expression.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.casted_expression]

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.casted_expression == other.casted_expression
                and self.target_type == other.target_type)

    def __str__(self) -> str:
        return f"CAST({self.casted_expression} AS {self.target_type})"


class MathematicalExpression(SqlExpression):
    """
    A mathematical expression computes a result value based on an arbitrary expression, an operator and potentially
    a number of additional expressions/arguments.

    The precise representation of mathematical expressions is not tightly standardized by PostBOUND and there will be
    multiple ways to represent the same expression.

    For example, the expression `R.a + S.b + 42` could be modeled as a single expression object with `R.a` as first
    argument and the sequence `S.b, 42` as second arguments. At the same time, the mathematical expression can also be
    used to represent logical expressions such as `R.a < 42` or `S.b IN (1, 2, 3)`. However, this should be used
    sparingly since logical expressions can be considered as predicates which are handled in the dedicated `predicates`
    package. Moving logical expressions into a mathematical expression object can break correct functionality in that
    package (e.g. determining joins and filters in a query).
    """

    def __init__(self, operator: SqlOperator, first_argument: SqlExpression,
                 second_argument: SqlExpression | Sequence[SqlExpression] | None = None) -> None:
        if not operator or not first_argument:
            raise ValueError("Operator and first argument are required!")
        self._operator = operator
        self._first_arg = first_argument
        self._second_arg: SqlExpression | tuple[SqlExpression] | None = (tuple(second_argument)
                                                                         if isinstance(second_argument, Sequence)
                                                                         else second_argument)

        if isinstance(self._second_arg, tuple) and len(self._second_arg) == 1:
            self._second_arg = self._second_arg[0]

        hash_val = hash((self._operator, self._first_arg, self._second_arg))
        super().__init__(hash_val)

    @property
    def operator(self) -> SqlOperator:
        """Get the operation to combine the input value(s)."""
        return self._operator

    @property
    def first_arg(self) -> SqlExpression:
        """Get the first argument to the operator. This is always specified."""
        return self._first_arg

    @property
    def second_arg(self) -> SqlExpression | Sequence[SqlExpression] | None:
        """Get the second argument to the operator.

        Depending on the operator, this can be a single expression (the most common case), but also a sequence of
        expressions (e.g. sum of multiple values) or no value at all (e.g. negation).
        """
        return self._second_arg

    def columns(self) -> set[base.ColumnReference]:
        all_columns = set(self.first_arg.columns())
        if isinstance(self.second_arg, list):
            for expression in self.second_arg:
                all_columns |= expression.columns()
        elif isinstance(self.second_arg, SqlExpression):
            all_columns |= self.second_arg.columns()
        return all_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        first_columns = list(self.first_arg.itercolumns())
        if not self.second_arg:
            return first_columns
        second_columns = (collection_utils.flatten(sub_arg.itercolumns() for sub_arg in self.second_arg)
                          if isinstance(self.second_arg, tuple) else list(self.second_arg.itercolumns()))
        return first_columns + second_columns

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.first_arg, self.second_arg]

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.operator == other.operator
                and self.first_arg == other.first_arg
                and self.second_arg == other.second_arg)

    def __str__(self) -> str:
        operator_str = self.operator.value
        if self.operator == MathematicalSqlOperators.Negate:
            return f"{operator_str}{self.first_arg}"
        if isinstance(self.second_arg, tuple):
            all_args = [self.first_arg] + list(self.second_arg)
            return operator_str.join(str(arg) for arg in all_args)
        return f"{self.first_arg} {operator_str} {self.second_arg}"


class ColumnExpression(SqlExpression):
    """A column expression wraps the reference to a column."""

    def __init__(self, column: base.ColumnReference) -> None:
        self._column = column
        super().__init__(hash(self._column))

    @property
    def column(self) -> base.ColumnReference:
        """Get the column that is wrapped by this expression."""
        return self._column

    def columns(self) -> set[base.ColumnReference]:
        return {self.column}

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return [self.column]

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.column == other.column

    def __str__(self) -> str:
        return str(self.column)


AggregateFunctions = {"COUNT", "SUM", "MIN", "MAX", "AVG"}
"""All aggregate functions specified in standard SQL."""


class FunctionExpression(SqlExpression):
    """The function expression indicates a call to an arbitrary function.

    The actual function might be one of the standard SQL functions, an aggregation function or a user-defined one.
    PostBOUND does not make any difference here.
    """

    def __init__(self, function: str, arguments: Optional[Sequence[SqlExpression]] = None, *,
                 distinct: bool = False) -> None:
        if not function:
            raise ValueError("Function is required")
        self._function = function.upper()
        self._arguments: tuple[SqlExpression] = () if arguments is None else tuple(arguments)
        self._distinct = distinct

        hash_val = hash((self._function, self._distinct, self._arguments))
        super().__init__(hash_val)

    @property
    def function(self) -> str:
        """Get the function name."""
        return self._function

    @property
    def arguments(self) -> Sequence[SqlExpression]:
        """Get all arguments that are supplied to the function."""
        return self._arguments

    @property
    def distinct(self) -> bool:
        """Get whether the function should only operate on distinct values.

        Whether this makes any sense for the function at hand is entirely dependend on the specific function and not
        enfored by PostBOUND. The runtime DBS has to check this.
        """
        return self._distinct

    def is_aggregate(self) -> bool:
        """Checks, whether the function is a well-known SQL aggregation function.

        Only standard functions are considered (e.g. no CORR for computing correlations).
        """
        return self._function.upper() in AggregateFunctions

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

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.function == other.function
                and self.arguments == other.arguments
                and self.distinct == other.distinct)

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._arguments)
        distinct_str = "DISTINCT " if self._distinct else ""
        return f"{self._function}({distinct_str}{args_str})"


class SubqueryExpression(SqlExpression):
    """A subquery expression wraps an arbitrary subquery."""

    def __init__(self, subquery: qal.SqlQuery) -> None:
        self._query = subquery
        super().__init__(hash(subquery))

    @property
    def query(self) -> qal.SqlQuery:
        """The (sub-) query that is wrapped by this expression."""
        return self._query

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

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.query == other.query

    def __str__(self) -> str:
        query_str = str(self.query).removesuffix(";")
        return f"({query_str})"


class StarExpression(SqlExpression):
    """A special expression used in SELECT clauses to select all columns."""

    def __init__(self) -> None:
        super().__init__(hash("*"))

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self))

    def __str__(self) -> str:
        return "*"


def as_expression(value: object) -> SqlExpression:
    """Transforms the given value into the most appropriate `SqlExpression` instance.

    The following rules are applied:

    - `ColumnReference` becomes `ColumnExpression`
    - `SqlQuery` becomes `SubqueryExpression`
    - the star-string `*` becomes a `StarExpression`

    All other values become a `StaticValueExpression`.
    """
    if isinstance(value, SqlExpression):
        return value

    if isinstance(value, base.ColumnReference):
        return ColumnExpression(value)
    elif isinstance(value, qal.SqlQuery):
        return SubqueryExpression(value)

    if value == "*":
        return StarExpression()
    return StaticValueExpression(value)
