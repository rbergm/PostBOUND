from __future__ import annotations

import abc
import numbers
import typing

from postbound.qal import base, qal

_T = typing.TypeVar("_T")


class SqlExpression(abc.ABC):

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced by this expression."""
        raise NotImplementedError

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are accessed by this expression."""
        return {column.table for column in self.columns() if column}

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)


class StaticValueExpression(SqlExpression, typing.Generic[_T]):
    def __init__(self, value: _T) -> None:
        self.value = value

    def columns(self) -> set[base.ColumnReference]:
        return set()

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


_MospMathFunctionsSQL = {"add": "+", "sub": "-", "neg": "-", "mul": "*", "div": "/", "mod": "%"}


class MathematicalExpression(SqlExpression):
    def __init__(self, operator: str, first_argument: SqlExpression,
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
        operator_str = _MospMathFunctionsSQL.get(self.operator, self.operator)
        if self.operator == "neg":
            return f"{operator_str}{self.first_arg}"
        if isinstance(self.second_arg, list):
            all_args = [self.first_arg] + self.second_arg
            return self.operator.join(str(arg) for arg in all_args)
        return f"{self.first_arg} {operator_str} {self.second_arg}"


class ColumnExpression(SqlExpression):
    def __init__(self, column: base.ColumnReference) -> None:
        self.column = column

    def columns(self) -> set[base.ColumnReference]:
        return {self.column}

    def __hash__(self) -> int:
        return hash(self.column)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.column == other.column

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return str(self.column)


class FunctionExpression(SqlExpression):
    def __init__(self, function: str, arguments: list[SqlExpression] | None = None) -> None:
        self.function = function
        self.arguments = [] if arguments is None else arguments

    def columns(self) -> set[base.ColumnReference]:
        all_columns = set()
        for arg in self.arguments:
            all_columns |= arg.columns()
        return all_columns

    def __hash__(self) -> int:
        return hash(tuple([self.function] + self.arguments))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.function == other.function and self.arguments == other.arguments

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        function_str = self.function.upper()
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{function_str}({args_str})"


class SubqueryExpression(SqlExpression):
    def __init__(self, subquery: qal.SqlQuery) -> None:
        self.query = subquery

    def columns(self) -> set[base.ColumnReference]:
        return self.query.predicates().root().columns()

    def __hash__(self) -> int:
        return hash(self.query)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.query == other.query

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return f"({self.query})"
