
from __future__ import annotations

import abc
import collections
import enum
import functools
import itertools
import numbers
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from typing import Generic, Literal, Optional, Type, TypeVar, Union

import networkx as nx

from .. import util
from .._core import TableReference
from ..util import StateError

T = TypeVar("T")
"""Typed expressions use this generic type variable."""


VisitorResult = TypeVar("VisitorResult")
"""Result of visitor invocations."""


class ColumnReference:
    """A column reference represents a specific column of a specific database table.

    This reference always consists of the name of the "physical" column (see below for special cases). In addition,
    each column can be bound to the table to which it belongs by providing the associated table reference.

    Column references can be sorted lexicographically and are designed as immutable data objects.

    Parameters
    ----------
    name : str
        The name of the column. Cannot be empty.
    table : Optional[TableReference], optional
        The table which provides the column. Can be ``None`` if the table is unknown.

    Raises
    ------
    ValueError
        If the name is empty (or ``None``)

    Notes
    -----
    A number of special cases arise when dealing with subqueries and common table expressions. The first one is the
    fact that columns can be bound to virtual tables, e.g. if they are exported by subqueries, etc. In the same vein,
    columns also do not always need to refer directly to physical columns. Consider the following example query:

    ::

        WITH cte_table AS (SELECT foo.f_id, foo.a + foo.b AS 'sum' FROM foo)
        SELECT *
        FROM bar JOIN cte_table ON bar.b_id = cte_table.f_id
        WHERE cte_table.sum < 42

    In this case, the CTE exports a column ``sum`` that is constructed based on two "actual" columns. Hence, the sum
    column itself does not have any physical representation but will be modelled as a column reference nevertheless.
    """

    def __init__(self, name: str, table: Optional[TableReference] = None) -> None:
        if not name:
            raise ValueError("Column name is required")
        self._name = name
        self._table = table
        self._normalized_name = self.name.lower()
        self._hash_val = hash((self._normalized_name, self._table))

    @property
    def name(self) -> str:
        """Get the name of this column. This is guaranteed to be set and will never be empty

        Returns
        -------
        str
            The name
        """
        return self._name

    @property
    def table(self) -> Optional[TableReference]:
        """Get the table to which this column belongs, if specified.

        Returns
        -------
        Optional[TableReference]
            The table or ``None``. The table can be an arbitrary reference, i.e. virtual or physical.
        """
        return self._table

    def is_bound(self) -> bool:
        """Checks, whether this column is bound to a table.

        Returns
        -------
        bool
            Whether a valid table reference is set
        """
        return self.table is not None

    def belongs_to(self, table: TableReference) -> bool:
        """Checks, whether the column is part of the given table.

        This check does not consult the schema of the actual database or the like, it merely matches the given table
        reference with the `table` attribute of this column.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the table's column is the same as the given one
        """
        return table == self.table

    def bind_to(self, table: TableReference) -> ColumnReference:
        """Binds this column to a new table.

        Parameters
        ----------
        table : TableReference
            The new table

        Returns
        -------
        ColumnReference
            The updated column reference, the original reference is not modified.
        """
        return ColumnReference(self.name, table)

    def __json__(self) -> object:
        return {"name": self._name, "table": self._table}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ColumnReference):
            return NotImplemented
        if self.table == other.table:
            return self._normalized_name < other._normalized_name
        if not self.table:
            return True
        if not other.table:
            return False
        return self.table < other.table

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._normalized_name == other._normalized_name and self.table == other.table

    def __repr__(self) -> str:
        return f"ColumnReference(name='{self.name}', table={repr(self.table)})"

    def __str__(self) -> str:
        if self.table and self.table.alias:
            return f"{self.table.alias}.{self.name}"
        elif self.table and self.table.full_name:
            return f"{self.table.full_name}.{self.name}"
        return self.name


class UnboundColumnError(StateError):
    """Indicates that a column is required to be bound to a table, but the provided column was not.

    Parameters
    ----------
    column : ColumnReference
        The column without the necessary table binding
    """

    def __init__(self, column: ColumnReference) -> None:
        super().__init__("Column is not bound to any table: " + str(column))
        self.column = column


class VirtualTableError(StateError):
    """Indicates that a table is required to correspond to a physical table, but the provided reference was not.

    Parameters
    ----------
    table : TableReference
        The virtual table
    """

    def __init__(self, table: TableReference) -> None:
        super().__init__("Table is virtual: " + str(table))
        self.table = table


class MathematicalSqlOperators(enum.Enum):
    """The supported mathematical operators."""
    Add = "+"
    Subtract = "-"
    Multiply = "*"
    Divide = "/"
    Modulo = "%"
    Negate = "-"


class LogicalSqlOperators(enum.Enum):
    """The supported unary and binary operators.

    Notice that the predicates which make heavy use of these operators are specified in the `predicates` module.
    """
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
    Exists = "EXISTS"
    Is = "IS"
    IsNot = "IS NOT"
    Between = "BETWEEN"


UnarySqlOperators = frozenset({LogicalSqlOperators.Exists, LogicalSqlOperators.Is, LogicalSqlOperators.IsNot})
"""The `LogicalSqlOperators` that can be used as unary operators."""


class CompoundOperators(enum.Enum):
    """The supported compound operators.

    Notice that predicates which make heavy use of these operators are specified in the `predicates` module.
    """
    And = "AND"
    Or = "OR"
    Not = "NOT"


SqlOperator = Union[MathematicalSqlOperators, LogicalSqlOperators, CompoundOperators]
"""Captures all different kinds of operators in one type."""


class SqlExpression(abc.ABC):
    """Base class for all expressions.

    Expressions form one of the central building blocks of representing a SQL query in the QAL. They specify how values
    from different columns are modified and combined, thereby forming larger (hierarchical) structures.

    Expressions can be inserted in many different places in a SQL query. For example, a ``SELECT`` clause produces
    columns such as in ``SELECT R.a FROM R``, but it can also modify the column values slightly, such as in
    ``SELECT R.a + 42 FROM R``. To account for all  these different situations, the `SqlExpression` is intended to form
    hierarchical trees and chains of expressions. In the first case, a `ColumnExpression` is used, whereas a
    `MathematicalExpression` can model the second case. Whereas column expressions represent leaves in the expression
    tree, mathematical expressions are intermediate nodes.

    As a more advanced example, a complicated expressions such as `my_udf(R.a::interval + 42)` which consists of a
    user-defined function, a value cast and a mathematical operation is represented the following way:
    `FunctionExpression(MathematicalExpression(CastExpression(ColumnExpression), StaticValueExpression))`. The methods
    provided by all expression instances enable a more convenient use and access to the expression hierarchies.

    The different kinds of expressions are represented using different subclasses of the `SqlExpression` interface.
    This really is an abstract interface, not a usable expression. All inheriting expression have to provide their own
    `__eq__` method and re-use the `__hash__` method provided by the base expression. Remember to explicitly set this
    up! The concrete hash value is constant since the clause itself is immutable. It is up to the implementing class to
    make sure that the equality/hash consistency is enforced.

    Parameters
    ----------
    hash_val : int
        The hash of the concrete expression object
    """

    def __init__(self, hash_val: int):
        self._hash_val = hash_val

    @abc.abstractmethod
    def tables(self) -> set[TableReference]:
        """Provides all tables that are accessed by this expression.

        Returns
        -------
        set[TableReference]
            All tables. This includes virtual tables if such tables are present in the expression.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced by this expression.

        Returns
        -------
        set[ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides all columns that are referenced by this expression.

        If a column is referenced multiple times, it is also returned multiple times.

        Returns
        -------
        Iterable[ColumnReference]
            All columns in exactly the order in which they are used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterchildren(self) -> Iterable[SqlExpression]:
        """Provides unified access to all child expressions of the concrete expression type.

        For *leaf* expressions such as static values, the iterable will not contain any elements. Otherwise, all
        *direct* children will be returned. For example, a mathematical expression could return both the left, as well
        as the right operand. This allows for access to nested expressions in a recursive manner.

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        """Enables processing of the current expression by an expression visitor.

        Parameters
        ----------
        visitor : SqlExpressionVisitor[VisitorResult]
            The visitor
        """
        raise NotImplementedError

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


class StaticValueExpression(SqlExpression, Generic[T]):
    """An expression that wraps a literal/static value.

    This is one of the leaf expressions that does not contain any further child expressions.

    **NULL** values are represented by **None**, you can use `StaticValueExpression.null()` to create such an expression and
    `is_null()` to check whether the value is **NULL**.

    Parameters
    ----------
    value : T
        The value that is wrapped by the expression

    Examples
    --------
    Consider the following SQL query: ``SELECT * FROM R WHERE R.a = 42``. In this case the comparison value of 42 will
    be represented as a static value expression. The reference to the column ``R.a`` cannot be a static value since its
    values depend on the actual column values. Hence, a `ColumnExpression` is used for it.
    """

    @staticmethod
    def null() -> StaticValueExpression[None]:
        """Create a static value expression that represents a **NULL** value."""
        return StaticValueExpression(None)

    def __init__(self, value: T) -> None:
        self._value = value
        super().__init__(hash(value))

    @property
    def value(self) -> T:
        """Get the value.

        Returns
        -------
        T
            The value, duh!
        """
        return self._value

    def is_null(self) -> bool:
        """Checks, whether the value is **NULL**.

        Returns
        -------
        bool
            Whether the value is **NULL**
        """
        return self._value is None

    def tables(self) -> set[TableReference]:
        return set()

    def columns(self) -> set[ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_static_value_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.value == other.value

    def __str__(self) -> str:
        if self.value is None:
            return "NULL"
        return f"{self.value}" if isinstance(self.value, numbers.Number) else f"'{self.value}'"


class CastExpression(SqlExpression):
    """An expression that casts the type of another nested expression.

    Note that PostBOUND itself does not know about the semantics of the actual types or casts. Eventual errors due to
    illegal casts are only caught at runtime by the actual database system.

    Parameters
    ----------
    expression : SqlExpression
        The expression that is casted to a different type.
    target_type : str
        The type to which the expression should be converted to. This cannot be empty.

    Raises
    ------
    ValueError
        If the `target_type` is empty.
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
        """Get the expression that is being casted.

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._casted_expression

    @property
    def target_type(self) -> str:
        """Get the type to which to cast to.

        Returns
        -------
        str
            The desired type. This is never empty.
        """
        return self._target_type

    def tables(self) -> set[TableReference]:
        return self._casted_expression.tables()

    def columns(self) -> set[ColumnReference]:
        return self.casted_expression.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.casted_expression.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.casted_expression]

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_cast_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.casted_expression == other.casted_expression
                and self.target_type == other.target_type)

    def __str__(self) -> str:
        return f"CAST({self.casted_expression} AS {self.target_type})"


class MathematicalExpression(SqlExpression):
    """A mathematical expression computes a result value based on a mathematical formula.

    The formula is based on an arbitrary expression, an operator and potentially a number of additional
    expressions/arguments.

    The precise representation of mathematical expressions is not tightly standardized by PostBOUND and there will be
    multiple ways to represent the same expression.

    For example, the expression ``R.a + S.b + 42`` could be modeled as a single expression object with ``R.a`` as first
    argument and the sequence ``S.b, 42`` as second arguments. At the same time, the mathematical expression can also
    be used to represent logical expressions such as ``R.a < 42`` or ``S.b IN (1, 2, 3)``. However, this should be used
    sparingly since logical expressions can be considered as predicates which are handled in the dedicated `predicates`
    module. Moving logical expressions into a mathematical expression object can break correct functionality in that
    module (e.g. determining joins and filters in a query).

    Parameters
    ----------
    operator : SqlOperator
        The operator that is used to combine the arguments.
    first_argument : SqlExpression
        The first argument. For unary expressions, this can also be the only argument
    second_argument : SqlExpression | Sequence[SqlExpression] | None, optional
        Additional arguments. For the most common case of a binary expression, this will be exactly one argument.
        Defaults to ``None`` to accomodate for unary expressions.
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
        """Get the operation to combine the input value(s).

        Returns
        -------
        SqlOperator
            The operator
        """
        return self._operator

    @property
    def first_arg(self) -> SqlExpression:
        """Get the first argument to the operator. This is always specified.

        Returns
        -------
        SqlExpression
            The argument
        """
        return self._first_arg

    @property
    def second_arg(self) -> SqlExpression | Sequence[SqlExpression] | None:
        """Get the second argument to the operator.

        Depending on the operator, this can be a single expression (the most common case), but also a sequence of
        expressions (e.g. sum of multiple values) or no value at all (e.g. negation).

        Returns
        -------
        SqlExpression | Sequence[SqlExpression] | None
            The argument(s)
        """
        return self._second_arg

    def tables(self) -> set[TableReference]:
        all_tables = set(self.first_arg.tables())
        if isinstance(self.second_arg, list):
            all_tables |= util.set_union(expr.tables() for expr in self.second_arg)
        elif isinstance(self.second_arg, SqlExpression):
            all_tables |= self.second_arg.tables()
        return all_tables

    def columns(self) -> set[ColumnReference]:
        all_columns = set(self.first_arg.columns())
        if isinstance(self.second_arg, list):
            for expression in self.second_arg:
                all_columns |= expression.columns()
        elif isinstance(self.second_arg, SqlExpression):
            all_columns |= self.second_arg.columns()
        return all_columns

    def itercolumns(self) -> Iterable[ColumnReference]:
        first_columns = list(self.first_arg.itercolumns())
        if not self.second_arg:
            return first_columns
        second_columns = (util.flatten(sub_arg.itercolumns() for sub_arg in self.second_arg)
                          if isinstance(self.second_arg, tuple) else list(self.second_arg.itercolumns()))
        return first_columns + second_columns

    def iterchildren(self) -> Iterable[SqlExpression]:
        return [self.first_arg, self.second_arg]

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_mathematical_expr(self)

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
    """A column expression wraps the reference to a column.

    This is a leaf expression, i.e. a column expression cannot have any more child expressions. It corresponds directly
    to an access to the values of the wrapped column with no modifications.

    Parameters
    ----------
    column : ColumnReference
        The column being wrapped
    """

    def __init__(self, column: ColumnReference) -> None:
        if column is None:
            raise ValueError("Column cannot be none")
        self._column = column
        super().__init__(hash(self._column))

    @property
    def column(self) -> ColumnReference:
        """Get the column that is wrapped by this expression.

        Returns
        -------
        ColumnReference
            The column
        """
        return self._column

    def tables(self) -> set[TableReference]:
        if not self.column.is_bound():
            return set()
        return {self.column.table}

    def columns(self) -> set[ColumnReference]:
        return {self.column}

    def itercolumns(self) -> Iterable[ColumnReference]:
        return [self.column]

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_column_expr(self)

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
    PostBOUND treats them all the same and it is up to the user to differentiate e.g. between UDFs and aggregations if
    this distinction is important. This can be easily achieved by introducing additional subclasses of the function
    expression and updating the queries to use the new function expressions where appropriate. The `transform` module
    provides utilities to make such updates easy.

    Parameters
    ----------
    function : str
        The name of the function that should be called. Cannot be empty.
    arguments : Optional[Sequence[SqlExpression]], optional
        The parameters that should be passed to the function. Can be ``None`` if the function does not take or does not
        need any arguments (e.g. ``CURRENT_TIME()``)
    distinct : bool, optional
        Whether the (aggregation) function should only operate on distinct column values and hence a duplicate
        elimination needs to be performed before passing the argument values (e.g. ``COUNT(DISTINCT *)``). Defaults to
        ``False``

    Raises
    ------
    ValueError
        If `function` is empty

    See Also
    --------
    postbound.qal.transform.replace_expressions
    """

    @staticmethod
    def all_func(subquery: SqlQuery | SubqueryExpression) -> FunctionExpression:
        """Create a function expression for the ``ALL`` function, as used for subqueries.

        Parameters
        ----------
        subquery : SqlQuery | SubqueryExpression
            The subquery whose result set should be checked. Will be wrapped in a `SubqueryExpression` if necessary.

        Returns
        -------
        FunctionExpression
            The ``ALL`` function expression
        """
        subquery = SubqueryExpression(subquery) if not isinstance(subquery, SubqueryExpression) else subquery
        return FunctionExpression("ALL", (subquery,))

    @staticmethod
    def any_func(subquery: SqlQuery) -> FunctionExpression:
        """Create a function expression for the ``ANY`` function, as used for subqueries.

        Parameters
        ----------
        subquery : SqlQuery | SubqueryExpression
            The subquery whose result set should be checked. Will be wrapped in a `SubqueryExpression` if necessary.

        Returns
        -------
        FunctionExpression
            The ``ANY`` function expression
        """
        subquery = SubqueryExpression(subquery) if not isinstance(subquery, SubqueryExpression) else subquery
        return FunctionExpression("ANY", (subquery,))

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
        """Get the function name.

        Returns
        -------
        str
            The function name. Will never be empty
        """
        return self._function

    @property
    def arguments(self) -> Sequence[SqlExpression]:
        """Get all arguments that are supplied to the function.

        Returns
        -------
        Sequence[SqlExpression]
            The arguments. Can be empty if no arguments are passed (but will never be ``None``).
        """
        return self._arguments

    @property
    def distinct(self) -> bool:
        """Get whether the function should only operate on distinct values.

        Whether this makes any sense for the function at hand is entirely dependend on the specific function and not
        enfored by PostBOUND. The runtime DBS has to check this.

        Generally speaking, this argument is intended for aggregation functions.

        Returns
        -------
        bool
            Whether a duplicate elimination has to be performed on the function arguments
        """
        return self._distinct

    def is_aggregate(self) -> bool:
        """Checks, whether the function is a well-known SQL aggregation function.

        Only standard functions are considered (e.g. no CORR for computing correlations).

        Returns
        -------
        bool
            Whether the function is a known aggregate function.
        """
        return self._function.upper() in AggregateFunctions

    def is_all(self) -> bool:
        """Checks, whether the function is an ``ALL`` function.

        If this is the case, the `subquery()` method can be used to directly retrieve the subquery being checked.

        Returns
        -------
        bool
            Whether the function is an ``ALL`` function.
        """
        return self._function.upper() == "ALL"

    def is_any(self) -> bool:
        """Checks, whether the function is an ``ANY`` function.

        If this is the case, the `subquery()` method can be used to directly retrieve the subquery being checked.

        Returns
        -------
        bool
            Whether the function is an ``ANY`` function.
        """
        return self._function.upper() == "ANY"

    def subquery(self) -> SqlQuery:
        """Get the subquery that is passed as argument to the function.

        This is only possible if the function is an ``ALL`` or ``ANY`` function. Otherwise, an error is raised.

        Returns
        -------
        SqlQuery
            The subquery

        Raises
        ------
        StateError
            If the function is not an ``ALL`` or ``ANY`` function
        """
        if not self.is_all() and not self.is_any():
            raise StateError("Function is not an ALL or ANY function")
        subquery: SubqueryExpression = self.arguments[0]
        return subquery.query

    def tables(self) -> set[TableReference]:
        return util.set_union(arg.tables() for arg in self.arguments)

    def columns(self) -> set[ColumnReference]:
        all_columns = set()
        for arg in self.arguments:
            all_columns |= arg.columns()
        return all_columns

    def itercolumns(self) -> Iterable[ColumnReference]:
        all_columns = []
        for arg in self.arguments:
            all_columns.extend(arg.itercolumns())
        return all_columns

    def iterchildren(self) -> Iterable[SqlExpression]:
        return list(self.arguments)

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_function_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.function == other.function
                and self.arguments == other.arguments
                and self.distinct == other.distinct)

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._arguments)
        distinct_str = "DISTINCT " if self._distinct else ""
        if self.is_all() or self.is_any():
            parameterization = f" {args_str}"
        else:
            parameterization = f"({distinct_str}{args_str})"
        return self._function + parameterization


class SubqueryExpression(SqlExpression):
    """A subquery expression wraps an arbitrary subquery.

    This expression can be used in two different contexts: as table source to produce a virtual temporary table for
    reference in the query (see the `clauses` module), or as a part of a predicate. In the latter scenario the
    subqueries' results are transient for the rest of the query. Therefore, this expression only represents the
    subquery part but no name under which the query result can be accessed. This is added by the different parts of the
    `clauses` module (e.g. `WithQuery` or `SubqueryTableSource`).

    This is a leaf expression, i.e. a subquery expression cannot have any more child expressions. However, the subquery itself
    likely consists of additional expressions.

    Parameters
    ----------
    subquery : SqlQuery
        The subquery that forms this expression

    """

    def __init__(self, subquery: SqlQuery) -> None:
        self._query = subquery
        super().__init__(hash(subquery))

    @property
    def query(self) -> SqlQuery:
        """The (sub)query that is wrapped by this expression.

        Returns
        -------
        SqlQuery
            The query
        """
        return self._query

    def tables(self) -> set[TableReference]:
        return self._query.tables()

    def columns(self) -> set[ColumnReference]:
        return self._query.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._query.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_subquery_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._query == other._query

    def __str__(self) -> str:
        query_str = str(self._query).removesuffix(";")
        return f"({query_str})"


class StarExpression(SqlExpression):
    """A special expression that is only used in ``SELECT`` clauses to select all columns."""

    def __init__(self) -> None:
        super().__init__(hash("*"))

    def tables(self) -> set[TableReference]:
        return set()

    def columns(self) -> set[ColumnReference]:
        return set()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def iterchildren(self) -> Iterable[SqlExpression]:
        return []

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_star_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self))

    def __str__(self) -> str:
        return "*"


class WindowExpression(SqlExpression):
    """Represents a window expression in SQL.

    Parameters
    ----------
    window_function : FunctionExpression
        The window function to be applied.
    partitioning : Optional[Sequence[SqlExpression]], optional
        The expressions used for partitioning the window. Defaults to None.
    ordering : Optional[OrderBy], optional
        The ordering of the window. Defaults to None.
    filter_condition : Optional[BooleanExpression], optional
        The filter condition for the window. Defaults to None.
    """

    def __init__(self, window_function: FunctionExpression, *,
                 partitioning: Optional[Sequence[SqlExpression]] = None,
                 ordering: Optional[OrderBy] = None,
                 filter_condition: Optional[BooleanExpression] = None) -> None:
        self._window_function = window_function
        self._partitioning = tuple(partitioning) if partitioning else tuple()
        self._ordering = ordering
        self._filter_condition = filter_condition

        hash_val = hash((self._window_function, self._partitioning, self._ordering, self._filter_condition))
        super().__init__(hash_val)

    @property
    def window_function(self) -> FunctionExpression:
        """Get the window function of the window expression.

        Returns
        -------
        FunctionExpression
            The window function.
        """
        return self._window_function

    @property
    def partitioning(self) -> Sequence[SqlExpression]:
        """Get the expressions used for partitioning the window.

        Returns
        -------
        Sequence[SqlExpression]
            The expressions used for partitioning the window. Can be empty if no partitioning is used.
        """
        return self._partitioning

    @property
    def ordering(self) -> Optional[OrderBy]:
        """Get the ordering of tuples in the current window.

        Returns
        -------
        Optional[OrderBy]
            The ordering of the tuples, or *None* if no ordering is specified.
        """
        return self._ordering

    @property
    def filter_condition(self) -> Optional[BooleanExpression]:
        """Get the filter condition for tuples in the current window.

        Returns:
            Optional[AbstractPredicate]:
            The filter condition for the expression, or *None* if all tuples are aggegrated.
        """
        return self._filter_condition

    def tables(self) -> set[TableReference]:
        return util.set_union(child.tables() for child in self.iterchildren())

    def columns(self) -> set[ColumnReference]:
        return util.set_union(child.columns() for child in self.iterchildren())

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(expr.itercolumns() for expr in self.iterchildren())

    def iterchildren(self) -> Iterable[SqlExpression]:
        function_children = list(self.window_function.iterchildren())
        partitioning_children = util.flatten(expr.iterchildren() for expr in self.partitioning)
        ordering_children = self.ordering.iterexpressions() if self.ordering else []
        filter_children = self.filter_condition.iterexpressions() if self.filter_condition else []
        return function_children + partitioning_children + ordering_children + filter_children

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_window_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.window_function == other.window_function
                and self.partitioning == other.partitioning
                and self.ordering == other.ordering
                and self.filter_condition == other.filter_condition)

    def __str__(self) -> str:
        filter_str = f" FILTER (WHERE {self.filter_condition})" if self.filter_condition else ""
        function_str = f"{self.window_function}{filter_str} OVER"
        window_grouping: list[str] = []
        if self.partitioning:
            partitioning_str = ", ".join(str(partition) for partition in self.partitioning)
            window_grouping.append(f"PARTITION BY {partitioning_str}")
        if self.ordering:
            window_grouping.append(str(self.ordering))
        window_str = " ".join(window_grouping)
        window_str = f"({window_str})" if window_str else "()"
        return f"{function_str} {window_str}"


class CaseExpression(SqlExpression):
    """Represents a case expression in SQL.

    Parameters:
    -----------
    cases : Sequence[tuple[AbstractPredicate, SqlExpression]]
        A sequence of tuples representing the cases in the case expression. The cases are passed as a sequence rather than a
        dictionary, because the evaluation order of the cases is important. The first case that evaluates to true determines
        the result of the entire case statement.
    else_expr : Optional[SqlExpression], optional
        The expression to be evaluated if none of the cases match. If no case matches and no else expression is provided, the
        entire case expression should evaluate to NULL.
    """
    def __init__(self, cases: Sequence[tuple[AbstractPredicate, SqlExpression]], *,
                 else_expr: Optional[SqlExpression] = None) -> None:
        if not cases:
            raise ValueError("At least one case is required")
        self._cases = tuple(cases)
        self._else_expr = else_expr

        hash_val = hash((self._cases, self._else_expr))
        super().__init__(hash_val)

    @property
    def cases(self) -> Sequence[tuple[AbstractPredicate, SqlExpression]]:
        """Get the different cases.

        Returns
        -------
        Sequence[tuple[AbstractPredicate, SqlExpression]]
            The cases. At least one case will be present.
        """
        return self._cases

    @property
    def else_expression(self) -> Optional[SqlExpression]:
        """Get the expression to use if none of the cases match.

        Returns
        -------
        Optional[SqlExpression]
            The expression. Can be ``None``, in which case the case expression evaluates to *NULL*.
        """
        return self._else_expr

    def tables(self) -> set[TableReference]:
        return util.set_union(child.tables() for child in self.iterchildren())

    def columns(self) -> set[ColumnReference]:
        return util.set_union(child.columns() for child in self.iterchildren())

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(expr.itercolumns() for expr in self.iterchildren())

    def iterchildren(self) -> Iterable[SqlExpression]:
        case_children = util.flatten(list(pred.iterexpressions()) + list(expr.iterchildren())
                                     for pred, expr in self.cases)
        else_children = self.else_expression.iterchildren() if self.else_expression else []
        return case_children + else_children

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_case_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.cases == other.cases
                and self.else_expression == other.else_expression)

    def __str__(self) -> str:
        cases_str = " ".join(f"WHEN {pred} THEN {expr}" for pred, expr in self.cases)
        else_str = f" ELSE {self.else_expression}" if self.else_expression else ""
        return f"CASE {cases_str}{else_str} END"


class BooleanExpression(SqlExpression):
    """Represents a boolean expression in SQL.

    Notice that this expression does not function as a replacement or alternative to the `predicates` module. Instead, boolean
    expressions appear in situations where other (non-predicate) clauses require a boolean expression. For example, when using
    a user-defined function in the ``SELECT`` clause, such as in ``SELECT my_udf(R.a > 42) FROM R``.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate to wrap.
    """

    def __init__(self, predicate: AbstractPredicate) -> None:
        self._predicate = predicate
        hash_val = hash(predicate)
        super().__init__(hash_val)

    @property
    def predicate(self) -> AbstractPredicate:
        """Get the predicate that is wrapped by this expression.

        Returns
        -------
        AbstractPredicate
            The predicate
        """
        return self._predicate

    def tables(self) -> set[TableReference]:
        return self._predicate.tables()

    def columns(self) -> set[ColumnReference]:
        return self._predicate.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._predicate.itercolumns()

    def iterchildren(self) -> Iterable[SqlExpression]:
        return self._predicate.iterexpressions()

    def accept_visitor(self, visitor: SqlExpressionVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_boolean_expr(self)

    __hash__ = SqlExpression.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.predicate == other.predicate

    def __str__(self) -> str:
        return str(self.predicate)


class SqlExpressionVisitor(abc.ABC, Generic[VisitorResult]):
    """Basic visitor to operator on arbitrary expression trees.

    See Also
    --------
    SqlExpression

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_static_value_expr(self, expr: StaticValueExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_cast_expr(self, expr: CastExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_mathematical_expr(self, expr: MathematicalExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_column_expr(self, expr: ColumnExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_function_expr(self, expr: FunctionExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_subquery_expr(self, expr: SubqueryExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_star_expr(self, expr: StarExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_window_expr(self, expr: WindowExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_case_expr(self, expr: CaseExpression) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_boolean_expr(self, expr: BooleanExpression) -> VisitorResult:
        return expr.predicate.accept_visitor(self)


class ExpressionCollector(SqlExpressionVisitor[set[SqlExpression]]):
    """Utility to traverse an arbitrarily deep expression hierarchy in order to collect specific expressions.

    Parameters
    ----------
    matcher : Callable[[SqlExpression], bool]
        Function to determine whether a specific expression matches the collection predicate. Should return *True* for matches
        and *False* otherwise.
    continue_after_match : bool, optional
        Whether the traversal of the current expression element should be continued if the current element matches the
        collection predicate. By default, traversal is stopped for the current element (but other branches in the expression
        tree could still produce more matches).
    """
    def __init__(self, matcher: Callable[[SqlExpression], bool], *, continue_after_match: bool = False) -> None:
        self.matcher = matcher
        self.continue_after_match = continue_after_match

    def visit_column_expr(self, expression: ColumnExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_cast_expr(self, expression: CastExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_function_expr(self, expression: FunctionExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_mathematical_expr(self, expression: MathematicalExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_star_expr(self, expression: StarExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_static_value_expr(self, expression: StaticValueExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_subquery_expr(self, expression: SubqueryExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_window_expr(self, expression: WindowExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_case_expr(self, expression: CaseExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def visit_boolean_expr(self, expression: BooleanExpression) -> set[SqlExpression]:
        return self._check_match(expression)

    def _check_match(self, expression: SqlExpression) -> set[SqlExpression]:
        """Handler to perform the actual traversal.

        Parameters
        ----------
        expression : SqlExpression
            The current expression

        Returns
        -------
        set[SqlExpression]
            All matching expressions
        """
        own_match = {expression} if self.matcher(expression) else set()
        if own_match and not self.continue_after_match:
            return own_match
        return own_match | util.set_union(child.accept_visitor(self) for child in expression.iterchildren())


def as_expression(value: object) -> SqlExpression:
    """Transforms the given value into the most appropriate `SqlExpression` instance.

    This is a heuristic utility method that applies the following rules:

    - `ColumnReference` becomes `ColumnExpression`
    - `SqlQuery` becomes `SubqueryExpression`
    - the star-string ``*`` becomes a `StarExpression`

    All other values become a `StaticValueExpression`.

    Parameters
    ----------
    value : object
        The object to be transformed into an expression

    Returns
    -------
    SqlExpression
        The most appropriate expression object according to the transformation rules
    """
    if isinstance(value, SqlExpression):
        return value

    if isinstance(value, ColumnReference):
        return ColumnExpression(value)
    elif isinstance(value, SqlQuery):
        return SubqueryExpression(value)

    if value == "*":
        return StarExpression()
    return StaticValueExpression(value)


def _normalize_join_pair(columns: tuple[ColumnReference, ColumnReference]
                         ) -> tuple[ColumnReference, ColumnReference]:
    """Normalizes the given join such that a pair ``(R.a, S.b)`` and ``(S.b, R.a)`` can be recognized as equal.

    Normalization in this context means that the order in which two appear is always the same. Therefore, this method
    essentially forces a swap of the columns if necessary, making use of the ability to sort columns lexicographically.

    Parameters
    ----------
    columns : tuple[ColumnReference, ColumnReference]
        The join pair to normalize

    Returns
    -------
    tuple[ColumnReference, ColumnReference]
        The normalized join pair.
    """
    first_col, second_col = columns
    return (second_col, first_col) if second_col < first_col else columns


class NoJoinPredicateError(StateError):
    """Error to indicate that a filter predicate was supplied at a place where a join predicate was expected.

    Parameters
    ----------
    predicate : AbstractPredicate | None, optional
        The predicate that caused the error, defaults to ``None``
    """

    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


class NoFilterPredicateError(StateError):
    """Error to indicate that a join predicate was supplied at a place where a filter predicate was expected.

    Parameters
    ----------
    predicate : AbstractPredicate | None, optional
        The predicate that caused the error, defaults to ``None``.
    """

    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


BaseExpression = Union[ColumnExpression, StaticValueExpression, SubqueryExpression]
"""Supertype that captures all expression types that can be considered base expressions for predicates."""


def _collect_base_expressions(expression: SqlExpression) -> Iterable[BaseExpression]:
    """Provides all base expressions that are contained in a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree.

    Parameters
    ----------
    expression : SqlExpression
        The expression to traverse

    Returns
    -------
    Iterable[BaseExpression]
        The base expressions
    """
    if isinstance(expression, (ColumnExpression, StaticValueExpression, SubqueryExpression)):
        return [expression]
    return util.flatten(_collect_base_expressions(child_expr) for child_expr in expression.iterchildren())


def _collect_subquery_expressions(expression: SqlExpression) -> Iterable[SubqueryExpression]:
    """Provides all subquery expressions that are contained in a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree.

    Parameters
    ----------
    expression : SqlExpression
        The expression to traverse

    Returns
    -------
    Iterable[SubqueryExpression]
        All subqueries that are contained in some level in the expression
    """
    return [child_expr for child_expr in _collect_base_expressions(expression)
            if isinstance(child_expr, SubqueryExpression)]


def _collect_column_expression_columns(expression: SqlExpression) -> set[ColumnReference]:
    """Provides all columns that are directly contained in `ColumnExpression` instances with a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree. Notice that it ignores all
    expressions that are part of subqueries.

    Parameters
    ----------
    expression : SqlExpression
        The expression to traverse

    Returns
    -------
    set[ColumnReference]
        All columns that are referenced in `ColumnExpression`s
    """
    return util.set_union(base_expr.columns() for base_expr in _collect_base_expressions(expression)
                          if isinstance(base_expr, ColumnExpression))


def _collect_column_expression_tables(expression: SqlExpression) -> set[TableReference]:
    """Provides all tables that are linked directly in `ColumnExpression` instances with a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree. Notice that it ignores all
    expressions that are part of subqueries.

    Parameters
    ----------
    expression : SqlExpression
        The expression to traverse

    Returns
    -------
    set[TableReference]
        All tables that are referenced in the columns of `ColumnExpression`s
    """
    return {column.table for column in _collect_column_expression_columns(expression) if column.is_bound()}


def _generate_join_pairs(first_columns: Iterable[ColumnReference], second_columns: Iterable[ColumnReference]
                         ) -> set[tuple[ColumnReference, ColumnReference]]:
    """Provides all possible pairs of columns where each column comes from a different iterable.

    Essentially, this produces the cross product of the two column sets. The join pairs are normalized and duplicate
    elimination is performed (this is necessary since columns can appear in both iterables). Likewise, "joins" over the same
    logical relations are also skipped.

    Parameters
    ----------
    first_columns : Iterable[ColumnReference]
        Candidate columns to be joined with `second_columns`
    second_columns : Iterable[ColumnReference]
        The join partner columns

    Returns
    -------
    set[tuple[ColumnReference, ColumnReference]]
        All normalized join pairs
    """
    return {_normalize_join_pair((first_col, second_col)) for first_col, second_col
            in itertools.product(first_columns, second_columns) if first_col.table != second_col.table}


class AbstractPredicate(abc.ABC):
    """Base class for all predicates.

    Predicates constitute the central building block for ``WHERE`` and ``HAVING`` clauses and model the join conditions in
    explicit joins using the ``JOIN ON`` syntax.

    The different kinds of predicates are represented as subclasses of the `AbstractPredicate` interface. This really is an
    abstract interface, not a usable predicate and it only specifies the behaviour that is shared among all specific predicate
    implementations. All inheriting classes have to implement their own `__eq__` method and inherit the `__hash__` method
    specified by the abstract predicate. Remember to explicitly set this up! The concrete hash value is constant since the
    clause itself is immutable. It is up to the implementing class to make sure that the equality/hash consistency is enforced.

    Possible implementations of the abstract predicate can model basic binary predicates such as  ``R.a = S.b`` or
    ``R.a = 42``, as well as compound predicates that are build form base predicates, e.g. conjunctions, disjunctions or
    negations.

    Parameters
    ----------
    hash_val : int
        The hash of the concrete predicate object
    """

    def __init__(self, hash_val: int) -> None:
        self._hash_val = hash_val

    @abc.abstractmethod
    def is_compound(self) -> bool:
        """Checks, whether this predicate combines the evaluation of other predicates to compute the overall evaluation.

        Operators to combine such predicates can be standard logical operators like conjunction, disjunction and negation. This
        method serves as a high-level check, preventing the usage of dedicated ``isinstance`` calls in some use-cases.

        Returns
        -------
        bool
            Whethter this predicate is a composite of other predicates.
        """
        raise NotImplementedError

    def is_base(self) -> bool:
        """Checks, whether this predicate forms a leaf in the predicate tree, i.e. does not contain any more child predicates.

        This is the case for basic binary predicates, ``IN`` predicates, etc. This method serves as a high-level check,
        preventing the usage of dedicated ``isinstance`` calls in some use-cases.

        Returns
        -------
        bool
            Whether this predicate is a base predicate
        """
        return not self.is_compound()

    @abc.abstractmethod
    def is_join(self) -> bool:
        """Checks, whether this predicate encodes a join between two tables.

        PostBOUND uses the following criteria to determine, whether a predicate is join or not:

        1. all predicates of the form ``<col 1> <operator> <col 2>`` where ``<col 1>`` and ``<col 2>`` come from different
           tables are joins. The columns can optionally be modified by value casts or static expressions, e.g.
           ``R.a::integer + 7``
        2. all functions that access columns from multiple tables are joins, e.g. ``my_udf(R.a, S.b)``
        3. all subqueries are treated as filters, no matter whether they are dependent subqueries or not. This means that both
           ``R.a = (SELECT MAX(S.b) FROM S)`` and ``R.a = (SELECT MAX(S.b) FROM S WHERE R.c = S.d)`` are treated as filters and
           not as joins, even though the second subquery will require some sort of the join in the query plan.
        4. ``BETWEEN`` and ``IN`` predicates are treated according to rule 1 since they can be emulated via base predicates
           (subqueries in ``IN`` predicates are evaluated according to rule 3.)

        Although these rules might seem a bit arbitrary at first, there is actually no clear consensus of what constitutes a
        join and the query optimizers of different industrial database systems treat different predicates as joins. For
        example, some systems might not recognize function calls that access columns form two or more tables as joins or do not
        recognize predicates that use non-equi joins as opertors as actual joins.

        If the specific join and filter recognition procedure breaks a specific use-case, subclasses of the predicate classes
        can be implemented. These subclasses can then apply the required rules. Using the tools in the `transformation`
        module, the queries can be updated. For some use-cases it can also be sufficient to change the join/filter recognition
        rules of the `QueryPredicates` objects. Consult its documentation for more details.

        Lastly, notice that the distinction between join and filter is not entirely binary. There may also be a third class of
        predicates, potentially called "post-join filters". These are filters that are applied after a join but cannot be
        included in the join predicate itself. This is usually the case due to limitations in the operator implementation of
        the actual database system. For example, invocations of user defined functions (case 2 above) usually fall in this
        category. Since the query abstraction layer is agnostic to specific details of database systems, we apply the binary
        categorization outlined above.

        Returns
        -------
        bool
            Whether the predicate is a join of different relations.
        """
        raise NotImplementedError

    def is_filter(self) -> bool:
        """Checks, whether this predicate encodes a filter on a base table rather than a join of base tables.

        This is the inverse method to `is_join`. Consult it for more details on the join/filter recognition procedure.

        Returns
        -------
        bool
            Whether the predicate is a filter of some relation
        """
        return not self.is_join()

    def tables(self) -> set[TableReference]:
        """Provides all tables that are accessed by this predicate.

        Notice that even for filters, the provided set might contain multiple entries, e.g. if the predicate contains
        subqueries.

        Returns
        -------
        set[TableReference]
            All tables. This can include virtual tables if such tables are referenced in the predicate.
        """
        return util.set_union(e.tables() for e in self.iterexpressions())

    @abc.abstractmethod
    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced by this predicate.

        Returns
        -------
        set[ColumnReference]
            The columns. If the predicate contains a subquery, all columns of that query are included.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides all columns that are referenced by this predicate.

        If a column is referenced multiple times, it is also returned multiple times.

        Returns
        -------
        Iterable[ColumnReference]
            All columns in exactly the order in which they are used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all expressions that are directly contained in this predicate.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    def contains_table(self, table: TableReference) -> bool:
        """Checks, whether this predicate filters or joins a column of a specific table.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the given `table` is referenced by any of the columns in the predicate.
        """
        return any(table == tab for tab in self.tables())

    def joins_table(self, table: TableReference) -> bool:
        """Checks, whether this predicate describes a join and one of the join partners is a specific table.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the given `table` is one of the join partners in the predicate.
        """
        if not self.is_join():
            return False
        return any(first_col.belongs_to(table) or second_col.belongs_to(table)
                   for first_col, second_col in self.join_partners())

    def columns_of(self, table: TableReference) -> set[ColumnReference]:
        """Retrieves all columns of a specific table that are referenced by this predicate.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        set[ColumnReference]
            All columns in this predicate that belong to the given `table`
        """
        return {col for col in self.columns() if col.belongs_to(table)}

    def join_partners_of(self, table: TableReference) -> set[ColumnReference]:
        """Retrieves all columns that are joined with a specific table.

        Parameters
        ----------
        table : TableReference
            The table for which the join partners should be searched

        Returns
        -------
        set[ColumnReference]
            The columns that are joined with the given `table`

        Raises
        ------
        NoJoinPredicateError
        """
        partners = []
        for first_col, second_col in self.join_partners():
            if first_col.belongs_to(table):
                partners.append(second_col)
            elif second_col.belongs_to(table):
                partners.append(first_col)
        return set(partners)

    @abc.abstractmethod
    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        """Provides all pairs of columns that are joined within this predicate.

        If multiple columns are joined or it is unclear which columns are involved in a join exactly, this method falls back to
        returning the cross-product of all potential join partners. For example, consider the following query:
        ``SELECT * FROM R, S WHERE my_udf(R.a, R.b, S.c)``. In this case, it cannot be determined which columns of ``R`` take
        part in the join. Therefore, `join_partners` will return the set ``{(R.a, S.c), (R.b, S.c)}``.

        Returns
        -------
        set[tuple[ColumnReference, ColumnReference]]
            The pairs of joined columns. These pairs are normalized, such that two predicates which join the same columns
            provide the join partners in the same order.

        Raises
        ------
        NoJoinPredicateError
        """
        raise NotImplementedError

    @abc.abstractmethod
    def base_predicates(self) -> Iterable[AbstractPredicate]:
        """Provides all base predicates that form this predicate.

        This allows to iterate over all leaves of a compound predicate, for base predicates it simply returns the predicate
        itself.

        Returns
        -------
        Iterable[AbstractPredicate]
            The base predicates, in an arbitrary order. If the predicate is a base predicate already, it will be the only item
            in the iterable.
        """
        raise NotImplementedError

    def required_tables(self) -> set[TableReference]:
        """Provides all tables that have to be "available" in order for this predicate to be executed.

        Availability in this context means that the table has to be scanned already. Therefore it can be accessed either as-is,
        or as part of an intermediate relation.

        The output of this method differs from the `tables` method in one central aspect: `tables` provides all tables that are
        accessed, which includes all tables from subqueries. In contrast, the `required_tables` remove all tables that are
        scanned by the subquery and only include those that must be "provided" by the query execution engine.

        Consider the following example predicate: ``R.a = (SELECT MIN(S.b) FROM S)``. Calling `tables` on this predicate would
        return the set ``{R, S}``. However, table ``S`` is already provided by the subquery. Therefore, `required_tables` only
        returns ``{R}``, since this is the only table that has to be provided by the context of this method.

        Returns
        -------
        set[TableReference]
            The tables that need to be provided by the query execution engine in order to run this predicate
        """
        subqueries = util.flatten(_collect_subquery_expressions(child_expr)
                                  for child_expr in self.iterexpressions())
        subquery_tables = util.set_union(subquery.query.unbound_tables() for subquery in subqueries)
        column_tables = util.set_union(_collect_column_expression_tables(child_expr)
                                       for child_expr in self.iterexpressions())
        return column_tables | subquery_tables

    @abc.abstractmethod
    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        """Enables processing of the current predicate by a predicate visitor.

        Parameters
        ----------
        visitor : PredicateVisitor[VisitorResult]
            The visitor
        """
        raise NotImplementedError

    def _assert_join_predicate(self) -> None:
        """Raises a `NoJoinPredicateError` if this predicate is not a join.

        Raises
        ------
        NoJoinPredicateError
        """
        if not self.is_join():
            raise NoJoinPredicateError(self)

    def _assert_filter_predicate(self) -> None:
        """Raises a `NoFilterPredicateError` if this predicate is not a filter.

        Raises
        ------
        NoFilterPredicateError
        """
        if not self.is_filter():
            raise NoFilterPredicateError(self)

    def __json__(self) -> str:
        return {"node_type": "predicate", "tables": self.tables(), "predicate": str(self)}

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class BasePredicate(AbstractPredicate, abc.ABC):
    """A base predicate is a predicate that is not composed of any additional child predicates, such as a binary predicate.

    It represents the smallest kind of condition that evaluates to ``TRUE`` or ``FALSE``.

    Parameters
    ----------
    operation : Optional[SqlOperator]
        The operation that compares the column value(s). For unary base predicates, this may be ``None`` if a
        predicate function is used to determine matching tuples.
    hash_val : int
        The hash of the entire predicate
    """

    def __init__(self, operation: Optional[SqlOperator], *, hash_val: int) -> None:
        self._operation = operation
        super().__init__(hash_val)

    @property
    def operation(self) -> Optional[SqlOperator]:
        """Get the operation that is used to obtain matching (pairs of) tuples.

        Most of the time, this operation will be set to one of the SQL operators. However, for unary predicates that filter
        based on a predicate function this might be ``None`` (e.g. a user-defined function such as in
        `SELECT * FROM R WHERE my_udf_predicate(R.a, R.b)`).

        Returns
        -------
        Optional[SqlOperator]
            The operation if it exists
        """
        return self._operation

    def is_compound(self) -> bool:
        return False

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]

    __hash__ = AbstractPredicate.__hash__


class BinaryPredicate(BasePredicate):
    """A binary predicate combines exactly two base expressions with a comparison operation.

    This is the most typical kind of predicate and appears in most joins (e.g. ``R.a = S.b``) and many filters
    (e.g. ``R.a = 42``).

    Parameters
    ----------
    operation : SqlOperator
        The operation that combines the input arguments
    first_argument : SqlExpression
        The first comparison value
    second_argument : SqlExpression
        The second comparison value
    """

    @staticmethod
    def equal(first_argument: SqlExpression, second_argument: SqlExpression) -> BinaryPredicate:
        """Generates an equality predicate between two arguments."""
        return BinaryPredicate(LogicalSqlOperators.Equal, first_argument, second_argument)

    def __init__(self, operation: SqlOperator, first_argument: SqlExpression,
                 second_argument: SqlExpression) -> None:
        if not first_argument or not second_argument:
            raise ValueError("First argument and second argument are required")
        self._first_argument = first_argument
        self._second_argument = second_argument

        hash_val = hash((operation, first_argument, second_argument))
        super().__init__(operation, hash_val=hash_val)

    @property
    def first_argument(self) -> SqlExpression:
        """Get the first argument of the predicate.

        Returns
        -------
        SqlExpression
            The argument
        """
        return self._first_argument

    @property
    def second_argument(self) -> SqlExpression:
        """Get the second argument of the predicate.

        Returns
        -------
        SqlExpression
            The argument
        """
        return self._second_argument

    def is_join(self) -> bool:
        first_tables = _collect_column_expression_tables(self.first_argument)
        if len(first_tables) > 1:
            return True

        second_tables = _collect_column_expression_tables(self.second_argument)
        if len(second_tables) > 1:
            return True

        return first_tables and second_tables and len(first_tables ^ second_tables) > 0

    def columns(self) -> set[ColumnReference]:
        return self.first_argument.columns() | self.second_argument.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return list(self.first_argument.itercolumns()) + list(self.second_argument.itercolumns())

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self.first_argument, self.second_argument]

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        self._assert_join_predicate()
        first_columns = _collect_column_expression_columns(self.first_argument)
        second_columns = _collect_column_expression_columns(self.second_argument)

        partners = _generate_join_pairs(first_columns, first_columns)
        partners |= _generate_join_pairs(second_columns, second_columns)
        partners |= _generate_join_pairs(first_columns, second_columns)
        return partners

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_binary_predicate(self)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.operation == other.operation
                and self.first_argument == other.first_argument
                and self.second_argument == other.second_argument)

    def __str__(self) -> str:
        return f"{self.first_argument} {self.operation.value} {self.second_argument}"


class BetweenPredicate(BasePredicate):
    """A ``BETWEEN`` predicate is a special case of a conjunction of two binary predicates.

    Each ``BETWEEN`` predicate has a structure of ``<col> BETWEEN <a> AND <b>``, where ``<col>`` describes the (column)
    expression to which the condition should apply and ``<a>`` and ``<b>`` are the expressions that denote the valid bounds.

    Each BETWEEN predicate can be represented by a conjunction of binary predicates: ``<col> BETWEEN <a> AND <b>`` is
    equivalent to ``<col> >= <a> AND <col> <= <b>``.

    Parameters
    ----------
    column : SqlExpression
        The value that is checked by the predicate
    interval : tuple[SqlExpression, SqlExpression]
        The allowed range in which the `column` values must lie. The range is inclusive at both endpoints. This has to be a
        pair (2-tuple) of expressions.

    Raises
    ------
    ValueError
        If the `interval` is not a pair of values.

    Notes
    -----
    A ``BETWEEN`` predicate can be a join predicate as in ``R.a BETWEEN 42 AND S.b``.
    Furthermore, some systems even allow the ``<col>`` part to be an arbitrary expression. For example, in Postgres this is a
    valid query:

    .. code-block:: sql

        SELECT *
        FROM R JOIN S ON R.a = S.b
        WHERE 42 BETWEEN R.c AND S.d
    """

    def __init__(self, column: SqlExpression, interval: tuple[SqlExpression, SqlExpression]) -> None:
        if not column or not interval or len(interval) != 2:
            raise ValueError("Column and interval must be set")
        self._column = column
        self._interval = interval
        self._interval_start, self._interval_end = self._interval

        hash_val = hash((LogicalSqlOperators.Between, self._column, self._interval_start, self._interval_end))
        super().__init__(LogicalSqlOperators.Between, hash_val=hash_val)

    @property
    def column(self) -> SqlExpression:
        """Get the column that is tested (``R.a`` in ``SELECT * FROM R WHERE R.a BETWEEN 1 AND 42``).

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._column

    @property
    def interval(self) -> tuple[SqlExpression, SqlExpression]:
        """Get the interval (as ``(lower, upper)``) that is tested against.

        Returns
        -------
        tuple[SqlExpression, SqlExpression]
            The allowed range of values. This interval is inclusive at both endpoints.
        """
        return self._interval

    @property
    def interval_start(self) -> SqlExpression:
        """Get the lower bound of the interval that is tested against.

        Returns
        -------
        SqlExpression
            The lower value. This value is inclusive, i.e. the comparison values must be greater or equal.
        """
        return self._interval_start

    @property
    def interval_end(self) -> SqlExpression:
        """Get the upper bound of the interval that is tested against.

        Returns
        -------
        SqlExpression
            The upper value. This value is inclusive, i.e. the comparison values must be less or equal.
        """
        return self._interval_end

    def is_join(self) -> bool:
        column_tables = _collect_column_expression_tables(self.column)
        interval_start_tables = _collect_column_expression_tables(self.interval_start)
        interval_end_tables = _collect_column_expression_tables(self.interval_end)
        return (len(column_tables) > 1
                or len(column_tables | interval_start_tables) > 1
                or len(column_tables | interval_end_tables) > 1)

    def columns(self) -> set[ColumnReference]:
        return self.column.columns() | self.interval_start.columns() | self.interval_end.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return (list(self.column.itercolumns())
                + list(self.interval_start.itercolumns())
                + list(self.interval_end.itercolumns()))

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self.column, self.interval_start, self.interval_end]

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        self._assert_join_predicate()
        predicate_columns = _collect_column_expression_columns(self.column)
        start_columns = _collect_column_expression_columns(self.interval_start)
        end_columns = _collect_column_expression_columns(self.interval_end)

        partners = _generate_join_pairs(predicate_columns, predicate_columns)
        partners |= _generate_join_pairs(predicate_columns, start_columns)
        partners |= _generate_join_pairs(predicate_columns, end_columns)
        return set(partners)

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_between_predicate(self)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.column == other.column and self.interval == other.interval

    def __str__(self) -> str:
        interval_start, interval_end = self.interval
        return f"{self.column} BETWEEN {interval_start} AND {interval_end}"


class InPredicate(BasePredicate):
    """An ``IN`` predicate lists the allowed values for a column.

    In most cases, such a predicate with *n* allowed values can be transformed into a disjunction of *n* equality
    predicates, i.e. ``R.a IN (1, 2, 3)`` is equivalent to ``R.a = 1 OR R.a = 2 OR R.a = 3``. Depending on the allowed
    values, an IN predicate can denote a join, e.g. ``R.a IN (S.b, S.c)``. An important special case arises if the
    allowed values are produced by a subquery, e.g. ``R.a IN (SELECT S.b FROM S)``. This does no longer allow for a
    transformation into binary predicates since it is unclear how many rows the subquery will produce.

    Parameters
    ----------
    column : SqlExpression
        The value that is checked by the predicate
    values : Sequence[SqlExpression]
        The allowed column values. The individual expressions are not limited to `StaticValueExpression` instances, but can
        also include subqueries, columns or complicated mathematical expressions.

    Raises
    ------
    ValueError
        If `values` is empty.

    Notes
    -----
    Some systems even allow the `column` part to be an arbitrary expression. For example, in Postgres this is a valid query:

    .. code-block:: sql

        SELECT *
        FROM R JOIN S ON R.a = S.b
        WHERE 42 IN (R.c, 40 * S.d)
    """

    @staticmethod
    def subquery(column: SqlExpression, subquery: SubqueryExpression | SqlQuery) -> InPredicate:
        """Generates an ``IN`` predicate that is based on a subquery.

        Such a predicate is of the form ``R.a IN (SELECT S.b FROM S)``.

        Parameters
        ----------
        column : SqlExpression
            The column that should be checked for being contained by the subquerie's result set.
        subquery : SubqueryExpression | SqlQuery
            The subquery to produce the allowed values.

        Returns
        -------
        InPredicate
            The predicate
        """
        subquery = subquery if isinstance(subquery, SubqueryExpression) else SubqueryExpression(subquery)
        return InPredicate(column, (subquery,))

    def __init__(self, column: SqlExpression, values: Sequence[SqlExpression]) -> None:
        if not column or not values:
            raise ValueError("Both column and values must be given")
        if not all(val for val in values):
            raise ValueError("No empty value allowed")
        self._column = column
        self._values = tuple(values)
        hash_val = hash((LogicalSqlOperators.In, self._column, self._values))
        super().__init__(LogicalSqlOperators.In, hash_val=hash_val)

    @property
    def column(self) -> SqlExpression:
        """Get the expression that is tested (``R.a`` in ``SELECT * FROM R WHERE R.a IN (1, 2, 3)``).

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._column

    @property
    def values(self) -> Sequence[SqlExpression]:
        """Get the allowed values of the tested expression.

        Returns
        -------
        Sequence[SqlExpression]
            The allowed values. This sequence always contains at least one entry.
        """
        return self._values

    def is_subquery_predicate(self) -> bool:
        """Checks, if this is a subquery-based **IN** predicate, i.e. a predicate of the form ``R.a IN (SELECT S.b FROM S)``.

        Returns
        -------
        bool
            Whether this predicate is based on a subquery
        """
        return len(self._values) == 1 and isinstance(self._values[0], SubqueryExpression)

    def is_join(self) -> bool:
        column_tables = _collect_column_expression_tables(self.column)
        if len(column_tables) > 1:
            return True
        for value in self.values:
            value_tables = _collect_column_expression_tables(value)
            if len(column_tables | value_tables) > 1:
                return True
        return False

    def columns(self) -> set[ColumnReference]:
        all_columns = self.column.columns()
        for val in self.values:
            all_columns |= val.columns()
        return all_columns

    def itercolumns(self) -> Iterable[ColumnReference]:
        return list(self.column.itercolumns()) + util.flatten(val.itercolumns() for val in self.values)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self.column] + list(self.values)

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        self._assert_join_predicate()
        predicate_columns = _collect_column_expression_columns(self.column)

        partners = _generate_join_pairs(predicate_columns, predicate_columns)
        for value in self.values:
            value_columns = _collect_column_expression_columns(value)
            partners |= _generate_join_pairs(predicate_columns, value_columns)
        return partners

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_in_predicate(self)

    def _stringify_values(self) -> str:
        """Converts the allowed values into a valid string representation."""
        # NOTE: part of this implementation is re-used in the __str__ method for NOT predicates to format NOT IN predicates
        # appropriately. These methods should be kept in sync.
        if len(self.values) == 1:
            value = util.simplify(self.values)
            vals = str(value) if isinstance(value, SubqueryExpression) else f"({value})"
        else:
            vals = "(" + ", ".join(str(val) for val in self.values) + ")"
        return vals

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.column == other.column and set(self.values) == set(other.values)

    def __str__(self) -> str:
        vals = self._stringify_values()
        return f"{self.column} IN {vals}"


class UnaryPredicate(BasePredicate):
    """A unary predicate is applied directly to an expression, evaluating to TRUE or FALSE.

    Examples of such predicates include ``R.a IS NOT NULL``, ``EXISTS (SELECT S.b FROM S WHERE R.a = S.b)``, or
    ``my_udf(R.a)``. In the last case, ``my_udf`` has to produce a boolean return value.

    Parameters
    ----------
    column : SqlExpression
        The expression that is tested. This can also be a user-defined function that produces a boolen return value.
    operation : Optional[SqlOperator], optional
        The operation that is used to generate the unary predicate. Only a small subset of operators can actually be used in
        this context (e.g. ``EXISTS`` or ``MISSING``). If the predicate does not require an operator (e.g. in the case of
        filtering UDFs), the operation can be ``None``. Notice however, that PostBOUND has no knowledge of the semantics of
        UDFs and can therefore not enforce, whether UDFs is actually valid in this context. This has to be done at runtime by
        the actual database system.

    Raises
    ------
    ValueError
        If the given operation is not a valid unary operator
    """

    @staticmethod
    def exists(subquery: SqlQuery | SubqueryExpression) -> UnaryPredicate:
        """Creates an ``EXISTS`` predicate for a subquery.

        Parameters
        ----------
        subquery : SqlQuery | SubqueryExpression
            The subquery. Will be wrapped in a `SubqueryExpression` if it is not already one.

        Returns
        -------
        UnaryPredicate
            The ``EXISTS`` predicate
        """
        subquery = subquery if isinstance(subquery, SubqueryExpression) else SubqueryExpression(subquery)
        return UnaryPredicate(subquery, LogicalSqlOperators.Exists)

    def __init__(self, column: SqlExpression, operation: Optional[SqlOperator] = None):
        if not column:
            raise ValueError("Column must be set")
        if operation is not None and operation not in UnarySqlOperators:
            raise ValueError(f"Not an allowed unary operator: {operation}")
        self._column = column
        super().__init__(operation, hash_val=hash((operation, column)))

    @property
    def column(self) -> SqlExpression:
        """The column that is checked by this predicate

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._column

    def is_exists(self) -> bool:
        """Checks, whether this predicate is an ``EXISTS`` predicate.

        Returns
        -------
        bool
            Whether this predicate is an ``EXISTS`` predicate
        """
        return self.operation == LogicalSqlOperators.Exists

    def is_join(self) -> bool:
        return len(_collect_column_expression_tables(self.column)) > 1

    def columns(self) -> set[ColumnReference]:
        return self.column.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.column.itercolumns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self.column]

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        columns = _collect_column_expression_columns(self.column)
        return _generate_join_pairs(columns, columns)

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_unary_predicate(self)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.operation == other.operation and self.column == other.column

    def __str__(self) -> str:
        if not self.operation:
            return str(self.column)

        if self.operation == LogicalSqlOperators.Exists:
            assert isinstance(self.column, SubqueryExpression)
            return f"EXISTS {self.column}"

        return f"{self.operation.value}{self.column}"


class CompoundPredicate(AbstractPredicate):
    """A compound predicate creates a composite hierarchical structure of other predicates.

    Currently, PostBOUND supports 3 kinds of compound predicates: negations, conjunctions and disjunctions. Depending on the
    specific compound operator, a diferent number of child predicates is allowed.

    Parameters
    ----------
    operation : LogicalSqlCompoundOperators
        The operation that glues together the individual child predicates.
    children : AbstractPredicate | Sequence[AbstractPredicate]
        The predicates that are combined by this composite. For conjunctions and disjunctions, at least two children are
        required. For negations, exactly one child is permitted (either directly or in the sequence).

    Raises
    ------
    ValueError
        If `operation` is a negation and a number of children unequal to 1 is passed
    ValueError
        If `operation` is a conjunction or a disjunction and less than 2 children are passed
    """

    @staticmethod
    def create(operation: CompoundOperators, parts: Collection[AbstractPredicate]) -> AbstractPredicate:
        """Creates an arbitrary compound predicate for a number of child predicates.

        If just a single child predicate is provided, but the operation requires multiple children, that child is returned
        directly instead of the compound predicate.

        Parameters
        ----------
        operation : LogicalSqlCompoundOperators
            The logical operator to combine the child predicates.
        parts : Collection[AbstractPredicate]
            The child predicates

        Returns
        -------
        AbstractPredicate
            A composite predicate of the given `parts`, if parts contains the appropriate number of items. Otherwise the
            supplied child predicate.

        Raises
        ------
        ValueError
            If a negation predicate should be created but a number child predicates unequal to one are supplied. Likewise, if
            a conjunction or disjunction is requested, but no child predicates are supplied.
        """
        if operation == CompoundOperators.Not and len(parts) != 1:
            raise ValueError(f"Can only create negations for exactly one predicate but received: '{parts}'")
        elif operation != CompoundOperators.Not and not parts:
            raise ValueError("Conjunctions/disjunctions require at least one predicate")

        match operation:
            case CompoundOperators.Not:
                return CompoundPredicate.create_not(parts[0])
            case CompoundOperators.And | CompoundOperators.Or:
                if len(parts) == 1:
                    return parts[0]
                return CompoundPredicate(operation, parts)
            case _:
                raise ValueError(f"Unknown operator: '{operation}'")

    @staticmethod
    def create_and(parts: Collection[AbstractPredicate]) -> AbstractPredicate:
        """Creates an ``AND`` predicate, combining a number of child predicates.

        If just a single child predicate is provided, that child is returned directly instead of wrapping it in an
        ``AND`` predicate.

        Parameters
        ----------
        parts : Collection[AbstractPredicate]
            The children that should be combined

        Returns
        -------
        AbstractPredicate
            A conjunctive predicate of the given `parts`, if `parts` contains at least two items. Otherwise the only passed
            predicate is returned.

        Raises
        ------
        ValueError
            If `parts` is empty
        """
        parts = list(parts)
        if not parts:
            raise ValueError("No predicates supplied.")
        if len(parts) == 1:
            return parts[0]
        return CompoundPredicate(CompoundOperators.And, parts)

    @staticmethod
    def create_not(predicate: AbstractPredicate) -> CompoundPredicate:
        """Builds a ``NOT`` predicate, wrapping a specific child predicate.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate that should be negated

        Returns
        -------
        CompoundPredicate
            The negated predicate. No logic checks or simplifications are performed. For example, it is possible to negate a
            negation and this will still be represented in the predicate hierarchy.
        """
        return CompoundPredicate(CompoundOperators.Not, predicate)

    @staticmethod
    def create_or(parts: Collection[AbstractPredicate]) -> AbstractPredicate:
        """Creates an ``OR`` predicate, combining a number of child predicates.

        If just a single child predicate is provided, that child is returned directly instead of wrapping it in an
        ``OR`` predicate.

        Parameters
        ----------
        parts : Collection[AbstractPredicate]
            The children that should be combined

        Returns
        -------
        AbstractPredicate
            A disjunctive predicate of the given `parts`, if `parts` contains at least two items. Otherwise the only passed
            predicate is returned.

        Raises
        ------
        ValueError
            If `parts` is empty
        """
        parts = list(parts)
        if not parts:
            raise ValueError("No predicates supplied.")
        if len(parts) == 1:
            return parts[0]
        return CompoundPredicate(CompoundOperators.Or, parts)

    def __init__(self, operation: CompoundOperators,
                 children: AbstractPredicate | Sequence[AbstractPredicate]) -> None:
        if not operation or not children:
            raise ValueError("Operation and children must be set")
        if operation == CompoundOperators.Not and len(util.enlist(children)) > 1:
            raise ValueError("NOT predicates can only have one child predicate")
        if operation != CompoundOperators.Not and len(util.enlist(children)) < 2:
            raise ValueError("AND/OR predicates require at least two child predicates.")
        self._operation = operation
        self._children = tuple(util.enlist(children))
        super().__init__(hash((self._operation, self._children)))

    @property
    def operation(self) -> CompoundOperators:
        """Get the operation used to combine the individual evaluations of the child predicates.

        Returns
        -------
        LogicalSqlCompoundOperators
            The operation
        """
        return self._operation

    @property
    def children(self) -> Sequence[AbstractPredicate] | AbstractPredicate:
        """Get the child predicates that are combined in this compound predicate.

        For conjunctions and disjunctions this will be a sequence of children with at least two children. For negations
        the child predicate will be returned directly (i.e. without being wrapped in a sequence).

        Returns
        -------
        Sequence[AbstractPredicate] | AbstractPredicate
            The sequence of child predicates for ``AND`` and ``OR`` predicates, or the negated predicate for ``NOT``
            predicates.
        """
        return self._children[0] if self.operation == CompoundOperators.Not else self._children

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self._children)

    def columns(self) -> set[ColumnReference]:
        return util.set_union(child.columns() for child in self._children)

    def iterchildren(self) -> Sequence[AbstractPredicate]:
        """Provides all children contained in this predicate.

        In contrast to the `children` property, this method always returns an iterable, even for *NOT* predicates. In the
        latter case the iterable contains just a single item.

        Returns
        -------
        Sequence[AbstractPredicate]
            The children
        """
        return self._children

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(child.itercolumns() for child in self._children)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(child.iterexpressions() for child in self._children)

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        return util.set_union(child.join_partners() for child in self._children)

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return util.set_union(set(child.base_predicates()) for child in self._children)

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        match self.operation:
            case CompoundOperators.Not:
                return visitor.visit_not_predicate(self, self.children)
            case CompoundOperators.And:
                return visitor.visit_and_predicate(self, self.children)
            case CompoundOperators.Or:
                return visitor.visit_or_predicate(self, self.children)
            case _:
                raise ValueError(f"Unknown operation: '{self.operation}'")

    def _stringify_not(self) -> str:
        if not isinstance(self.children, InPredicate):
            return f"NOT {self.children}"
        return f"{self.children.column} NOT IN {self.children._stringify_values()}"

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.operation == other.operation and self.children == other.children

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        if self.operation == CompoundOperators.Not:
            return self._stringify_not()
        elif self.operation == CompoundOperators.Or:
            return "(" + " OR ".join(f"({child})" if child.is_compound() else str(child)
                                     for child in self.iterchildren()) + ")"
        elif self.operation == CompoundOperators.And:
            return " AND ".join(str(child) for child in self.children)
        else:
            raise ValueError(f"Unknown operation: '{self.operation}'")


class PredicateVisitor(abc.ABC, Generic[VisitorResult]):
    """Basic visitor to operator on arbitrary predicate trees.

    As a modification to a strict vanilla interpretation of the design pattern, we provide dedicated matching methods for the
    different composite operators (i.e. for *AND*, *OR* and *NOT* predicates), rather than just matching on
    `CompoundPredicate`.

    See Also
    --------
    AbstractPredicate

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_binary_predicate(self, predicate: BinaryPredicate) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_between_predicate(self, predicate: BetweenPredicate) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_in_predicate(self, predicate: InPredicate) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_unary_predicate(self, predicate: UnaryPredicate) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_not_predicate(self, predicate: CompoundPredicate, child_predicate: AbstractPredicate) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_or_predicate(self, predicate: CompoundPredicate, components: Sequence[AbstractPredicate]) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_and_predicate(self, predicate: CompoundPredicate, components: Sequence[AbstractPredicate]) -> VisitorResult:
        raise NotImplementedError


def as_predicate(column: ColumnReference, operation: LogicalSqlOperators,
                 *arguments) -> BasePredicate:
    """Utility method to quickly construct instances of base predicates.

    The given arguments are transformed into appropriate expression objects as necessary.

    The specific type of generated predicate is determined by the given operation. The following rules are applied:

    - for ``BETWEEN`` predicates, the arguments can be either two values, or a tuple of values
      (additional arguments are ignored)
    - for ``IN`` predicates, the arguments can be either a number of arguments, or a (nested) iterable of arguments
    - for all other binary predicates exactly one additional argument must be given (and an error is raised if that
      is not the case)

    Parameters
    ----------
    column : ColumnReference
        The column that should become the first operand of the predicate
    operation : LogicalSqlOperators
        The operation that should be used to build the predicate. The actual return type depends on this value.
    *arguments
        Further operands for the predicate. The allowed values and their structure depend on the precise predicate (see rules
        above).

    Returns
    -------
    BasePredicate
        A predicate representing the given operation on the given operands

    Raises
    ------
    ValueError
        If a binary predicate is requested, but `*arguments` does not contain a single value
    """
    column = ColumnExpression(column)

    if operation == LogicalSqlOperators.Between:
        if len(arguments) == 1:
            lower, upper = arguments[0]
        else:
            lower, upper, *__ = arguments
        return BetweenPredicate(column, (as_expression(lower), as_expression(upper)))
    elif operation == LogicalSqlOperators.In:
        arguments = util.flatten(arguments)
        return InPredicate(column, [as_expression(value) for value in arguments])
    elif len(arguments) != 1:
        raise ValueError("Too many arguments for binary predicate: " + str(arguments))

    argument = arguments[0]
    return BinaryPredicate(operation, column, as_expression(argument))


def determine_join_equivalence_classes(predicates: Iterable[BinaryPredicate]) -> set[frozenset[ColumnReference]]:
    """Calculates all equivalence classes of equijoin predicates.

    Columns are in an equivalence class if they can all be compared with matching equality predicates. For example, consider
    two predicates *a = b* and *a = c*. From these predicates it follows that *b = c* and hence the set of columns *{a, b, c}*
    is an equivalence class. Likewise, the predicates *a = b* and *c = d* form two equivalence classes, namely *{a, b}* and
    *{c, d}*.

    Parameters
    ----------
    predicates : Iterable[BinaryPredicate]
        The predicates to check. Non-equijoin predicates are discarded automatically.

    Returns
    -------
    set[frozenset[ColumnReference]]
        The equivalence classes. Each element of the set describes a complete equivalence class.
    """
    join_predicates = {pred for pred in predicates
                       if isinstance(pred, BinaryPredicate) and pred.is_join()
                       and pred.operation == LogicalSqlOperators.Equal}

    equivalence_graph = nx.Graph()
    for predicate in join_predicates:
        columns = predicate.columns()
        if not len(columns) == 2:
            continue
        col_a, col_b = columns
        equivalence_graph.add_edge(col_a, col_b)

    equivalence_classes: set[set[ColumnReference]] = set()
    for equivalence_class in nx.connected_components(equivalence_graph):
        equivalence_classes.add(frozenset(equivalence_class))
    return equivalence_classes


def generate_predicates_for_equivalence_classes(equivalence_classes: set[frozenset[ColumnReference]]) -> set[BinaryPredicate]:
    """Provides all possible equijoin predicates for a set of equivalence classes.

    This function can be used in combination with `determine_join_equivalence_classes` to expand join predicates to also
    include additional joins that can be derived from the predicates.

    For example, consider two joins *a = b* and *b = c*. These joins form one equivalence class *{a, b, c}*. Based on the
    equivalence class, the predicates *a = b*, *b = c* and *a = c* can be generated.

    Parameters
    ----------
    equivalence_classes : set[frozenset[ColumnReference]]
        The equivalence classes. Each class is described by the columns it contains.

    Returns
    -------
    set[BinaryPredicate]
        The predicates

    See Also
    --------
    determine_join_equivalence_classes
    CompoundPredicate.create_and
    """
    equivalence_predicates: set[BinaryPredicate] = set()
    for equivalence_class in equivalence_classes:
        for first_col, second_col in util.collections.pairs(equivalence_class):
            equivalence_predicates.add(as_predicate(first_col, LogicalSqlOperators.Equal, second_col))
    return equivalence_predicates


def _unwrap_expression(expression: ColumnExpression | StaticValueExpression) -> ColumnReference | object:
    """Provides the column of a `ColumnExpression` or the value of a `StaticValueExpression`.

    This is a utility method to gain quick access to the values in simple predicates.

    Parameters
    ----------
    expression : SqlExpression
        The expression to unwrap.

    Returns
    -------
    ColumnReference | object
        The column or value contained in the expression.
    """
    if isinstance(expression, StaticValueExpression):
        return expression.value
    elif isinstance(expression, ColumnExpression):
        return expression.column
    else:
        raise ValueError("Cannot unwrap expression " + str(expression))


UnwrappedFilter = tuple[ColumnReference, LogicalSqlOperators, object]
"""Type that captures the main components of a filter predicate."""


def _attempt_filter_unwrap(predicate: AbstractPredicate) -> Optional[UnwrappedFilter]:
    """Extracts the main components of a simple filter predicate to make them more directly accessible.

    This is a preparatory step in order to create instances of `SimplifiedFilterView`. Therefore, it only works for predicates
    that match the requirements of "simple" filters. This boils down to being of the form ``<column> <operator> <values>``. If
    this condition is not met, the unwrapping fails.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate that should be simplified

    Returns
    -------
    Optional[UnwrappedFilter]
        A triple consisting of column, operator and value(s) if the `predicate` could be unwrapped, or ``None`` otherwise.

    Raises
    ------
    ValueError
        If `predicate` is not a base predicate
    """
    if not predicate.is_filter() or not predicate.is_base():
        return None

    if isinstance(predicate, BinaryPredicate):
        try:
            left, right = _unwrap_expression(predicate.first_argument), _unwrap_expression(predicate.second_argument)
            operation = predicate.operation
            left, right = (left, right) if isinstance(left, ColumnReference) else (right, left)
            return left, operation, right
        except ValueError:
            return None
    elif isinstance(predicate, BetweenPredicate):
        try:
            column = _unwrap_expression(predicate.column)
            start = _unwrap_expression(predicate.interval_start)
            end = _unwrap_expression(predicate.interval_end)
            return column, LogicalSqlOperators.Between, (start, end)
        except ValueError:
            return None
    elif isinstance(predicate, InPredicate):
        try:
            column = _unwrap_expression(predicate.column)
            values = [_unwrap_expression(val) for val in predicate.values]
            return column, LogicalSqlOperators.In, tuple(values)
        except ValueError:
            return None
    else:
        raise ValueError("Unknown predicate type: " + str(predicate))


class SimplifiedFilterView(AbstractPredicate):
    """The intent behind this view is to provide more streamlined and direct access to filter predicates.

    As the name suggests, the view is a read-only predicate, i.e. it cannot be created on its own and has to be derived
    from a base predicate (either a binary predicate, a ``BETWEEN`` predicate or an ``IN`` predicate). Afterward, it provides
    read-only access to the predicate being filtered, the filter operation, as well as the values used to restrict the allowed
    column instances.

    Note that not all base predicates can be represented as a simplified view. In order for the view to work, both the
    column as well as the filter values cannot be modified by other expressions such as casts or mathematical
    expressions. As a rule of thumb, a filter has to be of the form ``<column reference> <operator> <static values>`` in order
    for the representation to work.

    The static methods `wrap`, `can_wrap` and `wrap_all` can serve as high-level access points into the view. The components
    of the view are accessible via properties.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate that should be simplified

    Raises
    ------
    ValueError
        If the `predicate` cannot be represented by a simplified view.

    Examples
    --------
    The best way to construct simplified views is to start with the `QueryPredicates` and perform a list comprehension on all
    filter predicates, i.e. ``views = SimplifiedFilterView.wrap_all(query.predicates())``. Notice that this conversion can be
    "lossy", e.g. disjunctions of filter predicates are not contained in the sequence!

    The view can only be created for "simple" predicates. These are predicates that do not make use of any advanced
    expressions. For example, the following predicates can be represented as a simplified view: ``R.a = 42``,
    ``R.b BETWEEN 1 AND 2`` or ``R.c IN (11, 22, 33)``. On the other hand, the following predicates cannot be represented b/c
    they involve advanced operations: ``R.a + 10 = 42`` (contains a mathematical expression) and ``R.a::float < 11``
    (contains a value cast).
    """

    @staticmethod
    def wrap(predicate: AbstractPredicate) -> SimplifiedFilterView:
        """Transforms a specific predicate into a simplified view. Raises an error if that is not possible.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate to represent as a simplified view

        Returns
        -------
        SimplifiedFilterView
            A simplified view wrapping the given predicate

        Raises
        ------
        ValueError
            If the predicate cannot be represented as a simplified view.
        """
        return SimplifiedFilterView(predicate)

    @staticmethod
    def can_wrap(predicate: AbstractPredicate) -> bool:
        """Checks, whether a specific predicate can be represented as a simplified view

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate to check

        Returns
        -------
        bool
            Whether a representation as a simplified view is possible.
        """
        return _attempt_filter_unwrap(predicate) is not None

    @staticmethod
    def wrap_all(predicates: Iterable[AbstractPredicate]) -> Sequence[SimplifiedFilterView]:
        """Transforms specific predicates into simplified views.

        If individual predicates cannot be represented as views, they are ignored.

        Parameters
        ----------
        predicates : Iterable[AbstractPredicate]
            The predicates to represent as views. These can be arbitrary predicates, i.e. including joins and complex filters.

        Returns
        -------
        Sequence[SimplifiedFilterView]
            The simplified views for all predicates that can be represented this way. The sequence of the views matches the
            sequence in the `predicates`. If the representation fails for individual predicates, they simply do not appear in
            the result. Therefore, this sequence may be empty if none of the predicates are valid  simplified views.
        """
        views = []
        for pred in predicates:
            try:
                v = SimplifiedFilterView.wrap(pred)
                views.append(v)
            except ValueError:
                continue
        return views

    def __init__(self, predicate: AbstractPredicate) -> None:
        column, operation, value = _attempt_filter_unwrap(predicate)
        self._column = column
        self._operation = operation
        self._value = value
        self._predicate = predicate

        hash_val = hash((column, operation, value))
        super().__init__(hash_val)

    @property
    def column(self) -> ColumnReference:
        """Get the filtered column.

        Returns
        -------
        ColumnReference
            The column
        """
        return self._column

    @property
    def operation(self) -> LogicalSqlOperators:
        """Get the SQL operation that is used for the filter (e.g. ``IN`` or ``<>``).

        Returns
        -------
        LogicalSqlOperators
            the operator. This cannot be ``EXISTS`` or ``MISSING``, since subqueries cannot be represented in simplified
            views.
        """
        return self._operation

    @property
    def value(self) -> object | tuple[object] | Sequence[object]:
        """Get the filter value.

        Returns
        -------
        object | tuple[object] | Sequence[object]
            The value. For a binary predicate, this is just the value itself. For a ``BETWEEN`` predicate, this is tuple of the
            form ``(lower, upper)`` and for an ``IN`` predicate, this is a sequence of the allowed values.
        """
        return self._value

    def unwrap(self) -> AbstractPredicate:
        """Get the original predicate that is represented by this view.

        Returns
        -------
        AbstractPredicate
            The original predicate
        """
        return self._predicate

    def is_compound(self) -> bool:
        return False

    def is_join(self) -> bool:
        return False

    def columns(self) -> set[ColumnReference]:
        return {self.column}

    def itercolumns(self) -> Iterable[ColumnReference]:
        return [self.column]

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def join_partners(self) -> set[tuple[ColumnReference, ColumnReference]]:
        return set()

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        return self._predicate.accept_visitor(visitor)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.unwrap() == other.unwrap()

    def __str__(self) -> str:
        return str(self.unwrap())


def _collect_filter_predicates(predicate: AbstractPredicate) -> set[AbstractPredicate]:
    """Provides all base filter predicates that are contained in a specific predicate hierarchy.

    To determine, whether a given predicate is a join or a filter, the `AbstractPredicate.is_filter` method is used.

    This method handles compound predicates as follows:

    - conjunctions are un-nested, i.e. all predicates that form an ``AND`` predicate are collected individually
    - ``OR`` predicates are included with exactly those predicates from their children that are filters. If this is only true
       for a single predicate, that predicate will be returned directly.
    - ``NOT`` predicates are included if their child predicate is a filter

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate hierarchy to unwrap.

    Returns
    -------
    set[AbstractPredicate]
        The filter predicates that are contained in the `predicate`.

    Raises
    ------
    ValueError
        If a compound predicate has an unknown operation. This indicates a programming error or a broken invariant.
    ValueError
        If the given `predicate` is neither a `BasePredicate`, nor a `CompoundPredicate`. This indicates a modification of the
        predicate class hierarchy without the necessary adjustments to the consuming methods.

    See Also
    --------
    _collect_join_predicates
    """
    if isinstance(predicate, BasePredicate):
        return {predicate} if predicate.is_filter() else set()
    elif isinstance(predicate, CompoundPredicate):
        if predicate.operation == CompoundOperators.Or:
            or_filter_children = [child_pred for child_pred in predicate.children if child_pred.is_filter()]
            if len(or_filter_children) < 2:
                return set(or_filter_children)
            or_filters = CompoundPredicate(CompoundOperators.Or, or_filter_children)
            return {or_filters}
        elif predicate.operation == CompoundOperators.Not:
            not_filter_children = predicate.children if predicate.children.is_filter() else None
            if not not_filter_children:
                return set()
            return {predicate}
        elif predicate.operation == CompoundOperators.And:
            return util.set_union([_collect_filter_predicates(child) for child in predicate.children])
        else:
            raise ValueError(f"Unknown operation: '{predicate.operation}'")
    else:
        raise ValueError(f"Unknown predicate type: {predicate}")


def _collect_join_predicates(predicate: AbstractPredicate) -> set[AbstractPredicate]:
    """Provides all join predicates that are contained in a specific predicate hierarchy.

    To determine, whether a given predicate is a join or a filter, the `AbstractPredicate.is_join` method is used.

    This method handles compound predicates as follows:

    - conjunctions are un-nested, i.e. all predicates that form an ``AND`` predicate are collected individually
    - ``OR`` predicates are included with exactly those predicates from their children that are joins. If this is only true for
       a single predicate, that predicate will be returned directly.
    - ``NOT`` predicates are included if their child predicate is a join

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate hierarchy to unwrap.

    Returns
    -------
    set[AbstractPredicate]
        The join predicates that are contained in the `predicate`

    Raises
    ------
    ValueError
        If a compound predicate has an unknown operation. This indicates a programming error or a broken invariant.
    ValueError
        If the given `predicate` is neither a `BasePredicate`, nor a `CompoundPredicate`. This indicates a modification of the
        predicate class hierarchy without the necessary adjustments to the consuming methods.

    See Also
    --------
    _collect_filter_predicates
    """
    if isinstance(predicate, BasePredicate):
        return {predicate} if predicate.is_join() else set()
    elif isinstance(predicate, CompoundPredicate):
        if predicate.operation == CompoundOperators.Or:
            or_join_children = [child_pred for child_pred in predicate.children if child_pred.is_join()]
            if len(or_join_children) < 2:
                return set(or_join_children)
            or_joins = CompoundPredicate(CompoundOperators.Or, or_join_children)
            return {or_joins}
        elif predicate.operation == CompoundOperators.Not:
            not_join_children = predicate.children if predicate.children.is_join() else None
            if not not_join_children:
                return set()
            return {predicate}
        elif predicate.operation == CompoundOperators.And:
            return util.set_union([_collect_join_predicates(child) for child in predicate.children])
        else:
            raise ValueError(f"Unknown operation: '{predicate.operation}'")
    else:
        raise ValueError(f"Unknown predicate type: {predicate}")


class QueryPredicates:
    """The query predicates provide high-level access to all the different predicates in a query.

    Generally speaking, this class provides the most user-friendly access into the predicate hierarchy and should be sufficient
    for most use-cases. The provided methods revolve around identifying filter and join predicates easily, as well finding the
    predicates that are specified on specific tables.

    To distinguish between filter predicates and join predicates, the default logic provided by the `AbstractPredicate` is
    used (see `AbstractPredicate.is_join`). If this distinction does not work for a specific usage scenario, a custom strategy
    to recognize filters and joins can be provided. This is done by creating a subclass of `QueryPredicates` and setting the
    global `DefaultPredicateHandler` variable to that class. Afterwards, each call to `predicates` on a new `SqlQuery` object
    will use the updated handler. Notice however, that any change to the defaut handler should also involve changes to the
    join and filter check of the actual predicates. Otherwise, the query predicates might provide a predicate as a join
    predicate, but calling `is_join` on that predicate returns ``False``. This could break application behaviour.

    Parameters
    ----------
    root : Optional[AbstractPredicate]
        The root predicate of the predicate hierarchy that should be represented by the `QueryPredicates`. Typically, this is
        a conjunction of the actual predicates.
    """

    @staticmethod
    def empty_predicate() -> QueryPredicates:
        """Constructs a new predicates instance without any actual content.

        Returns
        -------
        QueryPredicates
            The predicates wrapper
        """
        return QueryPredicates(None)

    def __init__(self, root: Optional[AbstractPredicate]):
        self._root = root
        self._hash_val = hash(self._root)
        self._join_predicate_map = self._init_join_predicate_map()

    @property
    def root(self) -> AbstractPredicate:
        """Get the root predicate that represents the entire predicate hierarchy.

        Typically, this is a conjunction of the actual predicates. This conjunction can be used to start a custom traversal of
        the predicate hierarchy.

        Returns
        -------
        AbstractPredicate
            The root predicate

        Raises
        ------
        errors.StateError
            If the predicates warpper is empty and there is no root predicate.
        """
        self._assert_not_empty()
        return self._root

    def is_empty(self) -> bool:
        """Checks, whether this predicate wrapper contains any actual predicates.

        Returns
        -------
        bool
            Whether at least one predicate was specified.
        """
        return self._root is None

    @functools.cache
    def filters(self) -> Collection[AbstractPredicate]:
        """Provides all filter predicates that are contained in the predicate hierarchy.

        By default, the distinction between filters and joins that is defined in `AbstractPredicate.is_join` is used. However,
        this behaviour can be changed by subclasses.

        This method handles compound predicates as follows:

        - conjunctions are un-nested, i.e. all predicates that form an ``AND`` predicate are collected individually
        - ``OR`` predicates are included with exactly those predicates from their children that are filters. If this is only
          true for a single predicate, that predicate will be returned directly.
        - ``NOT`` predicates are included if their child predicate is a filter.

        Returns
        -------
        Collection[AbstractPredicate]
            The filter predicates.
        """
        return _collect_filter_predicates(self._root)

    @functools.cache
    def joins(self) -> Collection[AbstractPredicate]:
        """Provides all join predicates that are contained in the predicate hierarchy.

        By default, the distinction between filters and joins that is defined in `AbstractPredicate.is_join` is used. However,
        this behaviour can be changed by subclasses.

        This method handles compound predicates as follows:

        - conjunctions are un-nested, i.e. all predicates that form an ``AND`` predicate are collected individually
        - ``OR`` predicates are included with exactly those predicates from their children that are joins. If this is only true
          for a single predicate, that predicate will be returned directly.
        - ``NOT`` predicates are included if their child predicate is a join.

        Returns
        -------
        Collection[AbstractPredicate]
            The join predicates
        """
        return _collect_join_predicates(self._root)

    @functools.cache
    def join_graph(self) -> nx.Graph:
        """Provides the join graph for the predicates.

        A join graph is an undirected graph, where each node corresponds to a base table and each edge corresponds to a
        join predicate between two base tables. In addition, each node is annotated by a ``predicate`` key which is a
        conjunction of all filter predicates on that table (or ``None`` if the table is unfiltered). Likewise, each edge is
        annotated by a ``predicate`` key that corresponds to the join predicate (which can never be ``None``).

        Returns
        -------
        nx.Graph
            The join graph.

        Notes
        -----
        Since our implementation of a join graph is not based on a multigraph, only binary joins can be represented. The
        determination of edges is based on `AbstractPredicate.join_partners`.
        """
        join_graph = nx.Graph()
        if self.is_empty():
            return join_graph

        for table in self._root.tables():
            filter_predicates = self.filters_for(table)
            join_graph.add_node(table, predicate=filter_predicates)

        for join in self.joins():
            for first_col, second_col in join.join_partners():
                join_graph.add_edge(first_col.table, second_col.table, predicate=join)

        return join_graph.copy()

    @functools.cache
    def filters_for(self, table: TableReference) -> Optional[AbstractPredicate]:
        """Provides all filter predicates that reference a specific table.

        If multiple individual filter predicates are specified in the query, they will be combined in one large
        conjunction.

        The determination of matching filter predicates is the same as for the `filters()` method.

        Parameters
        ----------
        table : TableReference
            The table to retrieve the filters for

        Returns
        -------
        Optional[AbstractPredicate]
            A (conjunction of) the filter predicates of the `table`, or ``None`` if the table is unfiltered.
        """
        if self.is_empty():
            return None
        applicable_filters = [filter_pred for filter_pred in self.filters() if filter_pred.contains_table(table)]
        return CompoundPredicate.create_and(applicable_filters) if applicable_filters else None

    @functools.cache
    def joins_for(self, table: TableReference) -> Collection[AbstractPredicate]:
        """Provides all join predicates that reference a specific table.

        Each entry in the resulting collection is a join predicate between the given table and a (set of) partner tables, such
        that the partner tables in different entries in the collection are also different. If multiple join predicates are
        specified between the given table and a specific (set of) partner tables, these predicates are aggregated into one
        large conjunction.

        The determination of matching join predicates is the same as for the `joins()` method.

        Parameters
        ----------
        table : TableReference
            The table to retrieve the joins for

        Returns
        -------
        Collection[AbstractPredicate]
            The join predicates with `table`. If there are no such predicates, the collection is empty.
        """
        if self.is_empty():
            return []

        applicable_joins: list[AbstractPredicate] = [join_pred for join_pred in self.joins()
                                                     if join_pred.contains_table(table)]
        distinct_joins: dict[frozenset[TableReference], list[AbstractPredicate]] = collections.defaultdict(list)
        for join_predicate in applicable_joins:
            partners = {column.table for column in join_predicate.join_partners_of(table)}
            distinct_joins[frozenset(partners)].append(join_predicate)

        aggregated_predicates = []
        for join_group in distinct_joins.values():
            aggregated_predicates.append(CompoundPredicate.create_and(join_group))
        return aggregated_predicates

    def joins_between(self, first_table: TableReference | Iterable[TableReference],
                      second_table: TableReference | Iterable[TableReference], *,
                      _computation: Literal["legacy", "graph", "map"] = "map") -> Optional[AbstractPredicate]:
        """Provides the (conjunctive) join predicate that joins specific tables.

        The precise behaviour of this method depends on the provided parameters: If `first_table` or `second_table` contain
        multiple tables, all join predicates between tables from the different sets are returned (but joins from tables within
        `first_table` or from tables within `second_table` are not).

        Notice that the returned predicate might also include other tables, if they are part of a join predicate that
        also joins the given two tables.

        The performance of this method can be crucial for some applications, since the join check is often part of a very hot
        loop. Therefore, a number of different calculation strategies are implemented. If profiling shows that the application
        is slowed down heavily by the current strategy, it might be a good idea to switch to another algorithm. This can be
        achieved using the `_computation` parameter.

        Parameters
        ----------
        first_table : TableReference | Iterable[TableReference]
            The (set of) tables to join
        second_table : TableReference | Iterable[TableReference]
            The (set of) join partners for `first_table`.
        _computation : Literal[&quot;legacy&quot;, &quot;graph&quot;, &quot;map&quot;], optional
            The specific algorithm to use for determining the join partners. The algorithms have very different performance
            and memory charactersistics. The default ``"map"`` setting is usually the fastest. It should only really be changed
            if there are very good reasons for it. The following settings exist:

            - *legacy*: uses a recursive strategy in case `first_table` or `second_table` contain multiple references.
              This is pretty slow, but easy to debug
            - *graph*: builds an internal join graph and merges the involved nodes from `first_table` and `second_table` to
              extract the overall join predicate directly.
            - *map*: stores a mapping between tables and their join predicates and merges these mappings to determine the
              overall join predicate

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, ``None`` is returned.

        Raises
        ------
        ValueError
            If the `_computation` strategy is none of the allowed values
        """
        if self.is_empty():
            return None

        if _computation == "legacy":
            return self._legacy_joins_between(first_table, second_table)
        elif _computation == "graph":
            return self._graph_based_joins_between(first_table, second_table)
        elif _computation == "map":
            return self._map_based_joins_between(first_table, second_table)
        else:
            raise ValueError("Unknown computation method. Allowed values are 'legacy', 'graph', or 'map', ",
                             f"not '{_computation}'")

    def joins_tables(self, tables: TableReference | Iterable[TableReference],
                     *more_tables: TableReference) -> bool:
        """Checks, whether specific tables are all joined with each other.

        This does not mean that there has to be a join predicate between each pair of tables, but rather that all pairs
        of tables must at least be connected through a sequence of other tables. From a graph-theory centric point of
        view this means that the join (sub-) graph induced by the given tables is connected.

        Parameters
        ----------
        tables : TableReference | Iterable[TableReference]
            The tables to check.
        *more_tables
            Additional tables that also should be included in the check. This parameter is mainly for convenience usage in
            interactive scenarios.

        Returns
        -------
        bool
            Whether the given tables can be joined without any cross product

        Raises
        ------
        ValueError
            If `tables` and `more_tables` is both empty

        Examples
        --------
        The following calls are exactly equivalent:

        .. code-block:: python

            >>> predicates.joins_tables([table1, table2, table3])
            >>> predicates.joins_tables(table1, table2, table3)
            >>> predicates.joins_table([table1, table2], table3)

        """
        if self.is_empty():
            return False

        tables = [tables] if not isinstance(tables, Iterable) else list(tables)
        tables = frozenset(set(tables) | set(more_tables))

        if not tables:
            raise ValueError("Cannot perform check for empty tables")
        if len(tables) == 1:
            table = util.simplify(tables)
            return table in self._root.tables()

        return self._join_tables_check(tables)

    def and_(self, other_predicate: QueryPredicates | AbstractPredicate) -> QueryPredicates:
        """Combines the current predicates with additional predicates, creating a conjunction of the two predicates.

        The input predicates, as well as the current predicates object are not modified. All changes are applied to the
        resulting predicates object.

        Parameters
        ----------
        other_predicate : QueryPredicates | AbstractPredicate
            The predicates to combine. Can also be an `AbstractPredicate`, in which case this predicate is used as the root
            for the other predicates instance.

        Returns
        -------
        QueryPredicates
            The merged predicates wrapper. Its root is roughly equivalent to ``self.root AND other_predicate.root``.
        """
        if self.is_empty() and isinstance(other_predicate, QueryPredicates) and other_predicate.is_empty():
            return self
        elif isinstance(other_predicate, QueryPredicates) and other_predicate.is_empty():
            return self

        other_predicate = other_predicate._root if isinstance(other_predicate, QueryPredicates) else other_predicate
        if self.is_empty():
            return QueryPredicates(other_predicate)

        merged_predicate = CompoundPredicate.create_and([self._root, other_predicate])
        return QueryPredicates(merged_predicate)

    @functools.cache
    def _join_tables_check(self, tables: frozenset[TableReference]) -> bool:
        """Constructs the join graph for the given tables and checks, whether it is connected.

        Parameters
        ----------
        tables : frozenset[TableReference]
            The tables to check. Has to be a frozenset to enable caching via `functools.cache`

        Returns
        -------
        bool
            Whether the specified tables induce a connected subgraph of the full join graph.
        """
        join_graph = nx.Graph()
        join_graph.add_nodes_from(tables)
        for table in tables:
            for join in self.joins_for(table):
                partner_tables = set(col.table for col in join.join_partners_of(table)) & tables
                join_graph.add_edges_from(itertools.product([table], partner_tables))
        return nx.is_connected(join_graph)

    def _assert_not_empty(self) -> None:
        """Ensures that a root predicate is set

        Raises
        ------
        StateError
            If there is no root predicate
        """
        if self._root is None:
            raise StateError("No query predicates!")

    def _init_join_predicate_map(self) -> dict[frozenset[TableReference], AbstractPredicate]:
        """Generates the necessary mapping for `_map_based_joins_between`.

        This is a static data structure and hence can be pre-computed.

        Returns
        -------
        dict[frozenset[TableReference], AbstractPredicate]
            A mapping from a set of tables to the join predicate that is specified between those tables. If a set of tables
            does not appear in the dictionary, there is no join predicate between the specific tables.
        """
        if self.is_empty():
            return {}

        predicate_map: dict[frozenset[TableReference], AbstractPredicate] = {}
        for table in self._root.tables():
            join_partners = self.joins_for(table)
            for join_predicate in join_partners:
                partner_tables = {partner.table for partner in join_predicate.join_partners_of(table)}
                map_key = frozenset(partner_tables | {table})
                if map_key in predicate_map:
                    continue
                predicate_map[map_key] = join_predicate
        return predicate_map

    def _legacy_joins_between(self, first_table: TableReference | Iterable[TableReference],
                              second_table: TableReference | Iterable[TableReference]
                              ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined using the legacy recursive structure.

        .. deprecated::
            There is no real advantage of using this method, other than slightly easier debugging. Should be removed at some
            later point in time (in which case the old ``"legacy"`` strategy key will be re-mapped to a different) strategy.

        Parameters
        ----------
        first_table : TableReference | Iterable[TableReference]
            The (set of) tables to join
        second_table : TableReference | Iterable[TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, ``None`` is returned.
        """
        if isinstance(first_table, TableReference) and isinstance(second_table, TableReference):
            if first_table == second_table:
                return None
            first_joins: Collection[AbstractPredicate] = self.joins_for(first_table)
            matching_joins = {join for join in first_joins if join.joins_table(second_table)}
            return CompoundPredicate.create_and(matching_joins) if matching_joins else None

        matching_joins = set()
        first_table, second_table = util.enlist(first_table), util.enlist(second_table)
        for first in frozenset(first_table):
            for second in frozenset(second_table):
                join_predicate = self.joins_between(first, second)
                if not join_predicate:
                    continue
                matching_joins.add(join_predicate)
        return CompoundPredicate.create_and(matching_joins) if matching_joins else None

    def _graph_based_joins_between(self, first_table: TableReference | Iterable[TableReference],
                                   second_table: TableReference | Iterable[TableReference]
                                   ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined using a graph-based approach.

        Parameters
        ----------
        first_table : TableReference | Iterable[TableReference]
            The (set of) tables to join
        second_table : TableReference | Iterable[TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, ``None`` is returned.

        """
        join_graph = self.join_graph()
        first_table, second_table = util.enlist(first_table), util.enlist(second_table)

        if len(first_table) > 1:
            first_first_table, *remaining_first_tables = first_table
            for remaining_first_table in remaining_first_tables:
                join_graph = nx.contracted_nodes(join_graph, first_first_table, remaining_first_table)
            first_hook = first_first_table
        else:
            first_hook = util.simplify(first_table)

        if len(second_table) > 1:
            first_second_table, *remaining_second_tables = second_table
            for remaining_second_table in remaining_second_tables:
                join_graph = nx.contracted_nodes(join_graph, first_second_table, remaining_second_table)
            second_hook = first_second_table
        else:
            second_hook = util.simplify(second_table)

        if (first_hook, second_hook) not in join_graph.edges:
            return None
        return join_graph.edges[first_hook, second_hook]["predicate"]

    def _map_based_joins_between(self, first_table: TableReference | Iterable[TableReference],
                                 second_table: TableReference | Iterable[TableReference]
                                 ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined together using a map-based approach.

        This method is the preferred way of inferring the join predicate from the two candidate sets. It is based on static
        map data that was precomputed as part of `_init_join_predicate_map`.

        Parameters
        ----------
        first_table : TableReference | Iterable[TableReference]
            The (set of) tables to join
        second_table : TableReference | Iterable[TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, ``None`` is returned.

        """
        join_predicates = set()
        first_table, second_table = util.enlist(first_table), util.enlist(second_table)
        for first in first_table:
            for second in second_table:
                map_key = frozenset((first, second))
                if map_key not in self._join_predicate_map:
                    continue
                current_predicate = self._join_predicate_map[map_key]
                join_predicates.add(current_predicate)
        if not join_predicates:
            return None
        return CompoundPredicate.create_and(join_predicates)

    def __iter__(self) -> Iterator[AbstractPredicate]:
        return (list(self.filters()) + list(self.joins())).__iter__()

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._root == other._root

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._root)


DefaultPredicateHandler: Type[QueryPredicates] = QueryPredicates
"""
The DefaultPredicateHandler designates which QueryPredicates object should be constructed when a query is asked for its
predicates. Changing this variable means an instance of a different class will be instantiated.
This might be useful if another differentiation between join and filter predicates is required by a specific use case.

See Also
--------
QueryPredicates
AbstractPredicate.is_join
"""


class BaseClause(abc.ABC):
    """Basic interface shared by all supported clauses.

    This really is an abstract interface, not a usable clause. All inheriting clauses have to provide their own
    `__eq__` method and re-use the `__hash__` method provided by the base clause. Remember to explicitly set this up!
    The concrete hash value is constant since the clause itself is immutable. It is up to the implementing class to
    make sure that the equality/hash consistency is enforced.

    Parameters
    ----------
    hash_val : int
        The hash of the concrete clause object
    """

    def __init__(self, hash_val: int):
        self._hash_val = hash_val

    def tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in the clause.

        Returns
        -------
        set[TableReference]
            All tables. This includes virtual tables if such tables are present in the clause
        """
        return util.set_union(expression.tables() for expression in self.iterexpressions())

    @abc.abstractmethod
    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced in the clause.

        Returns
        -------
        set[ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all directly contained expressions in this clause.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides access to all column in this clause.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n*
        times in this clause, it will also be returned *n* times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this clause.

        Returns
        -------
        Iterable[ColumnReference]
            All columns exactly in the order in which they are used
        """
        raise NotImplementedError

    @abc.abstractmethod
    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        """Enables processing of the current clause by a visitor.

        Parameters
        ----------
        visitor : ClauseVisitor[VisitorResult]
            The visitor
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class Hint(BaseClause):
    """Hint block of a clause.

    Depending on the SQL dialect, these hints will be placed at different points in the query. Furthermore, the precise
    contents (i.e. syntax and semantic) vary from database system to system.

    Hints are differentiated in two parts:

    - preparatory statements can be executed as valid commands on the database system, e.g. optimizer settings, etc.
    - query hints are the actual hints. Typically, these will be inserted as comments at some place in the query.

    These two parts are set as parameters in the `__init__` method and are available as read-only properties
    afterwards.

    Parameters
    ----------
    preparatory_statements : str, optional
        Statements that configure the optimizer and have to be run *before* the actual query is executed. Such settings
        often configure the optimizer for the entire session and can thus influence other queries as well. Defaults to
        an empty string, which indicates that there are no such settings.
    query_hints : str, optional
        Hints that configure the optimizer, often for an individual join. These hints are executed as part of the
        actual query.

    Examples
    --------
    A hint clause for MySQL could look like this:

    .. code-block::sql

        SET optimizer_switch = 'block_nested_loop=off';
        SELECT /*+ HASH_JOIN(R S) */ R.a
        FROM R, S, T
        WHERE R.a = S.b AND S.b = T.c

    This enforces the join between tables *R* and *S* to be executed as a hash join (due to the query hint) and
    disables usage of the block nested-loop join for the entire query (which in this case only affects the join between
    tables *S* and *T*) due to the preparatory ``SET optimizer_switch`` statement.
    """

    def __init__(self, preparatory_statements: str = "", query_hints: str = ""):
        self._preparatory_statements = preparatory_statements
        self._query_hints = query_hints

        hash_val = hash((preparatory_statements, query_hints))
        super().__init__(hash_val)

    @property
    def preparatory_statements(self) -> str:
        """Get the string of preparatory statements. Can be empty.

        Returns
        -------
        str
            The preparatory statements. If these are multiple statements, they are concatenated into a single string
            with appropriate separator characters between them.
        """
        return self._preparatory_statements

    @property
    def query_hints(self) -> str:
        """Get the query hint text. Can be empty.

        Returns
        -------
        str
            The hints. The string has to be understood as-is by the target database system. If multiple hints are used,
            they have to be concatenated into a single string with appropriate separator characters between them.
            Correspondingly, if the hint blocks requires a specific prefix/suffix (e.g. comment syntax), this has to
            be part of the string as well.
        """
        return self._query_hints

    def columns(self) -> set[ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_hint_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.preparatory_statements == other.preparatory_statements
                and self.query_hints == other.query_hints)

    def __str__(self) -> str:
        if self.preparatory_statements and self.query_hints:
            return self.preparatory_statements + "\n" + self.query_hints
        elif self.preparatory_statements:
            return self.preparatory_statements
        return self.query_hints


class Explain(BaseClause):
    """``EXPLAIN`` block of a query.

    ``EXPLAIN`` queries change the execution mode of a query. Instead of focusing on the actual query result, an
    ``EXPLAIN`` query produces information about the internal processes of the database system. Typically, this
    includes which execution plan the DBS would choose for the query. Additionally, ``EXPLAIN ANALYZE`` (as for example
    supported by Postgres) provides the query plan and executes the actual query. The returned plan is then annotated
    by how the optimizer predictions match reality. Furthermore, such ``ANALYZE`` plans typically also contain some
    runtime statistics such as runtime of certain operators.

    Notice that there is no ``EXPLAIN`` keyword in the SQL standard, but all major database systems provide this
    functionality. Nevertheless, the precise syntax and semantic of an ``EXPLAIN`` statement depends on the actual DBS.
    The Explain clause object is modeled after Postgres and needs to adapted accordingly for different systems (see
    `db.HintService`). Especially the ``EXPLAIN ANALYZE`` variant is not supported by all systems.

    Parameters
    ----------
    analyze : bool, optional
        Whether the query should not only be executed as an ``EXPLAIN`` query, but rather as an ``EXPLAIN ANALYZE``
        query. Defaults to ``False`` which runs the query as a pure ``EXPLAIN`` query.
    target_format : Optional[str], optional
        The desired output format of the query plan, if this is supported by the database system. Defaults to ``None``
        which normally forces the default output format.

    See Also
    --------
    postbound.db.db.HintService.format_query

    References
    ----------

    .. PostgreSQL ``EXPLAIN`` command: https://www.postgresql.org/docs/current/sql-explain.html
    """

    @staticmethod
    def explain_analyze(format_type: str = "JSON") -> Explain:
        """Constructs an ``EXPLAIN ANALYZE`` clause with the specified output format.

        Parameters
        ----------
        format_type : str, optional
            The output format, by default ``"JSON"``

        Returns
        -------
        Explain
            The explain clause
        """
        return Explain(True, format_type)

    @staticmethod
    def plan(format_type: str = "JSON") -> Explain:
        """Constructs a pure ``EXPLAIN`` clause (i.e. without ``ANALYZE``) with the specified output format.

        Parameters
        ----------
        format_type : str, optional
            The output format, by default ``"JSON"``

        Returns
        -------
        Explain
            The explain clause
        """
        return Explain(False, format_type)

    def __init__(self, analyze: bool = False, target_format: Optional[str] = None):
        self._analyze = analyze
        self._target_format = target_format if target_format != "" else None

        hash_val = hash((analyze, target_format))
        super().__init__(hash_val)

    @property
    def analyze(self) -> bool:
        """Check, whether the query should be executed as ``EXPLAIN ANALYZE`` rather than just plain ``EXPLAIN``.

        Usually, ``EXPLAIN ANALYZE`` executes the query and gathers extensive runtime statistics (e.g. comparing
        estimated vs. true cardinalities for intermediate nodes).

        Returns
        -------
        bool
            Whether ``ANALYZE`` mode is enabled
        """
        return self._analyze

    @property
    def target_format(self) -> Optional[str]:
        """Get the target format in which the ``EXPLAIN`` plan should be provided.

        Returns
        -------
        Optional[str]
            The output format, or ``None`` if this is not specified. This is never an empty string.
        """
        return self._target_format

    def columns(self) -> set[ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_explain_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.analyze == other.analyze
                and self.target_format == other.target_format)

    def __str__(self) -> str:
        explain_prefix = "EXPLAIN"
        explain_body = ""
        if self.analyze and self.target_format:
            explain_body = f" (ANALYZE, FORMAT {self.target_format})"
        elif self.analyze:
            explain_body = " ANALYZE"
        elif self.target_format:
            explain_body = f" (FORMAT {self.target_format})"
        return explain_prefix + explain_body


class WithQuery:
    """A single common table expression that can be referenced in the actual query.

    Each ``WITH`` clause can consist of multiple auxiliary common table expressions. This class models exactly one
    such query. It consists of the query as well as the name under which the temporary table can be referenced
    in the actual SQL query.

    Parameters
    ----------
    query : SqlQuery
        The query that should be used to construct the temporary common table.
    target_name : str | TableReference
        The name under which the table should be made available. If a table reference is provided, its identifier will be used.

    Raises
    ------
    ValueError
        If the `target_name` is empty
    """
    def __init__(self, query: SqlQuery, target_name: str | TableReference) -> None:
        if not target_name:
            raise ValueError("Target name is required")
        self._query = query
        self._subquery_expression = SubqueryExpression(query)
        self._target_name = target_name if isinstance(target_name, str) else target_name.identifier()
        self._hash_val = hash((query, target_name))

    @property
    def query(self) -> SqlQuery:
        """The query that is used to construct the temporary table

        Returns
        -------
        SqlQuery
            The query
        """
        return self._query

    @property
    def subquery(self) -> SubqueryExpression:
        """Provides the query that constructsd the temporary table as a subquery object.

        Returns
        -------
        SubqueryExpression
            The subquery
        """
        return self._subquery_expression

    @property
    def target_name(self) -> str:
        """The table name under which the temporary table can be referenced in the actual SQL query

        Returns
        -------
        str
            The name. Will never be empty.
        """
        return self._target_name

    @property
    def target_table(self) -> TableReference:
        """The table under which the temporary CTE table can be referenced in the actual SQL query

        The only difference to `target_name` is the type of this property: it provides a proper (virtual) table
        reference object

        Returns
        -------
        TableReference
            The table. Will always be a virtual table.
        """
        return TableReference.create_virtual(self.target_name)

    def tables(self) -> set[TableReference]:
        return self._query.tables()

    def columns(self) -> set[ColumnReference]:
        return self._query.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self._subquery_expression]

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._query.itercolumns()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._target_name == other._target_name
                and self._query == other._query)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        query_str = self._query.stringify(trailing_delimiter=False)
        return f"{self._target_name} AS ({query_str})"


class CommonTableExpression(BaseClause):
    """The ``WITH`` clause of a query, consisting of at least one CTE query.

    Parameters
    ----------
    with_queries : Iterable[WithQuery]
        The common table expressions that form the WITH clause.

    Raises
    ------
    ValueError
        If `with_queries` does not contain any CTE

    Warnings
    --------
    The `tables()` method provides all tables that are referenced as part of the CTEs as well as their aliases.
    The `referenced_tables()` method does not include the CTE aliases. Likewise, the `aliases()` method provides all the
    aliases, but not the tables that are referenced within the CTEs.
    """
    def __init__(self, with_queries: Iterable[WithQuery]):
        self._with_queries = tuple(with_queries)
        if not self._with_queries:
            raise ValueError("With queries cannnot be empty")
        super().__init__(hash(self._with_queries))

    @property
    def queries(self) -> Sequence[WithQuery]:
        """Get CTEs that form the ``WITH`` clause

        Returns
        -------
        Sequence[WithQuery]
            The CTEs in the order in which they were originally specified.
        """
        return self._with_queries

    def tables(self) -> set[TableReference]:
        all_tables: set[TableReference] = set()
        for cte in self._with_queries:
            all_tables |= cte.tables() | {cte.target_table}
        return all_tables

    def referenced_tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in the CTEs. This does not include the CTE aliases.

        Returns
        -------
        set[TableReference]
            The tables.
        """
        return util.set_union(cte.tables() for cte in self._with_queries)

    def aliases(self) -> set[TableReference]:
        """Provides all aliases that are used in the CTEs.

        Returns
        -------
        set[TableReference]
            The aliases.
        """
        return {cte.target_table for cte in self._with_queries}

    def columns(self) -> set[ColumnReference]:
        return util.set_union(with_query.columns() for with_query in self._with_queries)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(with_query.iterexpressions() for with_query in self._with_queries)

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(with_query.itercolumns() for with_query in self._with_queries)

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_cte_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._with_queries == other._with_queries

    def __str__(self) -> str:
        query_str = ", ".join(str(with_query) for with_query in self._with_queries)
        return "WITH " + query_str


class BaseProjection:
    """The `BaseProjection` forms the fundamental building block of a ``SELECT`` clause.

    Each ``SELECT`` clause is composed of at least one base projection. Each projection can be an arbitrary
    `SqlExpression` (rules and restrictions of the SQL standard are not enforced here). In addition, each projection
    can receive a target name as in ``SELECT foo AS f FROM bar``.

    Parameters
    ----------
    expression : SqlExpression
        The expression that is used to calculate the column value. In the simplest case, this can just be a
        `ColumnExpression`, which provides the column values directly.
    target_name : str, optional
        An optional name under which the column should be accessible. Defaults to an empty string, which indicates that
        the original column value or a system-specific modification of that value should be used. The latter case
        mostly applies to columns which are modified in some way, e.g. by a mathematical expression or a function call.
        Depending on the specific database system, the default column name could just be the function name, or the
        function name along with all its parameters.

    """

    @staticmethod
    def count_star() -> BaseProjection:
        """Shortcut method to create a ``COUNT(*)`` projection.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(FunctionExpression("count", [StarExpression()]))

    @staticmethod
    def star() -> BaseProjection:
        """Shortcut method to create a ``*`` (as in ``SELECT * FROM R``) projection.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(StarExpression())

    @staticmethod
    def column(col: ColumnReference, target_name: str = "") -> BaseProjection:
        """Shortcut method to create a projection for a specific column.

        Parameters
        ----------
        col : ColumnReference
            The column that should be projected
        target_name : str, optional
            An optional name under which the column should be available.

        Returns
        -------
        BaseProjection
            The projection
        """
        return BaseProjection(ColumnExpression(col), target_name)

    def __init__(self, expression: SqlExpression, target_name: str = ""):
        if not expression:
            raise ValueError("Expression must be set")
        self._expression = expression
        self._target_name = target_name
        self._hash_val = hash((expression, target_name))

    @property
    def expression(self) -> SqlExpression:
        """Get the expression that forms the column.

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._expression

    @property
    def target_name(self) -> str:
        """Get the alias under which the column should be accessible.

        Can be empty to indicate the absence of a target name.

        Returns
        -------
        str
            The name
        """
        return self._target_name

    def columns(self) -> set[ColumnReference]:
        return self.expression.columns()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.expression.itercolumns()

    def tables(self) -> set[TableReference]:
        return self.expression.tables()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.expression == other.expression and self.target_name == other.target_name)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self.target_name:
            return str(self.expression)
        return f"{self.expression} AS {self.target_name}"


class SelectType(enum.Enum):
    """Indicates the specific type of the ``SELECT`` clause."""

    Select = "SELECT"
    """Plain projection without duplicate removal."""

    SelectDistinct = "SELECT DISTINCT"
    """Projection with duplicate elimination."""


class Select(BaseClause):
    """The ``SELECT`` clause of a query.

    This is the only required part of a query. Everything else is optional and can be left out. (Notice that PostBOUND
    is focused on SPJ-queries, hence there are no ``INSERT``, ``UPDATE``, or ``DELETE`` queries)

    A ``SELECT`` clause simply consists of a number of individual projections (see `BaseProjection`), its `targets`.

    Parameters
    ----------
    targets : BaseProjection | Sequence[BaseProjection]
        The individual projection(s) that form the ``SELECT`` clause
    projection_type : SelectType, optional
        The kind of projection that should be performed (i.e. with duplicate elimination or without). Defaults
        to a `SelectType.Select`, which is a plain projection without duplicate removal.

    Raises
    ------
    ValueError
        If the `targets` are empty.
    """

    @staticmethod
    def count_star() -> Select:
        """Shortcut method to create a ``SELECT COUNT(*)`` clause.

        Returns
        -------
        Select
            The clause
        """
        return Select(BaseProjection.count_star())

    @staticmethod
    def star() -> Select:
        """Shortcut to create a ``SELECT *`` clause.

        Returns
        -------
        Select
            The clause
        """
        return Select(BaseProjection.star())

    @staticmethod
    def create_for(columns: Iterable[ColumnReference],
                   projection_type: SelectType = SelectType.Select) -> Select:
        """Full factory method to accompany `star` and `count_star` factory methods.

        This is basically the same as calling the `__init__` method directly.

        Parameters
        ----------
        columns : Iterable[ColumnReference]
            The columns that should form the projection
        projection_type : SelectType, optional
            The kind of projection that should be performed, by default `SelectType.Select` which is a plain selection
            without duplicate removal

        Returns
        -------
        Select
            The clause
        """
        target_columns = [BaseProjection.column(column) for column in columns]
        return Select(target_columns, projection_type)

    def __init__(self, targets: BaseProjection | Sequence[BaseProjection],
                 projection_type: SelectType = SelectType.Select) -> None:
        if not targets:
            raise ValueError("At least one target must be specified")
        self._targets = tuple(util.enlist(targets))
        self._projection_type = projection_type

        hash_val = hash((self._projection_type, self._targets))
        super().__init__(hash_val)

    @property
    def targets(self) -> Sequence[BaseProjection]:
        """Get all projections.

        Returns
        -------
        Sequence[BaseProjection]
            The projections in the order in which they were originally specified
        """
        return self._targets

    @property
    def projection_type(self) -> SelectType:
        """Get the type of projection (with or without duplicate elimination).

        Returns
        -------
        SelectType
            The projection type
        """
        return self._projection_type

    def is_star(self) -> bool:
        """Checks, whether the clause is simply ``SELECT *``.

        Returns
        -------
        bool
            Whether this clause is a ``SELECT *`` clause.
        """
        return len(self._targets) == 1 and self._targets[0] == BaseProjection.star()

    def columns(self) -> set[ColumnReference]:
        return util.set_union(target.columns() for target in self.targets)

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(target.itercolumns() for target in self.targets)

    def tables(self) -> set[TableReference]:
        return util.set_union(target.tables() for target in self.targets)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [target.expression for target in self.targets]

    def output_names(self) -> dict[str, ColumnReference]:
        """Output names map the alias of each column to the actual column.

        For example, consider a query ``SELECT R.a AS foo, R.b AS bar FROM R``. Calling `output_names` on this query
        provides the dictionary ``{'foo': R.a, 'bar': R.b}``.

        Currently, this method only works for 1:1 mappings and other aliases are ignored. For example, consider a query
        ``SELECT my_udf(R.a, R.b) AS c FROM R``. Here, a user-defined function is used to combine the values of ``R.a``
        and ``R.b`` to form an output column ``c``. Such a projection is ignored by `output_names`.

        Returns
        -------
        dict[str, ColumnReference]
            A mapping from the column target name to the original column.
        """
        output = {}
        for projection in self.targets:
            if not projection.target_name:
                continue
            source_columns = projection.expression.columns()
            if len(source_columns) != 1:
                continue
            output[projection.target_name] = util.simplify(source_columns)
        return output

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_select_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.projection_type == other.projection_type
                and self.targets == other.targets)

    def __str__(self) -> str:
        select_str = self.projection_type.value
        parts_str = ", ".join(str(target) for target in self.targets)
        return f"{select_str} {parts_str}"


class TableSource(abc.ABC):
    """A table source models a relation that can be scanned by the database system, filtered, joined, ...

    This is what is commonly reffered to as a *table* or a *relation* and forms the basic item of a ``FROM`` clause. In
    an SQL query the items of the ``FROM`` clause can originate from a number of different concepts. In the simplest
    case, this is just a physical table (e.g. ``SELECT * FROM R, S, T WHERE ...``), but other queries might reference
    subqueries or common table expressions in the ``FROM`` clause (e.g.
    ``SELECT * FROM R, (SELECT * FROM S, T WHERE ...) WHERE ...``). This class models the similarities between these
    concepts. Specific sub-classes implement them for the concrete kind of source (e.g. `DirectTableSource` or
    `SubqueryTableSource`).
    """

    @abc.abstractmethod
    def tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in the source.

        For plain table sources this will just be the actual table. For more complicated structures, such as subquery
        sources, this will include all tables of the subquery as well.

        Returns
        -------
        set[TableReference]
            The tables.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[ColumnReference]:
        """Provides all column sthat are referenced in the source.

        For plain table sources this will be empty, but for more complicate structures such as subquery source, this
        will include all columns that are referenced in the subquery.

        Returns
        -------
        set[ColumnReference]
            The columns
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all directly contained expressions in the source.

        For plain table sources this will be empty, but for subquery sources, etc. all expressions are returned. Nested
        expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression` interface for
        details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides access to all column in the source.

        For plain table sources this will be empty, but for subquery sources, etc. all expressions are returned. In
        contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n*
        times in this source, it will also be returned *n* times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this clause.

        Returns
        -------
        Iterable[ColumnReference]
            All columns exactly in the order in which they are used
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predicates(self) -> QueryPredicates | None:
        """Provides all predicates that are contained in the source.

        For plain table sources this will be ``None``, but for subquery sources, etc. all predicates are returned.

        Returns
        -------
        QueryPredicates | None
            The predicates or ``None`` if the source does not allow predicates or simply does not contain any.
        """
        raise NotImplementedError


class DirectTableSource(TableSource):
    """Models a plain table that is directly referenced in a ``FROM`` clause, e.g. ``R`` in ``SELECT * FROM R, S``.

    Parameters
    ----------
    table : TableReference
        The table that is sourced
    """
    def __init__(self, table: TableReference) -> None:
        self._table = table

    @property
    def table(self) -> TableReference:
        """Get the table that is sourced.

        This can be a virtual table (e.g. for CTEs), but will most commonly be an actual table.

        Returns
        -------
        TableReference
            The table.
        """
        return self._table

    def tables(self) -> set[TableReference]:
        return {self._table}

    def columns(self) -> set[ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def predicates(self) -> QueryPredicates | None:
        return None

    def __hash__(self) -> int:
        return hash(self._table)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._table == other._table

    def __repr__(self) -> str:
        return str(self._table)

    def __str__(self) -> str:
        return str(self._table)


class SubqueryTableSource(TableSource):
    """Models subquery that is referenced as a virtual table in the ``FROM`` clause.

    Consider the example query ``SELECT * FROM R, (SELECT * FROM S, T WHERE S.a = T.b) AS s_t WHERE R.c = s_t.a``.
    In this query, the subquery ``s_t`` would be represented as a subquery table source.

    Parameters
    ----------
    query : SqlQuery | SubqueryExpression
        The query that is sourced as a subquery
    target_name : str
        The name under which the subquery should be made available

    Raises
    ------
    ValueError
        If the `target_name` is empty
    """

    def __init__(self, query: SqlQuery | SubqueryExpression, target_name: str) -> None:
        if not target_name:
            raise ValueError("Target name for subquery source is required")
        self._subquery_expression = (query if isinstance(query, SubqueryExpression)
                                     else SubqueryExpression(query))
        self._target_name = target_name
        self._hash_val = hash((self._subquery_expression, self._target_name))

    @property
    def query(self) -> SqlQuery:
        """Get the query that is sourced as a virtual table.

        Returns
        -------
        SqlQuery
            The query
        """
        return self._subquery_expression.query

    @property
    def target_name(self) -> str:
        """Get the name under which the virtual table can be accessed in the actual query.

        Returns
        -------
        str
            The name. This will never be empty.
        """
        return self._target_name

    @property
    def target_table(self) -> TableReference:
        """Get the name under which the virtual table can be accessed in the actual query.

        The only difference to `target_name` this return type: this property provides the name as a proper table
        reference, rather than a string.

        Returns
        -------
        TableReference
            The table. This will always be a virtual table
        """
        return TableReference.create_virtual(self._target_name)

    @property
    def expression(self) -> SubqueryExpression:
        """Get the query that is used to construct the virtual table, as a subquery expression.

        Returns
        -------
        SubqueryExpression
            The subquery.
        """
        return self._subquery_expression

    def tables(self) -> set[TableReference]:
        return self._subquery_expression.tables() | {self.target_table}

    def columns(self) -> set[ColumnReference]:
        return self._subquery_expression.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [self._subquery_expression]

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._subquery_expression.itercolumns()

    def predicates(self) -> QueryPredicates | None:
        return self._subquery_expression.query.predicates()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self)) and self._subquery_expression == other._subquery_expression
                and self._target_name == other._target_name)

    def __repr__(self) -> str:
        return str(self._subquery_expression)

    def __str__(self) -> str:
        query_str = self._subquery_expression.query.stringify(trailing_delimiter=False)
        return f"({query_str}) AS {self._target_name}"


ValuesList = Iterable[tuple[StaticValueExpression, ...]]


class ValuesTableSource(TableSource):
    def __init__(self, values: ValuesList, *, alias: str, columns: Iterable[str]) -> None:
        self._values = tuple(values)
        self._table = TableReference.create_virtual(alias) if alias else None
        self._columns = tuple(ColumnReference(column, self._table) for column in columns)
        self._hash_val = hash((self._table, self._columns, self._values))

    @property
    def rows(self) -> ValuesList:
        return self._values

    @property
    def table(self) -> Optional[TableReference]:
        return self._table

    @property
    def cols(self) -> Sequence[ColumnReference]:
        return self._columns

    def tables(self) -> set[TableReference]:
        return {self._table}

    def columns(self) -> set[ColumnReference]:
        return set(self._columns)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(row for row in self._values)

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._columns

    def predicates(self) -> QueryPredicates | None:
        return None

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._table == other._table
                and self._columns == other._columns
                and self._values == other._values)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        vals = []
        for row in self._values:
            current_str = ", ".join(str(val) for val in row)
            vals.append(f"({current_str})")

        complete_vals_str = ", ".join(vals)
        cols = ", ".join(col.name for col in self._columns)

        return f"VALUES ({complete_vals_str}) AS {self._table} ({cols})"


class JoinType(enum.Enum):
    """Indicates the type of a join using the explicit ``JOIN`` syntax, e.g. ``OUTER JOIN`` or ``NATURAL JOIN``.

    The names of the individual values should be pretty self-explanatory and correspond entirely to the names in the
    SQL standard.
    """
    InnerJoin = "JOIN"
    OuterJoin = "FULL OUTER JOIN"
    LeftJoin = "LEFT OUTER JOIN"
    RightJoin = "RIGHT OUTER JOIN"
    CrossJoin = "CROSS JOIN"

    NaturalInnerJoin = "NATURAL JOIN"
    NaturalOuterJoin = "NATURAL OUTER JOIN"
    NaturalLeftJoin = "NATURAL LEFT JOIN"
    NaturalRightJoin = "NATURAL RIGHT JOIN"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.value


AnonymousJoins = {JoinType.CrossJoin,
                  JoinType.NaturalInnerJoin,
                  JoinType.NaturalOuterJoin,
                  JoinType.NaturalLeftJoin,
                  JoinType.NaturalRightJoin}
"""Anonymous joins are those joins that use the **JOIN** syntax, but do not require a predicate to work.

Examples include **CROSS JOIN** and **NATURAL JOIN**.
"""


class JoinTableSource(TableSource):
    """Models a table that is referenced in a *FROM* clause using the explicit *JOIN* syntax.

    Such a table source consists of three parts:

    1. the left-hand side table source
    2. the right-hand side table source being joined to the left-hand side
    3. an (optional) join condition that specifies how the two tables are joined, if the specific join requires such a
       predicate

    Parameters
    ----------
    left : TableSource
        The left-hand side of the join
    right : TableSource
        The right-hand side of the join
    join_condition : Optional[AbstractPredicate], optional
        The predicate that is used to join the specified table with the other tables of the *FROM* clause. For most
        joins this is a required argument in order to create a valid SQL query (e.g. *LEFT JOIN* or *INNER JOIN*),
        but there are some joins without a condition (e.g. *CROSS JOIN* and *NATURAL JOIN*).
    join_type : JoinType, optional
        The specific join that should be performed. Defaults to `JoinType.InnerJoin`.
    """

    def __init__(self, left: TableSource, right: TableSource, *,
                 join_condition: Optional[AbstractPredicate] = None,
                 join_type: JoinType = JoinType.InnerJoin) -> None:
        if join_condition is None and join_type not in AnonymousJoins:
            raise ValueError("Join condition is required for this join type: " + str(join_type))

        self._left = left
        self._right = right
        self._join_condition = join_condition
        self._join_type = join_type if join_condition else JoinType.CrossJoin
        self._hash_val = hash((self._left, self._right, self._join_condition, self._join_type))

    @property
    def left(self) -> TableSource:
        """Get the left-hand side of the join.

        Returns
        -------
        TableSource
            The join partner. Can be anything from a plain base table, to a subquery, to another join.
        """
        return self._left

    @property
    def right(self) -> TableSource:
        """Get the right-hand side of the join.

        Returns
        -------
        TableSource
            The join partner. Can be anything from a plain base table, to a subquery, to another join.
        """
        return self._right

    @property
    def source(self) -> TableSource:
        """Get the actual table being joined. This can be a proper table or a subquery.

        .. deprecated:: 0.10.0
            Use `left` instead. This is an artifact of the old mosql-based query representation

        Returns
        -------
        TableSource
            The table
        """
        return self._left

    @property
    def joined_table(self) -> JoinTableSource:
        """Get the nested join statements contained in this join.

        .. deprecated:: 0.10.0
            Use `right` instead. This is an artifact of the old mosql-based query representation

        A nested join is a *JOIN* statement within a *JOIN* statement, as in
        ``SELECT * FROM R JOIN (S JOIN T ON a = b) ON a = c``.

        Returns
        -------
        JoinTableSource
            The nested joins, can be empty if there are no such joins.
        """
        return self._right

    @property
    def join_condition(self) -> Optional[AbstractPredicate]:
        """Get the predicate that is used to determine matching tuples from the table.

        This can be ``None`` if the specific `join_type` does not require or allow a join condition (e.g.
        ``NATURAL JOIN``).

        Returns
        -------
        Optional[AbstractPredicate]
            The condition if it is specified, ``None`` otherwise.
        """
        return self._join_condition

    @property
    def join_type(self) -> JoinType:
        """Get the kind of join that should be performed.

        Returns
        -------
        JoinType
            The join type
        """
        return self._join_type

    def base_table(self) -> TableReference:
        """Provide the table that is farthest to the left in the join chain.

        For subqueries or **VALUES** clauses, this will return the alias of the expression, i.e. the name of the virtual table
        that is created for the subquery or **VALUES** clause.

        Returns
        -------
        TableReference
            The table
        """
        match self._left:
            case DirectTableSource():
                return self._left.table
            case JoinTableSource():
                return self._left.base_table()
            case SubqueryTableSource():
                return self._left.target_table
            case ValuesTableSource():
                return self._left.table
            case _:
                raise TypeError("Unknown table source type: " + str(type(self._left)))

    def tables(self) -> set[TableReference]:
        return self._left.tables() | self._right.tables()

    def columns(self) -> set[ColumnReference]:
        condition_columns = self._join_condition.columns() if self._join_condition else set()
        return self._left.columns() | self._right.columns() | condition_columns

    def iterexpressions(self) -> Iterable[SqlExpression]:
        left_expressions = list(self._left.iterexpressions())
        right_expressions = list(self._right.iterexpressions())
        condition_expressions = list(self._join_condition.iterexpressions()) if self._join_condition else []
        return left_expressions + right_expressions + condition_expressions

    def itercolumns(self) -> Iterable[ColumnReference]:
        left_columns = list(self._left.itercolumns())
        right_columns = list(self._right.itercolumns())
        condition_columns = list(self._join_condition.itercolumns()) if self._join_condition else []
        return left_columns + right_columns + condition_columns

    def predicates(self) -> Optional[QueryPredicates]:
        all_predicates: list[AbstractPredicate] = []

        left_predicates = self._left.predicates()
        if left_predicates:
            all_predicates.append(left_predicates.root)
        right_predicates = self._right.predicates()
        if right_predicates:
            all_predicates.append(right_predicates.root)
        if self._join_condition:
            all_predicates.append(self._join_condition)

        return QueryPredicates(CompoundPredicate.create_and(all_predicates)) if all_predicates else None

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self)) and self._left == other._left
                and self._right == other._right
                and self._join_condition == other._join_condition
                and self._join_type == other._join_type)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.join_type in AnonymousJoins:
            return f"{self.source} {self.join_type} {self.joined_table}"
        return f"{self.source} {self.join_type} {self.joined_table} ON {self.join_condition}"


TableType = TypeVar("TableType", bound=TableSource)


class From(BaseClause, Generic[TableType]):
    """The ``FROM`` clause models which tables should be selected and potentially how they are combined.

    A ``FROM`` clause permits arbitrary source items and does not enforce a specific structure or semantic on them.
    This puts the user in charge to generate a valid and meaningful structure. For example, the model allows for the
    first item to be a `JoinTableSource`, even though this is not valid SQL. Likewise, no duplicate checks are
    performed.

    To represent ``FROM`` clauses with a bit more structure, the `ImplicitFromClause` and `ExplicitFromClause`
    subclasses exist and should generally be preffered over direct usage of the raw `From` clause class.

    Parameters
    ----------
    items : TableSource | Iterable[TableSource]
        The tables that should be sourced in the ``FROM`` clause

    Raises
    ------
    ValueError
        If no items are specified
    """
    def __init__(self, items: TableSource | Iterable[TableSource]):
        items = util.enlist(items)
        if not items:
            raise ValueError("At least one source is required")
        self._items: tuple[TableSource] = tuple(items)
        super().__init__(hash(self._items))

    @property
    def items(self) -> Sequence[TableType]:
        """Get the tables that are sourced in the ``FROM`` clause

        Returns
        -------
        Sequence[TableSource]
            The sources in exactly the sequence in which they were specified
        """
        return self._items

    def tables(self) -> set[TableReference]:
        return util.set_union(src.tables() for src in self._items)

    def columns(self) -> set[ColumnReference]:
        return util.set_union(src.columns() for src in self._items)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return util.flatten(src.iterexpressions() for src in self._items)

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(src.itercolumns() for src in self._items)

    def predicates(self) -> Optional[QueryPredicates]:
        source_predicates = [src.predicates() for src in self._items]
        if not any(source_predicates):
            return None
        actual_predicates = [src_pred.root for src_pred in source_predicates if src_pred]
        merged_predicate = CompoundPredicate.create_and(actual_predicates)
        return QueryPredicates(merged_predicate)

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_from_clause(visitor)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._items == other._items

    def __str__(self) -> str:
        fixture = "FROM "
        contents_str = []
        for src in self._items:
            if isinstance(src, JoinTableSource):
                contents_str.append(" " + str(src))
            elif contents_str:
                contents_str.append(", " + str(src))
            else:
                contents_str.append(str(src))
        return fixture + "".join(contents_str)


class ImplicitFromClause(From[DirectTableSource]):
    """Represents a special case of ``FROM`` clause that only allows for pure tables to be selected.

    Specifically, this means that subqueries or explicit joins using the ``JOIN ON`` syntax are not allowed. Just
    plain old ``SELECT ... FROM R, S, T WHERE ...`` queries.

    As a special case, all ``FROM`` clauses that consist of a single (non-subquery) table can be represented as
    implicit clauses.

    Parameters
    ----------
    tables : DirectTableSource | Iterable[DirectTableSource]
        The tables that should be selected
    """

    @staticmethod
    def create_for(tables: TableReference | Iterable[TableReference]) -> ImplicitFromClause:
        """Shorthand method to create a ``FROM`` clause for a set of table references.

        This saves the user from creating the `DirectTableSource` instances before instantiating a implicit ``FROM``
        clause.

        Parameters
        ----------
        tables : TableReference | Iterable[TableReference]
            The tables that should be sourced

        Returns
        -------
        ImplicitFromClause
            The ``FROM`` clause
        """
        tables = util.enlist(tables)
        return ImplicitFromClause([DirectTableSource(tab) for tab in tables])

    def __init__(self, tables: DirectTableSource | Iterable[DirectTableSource]):
        super().__init__(tables)

    def itertables(self) -> Sequence[TableReference]:
        """Provides all tables in the ``FROM`` clause exactly in the sequence in which they were specified.

        This utility saves the user from unwrapping all the `DirectTableSource` objects by herself.

        Returns
        -------
        Sequence[TableReference]
            The tables.
        """
        return [src.table for src in self.items]


class ExplicitFromClause(From[JoinTableSource]):
    """Represents a special kind of ``FROM`` clause that requires all tables to be joined using the ``JOIN ON`` syntax.

    Parameters
    ----------
    joins : JoinTableSource | Iterable[JoinTableSource]
        The joins that should be performed
    """

    def __init__(self, joins: JoinTableSource | Iterable[JoinTableSource]) -> None:
        if isinstance(joins, Iterable) and len(joins) != 1:
            raise ValueError("Explicit FROM clauses can only contain a single join")
        super().__init__(joins)

    @property
    def root(self) -> JoinTableSource:
        """Get the root join of the ``FROM`` clause.

        Returns
        -------
        JoinTableSource
            The root join
        """
        return self.items[0]

    def base_table(self) -> TableReference:
        """Get the table that is farthest to the left in the join chain.

        For subqueries or **VALUES** clauses, this will return the alias of the expression, i.e. the name of the virtual table
        that is created for the subquery or **VALUES** clause.

        Returns
        -------
        TableReference
            The table
        """
        return self.root.base_table()

    def iterpredicates(self) -> Iterable[AbstractPredicate]:
        """Provides all join conditions that are contained in the ``FROM`` clause.

        Returns
        -------
        Iterable[AbstractPredicate]
            The join conditions.
        """
        return util.flatten(join.predicates() for join in self.items)


class Where(BaseClause):
    """The ``WHERE`` clause specifies conditions that result rows must satisfy.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.

    Parameters
    ----------
    predicate : AbstractPredicate
        The root predicate that specifies all conditions
    """

    def __init__(self, predicate: AbstractPredicate) -> None:
        if not predicate:
            raise ValueError("Predicate must be set")
        self._predicate = predicate
        super().__init__(hash(predicate))

    @property
    def predicate(self) -> AbstractPredicate:
        """Get the root predicate that contains all filters and joins in the ``WHERE`` clause.

        Returns
        -------
        AbstractPredicate
            The condition
        """
        return self._predicate

    def tables(self) -> set[TableReference]:
        return self._predicate.tables()

    def columns(self) -> set[ColumnReference]:
        return self.predicate.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self.predicate.iterexpressions()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.predicate.itercolumns()

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_where_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.predicate == other.predicate

    def __str__(self) -> str:
        return f"WHERE {self.predicate}"


class GroupBy(BaseClause):
    """The ``GROUP BY`` clause combines rows that match a grouping criterion to enable aggregation on these groups.

    Despite their names, all grouped columns can be arbitrary `SqlExpression` instances, rules and restrictions of the SQL
    standard are not enforced by PostBOUND.

    Parameters
    ----------
    group_columns : Sequence[SqlExpression]
        The expressions that should be used to perform the grouping
    distinct : bool, optional
        Whether the grouping should perform duplicate elimination, by default ``False``

    Raises
    ------
    ValueError
        If `group_columns` is empty.
    """

    def __init__(self, group_columns: Sequence[SqlExpression], distinct: bool = False) -> None:
        if not group_columns:
            raise ValueError("At least one group column must be specified")
        self._group_columns = tuple(group_columns)
        self._distinct = distinct

        hash_val = hash((self._group_columns, self._distinct))
        super().__init__(hash_val)

    @property
    def group_columns(self) -> Sequence[SqlExpression]:
        """Get all expressions that should be used to determine the grouping.

        Returns
        -------
        Sequence[SqlExpression]
            The grouping expressions in exactly the sequence in which they were specified.
        """
        return self._group_columns

    @property
    def distinct(self) -> bool:
        """Get whether the grouping should eliminate duplicates.

        Returns
        -------
        bool
            Whether duplicate removal is performed.
        """
        return self._distinct

    def columns(self) -> set[ColumnReference]:
        return util.set_union(column.columns() for column in self.group_columns)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self.group_columns

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(column.itercolumns() for column in self.group_columns)

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_groupby_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.group_columns == other.group_columns and self.distinct == other.distinct)

    def __str__(self) -> str:
        columns_str = ", ".join(str(col) for col in self.group_columns)
        distinct_str = " DISTINCT" if self.distinct else ""
        return f"GROUP BY{distinct_str} {columns_str}"


class Having(BaseClause):
    """The ``HAVING`` clause enables filtering of the groups constructed by a ``GROUP BY`` clause.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.

    The structure of this clause is similar to the `Where` clause, but its scope is different (even though PostBOUND
    does no semantic validation to enforce this): predicates of the ``HAVING`` clause are only checked on entire groups
    of values and have to be valid their, instead of on individual tuples.

    Parameters
    ----------
    condition : AbstractPredicate
        The root predicate that contains all actual conditions
    """

    def __init__(self, condition: AbstractPredicate) -> None:
        if not condition:
            raise ValueError("Condition must be set")
        self._condition = condition
        super().__init__(hash(condition))

    @property
    def condition(self) -> AbstractPredicate:
        """Get the root predicate that is used to form the ``HAVING`` clause.

        Returns
        -------
        AbstractPredicate
            The condition
        """
        return self._condition

    def columns(self) -> set[ColumnReference]:
        return self.condition.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self.condition.iterexpressions()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self.condition.itercolumns()

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_having_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.condition == other.condition

    def __str__(self) -> str:
        return f"HAVING {self.condition}"


class OrderByExpression:
    """The `OrderByExpression` is the fundamental building block for an ``ORDER BY`` clause.

    Each expression consists of the actual column (which might be an arbitrary `SqlExpression`, rules and restrictions
    by the SQL standard are not enforced here) as well as information regarding the ordering of the column. Setting
    this information to `None` falls back to the default interpretation by the target database system.

    Parameters
    ----------
    column : SqlExpression
        The column that should be used for ordering
    ascending : Optional[bool], optional
        Whether the column values should be sorted in ascending order. Defaults to ``None``, which indicates that the
        system-default ordering should be used.
    nulls_first : Optional[bool], optional
        Whether ``NULL`` values should be placed at beginning or at the end of the sorted list. Defaults to ``None``,
        which indicates that the system-default behaviour should be used.
    """

    def __init__(self, column: SqlExpression, ascending: Optional[bool] = None,
                 nulls_first: Optional[bool] = None) -> None:
        if not column:
            raise ValueError("Column must be specified")
        self._column = column
        self._ascending = ascending
        self._nulls_first = nulls_first
        self._hash_val = hash((self._column, self._ascending, self._nulls_first))

    @property
    def column(self) -> SqlExpression:
        """Get the expression used to specify the current grouping.

        In the simplest case this can just be a `ColumnExpression` which sorts directly by the column values. More
        complicated constructs like mathematical expressions over the column values are also possible.

        Returns
        -------
        SqlExpression
            The expression
        """
        return self._column

    @property
    def ascending(self) -> Optional[bool]:
        """Get the desired ordering of the output rows.

        Returns
        -------
        Optional[bool]
            Whether to sort in ascending order. ``None`` indicates that the default behaviour of the system should be
            used.
        """
        return self._ascending

    @property
    def nulls_first(self) -> Optional[bool]:
        """Get where to place ``NULL`` values in the result set.

        Returns
        -------
        Optional[bool]
            Whether to put ``NULL`` values at the beginning of the result set (or at the end). ``None`` indicates that
            the default behaviour of the system should be used.
        """
        return self._nulls_first

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.column == other.column
                and self.ascending == other.ascending
                and self.nulls_first == other.nulls_first)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        ascending_str = "" if self.ascending is None else (" ASC" if self.ascending else " DESC")
        nulls_first = "" if self.nulls_first is None else (" NULLS FIRST " if self.nulls_first else " NULLS LAST")
        return f"{self.column}{ascending_str}{nulls_first}"


class OrderBy(BaseClause):
    """The ``ORDER BY`` clause specifies how result rows should be sorted.

    This clause has a similar structure like a `Select` clause and simply consists of an arbitrary number of
    `OrderByExpression` objects.

    Parameters
    ----------
    expressions : Iterable[OrderByExpression]
        The terms that should be used to determine the ordering. At least one expression is required

    Raises
    ------
    ValueError
        If no `expressions` are provided
    """

    def __init__(self, expressions: Iterable[OrderByExpression]) -> None:
        if not expressions:
            raise ValueError("At least one ORDER BY expression required")
        self._expressions = tuple(expressions)
        super().__init__(hash(self._expressions))

    @property
    def expressions(self) -> Sequence[OrderByExpression]:
        """Get the expressions that form this ``ORDER BY`` clause.

        Returns
        -------
        Sequence[OrderByExpression]
            The individual terms that make up the ordering in exactly the sequence in which they were specified (which
            is the only valid sequence since all other orders could change the ordering of the result set).
        """
        return self._expressions

    def columns(self) -> set[ColumnReference]:
        return util.set_union(expression.column.columns() for expression in self.expressions)

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return [expression.column for expression in self.expressions]

    def itercolumns(self) -> Iterable[ColumnReference]:
        return util.flatten(expression.itercolumns() for expression in self.iterexpressions())

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_orderby_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.expressions == other.expressions

    def __str__(self) -> str:
        return "ORDER BY " + ", ".join(str(order_expr) for order_expr in self.expressions)


class Limit(BaseClause):
    """The ``FETCH FIRST`` or ``LIMIT`` clause restricts the number of output rows returned by the database system.

    Each clause can specify an offset (which is probably only meaningful if there is also an ``ORDER BY`` clause)
    and the actual limit. Notice that although many database systems use a non-standard syntax for this clause, our
    implementation is modelled after the actual SQL standard version (i.e. it produces a ``FETCH ...`` string output).

    Parameters
    ----------
    limit : Optional[int], optional
        The maximum number of tuples to put in the result set. Defaults to ``None`` which indicates that all tuples
        should be returned.
    offset : Optional[int], optional
        The number of tuples that should be skipped from the beginning of the result set. If no `OrderBy` clause is
        defined, this makes the result set's contents non-deterministic (at least in theory). Defaults to ``None``
        which indicates that no tuples should be skipped.

    Raises
    ------
    ValueError
        If neither a `limit`, nor an `offset` are specified
    """

    def __init__(self, *, limit: Optional[int] = None, offset: Optional[int] = None) -> None:
        if limit is None and offset is None:
            raise ValueError("Limit and offset cannot be both unspecified")
        self._limit = limit
        self._offset = offset

        hash_val = hash((self._limit, self._offset))
        super().__init__(hash_val)

    @property
    def limit(self) -> Optional[int]:
        """Get the maximum number of rows in the result set.

        Returns
        -------
        Optional[int]
            The limit or ``None``, if all rows should be returned.
        """
        return self._limit

    @property
    def offset(self) -> Optional[int]:
        """Get the offset within the result set (i.e. number of first rows to skip).

        Returns
        -------
        Optional[int]
            The offset or ``None`` if no rows should be skipped.
        """
        return self._offset

    def columns(self) -> set[ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[ColumnReference]:
        return []

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_limit_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.limit == other.limit and self.offset == other.offset

    def __str__(self) -> str:
        offset_str = f"OFFSET {self.offset} ROWS" if self.offset is not None else ""
        limit_str = f"FETCH FIRST {self.limit} ROWS ONLY" if self.limit is not None else ""
        if offset_str and limit_str:
            return offset_str + " " + limit_str
        elif offset_str:
            return offset_str
        elif limit_str:
            return limit_str
        return ""


class UnionClause(BaseClause):
    """The ``UNION`` or ``UNION ALL`` clause of a query.

    Parameters
    ----------
    partner_query : SqlQuery
        The query whose result set should be combined with the result set of the current query.
    union_all : bool, optional
        Whether the ``UNION`` operation should eliminate duplicates or not. Defaults to ``False`` which indicates that
        duplicates should be eliminated.
    """

    def __init__(self, partner_query: SqlQuery, *, union_all: bool = False) -> None:
        self._partner_query = partner_query
        self._union_all = union_all
        hash_val = hash((partner_query, union_all))
        super().__init__(hash_val)

    @property
    def query(self) -> SqlQuery:
        """Get the query that is combined with the current query.

        Returns
        -------
        SqlQuery
            The SQL query being "unioned"
        """
        return self._partner_query

    @property
    def union_all(self) -> bool:
        """Get whether this is a ``UNION`` or ``UNION ALL`` clause.

        Returns
        -------
        bool
            ``True`` if duplicates are eliminated, ``False`` otherwise.
        """
        return self._union_all

    def is_union_all(self) -> bool:
        """Whether this is a ``UNION`` or ``UNION ALL`` clause.

        Returns
        -------
        bool
            ``True`` if duplicates are eliminated, ``False`` otherwise.
        """
        return self._union_all

    def tables(self) -> set[TableReference]:
        return self._partner_query.tables()

    def columns(self) -> set[ColumnReference]:
        return self._partner_query.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self._partner_query.iterexpressions()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._partner_query.itercolumns()

    def accept_visitor(self, visitor) -> VisitorResult:
        return visitor.visit_union_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._partner_query == other._partner_query
                and self._union_all == other._union_all)

    def __str__(self) -> str:
        prefix = "UNION ALL" if self._union_all else "UNION"
        return f"\n  {prefix}\n{self._partner_query.stringify(trailing_delimiter=False)}"


class ExceptClause(BaseClause):
    """The ``EXCEPT`` clause of a query.

    Parameters
    ----------
    partner_query : SqlQuery
        The query whose result set should be subtracted from the result set of the current query.
    """

    def __init__(self, partner_query: SqlQuery) -> None:
        self._partner_query = partner_query
        super().__init__(hash(partner_query))

    @property
    def query(self) -> SqlQuery:
        """Get the query that is subtracted from the current query.

        Returns
        -------
        SqlQuery
            The SQL query being subtracted
        """
        return self._partner_query

    def tables(self) -> set[TableReference]:
        return self._partner_query.tables()

    def columns(self) -> set[ColumnReference]:
        return self._partner_query.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self._partner_query.iterexpressions()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._partner_query.itercolumns()

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_except_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._partner_query == other._partner_query

    def __str__(self) -> str:
        return f"\n  EXCEPT\n{self._partner_query.stringify(trailing_delimiter=False)}"


class IntersectClause(BaseClause):
    """The ``INTERSECT`` clause of a query.

    Parameters
    ----------
    partner_query : SqlQuery
        The query whose result set should be intersected with the result set of the current query.
    """

    def __init__(self, partner_query: SqlQuery) -> None:
        self._partner_query = partner_query
        super().__init__(hash(partner_query))

    @property
    def query(self) -> SqlQuery:
        """Get the query that is intersected with the current query.

        Returns
        -------
        SqlQuery
            The SQL query being intersected
        """
        return self._partner_query

    def tables(self) -> set[TableReference]:
        return self._partner_query.tables()

    def columns(self) -> set[ColumnReference]:
        return self._partner_query.columns()

    def iterexpressions(self) -> Iterable[SqlExpression]:
        return self._partner_query.iterexpressions()

    def itercolumns(self) -> Iterable[ColumnReference]:
        return self._partner_query.itercolumns()

    def accept_visitor(self, visitor: ClauseVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_intersect_clause(self)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._partner_query == other._partner_query

    def __str__(self) -> str:
        return f"\n  INTERSECT\n{self._partner_query.stringify(trailing_delimiter=False)}"


SetOperationClause = Union[UnionClause, ExceptClause, IntersectClause]
"""Supertype for all possible set operation clauses (**UNION**, **UNION ALL**, **INTERSECT**, **EXCEPT**)."""


class ClauseVisitor(abc.ABC, Generic[VisitorResult]):
    """Basic visitor to operate on arbitrary clause lists.

    See Also
    --------
    BaseClause

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_hint_clause(self, clause: Hint) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_explain_clause(self, clause: Explain) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_cte_clause(self, clause: WithQuery) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_select_clause(self, clause: Select) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_from_clause(self, clause: From) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_where_clause(self, clause: Where) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_groupby_clause(self, clause: GroupBy) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_having_clause(self, clause: Having) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_orderby_clause(self, clause: OrderBy) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_limit_clause(self, clause: Limit) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_union_clause(self, clause: UnionClause) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_except_clause(self, clause: ExceptClause) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_intersect_clause(self, clause: IntersectClause) -> VisitorResult:
        raise NotImplementedError


def _stringify_clause(clause: BaseClause) -> str:
    """Handler method to provide a refined string for a specific given clause, to be used by the `SqlQuery` ``__str__`` method.

    This method is slightly smarter than calling ``__str__`` directly, because it inserts newlines at sensible places in a
    query, e.g. after the hint block.

    Parameters
    ----------
    clause : BaseClause
        The clause to build the string representation

    Returns
    -------
    str
        The string representation of the clause
    """
    if isinstance(clause, Hint):
        return str(clause) + "\n"
    return str(clause) + " "


def collect_subqueries_in_expression(expression: SqlExpression) -> set[SqlQuery]:
    """Handler method to provide all the subqueries that are contained in a specific expression.

    Parameters
    ----------
    expression : SqlExpression
        The expression to analyze

    Returns
    -------
    set[SqlQuery]
        The subqueries from the `expression`

    See Also
    --------
    SqlQuery.subqueries
    """
    if isinstance(expression, SubqueryExpression):
        return {expression.query}
    return util.set_union(collect_subqueries_in_expression(child_expr)
                          for child_expr in expression.iterchildren())


def _collect_subqueries_in_table_source(table_source: TableSource) -> set[SqlQuery]:
    """Handler method to provide all subqueries that are contained in a specific table.

    This does not collect the subqueries in a recursive manner: once a subquery has been found, the collection stops.
    Therefore, subqueries that are contained in subqueries are not collected.

    Parameters
    ----------
    table_source : TableSource
        The table to analyze

    Returns
    -------
    set[SqlQuery]
        The subqueries that were found

    See Also
    --------
    SqlQuery.subqueries
    """
    if isinstance(table_source, SubqueryTableSource):
        return {table_source.query}
    elif isinstance(table_source, JoinTableSource):
        source_subqueries = _collect_subqueries_in_table_source(table_source.source)
        nested_subqueries = util.set_union(_collect_subqueries_in_table_source(nested_join)
                                           for nested_join in table_source.joined_table)
        condition_subqueries = (util.set_union(collect_subqueries_in_expression(cond_expr) for cond_expr
                                               in table_source.join_condition.iterexpressions())
                                if table_source.join_condition else set())
        return source_subqueries | nested_subqueries | condition_subqueries
    else:
        return set()


def _collect_subqueries(clause: BaseClause) -> set[SqlQuery]:
    """Handler method to provide all the subqueries that are contained in a specific clause.

    Following the definitions of `SqlQuery.subqueries`, this completely ignores CTEs. Therefore, subqueries that are defined
    within CTEs are not detected.

    Parameters
    ----------
    clause : BaseClause
        The clause to check

    Returns
    -------
    set[SqlQuery]
        All subqueries that have been found in the clause.

    Raises
    ------
    ValueError
        If the given clause is unknown. This indicates that this method is missing a handler for a specific clause type that
        was added later on.

    See Also
    --------
    SqlQuery.subqueries
    """
    if isinstance(clause, Hint) or isinstance(clause, Limit) or isinstance(clause, Explain):
        return set()

    if isinstance(clause, CommonTableExpression):
        return set()
    elif isinstance(clause, Select):
        return util.set_union(collect_subqueries_in_expression(target.expression) for target in clause.targets)
    elif isinstance(clause, ImplicitFromClause):
        return set()
    elif isinstance(clause, From):
        return util.set_union(_collect_subqueries_in_table_source(src) for src in clause.items)
    elif isinstance(clause, Where):
        where_predicate = clause.predicate
        return util.set_union(collect_subqueries_in_expression(expression) for expression in where_predicate.iterexpressions())
    elif isinstance(clause, GroupBy):
        return util.set_union(collect_subqueries_in_expression(column) for column in clause.group_columns)
    elif isinstance(clause, Having):
        having_predicate = clause.condition
        return util.set_union(collect_subqueries_in_expression(expression)
                              for expression in having_predicate.iterexpressions())
    elif isinstance(clause, OrderBy):
        return util.set_union(collect_subqueries_in_expression(expression.column) for expression in clause.expressions)
    elif isinstance(clause, UnionClause):
        return clause.query.subqueries()
    elif isinstance(clause, ExceptClause):
        return clause.query.subqueries()
    elif isinstance(clause, IntersectClause):
        return clause.query.subqueries()
    else:
        raise ValueError(f"Unknown clause type: {clause}")


def _collect_bound_tables_from_source(table_source: TableSource) -> set[TableReference]:
    """Handler method to provide all tables that are "produced" by a table source.

    "Produced" tables are tables that are either directly referenced in the ``FROM`` clause (e.g. ``FROM R``), referenced as
    part of joins (e.g. ``FROM R JOIN S ON ...``), or part of the ``FROM`` clauses of subqueries. In contrast, an unbound table
    is one that has to be provided by "context", such as the dependent table in a dependent subquery.

    Parameters
    ----------
    table_source : TableSource
        The table to check

    Returns
    -------
    set[TableReference]
        The "produced" tables.

    See Also
    --------
    SqlQuery.bound_tables
    """
    if isinstance(table_source, DirectTableSource):
        return {table_source.table}
    elif isinstance(table_source, SubqueryTableSource):
        return _collect_bound_tables(table_source.query.from_clause)
    elif isinstance(table_source, JoinTableSource):
        direct_tables = _collect_bound_tables_from_source(table_source.source)
        nested_tables = util.set_union(_collect_bound_tables_from_source(nested_join)
                                       for nested_join in table_source.joined_table)
        return direct_tables | nested_tables


def _collect_bound_tables(from_clause: From) -> set[TableReference]:
    """Handler method to provide all tables that are "produced" in the given clause.

    "Produced" tables are tables that are either directly referenced in the ``FROM`` clause (e.g. ``FROM R``), referenced as
    part of joins (e.g. ``FROM R JOIN S ON ...``), or part of the ``FROM`` clauses of subqueries. In contrast, an unbound table
    is one that has to be provided by "context", such as the dependent table in a dependent subquery.

    Parameters
    ----------
    from_clause : From
        The clause to check

    Returns
    -------
    set[TableReference]
        The "produced" tables

    See Also
    --------
    SqlQuery.bound_tables
    """
    if isinstance(from_clause, ImplicitFromClause):
        return from_clause.tables()
    else:
        return util.set_union(_collect_bound_tables_from_source(src) for src in from_clause.items)


def _assert_sound_set_operation(union_with: Optional[SqlQuery | UnionClause],
                                union_with_all: Optional[SqlQuery | UnionClause],
                                intersect_with: Optional[SqlQuery | IntersectClause],
                                except_from: Optional[SqlQuery | ExceptClause]) -> None:
    """Helper method to ensure that only one set operation is specified for a query and its correctly initialized.

    Parameters
    ----------
    union_with : Optional[SqlQuery | UnionClause]
        The query that should be unioned with this query
    union_with_all : Optional[SqlQuery | UnionClause]
        The query that should be unioned with this query using the ``UNION ALL`` operator
    intersect_with : Optional[SqlQuery | IntersectClause]
        The query that should be intersected with this query
    except_from : Optional[SqlQuery | ExceptClause]
        The query that should be subtracted from this query

    Raises
    ------
    ValueError
        If more than one set operation is specified
    """
    n_set_ops = sum(1 for query in (union_with, union_with_all, intersect_with, except_from) if query)
    if n_set_ops > 1:
        raise ValueError("Only one set operation is allowed")
    if isinstance(union_with, UnionClause) and union_with.union_all:
        raise ValueError("UNION clause specified but UNION all set")
    if isinstance(union_with_all, UnionClause) and not union_with_all.union_all:
        raise ValueError("UNION ALL clause specified but only UNION set")


FromClauseType = TypeVar("FromClauseType", bound=From)


def _create_ast(item: SqlQuery | BaseClause | TableSource | AbstractPredicate | SqlExpression, *, indentation: int = 0) -> str:
    prefix = " " * indentation
    item_str = type(item).__name__
    match item:
        case ColumnExpression():
            return f"{prefix}+ {item_str} [{item.column}]"
        case MathematicalExpression() | BooleanExpression() | SubqueryExpression():
            expressions = [_create_ast(e, indentation=indentation + 2) for e in item.iterchildren()]
            expression_str = "\n".join(expressions)
            return f"{prefix}+-{item_str}\n{expression_str}"
        case SqlExpression():
            return f"{prefix}+-{item_str} [{item}]"
        case CompoundPredicate():
            children = [_create_ast(c, indentation=indentation + 2) for c in item.children]
            child_str = "\n".join(children)
            return f"{prefix}+-{item_str}\n{child_str}"
        case AbstractPredicate():
            expressions = [_create_ast(e, indentation=indentation + 2) for e in item.iterexpressions()]
            expression_str = "\n".join(expressions)
            return f"{prefix}+-{item_str}\n{expression_str}"
        case Where() | Having():
            predicate_str = _create_ast(item.predicate, indentation=indentation + 2)
            return f"{prefix}+-{item_str}\n{predicate_str}"
        case DirectTableSource():
            return f"{prefix}+-{item_str} [{item.table}]"
        case JoinTableSource():
            source = _create_ast(item.source, indentation=indentation + 2)
            joins = [_create_ast(j, indentation=indentation + 2) for j in item.joined_table]
            join_str = "\n".join(joins)
            return f"{prefix}+-{item_str}\n{source}\n{join_str}"
        case ValuesTableSource():
            return f"{prefix}+-{item_str}"
        case From():
            tables = [_create_ast(t, indentation=indentation + 2) for t in item.items]
            table_str = "\n".join(tables)
            return f"{prefix}+-{item_str}\n{table_str}"
        case BaseClause():
            expressions = [_create_ast(e, indentation=indentation + 2) for e in item.iterexpressions()]
            expression_str = "\n".join(expressions)
            return f"{prefix}+-{item_str}\n{expression_str}"
        case SqlQuery():
            clauses = [_create_ast(c, indentation=indentation + 2) for c in item.clauses()]
            clause_str = "\n".join(clauses)
            return f"{prefix}+-{item_str}\n{clause_str}"


class SqlQuery:
    """Represents an arbitrary SQL query, providing direct access to the different clauses in the query.

    At a basic level, PostBOUND differentiates between two types of queries:

    - implicit SQL queries specify all referenced tables in the ``FROM`` clause and the join predicates in the ``WHERE``
      clause, e.g. ``SELECT * FROM R, S WHERE R.a = S.b AND R.c = 42``. This is the traditional way of writing SQL queries.
    - explicit SQL queries use the ``JOIN ON`` syntax to reference tables, e.g.
      ``SELECT * FROM R JOIN S ON R.a = S.b WHERE R.c = 42``. This is the more "modern" way of writing SQL queries.

    There is also a third possibility of mixing the implicit and explicit syntax. For each of these cases, designated
    subqueries exist. They all provide the same functionality and only differ in the (sub-)types of their ``FROM`` clauses.
    Therefore, these classes can be considered as "marker" types to indicate that at a certain point of a computation, a
    specific kind of query is required. The `SqlQuery` class acts as a superclass that specifies the general behaviour of all
    query instances and can act as the most general type of query.

    The clauses of each query can be accessed via properties. If a clause is optional, the absence of the clause is indicated
    through a ``None`` value. All additional behaviour of the queries is provided by the different methods. These are mostly
    focused on an easy introspection of the query's structure.

    Notice that PostBOUND does not enforce any semantics on the queries (e.g. regarding data types, access to values, the
    cardinality of subquery results, or the connection between different clauses). This has to be done by the user, or by the
    actual database system.

    Limitations
    -----------

    While the query abstraction is quite powerful, it is cannot represent the full range of SQL statements. Noteworthy
    limitations include:

    - no DDL or DML statements. The query abstraction is really only focused on *queries*, i.e. ``SELECT``
      statements.
    - set operations cannot be nested, i.e. it is currently not possible to express a query like
      ``SELECT 1 EXCEPT (SELECT 2 UNION SELECT 3)``.
    - no recursive CTEs. While CTEs are supported, recursive CTEs are not. While this would be an easy addition, there simply
      was no need for it so far. If you need recursive CTEs, PRs are always welcome!
    - no support for GROUPING SETS, including CUBE() and ROLLUP(). Conceptually speaking, these would not be hard to add, but
      there simply was no need for them so far. If you need them, PRs are always welcome!

    Parameters
    ----------
    select_clause : Select
        The ``SELECT`` part of the query. This is the only required part of a query. Notice however, that some database systems
        do not allow queries without a ``FROM`` clause.
    from_clause : Optional[From], optional
        The ``FROM`` part of the query, by default ``None``
    where_clause : Optional[Where], optional
        The ``WHERE`` part of the query, by default ``None``
    groupby_clause : Optional[GroupBy], optional
        The ``GROUP BY`` part of the query, by default ``None``
    having_clause : Optional[Having], optional
        The ``HAVING`` part of the query, by default ``None``.
    orderby_clause : Optional[OrderBy], optional
        The ``ORDER BY`` part of the query, by default ``None``
    limit_clause : Optional[Limit], optional
        The ``LIMIT`` and ``OFFSET`` part of the query. In standard SQL, this is designated using the ``FETCH FIRST`` syntax.
        Defaults to ``None``.
    cte_clause : Optional[CommonTableExpression], optional
        The ``WITH`` part of the query, by default ``None``
    union_with : Optional[SqlQuery | UnionClause], optional
        The ``UNION`` part of the query. Defaults to ``None``.
    union_with_all : Optional[SqlQuery | UnionClause], optional
        The ``UNION ALL`` part of the query. Defaults to ``None``.
    intersect_with : Optional[SqlQuery | IntersectClause], optional
        The ``INTERSECT`` part of the query. Defaults to ``None``.
    except_from : Optional[SqlQuery | ExceptClause], optional
        The ``EXCEPT`` part of the query. Defaults to ``None``.
    hints : Optional[Hint], optional
        The hint block of the query. Hints are not part of standard SQL and follow a completely system-specific syntax. Even
        their placement in within the query varies from system to system and from extension to extension. Defaults to ``None``.
    explain : Optional[Explain], optional
        The ``EXPLAIN`` part of the query. Like hints, this is not part of standard SQL. However, most systems provide
        ``EXPLAIN`` functionality. The specific features and syntax are quite similar, but still system specific. Defaults to
        ``None``.

    Warnings
    --------
    See the `Limitations` section for unsupported SQL features.
    """

    def __init__(self, *, select_clause: Select,
                 from_clause: Optional[From] = None, where_clause: Optional[Where] = None,
                 groupby_clause: Optional[GroupBy] = None, having_clause: Optional[Having] = None,
                 orderby_clause: Optional[OrderBy] = None, limit_clause: Optional[Limit] = None,
                 cte_clause: Optional[CommonTableExpression] = None,
                 union_with: Optional[SqlQuery | UnionClause] = None,
                 union_with_all: Optional[SqlQuery | UnionClause] = None,
                 intersect_with: Optional[SqlQuery | IntersectClause] = None,
                 except_from: Optional[SqlQuery | ExceptClause] = None,
                 hints: Optional[Hint] = None, explain: Optional[Explain] = None) -> None:
        _assert_sound_set_operation(union_with, union_with_all, intersect_with, except_from)

        self._cte_clause = cte_clause
        self._select_clause = select_clause
        self._from_clause = from_clause
        self._where_clause = where_clause
        self._groupby_clause = groupby_clause
        self._having_clause = having_clause
        self._orderby_clause = orderby_clause
        self._limit_clause = limit_clause
        self._hints = hints
        self._explain = explain

        # for all set operations, there are three possible input values: an actual query, a clause object, or None
        # by testing for a query object (which None fails), we can ensure that we always store a clause object internally
        self._union_clause = UnionClause(union_with) if isinstance(union_with, SqlQuery) else union_with
        self._union_all_clause = (UnionClause(union_with_all, union_all=True) if isinstance(union_with_all, SqlQuery)
                                  else union_with_all)
        self._intersect_clause = IntersectClause(intersect_with) if isinstance(intersect_with, SqlQuery) else intersect_with
        self._except_clause = ExceptClause(except_from) if isinstance(except_from, SqlQuery) else except_from

        self._hash_val = hash((self._hints, self._explain,
                               self._cte_clause,
                               self._select_clause, self._from_clause, self._where_clause,
                               self._groupby_clause, self._having_clause,
                               self._orderby_clause, self._limit_clause,
                               self._union_clause, self._union_all_clause,
                               self._intersect_clause, self._except_clause))

    @property
    def cte_clause(self) -> Optional[CommonTableExpression]:
        """Get the ``WITH`` clause of the query.

        Returns
        -------
        Optional[CommonTableExpression]
            The ``WITH`` clause if it was specified, or ``None`` otherwise.
        """
        return self._cte_clause

    @property
    def select_clause(self) -> Select:
        """Get the ``SELECT`` clause of the query. Will always be set.

        Returns
        -------
        Select
            The ``SELECT`` clause
        """
        return self._select_clause

    @property
    def from_clause(self) -> Optional[From]:
        """Get the ``FROM`` clause of the query.

        Returns
        -------
        Optional[From]
            The ``FROM`` clause if it was specified, or ``None`` otherwise.
        """
        return self._from_clause

    @property
    def where_clause(self) -> Optional[Where]:
        """Get the ``WHERE`` clause of the query.

        Returns
        -------
        Optional[Where]
            The ``WHERE`` clause if it was specified, or ``None`` otherwise.
        """
        return self._where_clause

    @property
    def groupby_clause(self) -> Optional[GroupBy]:
        """Get the ``GROUP BY`` clause of the query.

        Returns
        -------
        Optional[GroupBy]
            The ``GROUP BY`` clause if it was specified, or ``None`` otherwise.
        """
        return self._groupby_clause

    @property
    def having_clause(self) -> Optional[Having]:
        """Get the ``HAVING`` clause of the query.

        Returns
        -------
        Optional[Having]
            The ``HAVING`` clause if it was specified, or ``None`` otherwise.
        """
        return self._having_clause

    @property
    def orderby_clause(self) -> Optional[OrderBy]:
        """Get the ``ORDER BY`` clause of the query.

        Returns
        -------
        Optional[OrderBy]
            The ``ORDER BY`` clause if it was specified, or ``None`` otherwise.
        """
        return self._orderby_clause

    @property
    def limit_clause(self) -> Optional[Limit]:
        """Get the combined ``LIMIT`` and ``OFFSET`` clauses of the query.

        According to the SQL standard, these clauses should use the ``FETCH FIRST`` syntax. However, many systems use
        ``OFFSET`` and ``LIMIT`` instead.

        Returns
        -------
        Optional[Limit]
            The ``FETCH FIRST`` clause if it was specified, or ``None`` otherwise.
        """
        return self._limit_clause

    @property
    def union_with(self) -> Optional[SqlQuery]:
        """Get the ``UNION`` clause of the query.

        Returns
        -------
        Optional[SqlQuery]
            The ``UNION`` clause if it was specified, or ``None`` otherwise.
        """
        return self._union_clause.query if self._union_clause else None

    @property
    def union_with_all(self) -> Optional[SqlQuery]:
        """Get the ``UNION ALL`` clause of the query.

        Returns
        -------
        Optional[SqlQuery]
            The ``UNION ALL`` clause if it was specified, or ``None`` otherwise.
        """
        return self._union_all_clause if self._union_all_clause else None

    @property
    def intersect_with(self) -> Optional[SqlQuery]:
        """Get the ``INTERSECT`` clause of the query.

        Returns
        -------
        Optional[SqlQuery]
            The ``INTERSECT`` clause if it was specified, or ``None`` otherwise.
        """
        return self._intersect_clause if self._intersect_clause else None

    @property
    def except_from(self) -> Optional[SqlQuery]:
        """Get the ``EXCEPT`` clause of the query.

        Returns
        -------
        Optional[SqlQuery]
            The ``EXCEPT`` clause if it was specified, or ``None`` otherwise.
        """
        return self._except_clause if self._except_clause else None

    @property
    def hints(self) -> Optional[Hint]:
        """Get the hint block of the query.

        The hints can specify preparatory statements that have to be executed before the actual query is run in addition to the
        hints themselves.

        Returns
        -------
        Optional[Hint]
            The hint block if it was specified, or ``None`` otherwise.
        """
        return self._hints

    @property
    def explain(self) -> Optional[Explain]:
        """Get the ``EXPLAIN`` block of the query.

        Returns
        -------
        Optional[Explain]
            The ``EXPLAIN`` settings if specified, or ``None`` otherwise.
        """
        return self._explain

    @abc.abstractmethod
    def is_implicit(self) -> bool:
        """Checks, whether this query has an implicit ``FROM`` clause.

        The implicit ``FROM`` clause only consists of the source tables that should be scanned for the query. No subqueries or
        joins are contained in the clause. All join predicates must be part of the ``WHERE`` clause.

        Returns
        -------
        bool
            Whether the query is implicit

        See Also
        --------
        ImplicitSqlQuery
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_explicit(self) -> bool:
        """Checks, whether this query has an explicit ``FROM`` clause.

        The explicit ``FROM`` clause exclusively makes use of the ``JOIN ON`` syntax to denote both the tables that should be
        scanned, and the predicates that should be used to join the tables together. Therefore, the ``WHERE`` clause should
        only consist of filter predicates on the base tables. However, this is not enforced and the contents of the ``ON``
        conditions as well as the ``WHERE`` clause can be arbitrary predicates.

        Returns
        -------
        bool
            Whether the query is explicit

        See Also
        --------
        ExplicitSqlQuery
        """
        raise NotImplementedError

    def is_explain(self) -> bool:
        """Checks, whether this query is an ``EXPLAIN`` query rather than a normal SQL query.

        An ``EXPLAIN`` query is not executed like a normal ``SELECT ...`` query. Instead of actually calculating a result set,
        the database system only provides a query plan. This plan is the execution plan that would be used, had the query been
        entered as a normal SQL query.

        Returns
        -------
        bool
            Whether this query should be explained, rather than executed.
        """
        return self.explain is not None

    @functools.cache
    def tables(self) -> set[TableReference]:
        """Provides all tables that are referenced at any point in the query.

        This includes tables from all clauses. Virtual tables will be included and tables that are only scanned within
        subqueries are included as well. Notice however, that some database systems might not support subqueries to be put
        at arbitrary positions in the query (e.g. ``GROUP BY`` clause).

        Returns
        -------
        set[TableReference]
            All tables that are referenced in the query.
        """
        relevant_clauses: list[BaseClause] = [self._select_clause,
                                              self._from_clause,
                                              self._where_clause, self._groupby_clause, self._having_clause,
                                              self._orderby_clause, self._limit_clause,
                                              self.set_clause()]

        tabs = set()
        tabs |= self.cte_clause.referenced_tables() if self.cte_clause else set()
        for clause in relevant_clauses:
            if clause is None:
                continue
            tabs |= clause.tables()

        return tabs

    def columns(self) -> set[ColumnReference]:
        """Provides all columns that are referenced at any point in the query.

        This includes columns from all clauses and does not account for renamed columns from subqueries. For example, consider
        the query ``SELECT R.a, my_sq.b FROM R JOIN (SELECT b FROM S) my_sq ON R.a < my_sq.b``. `columns` would return the
        following set: ``{R.a, S.b, my_sq.b}``, even though ``my_sq.b`` can be considered as just an alias for ``S.b``.

        Returns
        -------
        set[ColumnReference]
            All columns that are referenced in the query.
        """
        return util.set_union(clause.columns() for clause in self.clauses())

    @functools.cache
    def predicates(self) -> QueryPredicates:
        """Provides all predicates in this query.

        *All* predicates really means *all* predicates: this includes predicates that appear in the ``FROM`` clause, the
        ``WHERE`` clause, as well as any predicates from CTEs.

        Returns
        -------
        QueryPredicates
            A predicates wrapper around the conjunction of all individual predicates.
        """
        predicate_handler = DefaultPredicateHandler
        current_predicate = predicate_handler.empty_predicate()

        if self.cte_clause:
            for with_query in self.cte_clause.queries:
                current_predicate = current_predicate.and_(with_query.query.predicates())

        if self.where_clause:
            current_predicate = current_predicate.and_(self.where_clause.predicate)

        from_predicates = self.from_clause.predicates()
        if from_predicates:
            current_predicate = current_predicate.and_(from_predicates)

        return current_predicate

    def subqueries(self) -> Collection[SqlQuery]:
        """Provides all subqueries that are referenced in this query.

        Notice that CTEs are ignored by this method, since they can be accessed directly via the `cte_clause` property.

        Returns
        -------
        Collection[SqlQuery]
            All subqueries that appear in any of the "inner" clauses of the query
        """
        return util.set_union(_collect_subqueries(clause) for clause in self.clauses())

    def clauses(self) -> Sequence[BaseClause]:
        """Provides all the clauses that are defined (i.e. not ``None``) in this query.

        Returns
        -------
        Sequence[BaseClause]
            The clauses. The current order of the clauses is as follows: hints, explain, cte, select, from, where, group by,
            having, order by, limit. Notice however, that this order is not strictly standardized and may change in the future.
            All clauses that are not specified on the query will be skipped.
        """
        all_clauses = [self.hints, self.explain, self.cte_clause,
                       self.select_clause, self.from_clause, self.where_clause,
                       self.groupby_clause, self.having_clause,
                       self.orderby_clause, self.limit_clause,
                       self.set_clause()]
        return [clause for clause in all_clauses if clause is not None]

    def set_clause(self) -> Optional[UnionClause | ExceptClause | IntersectClause]:
        """Provides the set operation clause of the query.

        Returns
        -------
        Optional[UnionClause | ExceptClause | IntersectClause]
            The set operation if one was specified, or ``None`` otherwise.
        """
        if self._union_clause:
            return self._union_clause
        elif self._union_all_clause:
            return self._union_all_clause
        elif self._intersect_clause:
            return self._intersect_clause
        elif self._except_clause:
            return self._except_clause
        else:
            return None

    def bound_tables(self) -> set[TableReference]:
        """Provides all tables that can be assigned to a physical or virtual table reference in this query.

        Bound tables are those tables, that are selected in the ``FROM`` clause of the query, or a subquery. Conversely,
        unbound tables are those that have to be "injected" by an outer query, as is the case for dependent subqueries.

        For example, the query ``SELECT * FROM R, S WHERE R.a = S.b`` has two bound tables: ``R`` and ``S``.
        On the other hand, the query ``SELECT * FROM R WHERE R.a = S.b`` has only bound ``R``, whereas ``S`` has to be bound in
        a surrounding query.

        Returns
        -------
        set[TableReference]
            All tables that are bound (i.e. listed in the ``FROM`` clause) of the query.
        """
        subquery_produced_tables = util.set_union(subquery.bound_tables()
                                                  for subquery in self.subqueries())
        cte_produced_tables = self.cte_clause.tables() if self.cte_clause else set()
        own_produced_tables = _collect_bound_tables(self.from_clause)
        set_query_produced_tables = self.set_clause().query.tables() if self.is_set_query() else set()
        return own_produced_tables | subquery_produced_tables | cte_produced_tables | set_query_produced_tables

    def unbound_tables(self) -> set[TableReference]:
        """Provides all tables that are referenced in this query but not bound.

        While `tables()` provides all tables that are referenced in this query in any way, `bound_tables` restricts
        these tables. This method provides the complementary set to `bound_tables` i.e.
        ``tables = bound_tables ⊕ unbound_tables``.

        Returns
        -------
        set[TableReference]
            The unbound tables that have to be supplied as part of an outer query
        """
        if self.from_clause:
            virtual_subquery_targets = {subquery_source.target_table for subquery_source in self.from_clause.items
                                        if isinstance(subquery_source, SubqueryTableSource)}
        else:
            virtual_subquery_targets = set()

        if self.cte_clause:
            virtual_cte_targets = {with_query.target_table for with_query in self.cte_clause.queries}
        else:
            virtual_cte_targets = set()

        return self.tables() - self.bound_tables() - virtual_subquery_targets - virtual_cte_targets

    def is_ordered(self) -> bool:
        """Checks, whether this query produces its result tuples in order.

        Returns
        -------
        bool
            Whether a valid ``ORDER BY`` clause was specified on the query.
        """
        return self.orderby_clause is not None

    def is_dependent(self) -> bool:
        """Checks, whether all columns that are referenced in this query are provided by the tables from this query.

        In order for this check to work, all columns have to be bound to actual tables, i.e. the `tables` attribute of all
        column references have to be set to a valid object.

        Returns
        -------
        bool
            Whether all columns belong to tables that are bound by this query
        """
        return not (self.tables() <= self.bound_tables())

    def is_scalar(self) -> bool:
        """Checks, whether the query is guaranteed to provide a single scalar value as a result.

        Scalar results can only be calculated by queries with a single projection in the *SELECT* clause and if that projection
        is an aggregate function, e.g. *SELECT min(R.a) FROM R*. However, there are other queries which could also be scalar
        "by chance", e.g. *SELECT R.b FROM R WHERE R.a = 1*  if *R.a* is the primary key of *R*. Notice that such cases are not
        recognized by this method.

        Returns
        -------
        bool
            Whether the query will always return a single scalar value
        """
        if not len(self.select_clause.targets) == 1 or self.select_clause.projection_type != SelectType.Select:
            return False
        target: SqlExpression = util.simplify(self.select_clause.targets).expression
        return isinstance(target, FunctionExpression) and target.is_aggregate() and not self._groupby_clause

    def is_set_query(self) -> bool:
        """Checks, whether this query is a set query.

        A set query is a query that combines the results of two or more queries into a single result set. This can be done
        by combining the tuples from both sets using a ``UNION`` clause (which removes duplicates), or a ``UNION ALL`` clause
        (which retains duplicates). Alternatively, only tuples that are present in both sets can be retained using an
        ``INTERSECT`` clause. Finally, all tuples from the first result set that are not part of the second result set can be
        computed using an ``EXCEPT`` clause.

        Notice that only one of the set operators can be used at a time, but the input query of one set operation can itself
        use another set operation.

        Returns
        -------
        bool
            Whether this query is a set query
        """
        return any([self._union_clause, self._union_all_clause, self._intersect_clause, self._except_clause])

    def contains_cross_product(self) -> bool:
        """Checks, whether this query has at least one cross product.

        Returns
        -------
        bool
            Whether this query has cross products.
        """
        if not self._from_clause:
            return False
        if self.predicates().empty_predicate():
            return True
        join_graph = self.predicates().join_graph()
        return len(nx.connected_components(join_graph)) > 1

    def iterexpressions(self) -> Iterable[SqlExpression]:
        """Provides access to all expressions that are directly contained in this query.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        return util.flatten(clause.iterexpressions() for clause in self.clauses())

    def itercolumns(self) -> Iterable[ColumnReference]:
        """Provides access to all column in this query.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced *n* times
        in this query, it will also be returned *n* times by this method. Furthermore, the order in which columns are provided
        by the iterable matches the order in which they appear in this query.

        Returns
        -------
        Iterable[ColumnReference]
            The columns
        """
        return util.flatten(clause.itercolumns() for clause in self.clauses())

    def stringify(self, *, trailing_delimiter: bool = True) -> str:
        """Provides a string representation of this query.

        The only difference to calling `str` directly, is that the `stringify` method provides control over whether a trailing
        delimiter should be appended to the query.

        Parameters
        ----------
        trailing_delimiter : bool, optional
            Whether a delimiter should be appended to the query. Defaults to ``True``.

        Returns
        -------
        str
            A string representation of this query
        """
        delim = ";" if trailing_delimiter else ""
        return "".join(_stringify_clause(clause) for clause in self.clauses()).rstrip() + delim

    def ast(self) -> str:
        """Provides a human-readable representation of the abstract syntax tree for this query.

        The AST is a textual representation of the query that shows the structure of the query in a tree-like manner.

        Returns
        -------
        str
            The abstract syntax tree of this query
        """
        return _create_ast(self)

    def accept_visitor(self, clause_visitor: ClauseVisitor) -> None:
        """Applies a visitor over all clauses in the current query.

        Parameters
        ----------
        clause_visitor : ClauseVisitor
            The visitor algorithm to use.
        """
        for clause in self.clauses():
            clause.accept_visitor(clause_visitor)

    def __json__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.clauses() == other.clauses()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.stringify(trailing_delimiter=True)


class ImplicitSqlQuery(SqlQuery):
    """An implicit query restricts the constructs that may appear in the ``FROM`` clause.

    For implicit queries, the ``FROM`` clause may only consist of simple table sources. All join conditions have to be put in
    the ``WHERE`` clause. Notice that this does not restrict the structure of other clauses. For example, the ``WHERE`` clause
    can still contain subqueries. As a special case, queries without a ``FROM`` clause are also considered implicit.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    See Also
    --------
    ImplicitFromClause
    ExplicitSqlQuery

    Examples
    --------
    The following queries are considered as implicit queries:

    .. code-block:: sql

        SELECT *
        FROM R, S, T
        WHERE R.a = S.b
            AND S.b = T.c
            AND R.a < 42

    .. code-block:: sql

        SELECT *
        FROM R, S, T
        WHERE R.a = S.b
            AND S.b = T.c
            AND R.a = (SELECT MIN(R.a) FROM R)
    """

    def __init__(self, *, select_clause: Select,
                 from_clause: Optional[ImplicitFromClause] = None,
                 where_clause: Optional[Where] = None,
                 groupby_clause: Optional[GroupBy] = None,
                 having_clause: Optional[Having] = None,
                 orderby_clause: Optional[OrderBy] = None,
                 limit_clause: Optional[Limit] = None,
                 cte_clause: Optional[CommonTableExpression] = None,
                 union_with: Optional[SqlQuery | UnionClause] = None,
                 union_with_all: Optional[SqlQuery | UnionClause] = None,
                 intersect_with: Optional[SqlQuery | IntersectClause] = None,
                 except_from: Optional[SqlQuery | ExceptClause] = None,
                 explain_clause: Optional[Explain] = None,
                 hints: Optional[Hint] = None) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         cte_clause=cte_clause,
                         union_with=union_with, union_with_all=union_with_all,
                         intersect_with=intersect_with, except_from=except_from,
                         explain=explain_clause, hints=hints)

    @property
    def from_clause(self) -> Optional[ImplicitFromClause]:
        return self._from_clause

    def is_implicit(self) -> bool:
        return True

    def is_explicit(self) -> bool:
        return False


class ExplicitSqlQuery(SqlQuery):
    """An explicit query restricts the constructs that may appear in the ``FROM`` clause.

    For explicit queries, the ``FROM`` clause must utilize the ``JOIN ON`` syntax for all tables. The join conditions should
    be put into the ``ON`` blocks. Notice however, that PostBOUND does not perform any sanity checks here. Therefore, it is
    possible to put mix joins and filters in the ``ON`` blocks, move all joins to the ``WHERE`` clause or scatter the join
    conditions between the two clauses. Whether this is good style is up for debate, but at least PostBOUND does allow it. In
    contrast to the implicit query, subqueries are also allowed as table sources.

    Notice that each explicit query must join at least two tables in its ``FROM`` clause.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    See Also
    --------
    ExplicitFromClause
    ImplicitSqlQuery

    Examples
    --------
    The following queries are considered as explicit queries:

    .. code-block:: sql

        SELECT *
        FROM R
            JOIN S ON R.a = S.b
            JOIN T ON S.b = T.c
        WHERE R.a < 42

    .. code-block:: sql

        SELECT *
        FROM R
            JOIN S ON R.a = S.b AND R.a = (SELECT MIN(R.a) FROM R)
            JOIN T ON S.b = T.c
    """

    def __init__(self, *, select_clause: Select,
                 from_clause: Optional[ExplicitFromClause] = None,
                 where_clause: Optional[Where] = None,
                 groupby_clause: Optional[GroupBy] = None,
                 having_clause: Optional[Having] = None,
                 orderby_clause: Optional[OrderBy] = None,
                 limit_clause: Optional[Limit] = None,
                 cte_clause: Optional[CommonTableExpression] = None,
                 union_with: Optional[SqlQuery | UnionClause] = None,
                 union_with_all: Optional[SqlQuery | UnionClause] = None,
                 intersect_with: Optional[SqlQuery | IntersectClause] = None,
                 except_from: Optional[SqlQuery | ExceptClause] = None,
                 explain_clause: Optional[Explain] = None,
                 hints: Optional[Hint] = None) -> None:
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         cte_clause=cte_clause,
                         union_with=union_with, union_with_all=union_with_all,
                         intersect_with=intersect_with, except_from=except_from,
                         explain=explain_clause, hints=hints)

    @property
    def from_clause(self) -> Optional[ExplicitFromClause]:
        return self._from_clause

    def is_implicit(self) -> bool:
        return False

    def is_explicit(self) -> bool:
        return True


class MixedSqlQuery(SqlQuery):
    """A mixed query allows for both the explicit as well as the implicit syntax to be used within the same ``FROM`` clause.

    The mixed query complements `ImplicitSqlQuery` and `ExplicitSqlQuery` by removing the "purity" restriction: the tables that
    appear in the ``FROM`` clause can be described using either plain references or subqueries and they are free to use the
    ``JOIN ON`` syntax. The only thing that is not allowed as a ``FROM`` clause is an instance of `ImplicitFromClause` or an
    instance of `ExplicitFromClause`, since those cases are already covered by their respective query classes.

    Notice however, that we currently do not enforce the `From` clause to not be a valid explicit or implicit clause. All
    checks happen on a type level. If the contents of a general `From` clause just happen to also be a valid
    `ImplicitFromClause`, this is fine.

    The attributes and parameters for this query type are the same as for `SqlQuery`, only the type of the `From` clause is
    restricted.

    Raises
    ------
    ValueError
        If the given `from_clause` is either an implicit ``FROM`` clause or an explicit one.
    """
    def __init__(self, *, select_clause: Select,
                 from_clause: Optional[From] = None,
                 where_clause: Optional[Where] = None,
                 groupby_clause: Optional[GroupBy] = None,
                 having_clause: Optional[Having] = None,
                 orderby_clause: Optional[OrderBy] = None,
                 limit_clause: Optional[Limit] = None,
                 cte_clause: Optional[CommonTableExpression] = None,
                 union_with: Optional[SqlQuery | UnionClause] = None,
                 union_with_all: Optional[SqlQuery | UnionClause] = None,
                 intersect_with: Optional[SqlQuery | IntersectClause] = None,
                 except_from: Optional[SqlQuery | ExceptClause] = None,
                 explain_clause: Optional[Explain] = None,
                 hints: Optional[Hint] = None) -> None:
        if isinstance(from_clause, ExplicitFromClause) or isinstance(from_clause, ImplicitFromClause):
            raise ValueError("MixedSqlQuery cannot be combined with explicit/implicit FROM clause")
        super().__init__(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                         groupby_clause=groupby_clause, having_clause=having_clause,
                         orderby_clause=orderby_clause, limit_clause=limit_clause,
                         cte_clause=cte_clause,
                         union_with=union_with, union_with_all=union_with_all,
                         intersect_with=intersect_with, except_from=except_from,
                         explain=explain_clause, hints=hints)

    def is_implicit(self) -> bool:
        return False

    def is_explicit(self) -> bool:
        return False


def build_query(query_clauses: Iterable[BaseClause]) -> SqlQuery:
    """Constructs an SQL query based on specific clauses.

    No validation is performed. If clauses appear multiple times, later clauses overwrite former ones. The specific
    type of query (i.e. implicit, explicit or mixed) is inferred from the clauses (i.e. occurrence of an implicit ``FROM``
    clause enforces an `ImplicitSqlQuery` and vice-versa). The overwriting rules apply here as well: a later `From` clause
    overwrites a former one and can change the type of the produced query.

    Parameters
    ----------
    query_clauses : Iterable[BaseClause]
        The clauses that should be used to construct the query. If any of the clauses are **None**, they will simply be
        skipped.

    Returns
    -------
    SqlQuery
        A query consisting of the specified clauses

    Raises
    ------
    ValueError
        If `query_clauses` does not contain a `Select` clause
    ValueError
        If any of the clause types is unknown. This indicates that this method is missing a handler for a specific clause type
        that was added later on.
    """
    build_implicit_query, build_explicit_query = True, True

    cte_clause = None
    select_clause, from_clause, where_clause = None, None, None
    groupby_clause, having_clause = None, None
    orderby_clause, limit_clause = None, None
    union_clause, union_all_clause = None, None
    intersect_clause, except_clause = None, None
    explain_clause, hints_clause = None, None
    for clause in query_clauses:
        if not clause:
            continue

        if isinstance(clause, CommonTableExpression):
            cte_clause = clause
        elif isinstance(clause, Select):
            select_clause = clause
        elif isinstance(clause, ImplicitFromClause):
            from_clause = clause
            build_implicit_query, build_explicit_query = True, False
        elif isinstance(clause, ExplicitFromClause):
            from_clause = clause
            build_implicit_query, build_explicit_query = False, True
        elif isinstance(clause, From):
            from_clause = clause
            build_implicit_query, build_explicit_query = False, False
        elif isinstance(clause, Where):
            where_clause = clause
        elif isinstance(clause, GroupBy):
            groupby_clause = clause
        elif isinstance(clause, Having):
            having_clause = clause
        elif isinstance(clause, OrderBy):
            orderby_clause = clause
        elif isinstance(clause, Limit):
            limit_clause = clause
        elif isinstance(clause, UnionClause):
            if clause.is_union_all():
                union_all_clause = clause
            else:
                union_clause = clause
        elif isinstance(clause, ExceptClause):
            except_clause = clause
        elif isinstance(clause, IntersectClause):
            intersect_clause = clause
        elif isinstance(clause, Explain):
            explain_clause = clause
        elif isinstance(clause, Hint):
            hints_clause = clause
        else:
            raise ValueError("Unknown clause type: " + str(clause))

    if select_clause is None:
        raise ValueError("No SELECT clause detected")

    if build_implicit_query:
        return ImplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                groupby_clause=groupby_clause, having_clause=having_clause,
                                orderby_clause=orderby_clause, limit_clause=limit_clause,
                                cte_clause=cte_clause,
                                union_with=union_clause, union_with_all=union_all_clause,
                                intersect_with=intersect_clause, except_from=except_clause,
                                hints=hints_clause, explain_clause=explain_clause)
    elif build_explicit_query:
        return ExplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                groupby_clause=groupby_clause, having_clause=having_clause,
                                orderby_clause=orderby_clause, limit_clause=limit_clause,
                                cte_clause=cte_clause,
                                union_with=union_clause, union_with_all=union_all_clause,
                                intersect_with=intersect_clause, except_from=except_clause,
                                hints=hints_clause, explain_clause=explain_clause)
    else:
        return MixedSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                             groupby_clause=groupby_clause, having_clause=having_clause,
                             orderby_clause=orderby_clause, limit_clause=limit_clause,
                             cte_clause=cte_clause,
                             union_with=union_clause, union_with_all=union_all_clause,
                             intersect_with=intersect_clause, except_from=except_clause,
                             hints=hints_clause, explain_clause=explain_clause)
