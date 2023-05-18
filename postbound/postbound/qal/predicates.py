"""Contains those parts of the `qal`, that are dedicated to representing predicates of SQL queries."""
from __future__ import annotations

import abc
import collections
import functools
import itertools
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import Type, Union, Optional

import networkx as nx

from postbound.qal import base, expressions as expr
from postbound.util import errors, collections as collection_utils


def _normalize_join_pair(columns: tuple[base.ColumnReference, base.ColumnReference]
                         ) -> tuple[base.ColumnReference, base.ColumnReference]:
    """Normalizes the given join such that a pair (R.a, S.b) and (S.b, R.a) can be recognized as equal."""
    first_col, second_col = columns
    return (second_col, first_col) if second_col < first_col else columns


class NoJoinPredicateError(errors.StateError):
    """Indicates that the given predicate is not a join predicate, but was expected to be one."""

    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


class NoFilterPredicateError(errors.StateError):
    """Indicates that the given predicate is not a filter predicate, but was expected to be one."""

    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


BaseExpression = Union[expr.ColumnExpression, expr.StaticValueExpression, expr.SubqueryExpression]
"""Supertype that captures all expression types that can be considered base expressions for predicates."""


def _collect_base_expressions(expression: expr.SqlExpression) -> Iterable[BaseExpression]:
    """Provides all base expressions that are contained in the given expression."""
    if isinstance(expression, (expr.ColumnExpression, expr.StaticValueExpression, expr.SubqueryExpression)):
        return [expression]
    return collection_utils.flatten(_collect_base_expressions(child_expr) for child_expr in expression.iterchildren())


def _collect_subquery_expressions(expression: expr.SqlExpression) -> Iterable[expr.SubqueryExpression]:
    """Provides all subquery expressions that are contained in the given expression."""
    return [child_expr for child_expr in _collect_base_expressions(expression)
            if isinstance(child_expr, expr.SubqueryExpression)]


def _collect_column_expression_columns(expression: expr.SqlExpression) -> set[base.ColumnReference]:
    """Provides all the columns in column expressions that are contained in the given expression."""
    return collection_utils.set_union(base_expr.columns() for base_expr in _collect_base_expressions(expression)
                                      if isinstance(base_expr, expr.ColumnExpression))


def _collect_column_expression_tables(expression: expr.SqlExpression) -> set[base.TableReference]:
    """Provides all tables of columns from the column expressions that are contained in the given expression."""
    return {column.table for column in _collect_column_expression_columns(expression) if column.is_bound()}


def _generate_join_pairs(first_columns: Iterable[base.ColumnReference],
                         second_columns: Iterable[base.ColumnReference]
                         ) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
    """Provides all possible join pairs with columns from `first_columns` joined by columns from `second_columns`."""
    return {_normalize_join_pair((first_col, second_col)) for first_col, second_col
            in itertools.product(first_columns, second_columns) if first_col.table != second_col.table}


class AbstractPredicate(abc.ABC):
    """`AbstractPredicate` defines the basic interface that is shared by all possible predicates.

    Possible predicates include basic binary predicates such as `R.a = S.b` or `R.a = 42`, as well as compound
    predicates that are build from base predicates, e.g. conjunctions, disjunctions or negations.
    """

    def __init__(self, hash_val: int) -> None:
        self._hash_val = hash_val

    @abc.abstractmethod
    def is_compound(self) -> bool:
        """Checks, whether this predicate is a conjunction/disjunction/negation of base predicates."""
        raise NotImplementedError

    def is_base(self) -> bool:
        """Checks, whether this predicate is a base predicate i.e. not a conjunction/disjunction/negation."""
        return not self.is_compound()

    @abc.abstractmethod
    def is_join(self) -> bool:
        """Checks, whether this predicate describes a join between two tables.

        PostBOUND uses the following criteria to determine, whether a predicate is join or not:

        1. all predicates of the form <col 1> <operator> <col 2> where <col 1> and <col 2> come from different tables
        are joins. The columns can optionally be modified by value casts or static expressions, e.g. `R.a::integer + 7`
        2. all functions that access columns from multiple tables are joins, e.g. `my_udf(R.a, S.b)`
        3. all subqueries are treated as filters, no matter whether they are dependent subqueries or not.
        This means that both `R.a = (SELECT MAX(S.b) FROM S)` and `R.a = (SELECT MAX(S.b) FROM S WHERE R.c = S.d)`
        are treated as filters and not as joins, even though the second subquery will require some sort of the join
        in the query plan.
        4. BETWEEN and IN predicates are treated according to rule 1 since they can be emulated via base predicates
        (subqueries in IN predicates are evaluated according to rule 3.)
        """
        raise NotImplementedError

    def is_filter(self) -> bool:
        """Checks, whether this predicate is a filter on a base table rather than a join of base tables."""
        return not self.is_join()

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced by this predicate."""
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides all columns that are referenced by this predicate.

        If a column is referenced multiple times, it is also returned multiple times.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all directly contained expressions in this predicate.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).
        """
        raise NotImplementedError

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are accessed by this predicate."""
        return {column.table for column in self.columns() if column.is_bound()}

    def contains_table(self, table: base.TableReference) -> bool:
        """Checks, whether this predicate filters or joins a column of the given table."""
        return any(table == tab for tab in self.tables())

    def joins_table(self, table: base.TableReference) -> bool:
        """Checks, whether this predicate describes a join and one of the join partners is the given table."""
        if not self.is_join():
            return False
        return any(first_col.table == table or second_col.table == table
                   for first_col, second_col in self.join_partners())

    def columns_of(self, table: base.TableReference) -> set[base.ColumnReference]:
        """Retrieves all columns of the given table that are referenced by this predicate."""
        return {attr for attr in self.columns() if attr.table == table}

    def join_partners_of(self, table: base.TableReference) -> set[base.ColumnReference]:
        """Retrieves all columns that are joined with the given table.

        If this predicate is not a join, an error will be raised.
        """
        partners = []
        for first_col, second_col in self.join_partners():
            if first_col.table == table:
                partners.append(second_col)
            elif second_col.table == table:
                partners.append(first_col)
        return set(partners)

    @abc.abstractmethod
    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        """Provides all pairs of columns that are joined within this predicate.

        If multiple columns are joined or it is unclear which columns are involved in a join exactly, this method
        falls back to returning the cross-product of all potential join partners.
        For example, consider the following query: `SELECT * FROM R, S WHERE my_udf(R.a, R.b, S.c)`.
        In this case, it cannot be determined which columns of R take part in the join. Therefore, `join_partners`
        will return the set `{(R.a, S.c), (R.b, S.c)}`.

        If this predicate is not a join, an error will be raised.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def base_predicates(self) -> Iterable[AbstractPredicate]:
        """Provides all base predicates that form this predicate.

        This is most useful to iterate over all leaves of a compound predicate, for base predicates it simply returns
        the predicate itself.
        """
        raise NotImplementedError

    def required_tables(self) -> set[base.TableReference]:
        """Provides all tables that have to be available in order for this predicate to be executed.

        The return value of this method differs from `tables` in one central aspect: `tables` provides all tables
        that are accessed, which includes all tables from subqueries.

        Consider the following example predicate: `R.a = (SELECT MIN(S.b) FROM S)`. Calling `tables` on this predicate
        would return the set `{R, S}`. However, table `S` is already provided by the subquery. Therefore,
        `required_tables` only returns `{R}`, since this is the only table that has to be provided by the context of
        this method.
        """
        subqueries = collection_utils.flatten(_collect_subquery_expressions(child_expr)
                                              for child_expr in self.iterexpressions())
        subquery_tables = collection_utils.set_union(subquery.query.unbound_tables() for subquery in subqueries)
        column_tables = collection_utils.set_union(_collect_column_expression_tables(child_expr)
                                                   for child_expr in self.iterexpressions())
        return column_tables | subquery_tables

    def _assert_join_predicate(self) -> None:
        """Raises a `NoJoinPredicateError` if `self` is not a join."""
        if not self.is_join():
            raise NoJoinPredicateError(self)

    def _assert_filter_predicate(self) -> None:
        """Raises a `NoFilterPredicateError` if `self` is not a filter."""
        if not self.is_filter():
            raise NoFilterPredicateError(self)

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
    """A `BasePredicate` is a predicate that is not composed of other predicates anymore.

    It represents the smallest kind of condition that evaluates to TRUE or FALSE.
    """

    def __init__(self, operation: Optional[expr.SqlOperator], *, hash_val: int) -> None:
        self._operation = operation
        super().__init__(hash_val)

    @property
    def operation(self) -> Optional[expr.SqlOperator]:
        """Provides the operation that is used to obtain matching (pairs of) tuples.

        Most of the time, this operation will be set to one of the SQL operators. However, in a special case there
        might not be an operation. This occurs for unary predicates that filter based on a predicate function (e.g.
        a user-defined function such as in `SELECT * FROM R WHERE my_udf_predicate(R.a, R.b)`).
        """
        return self._operation

    def is_compound(self) -> bool:
        return False

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]

    __hash__ = AbstractPredicate.__hash__


class BinaryPredicate(BasePredicate):
    """A `BinaryPredicate` combines exactly two base expressions with a comparison operation.

    This will be the most typical kind of join, as in `R.a = S.b` or filter, as in `R.a = 42`.
    """

    def __init__(self, operation: expr.SqlOperator, first_argument: expr.SqlExpression,
                 second_argument: expr.SqlExpression) -> None:
        if not first_argument or not second_argument:
            raise ValueError("First argument and second argument are required")
        self._first_argument = first_argument
        self._second_argument = second_argument

        hash_val = hash((operation, first_argument, second_argument))
        super().__init__(operation, hash_val=hash_val)

    @property
    def first_argument(self) -> expr.SqlExpression:
        """Get the first argument of the predicate."""
        return self._first_argument

    @property
    def second_argument(self) -> expr.SqlExpression:
        """Get the second argument of the predicate."""
        return self._second_argument

    def is_join(self) -> bool:
        first_tables = _collect_column_expression_tables(self.first_argument)
        if len(first_tables) > 1:
            return True

        second_tables = _collect_column_expression_tables(self.second_argument)
        if len(second_tables) > 1:
            return True

        return first_tables and second_tables and len(first_tables ^ second_tables) > 0

    def columns(self) -> set[base.ColumnReference]:
        return self.first_argument.columns() | self.second_argument.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return list(self.first_argument.itercolumns()) + list(self.second_argument.itercolumns())

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [self.first_argument, self.second_argument]

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        self._assert_join_predicate()
        first_columns = _collect_column_expression_columns(self.first_argument)
        second_columns = _collect_column_expression_columns(self.second_argument)

        partners = _generate_join_pairs(first_columns, first_columns)
        partners |= _generate_join_pairs(second_columns, second_columns)
        partners |= _generate_join_pairs(first_columns, second_columns)
        return partners

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.operation == other.operation
                and self.first_argument == other.first_argument
                and self.second_argument == other.second_argument)

    def __str__(self) -> str:
        return f"{self.first_argument} {self.operation.value} {self.second_argument}"


class BetweenPredicate(BasePredicate):
    """A BETWEEN predicate is a special case of a conjunction of two binary predicates.

    A BETWEEN predicate as the structure `<col> BETWEEN <a> AND <b>`, where `<col>` denotes the column expression to
    which the condition should apply and `<a>` and `<b>` are the expressions that denote the valid bounds.

    Each BETWEEN predicate can be represented by a conjunction of binary predicates: `<col> BETWEEN <a> AND <b>` is
    equivalent to `<col> >= <a> AND <col> <= <b>`.

    Notice that a BETWEEN predicate can be a join predicate as in `R.a BETWEEN 42 AND S.b`.
    """

    def __init__(self, column: expr.SqlExpression, interval: tuple[expr.SqlExpression, expr.SqlExpression]) -> None:
        if not column or not interval:
            raise ValueError("Column and interval must be set")
        self._column = column
        self._interval = interval
        self._interval_start, self._interval_end = self._interval

        hash_val = hash((expr.LogicalSqlOperators.Between, self._column, self._interval_start, self._interval_end))
        super().__init__(expr.LogicalSqlOperators.Between, hash_val=hash_val)

    @property
    def column(self) -> expr.SqlExpression:
        """Get the column that is tested (`R.a` in `SELECT * FROM R WHERE R.a BETWEEN 1 AND 42`)."""
        return self._column

    @property
    def interval(self) -> tuple[expr.SqlExpression, expr.SqlExpression]:
        """Get the interval (as lower, upper) that is tested against."""
        return self._interval

    @property
    def interval_start(self) -> expr.SqlExpression:
        """Get the lower bound of the interval that is tested against."""
        return self._interval_start

    @property
    def interval_end(self) -> expr.SqlExpression:
        """Get the upper bound of the interval that is tested against."""
        return self._interval_end

    def is_join(self) -> bool:
        column_tables = _collect_column_expression_tables(self.column)
        interval_start_tables = _collect_column_expression_tables(self.interval_start)
        interval_end_tables = _collect_column_expression_tables(self.interval_end)
        return (len(column_tables) > 1
                or len(column_tables | interval_start_tables) > 1
                or len(column_tables | interval_end_tables) > 1)

    def columns(self) -> set[base.ColumnReference]:
        return self.column.columns() | self.interval_start.columns() | self.interval_end.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return (list(self.column.itercolumns())
                + list(self.interval_start.itercolumns())
                + list(self.interval_end.itercolumns()))

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [self.column, self.interval_start, self.interval_end]

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        self._assert_join_predicate()
        predicate_columns = _collect_column_expression_columns(self.column)
        start_columns = _collect_column_expression_columns(self.interval_start)
        end_columns = _collect_column_expression_columns(self.interval_end)

        partners = _generate_join_pairs(predicate_columns, predicate_columns)
        partners |= _generate_join_pairs(predicate_columns, start_columns)
        partners |= _generate_join_pairs(predicate_columns, end_columns)
        return set(partners)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.column == other.column and self.interval == other.interval

    def __str__(self) -> str:
        interval_start, interval_end = self.interval
        return f"{self.column} BETWEEN {interval_start} AND {interval_end}"


class InPredicate(BasePredicate):
    """An IN predicate lists the allowed values for a column.

    In most cases, such a predicate with `n` allowed values can be transformed into a disjunction of `n` equality
    predicates, i.e. `R.a IN (1, 2, 3)` is equivalent to `R.a = 1 OR R.a = 2 OR R.a = 3`. Depending on the allowed
    values, an IN predicate can denote a join, e.g. `R.a IN (S.b, S.c)`. An important special case arises if the
    allowed values are produced by a subquery, e.g. `R.a IN (SELECT S.b FROM S)`. This does no longer allow for a
    transformation into binary predicates since it is unclear how many rows the subquery will produce.
    """

    def __init__(self, column: expr.SqlExpression, values: Sequence[expr.SqlExpression]) -> None:
        if not column or not values:
            raise ValueError("Both column and values must be given")
        if not all(val for val in values):
            raise ValueError("No empty value allowed")
        self._column = column
        self._values = tuple(values)
        hash_val = hash((expr.LogicalSqlOperators.In, self._column, self._values))
        super().__init__(expr.LogicalSqlOperators.In, hash_val=hash_val)

    @property
    def column(self) -> expr.SqlExpression:
        """Get the column that is tested (`R.a` in `SELECT * FROM R WHERE R.a IN (1, 2, 3)`)."""
        return self._column

    @property
    def values(self) -> Sequence[expr.SqlExpression]:
        """Get the values that possible values that are tested against."""
        return self._values

    def is_join(self) -> bool:
        column_tables = _collect_column_expression_tables(self.column)
        if len(column_tables) > 1:
            return True
        for value in self.values:
            value_tables = _collect_column_expression_tables(value)
            if len(column_tables | value_tables) > 1:
                return True
        return False

    def columns(self) -> set[base.ColumnReference]:
        all_columns = self.column.columns()
        for val in self.values:
            all_columns |= val.columns()
        return all_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return list(self.column.itercolumns()) + collection_utils.flatten(val.itercolumns() for val in self.values)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [self.column] + list(self.values)

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        self._assert_join_predicate()
        predicate_columns = _collect_column_expression_columns(self.column)

        partners = _generate_join_pairs(predicate_columns, predicate_columns)
        for value in self.values:
            value_columns = _collect_column_expression_columns(value)
            partners |= _generate_join_pairs(predicate_columns, value_columns)
        return partners

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.column == other.column and set(self.values) == set(other.values)

    def __str__(self) -> str:
        if len(self.values) == 1:
            value = collection_utils.simplify(self.values)
            vals = str(value) if isinstance(value, expr.SubqueryExpression) else f"({value})"
        else:
            vals = "(" + ", ".join(str(val) for val in self.values) + ")"
        return f"{self.column} IN {vals}"


class UnaryPredicate(BasePredicate):
    """A unary predicate is applied directly to an expression, evaluating to TRUE or FALSE.

    For example, `R.a IS NOT NULL`, `EXISTS (SELECT S.b FROM S WHERE R.a = S.b)` or `my_udf(R.a)`.
    """

    def __init__(self, column: expr.SqlExpression, operation: Optional[expr.SqlOperator] = None):
        if not column:
            raise ValueError("Column must be set")
        self._column = column
        super().__init__(operation, hash_val=hash((operation, column)))

    @property
    def column(self) -> expr.SqlExpression:
        """Get the expression (without the operation) that forms the predicate."""
        return self._column

    def is_join(self) -> bool:
        return len(_collect_column_expression_tables(self.column)) > 1

    def columns(self) -> set[base.ColumnReference]:
        return self.column.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.column.itercolumns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [self.column]

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        columns = _collect_column_expression_columns(self.column)
        return _generate_join_pairs(columns, columns)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.operation == other.operation and self.column == other.column

    def __str__(self) -> str:
        if not self.operation:
            return str(self.column)

        if isinstance(self.column, expr.SubqueryExpression) and self.operation == expr.LogicalSqlOperators.Exists:
            return f"EXISTS {self.column}"
        elif isinstance(self.column, expr.SubqueryExpression) and self.operation == expr.LogicalSqlOperators.Missing:
            return f"MISSING {self.column}"

        if self.operation == expr.LogicalSqlOperators.Exists:
            return f"{self.column} IS NOT NULL"
        elif self.operation == expr.LogicalSqlOperators.Missing:
            return f"{self.column} IS NULL"
        return f"{self.operation.value}{self.column}"


class CompoundPredicate(AbstractPredicate):
    """A `CompoundPredicate` contains other predicates in a hierarchical structure.

    Currently, PostBOUND supports 3 kinds of compound predicates: negations, conjunctions and disjunctions.
    """

    @staticmethod
    def create_and(parts: Collection[AbstractPredicate]) -> AbstractPredicate:
        """Creates an AND predicate joining the giving child predicates.

        If just a single child predicate is provided, returns that child directly instead of wrapping it in an
        AND predicate.
        """
        parts = list(parts)
        if not parts:
            raise ValueError("No predicates supplied.")
        if len(parts) == 1:
            return parts[0]
        return CompoundPredicate(expr.LogicalSqlCompoundOperators.And, list(parts))

    def __init__(self, operation: expr.LogicalSqlCompoundOperators,
                 children: AbstractPredicate | Sequence[AbstractPredicate]):
        if not operation or not children:
            raise ValueError("Operation and children must be set")
        if operation == expr.LogicalSqlCompoundOperators.Not and len(collection_utils.enlist(children)) > 1:
            raise ValueError("NOT predicates can only have one child predicate")
        if operation != expr.LogicalSqlCompoundOperators.Not and len(collection_utils.enlist(children)) < 2:
            raise ValueError("AND/OR predicates require at least two child predicates.")
        self._operation = operation
        self._children = tuple(collection_utils.enlist(children))
        super().__init__(hash((self._operation, self._children)))

    @property
    def operation(self) -> expr.LogicalSqlCompoundOperators:
        """Get the operation used to combine the individual evaluations of the child predicates."""
        return self._operation

    @property
    def children(self) -> Sequence[AbstractPredicate] | AbstractPredicate:
        """Get the child predicates that are combined in this compound predicate.

        For conjunctions and disjunctions this will be a sequence of children with at least two children. For negations
        the child predicate will be returned directly (i.e. without being wrapped in a sequence).
        """
        return self._children[0] if self.operation == expr.LogicalSqlCompoundOperators.Not else self._children

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self._children)

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(child.columns() for child in self._children)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(child.itercolumns() for child in self._children)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return collection_utils.flatten(child.iterexpressions() for child in self._children)

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        return collection_utils.set_union(child.join_partners() for child in self._children)

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return collection_utils.set_union(set(child.base_predicates()) for child in self._children)

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.operation == other.operation and self.children == other.children

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        if self.operation == expr.LogicalSqlCompoundOperators.Not:
            return f"NOT {self.children}"
        elif self.operation == expr.LogicalSqlCompoundOperators.Or:
            return "(" + " OR ".join(str(child) for child in self.children) + ")"
        elif self.operation == expr.LogicalSqlCompoundOperators.And:
            return " AND ".join(str(child) for child in self.children)
        else:
            raise ValueError(f"Unknown operation: '{self.operation}'")


def as_predicate(column: base.ColumnReference, operation: expr.LogicalSqlOperators,
                 *arguments) -> AbstractPredicate:
    """Utility method to quickly construct predicate instances.

    The given arguments are transformed into appropriate expression objects as necessary.

    The specific type of generated predicate is determined by the given operation. The following rules are applied:

    - for BETWEEN predicates, the arguments can be either two values, or a tuple of values
    (additional arguments are ignored)
    - for IN predicates, the arguments can be either a number of arguments, or a (nested) iterable of arguments
    - for all other binary predicates exactly one additional argument must be given (and an error is raised if that
    is not the case)
    """
    column = expr.ColumnExpression(column)

    if operation == expr.LogicalSqlOperators.Between:
        if len(arguments) == 1:
            lower, upper = arguments[0]
        else:
            lower, upper, *__ = arguments
        return BetweenPredicate(column, (expr.as_expression(lower), expr.as_expression(upper)))
    elif operation == expr.LogicalSqlOperators.In:
        arguments = collection_utils.flatten(arguments)
        return InPredicate(column, [expr.as_expression(value) for value in arguments])
    elif len(arguments) != 1:
        raise ValueError("Too many arguments for binary predicate: " + str(arguments))

    argument = arguments[0]
    return BinaryPredicate(operation, column, expr.as_expression(argument))


def _unwrap_expression(expression: expr.SqlExpression) -> base.ColumnReference | object:
    if isinstance(expression, expr.StaticValueExpression):
        return expression.value
    elif isinstance(expression, expr.ColumnExpression):
        return expression.column
    else:
        raise ValueError("Cannot unwrap expression " + str(expression))


def _attempt_filter_unwrap(predicate: AbstractPredicate
                           ) -> Optional[tuple[base.ColumnReference, expr.LogicalSqlOperators, object]]:
    if not predicate.is_filter() or not isinstance(predicate, BasePredicate):
        return None

    if isinstance(predicate, BinaryPredicate):
        try:
            left, right = _unwrap_expression(predicate.first_argument), _unwrap_expression(predicate.second_argument)
            operation = predicate.operation
            left, right = (left, right) if isinstance(left, base.ColumnReference) else (right, left)
            return left, operation, right
        except ValueError:
            return None
    elif isinstance(predicate, BetweenPredicate):
        try:
            column = _unwrap_expression(predicate.column)
            start = _unwrap_expression(predicate.interval_start)
            end = _unwrap_expression(predicate.interval_end)
            return column, expr.LogicalSqlOperators.Between, (start, end)
        except ValueError:
            return None
    elif isinstance(predicate, InPredicate):
        try:
            column = _unwrap_expression(predicate.column)
            values = [_unwrap_expression(val) for val in predicate.values]
            return column, expr.LogicalSqlOperators.In, tuple(values)
        except ValueError:
            return None
    else:
        raise ValueError("Unknown predicate type: " + str(predicate))


class SimplifiedFilterView(AbstractPredicate):
    """The intent behind this view is to provide more streamlined and direct access to filter predicates.

    As the name suggests, the view is a read-only predicate, i.e. it cannot be created on its own and has to be derived
    from a base predicate (either a binary predicate, a BETWEEN predicate or an IN predicate). Afterward, it provides
    read-only access to the predicate being filtered, the filter operation, as well as the values used to restrict
    the allowed column instances.

    Note that not all base predicates can be represented as a simplified view. In order for the view to work, both the
    column as well as the filter values cannot be modified by other expressions such as casts or mathematical
    expressions.
    """

    @staticmethod
    def wrap(predicate: AbstractPredicate) -> SimplifiedFilterView:
        """Transforms the given predicate into a simplified view. Raises an error if that is not possible."""
        return SimplifiedFilterView(predicate)

    @staticmethod
    def can_wrap(predicate: AbstractPredicate) -> bool:
        """Checks, whether the given predicate can be represented as a simplified view"""
        return _attempt_filter_unwrap(predicate) is not None

    @staticmethod
    def wrap_all(predicates: Iterable[AbstractPredicate]) -> Sequence[AbstractPredicate]:
        """Transforms each of the given predicates into a simplified view if that is possible.

        If none of the predicates can be simplified, an empty sequence is returned.
        """
        return [SimplifiedFilterView.wrap(pred) for pred in predicates if SimplifiedFilterView.can_wrap(pred)]

    def __init__(self, predicate: AbstractPredicate) -> None:
        column, operation, value = _attempt_filter_unwrap(predicate)
        self._column = column
        self._operation = operation
        self._value = value
        self._predicate = predicate

        hash_val = hash((column, operation, value))
        super().__init__(hash_val)

    @property
    def column(self) -> base.ColumnReference:
        """Get the filtered column."""
        return self._column

    @property
    def operation(self) -> expr.LogicalSqlOperators:
        """Get the SQL operation used for the filter (e.g. IN or <>)."""
        return self._operation

    @property
    def value(self) -> object | tuple[object] | Sequence[object]:
        """Get the filter value.

        For a binary predicate, this is just the value itself. For a BETWEEN predicate, this is tuple in the form
        `(lower, upper)` and for an IN predicate, this is a sequence of the allowed values.
        """
        return self._value

    def unwrap(self) -> AbstractPredicate:
        """Get the original predicate that is represented by this view."""
        return self._predicate

    def is_compound(self) -> bool:
        return False

    def is_join(self) -> bool:
        return False

    def columns(self) -> set[base.ColumnReference]:
        return {self.column}

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return [self.column]

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        return set()

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]

    __hash__ = AbstractPredicate.__hash__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.unwrap() == other.unwrap()

    def __str__(self) -> str:
        return str(self.unwrap())


def _collect_filter_predicates(predicate: AbstractPredicate) -> set[AbstractPredicate]:
    """Provides all base filter predicates that are contained in the given predicate hierarchy.

    This method handles compound predicates as follows:

    - conjunctions are un-nested, i.e. all predicates that form an AND predicate are collected individually
    - OR predicates are included with exactly those predicates from their children that are filters. If this is only
    true for a single predicate, that predicate will be returned directly.
    - NOT predicates are included if their child predicate is a filter
    """
    if isinstance(predicate, BasePredicate):
        return set() if predicate.is_join() else {predicate}
    elif isinstance(predicate, CompoundPredicate):
        if predicate.operation == expr.LogicalSqlCompoundOperators.Or:
            or_filter_children = [child_pred for child_pred in predicate.children if child_pred.is_filter()]
            if len(or_filter_children) < 2:
                return set(or_filter_children)
            or_filters = CompoundPredicate(expr.LogicalSqlCompoundOperators.Or, or_filter_children)
            return {or_filters}
        elif predicate.operation == expr.LogicalSqlCompoundOperators.Not:
            not_filter_children = predicate.children if predicate.children.is_filter() else None
            if not not_filter_children:
                return set()
            return {predicate}
        elif predicate.operation == expr.LogicalSqlCompoundOperators.And:
            return collection_utils.set_union([_collect_filter_predicates(child) for child in predicate.children])
        else:
            raise ValueError(f"Unknown operation: '{predicate.operation}'")
    else:
        raise ValueError(f"Unknown predicate type: {predicate}")


def _collect_join_predicates(predicate: AbstractPredicate) -> set[AbstractPredicate]:
    """Provides all join predicates that are contained in the given predicate hierarchy.

    This method handles compound predicates as follows:

    - conjunctions are un-nested, i.e. all predicates that form an AND predicate are collected individually
    - OR predicates are included with exactly those predicates from their children that are joins. If this is only
    true for a single predicate, that predicate will be returned directly.
    - NOT predicates are included if their child predicate is a join
    """
    if isinstance(predicate, BasePredicate):
        return {predicate} if predicate.is_join() else set()
    elif isinstance(predicate, CompoundPredicate):
        if predicate.operation == expr.LogicalSqlCompoundOperators.Or:
            or_join_children = [child_pred for child_pred in predicate.children if child_pred.is_join()]
            if len(or_join_children) < 2:
                return set(or_join_children)
            or_joins = CompoundPredicate(expr.LogicalSqlCompoundOperators.Or, or_join_children)
            return {or_joins}
        elif predicate.operation == expr.LogicalSqlCompoundOperators.Not:
            not_join_children = predicate.children if predicate.children.is_join() else None
            if not not_join_children:
                return set()
            return {predicate}
        elif predicate.operation == expr.LogicalSqlCompoundOperators.And:
            return collection_utils.set_union([_collect_join_predicates(child) for child in predicate.children])
        else:
            raise ValueError(f"Unknown operation: '{predicate.operation}'")
    else:
        raise ValueError(f"Unknown predicate type: {predicate}")


class QueryPredicates:
    """`QueryPredicates` provide high-level access to all the different predicates in a query.

    This access mainly revolves around identifying filter and join predicates easily, as well the appropriate
    predicates for specific classes. In this sense, `QueryPredicates` act as a wrapper around the actual query
    predicates.

    Currently, all usages of `QueryPredicates` by PostBOUND are part of the `SqlQuery` interface. If another
    distinction between joins and filters is necessary to satisfy some use case, this can be achieved by setting the
    `DefaultPredicateHandler` to a subclass of `QueryPredicates` that implements the required logic. The value of
    `DefaultPredicateHandler` is respected when `SqlQuery` object create their `QueryPredicates` in the `predicates`
    method.
    """

    @staticmethod
    def empty_predicate() -> QueryPredicates:
        """Constructs a `QueryPredicates` instance without any actual content."""
        return QueryPredicates(None)

    def __init__(self, root: Optional[AbstractPredicate]):
        self._root = root
        self._hash_val = hash(self._root)

    def is_empty(self) -> bool:
        """Checks, whether this predicate handler contains any actual predicates."""
        return self._root is None

    def root(self) -> AbstractPredicate:
        """Provides the root predicate that is wrapped by this `QueryPredicates` instance."""
        self._assert_not_empty()
        return self._root

    @functools.cache
    def filters(self) -> Collection[AbstractPredicate]:
        """Provides all filter predicates that are contained in the predicate hierarchy.

        This method handles compound predicates as follows:

        - conjunctions are un-nested, i.e. all predicates that form an AND predicate are collected individually
        - OR predicates are included with exactly those predicates from their children that are filters. If this is
        only true for a single predicate, that predicate will be returned directly.
        - NOT predicates are included if their child predicate is a filter.
        """
        return _collect_filter_predicates(self._root)

    @functools.cache
    def joins(self) -> Collection[AbstractPredicate]:
        """Provides all join predicates that are contained in the predicate hierarchy.

        This method handles compound predicates as follows:

        - conjunctions are un-nested, i.e. all predicates that form an AND predicate are collected individually
        - OR predicates are included with exactly those predicates from their children that are joins. If this is only
        true for a single predicate, that predicate will be returned directly.
        - NOT predicates are included if their child predicate is a join.
        """
        return _collect_join_predicates(self._root)

    @functools.cache
    def join_graph(self) -> nx.Graph:
        """Provides the join graph for the predicates.

        Nodes correspond to tables and edges to predicates that join those tables.
        """
        join_graph = nx.Graph()
        if self.is_empty():
            return join_graph

        tables = self._root.tables()
        join_graph.add_nodes_from(tables)
        for join in self.joins():
            for first_col, second_col in join.join_partners():
                join_graph.add_edge(first_col.table, second_col.table)
        return join_graph

    @functools.cache
    def filters_for(self, table: base.TableReference) -> Optional[AbstractPredicate]:
        """Provides all filter predicates that reference the given table.

        If multiple individual filter predicates are specified in the query, they will be combined in one large
        conjunction.

        The determination of matching filter predicates is the same as for the `filters()` method.
        """
        if self.is_empty():
            return None
        applicable_filters = [filter_pred for filter_pred in self.filters() if filter_pred.contains_table(table)]
        return CompoundPredicate.create_and(applicable_filters) if applicable_filters else None

    @functools.cache
    def joins_for(self, table: base.TableReference) -> Collection[AbstractPredicate]:
        """Provides all join predicates that reference the given table.

        Each entry in the resulting collection is a join predicate between the given table and a (set of) partner
        tables, such that the partner tables between different entries in the collection are also different. If
        multiple join predicates are specified between the given table and a specific (set of) partner tables, these
        predicates are aggregated into one large conjunction.

        The determination of matching join predicates is the same as for the `joins()` method.
        """
        if self.is_empty():
            return []

        applicable_joins: list[AbstractPredicate] = [join_pred for join_pred in self.joins()
                                                     if join_pred.contains_table(table)]
        distinct_joins: dict[frozenset[base.TableReference], list[AbstractPredicate]] = collections.defaultdict(list)
        for join_predicate in applicable_joins:
            partners = {column.table for column in join_predicate.join_partners_of(table)}
            distinct_joins[frozenset(partners)].append(join_predicate)

        aggregated_predicates = []
        for join_group in distinct_joins.values():
            aggregated_predicates.append(CompoundPredicate.create_and(join_group))
        return aggregated_predicates

    @functools.cache
    def joins_between(self, first_table: base.TableReference | Iterable[base.TableReference],
                      second_table: base.TableReference | Iterable[base.TableReference]) -> Optional[AbstractPredicate]:
        """Provides the (conjunctive) join predicate that joins the given tables.

        If `first_table` or `second_table` contain multiple tables, all join predicates between tables from the
        different sets are returned (but joins between tables from `first_table` or from `second_table` are not).

        Notice that the returned predicate might also include other tables, if they are part of a join predicate that
        also joins the given two tables.
        """
        if self.is_empty():
            return None

        if isinstance(first_table, base.TableReference) and isinstance(second_table, base.TableReference):
            first_joins: Collection[AbstractPredicate] = self.joins_for(first_table)
            matching_joins = [join for join in first_joins if join.joins_table(second_table)]
            return CompoundPredicate.create_and(matching_joins) if matching_joins else None

        matching_joins = []
        first_table, second_table = collection_utils.enlist(first_table), collection_utils.enlist(second_table)
        for first in frozenset(first_table):
            for second in frozenset(second_table):
                join_predicate = self.joins_between(first, second)
                if not join_predicate:
                    continue
                matching_joins.append(join_predicate)
        return CompoundPredicate.create_and(matching_joins) if matching_joins else None

    def joins_tables(self, tables: base.TableReference | Iterable[base.TableReference],
                     *more_tables: base.TableReference) -> bool:
        """Checks, whether the given tables are all joined with each other.

        This does not mean that there has to be a join predicate between each pair of tables, but rather that all pairs
        of tables must at least be connected through a sequence of other tables. From a graph theory-centric point of
        view this means that the join (sub-) graph induced by the given tables is connected.
        """
        if self.is_empty():
            return False
        tables = [tables] if not isinstance(tables, Iterable) else list(tables)
        tables = frozenset(set(tables) | set(more_tables))
        return self._join_tables_check(tables)

    def and_(self, other_predicate: QueryPredicates | AbstractPredicate) -> QueryPredicates:
        """Combines the given predicates with the predicates in this query via a conjunction."""
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
    def _join_tables_check(self, tables: frozenset[base.TableReference]) -> bool:
        """Constructs the join graph for the given tables and checks, whether it is connected."""
        join_graph = nx.Graph()
        join_graph.add_nodes_from(tables)
        for table in tables:
            for join in self.joins_for(table):
                partner_tables = set(col.table for col in join.join_partners_of(table)) & tables
                join_graph.add_edges_from(itertools.product([table], partner_tables))
        return nx.is_connected(join_graph)

    def _assert_not_empty(self) -> None:
        """Raises a `StateError` if this predicates instance is empty."""
        if self._root is None:
            raise errors.StateError("No query predicates!")

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
"""
