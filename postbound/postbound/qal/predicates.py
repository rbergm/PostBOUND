"""Predicates are the central building block to represent filter conditions for SQL queries.

A predicate is a boolean expression that can be applied to a tuple to determine whether it should be kept in the intermediate
result or thrown away. PostBOUND distinguishes between two kinds of predicates, even though they are both represented by the
same class: there are filter predicates, which - as a rule of thumb - can be applied directly to base table relations.
Furthermore, there are join predicates that access tuples from different relations and determine whether the join of both
tuples should become part of the intermediate result.

PostBOUND's implementation of predicates is structured using a composite-style layout. At the core there of the module there
is the `AbstractPredicate` interface that describes all the behaviour that is common to all concrete predicate types. Then,
there are `BasePredicate`s, which are built directly on `expressions`. Lastly, the `CompoundPredicate` is used to nest
different predicates, thereby creating tree-shaped hierarchies.

In line with the other parts of the query abstraction layer, predicates are designed as read-only data objects. Any forced
modifications on predicates will break the entire qal and result in unpredictable behaviour.

In addition to the predicate representation, this module also provides a utility for streamlined access to the important parts
of simple filter predicates via the `SimplifiedFilterView`.

Lastly, the `QueryPredicates` provide high-level access to all predicates (join and filter) that are specified in a query.
From a user perspective, this is probably the best entry point to work with predicates. Alternatively, the predicate tree can
also be traversed using custom functions.
"""
from __future__ import annotations

import abc
import collections
import functools
import itertools
import typing
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import Type, Union, Optional, Literal

import networkx as nx

from postbound.qal import base, expressions as expr
from postbound.util import errors, collections as collection_utils


def _normalize_join_pair(columns: tuple[base.ColumnReference, base.ColumnReference]
                         ) -> tuple[base.ColumnReference, base.ColumnReference]:
    """Normalizes the given join such that a pair ``(R.a, S.b)`` and ``(S.b, R.a)`` can be recognized as equal.

    Normalization in this context means that the order in which two appear is always the same. Therefore, this method
    essentially forces a swap of the columns if necessary, making use of the ability to sort columns lexicographically.

    Parameters
    ----------
    columns : tuple[base.ColumnReference, base.ColumnReference]
        The join pair to normalize

    Returns
    -------
    tuple[base.ColumnReference, base.ColumnReference]
        The normalized join pair.
    """
    first_col, second_col = columns
    return (second_col, first_col) if second_col < first_col else columns


class NoJoinPredicateError(errors.StateError):
    """Error to indicate that a filter predicate was supplied at a place where a join predicate was expected.

    Parameters
    ----------
    predicate : AbstractPredicate | None, optional
        The predicate that caused the error, defaults to ``None``
    """

    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


class NoFilterPredicateError(errors.StateError):
    """Error to indicate that a join predicate was supplied at a place where a filter predicate was expected.

    Parameters
    ----------
    predicate : AbstractPredicate | None, optional
        The predicate that caused the error, defaults to ``None``.
    """

    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


BaseExpression = Union[expr.ColumnExpression, expr.StaticValueExpression, expr.SubqueryExpression]
"""Supertype that captures all expression types that can be considered base expressions for predicates."""


def _collect_base_expressions(expression: expr.SqlExpression) -> Iterable[BaseExpression]:
    """Provides all base expressions that are contained in a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree.

    Parameters
    ----------
    expression : expr.SqlExpression
        The expression to traverse

    Returns
    -------
    Iterable[BaseExpression]
        The base expressions
    """
    if isinstance(expression, (expr.ColumnExpression, expr.StaticValueExpression, expr.SubqueryExpression)):
        return [expression]
    return collection_utils.flatten(_collect_base_expressions(child_expr) for child_expr in expression.iterchildren())


def _collect_subquery_expressions(expression: expr.SqlExpression) -> Iterable[expr.SubqueryExpression]:
    """Provides all subquery expressions that are contained in a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree.

    Parameters
    ----------
    expression : expr.SqlExpression
        The expression to traverse

    Returns
    -------
    Iterable[expr.SubqueryExpression]
        All subqueries that are contained in some level in the expression
    """
    return [child_expr for child_expr in _collect_base_expressions(expression)
            if isinstance(child_expr, expr.SubqueryExpression)]


def _collect_column_expression_columns(expression: expr.SqlExpression) -> set[base.ColumnReference]:
    """Provides all columns that are directly contained in `ColumnExpression` instances with a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree. Notice that it ignores all
    expressions that are part of subqueries.

    Parameters
    ----------
    expression : expr.SqlExpression
        The expression to traverse

    Returns
    -------
    set[base.ColumnReference]
        All columns that are referenced in `ColumnExpression`s
    """
    return collection_utils.set_union(base_expr.columns() for base_expr in _collect_base_expressions(expression)
                                      if isinstance(base_expr, expr.ColumnExpression))


def _collect_column_expression_tables(expression: expr.SqlExpression) -> set[base.TableReference]:
    """Provides all tables that are linked directly in `ColumnExpression` instances with a specific expression.

    This method is a shorthand to take care of the necessary traversal of the expression tree. Notice that it ignores all
    expressions that are part of subqueries.

    Parameters
    ----------
    expression : expr.SqlExpression
        The expression to traverse

    Returns
    -------
    set[base.TableReference]
        All tables that are referenced in the columns of `ColumnExpression`s
    """
    return {column.table for column in _collect_column_expression_columns(expression) if column.is_bound()}


def _generate_join_pairs(first_columns: Iterable[base.ColumnReference], second_columns: Iterable[base.ColumnReference]
                         ) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
    """Provides all possible pairs of columns where each column comes from a different iterable.

    Essentially, this produces the cross product of the two column sets. The join pairs are normalized and duplicate
    elimination is performed (this is necessary since columns can appear in both iterables). Likewise, "joins" over the same
    logical relations are also skipped.

    Parameters
    ----------
    first_columns : Iterable[base.ColumnReference]
        Candidate columns to be joined with `second_columns`
    second_columns : Iterable[base.ColumnReference]
        The join partner columns

    Returns
    -------
    set[tuple[base.ColumnReference, base.ColumnReference]]
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

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are accessed by this predicate.

        Notice that even for filters, the provided set might contain multiple entries, e.g. if the predicate contains
        subqueries.

        Returns
        -------
        set[base.TableReference]
            All tables. This can include virtual tables if such tables are referenced in the predicate.
        """
        return collection_utils.set_union(e.tables() for e in self.iterexpressions())

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced by this predicate.

        Returns
        -------
        set[base.ColumnReference]
            The columns. If the predicate contains a subquery, all columns of that query are included.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides all columns that are referenced by this predicate.

        If a column is referenced multiple times, it is also returned multiple times.

        Returns
        -------
        Iterable[base.ColumnReference]
            All columns in exactly the order in which they are used.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all expressions that are directly contained in this predicate.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).

        Returns
        -------
        Iterable[SqlExpression]
            The expressions
        """
        raise NotImplementedError

    def contains_table(self, table: base.TableReference) -> bool:
        """Checks, whether this predicate filters or joins a column of a specific table.

        Parameters
        ----------
        table : base.TableReference
            The table to check

        Returns
        -------
        bool
            Whether the given `table` is referenced by any of the columns in the predicate.
        """
        return any(table == tab for tab in self.tables())

    def joins_table(self, table: base.TableReference) -> bool:
        """Checks, whether this predicate describes a join and one of the join partners is a specific table.

        Parameters
        ----------
        table : base.TableReference
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

    def columns_of(self, table: base.TableReference) -> set[base.ColumnReference]:
        """Retrieves all columns of a specific table that are referenced by this predicate.

        Parameters
        ----------
        table : base.TableReference
            The table to check

        Returns
        -------
        set[base.ColumnReference]
            All columns in this predicate that belong to the given `table`
        """
        return {col for col in self.columns() if col.belongs_to(table)}

    def join_partners_of(self, table: base.TableReference) -> set[base.ColumnReference]:
        """Retrieves all columns that are joined with a specific table.

        Parameters
        ----------
        table : base.TableReference
            The table for which the join partners should be searched

        Returns
        -------
        set[base.ColumnReference]
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
    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        """Provides all pairs of columns that are joined within this predicate.

        If multiple columns are joined or it is unclear which columns are involved in a join exactly, this method falls back to
        returning the cross-product of all potential join partners. For example, consider the following query:
        ``SELECT * FROM R, S WHERE my_udf(R.a, R.b, S.c)``. In this case, it cannot be determined which columns of ``R`` take
        part in the join. Therefore, `join_partners` will return the set ``{(R.a, S.c), (R.b, S.c)}``.

        Returns
        -------
        set[tuple[base.ColumnReference, base.ColumnReference]]
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

    def required_tables(self) -> set[base.TableReference]:
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
        set[base.TableReference]
            The tables that need to be provided by the query execution engine in order to run this predicate
        """
        subqueries = collection_utils.flatten(_collect_subquery_expressions(child_expr)
                                              for child_expr in self.iterexpressions())
        subquery_tables = collection_utils.set_union(subquery.query.unbound_tables() for subquery in subqueries)
        column_tables = collection_utils.set_union(_collect_column_expression_tables(child_expr)
                                                   for child_expr in self.iterexpressions())
        return column_tables | subquery_tables

    @abc.abstractmethod
    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        """Enables processing of the current predicate by a predicate visitor.

        Parameters
        ----------
        visitor : PredicateVisitor
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
    operation : Optional[expr.SqlOperator]
        The operation that compares the column value(s). For unary base predicates, this may be ``None`` if a
        predicate function is used to determine matching tuples.
    hash_val : int
        The hash of the entire predicate
    """

    def __init__(self, operation: Optional[expr.SqlOperator], *, hash_val: int) -> None:
        self._operation = operation
        super().__init__(hash_val)

    @property
    def operation(self) -> Optional[expr.SqlOperator]:
        """Get the operation that is used to obtain matching (pairs of) tuples.

        Most of the time, this operation will be set to one of the SQL operators. However, for unary predicates that filter
        based on a predicate function this might be ``None`` (e.g. a user-defined function such as in
        `SELECT * FROM R WHERE my_udf_predicate(R.a, R.b)`).

        Returns
        -------
        Optional[expr.SqlOperator]
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
    operation : expr.SqlOperator
        The operation that combines the input arguments
    first_argument : expr.SqlExpression
        The first comparison value
    second_argument : expr.SqlExpression
        The second comparison value
    """

    @staticmethod
    def equal(first_argument: expr.SqlExpression, second_argument: expr.SqlExpression) -> BinaryPredicate:
        """Generates an equality predicate between two arguments."""
        return BinaryPredicate(expr.LogicalSqlOperators.Equal, first_argument, second_argument)

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
        """Get the first argument of the predicate.

        Returns
        -------
        expr.SqlExpression
            The argument
        """
        return self._first_argument

    @property
    def second_argument(self) -> expr.SqlExpression:
        """Get the second argument of the predicate.

        Returns
        -------
        expr.SqlExpression
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
    column : expr.SqlExpression
        The value that is checked by the predicate
    interval : tuple[expr.SqlExpression, expr.SqlExpression]
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

    def __init__(self, column: expr.SqlExpression, interval: tuple[expr.SqlExpression, expr.SqlExpression]) -> None:
        if not column or not interval or len(interval) != 2:
            raise ValueError("Column and interval must be set")
        self._column = column
        self._interval = interval
        self._interval_start, self._interval_end = self._interval

        hash_val = hash((expr.LogicalSqlOperators.Between, self._column, self._interval_start, self._interval_end))
        super().__init__(expr.LogicalSqlOperators.Between, hash_val=hash_val)

    @property
    def column(self) -> expr.SqlExpression:
        """Get the column that is tested (``R.a`` in ``SELECT * FROM R WHERE R.a BETWEEN 1 AND 42``).

        Returns
        -------
        expr.SqlExpression
            The expression
        """
        return self._column

    @property
    def interval(self) -> tuple[expr.SqlExpression, expr.SqlExpression]:
        """Get the interval (as ``(lower, upper)``) that is tested against.

        Returns
        -------
        tuple[expr.SqlExpression, expr.SqlExpression]
            The allowed range of values. This interval is inclusive at both endpoints.
        """
        return self._interval

    @property
    def interval_start(self) -> expr.SqlExpression:
        """Get the lower bound of the interval that is tested against.

        Returns
        -------
        expr.SqlExpression
            The lower value. This value is inclusive, i.e. the comparison values must be greater or equal.
        """
        return self._interval_start

    @property
    def interval_end(self) -> expr.SqlExpression:
        """Get the upper bound of the interval that is tested against.

        Returns
        -------
        expr.SqlExpression
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
    column : expr.SqlExpression
        The value that is checked by the predicate
    values : Sequence[expr.SqlExpression]
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
        """Get the expression that is tested (``R.a`` in ``SELECT * FROM R WHERE R.a IN (1, 2, 3)``).

        Returns
        -------
        expr.SqlExpression
            The expression
        """
        return self._column

    @property
    def values(self) -> Sequence[expr.SqlExpression]:
        """Get the allowed values of the tested expression.

        Returns
        -------
        Sequence[expr.SqlExpression]
            The allowed values. This sequence always contains at least one entry.
        """
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

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_in_predicate(self)

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

    Examples of such predicates include ``R.a IS NOT NULL``, ``EXISTS (SELECT S.b FROM S WHERE R.a = S.b)``, or
    ``my_udf(R.a)``. In the last case, ``my_udf`` has to produce a boolean return value.

    Parameters
    ----------
    column : expr.SqlExpression
        The expression that is tested. This can also be a user-defined function that produces a boolen return value.
    operation : Optional[expr.SqlOperator], optional
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

    def __init__(self, column: expr.SqlExpression, operation: Optional[expr.SqlOperator] = None):
        if not column:
            raise ValueError("Column must be set")
        if operation is not None and operation not in expr.UnarySqlOperators:
            raise ValueError(f"Not an allowed unary operator: {operation}")
        self._column = column
        super().__init__(operation, hash_val=hash((operation, column)))

    @property
    def column(self) -> expr.SqlExpression:
        """The column that is checked by this predicate

        Returns
        -------
        expr.SqlExpression
            The expression
        """
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

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        return visitor.visit_unary_predicate(self)

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
    """A compound predicate creates a composite hierarchical structure of other predicates.

    Currently, PostBOUND supports 3 kinds of compound predicates: negations, conjunctions and disjunctions. Depending on the
    specific compound operator, a diferent number of child predicates is allowed.

    Parameters
    ----------
    operation : expr.LogicalSqlCompoundOperators
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
    def create(operation: expr.LogicalSqlCompoundOperators, parts: Collection[AbstractPredicate]) -> AbstractPredicate:
        """Creates an arbitrary compound predicate for a number of child predicates.

        If just a single child predicate is provided, but the operation requires multiple children, that child is returned
        directly instead of the compound predicate.

        Parameters
        ----------
        operation : expr.LogicalSqlCompoundOperators
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
        if operation == expr.LogicalSqlCompoundOperators.Not and len(parts) != 1:
            raise ValueError(f"Can only create negations for exactly one predicate but received: '{parts}'")
        elif operation != expr.LogicalSqlCompoundOperators.Not and not parts:
            raise ValueError("Conjunctions/disjunctions require at least one predicate")

        match operation:
            case expr.LogicalSqlCompoundOperators.Not:
                return CompoundPredicate.create_not(parts[0])
            case expr.LogicalSqlCompoundOperators.And | expr.LogicalSqlCompoundOperators.Or:
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
        return CompoundPredicate(expr.LogicalSqlCompoundOperators.And, parts)

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
        return CompoundPredicate(expr.LogicalSqlCompoundOperators.Not, predicate)

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
        return CompoundPredicate(expr.LogicalSqlCompoundOperators.Or, parts)

    def __init__(self, operation: expr.LogicalSqlCompoundOperators,
                 children: AbstractPredicate | Sequence[AbstractPredicate]) -> None:
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
        """Get the operation used to combine the individual evaluations of the child predicates.

        Returns
        -------
        expr.LogicalSqlCompoundOperators
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
        return self._children[0] if self.operation == expr.LogicalSqlCompoundOperators.Not else self._children

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self._children)

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(child.columns() for child in self._children)

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

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(child.itercolumns() for child in self._children)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return collection_utils.flatten(child.iterexpressions() for child in self._children)

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        return collection_utils.set_union(child.join_partners() for child in self._children)

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return collection_utils.set_union(set(child.base_predicates()) for child in self._children)

    def accept_visitor(self, visitor: PredicateVisitor[VisitorResult]) -> VisitorResult:
        match self.operation:
            case expr.LogicalSqlCompoundOperators.Not:
                return visitor.visit_not_predicate(self, self.children)
            case expr.LogicalSqlCompoundOperators.And:
                return visitor.visit_and_predicate(self, self.children)
            case expr.LogicalSqlCompoundOperators.Or:
                return visitor.visit_or_predicate(self, self.children)
            case _:
                raise ValueError(f"Unknown operation: '{self.operation}'")

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


VisitorResult = typing.TypeVar("VisitorResult")
"""Result of visitor invocations."""


class PredicateVisitor(abc.ABC, typing.Generic[VisitorResult]):
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


def as_predicate(column: base.ColumnReference, operation: expr.LogicalSqlOperators,
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
    column : base.ColumnReference
        The column that should become the first operand of the predicate
    operation : expr.LogicalSqlOperators
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


def determine_join_equivalence_classes(predicates: Iterable[BinaryPredicate]) -> set[frozenset[base.ColumnReference]]:
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
    set[frozenset[base.ColumnReference]]
        The equivalence classes. Each element of the set describes a complete equivalence class.
    """
    join_predicates = {pred for pred in predicates
                       if isinstance(pred, BinaryPredicate) and pred.is_join()
                       and pred.operation == expr.LogicalSqlOperators.Equal}

    equivalence_graph = nx.Graph()
    for predicate in join_predicates:
        columns = predicate.columns()
        if not len(columns) == 2:
            continue
        col_a, col_b = columns
        equivalence_graph.add_edge(col_a, col_b)

    equivalence_classes: set[set[base.ColumnReference]] = set()
    for equivalence_class in nx.connected_components(equivalence_graph):
        equivalence_classes.add(frozenset(equivalence_class))
    return equivalence_classes


def generate_predicates_for_equivalence_classes(equivalence_classes: set[frozenset[base.ColumnReference]]
                                                ) -> set[BinaryPredicate]:
    """Provides all possible equijoin predicates for a set of equivalence classes.

    This function can be used in combination with `determine_join_equivalence_classes` to expand join predicates to also
    include additional joins that can be derived from the predicates.

    For example, consider two joins *a = b* and *b = c*. These joins form one equivalence class *{a, b, c}*. Based on the
    equivalence class, the predicates *a = b*, *b = c* and *a = c* can be generated.

    Parameters
    ----------
    equivalence_classes : set[frozenset[base.ColumnReference]]
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
        for first_col, second_col in collection_utils.pairs(equivalence_class):
            equivalence_predicates.add(as_predicate(first_col, expr.LogicalSqlOperators.Equal, second_col))
    return equivalence_predicates


def _unwrap_expression(expression: expr.ColumnExpression | expr.StaticValueExpression) -> base.ColumnReference | object:
    """Provides the column of a `ColumnExpression` or the value of a `StaticValueExpression`.

    This is a utility method to gain quick access to the values in simple predicates.

    Parameters
    ----------
    expression : expr.SqlExpression
        The expression to unwrap.

    Returns
    -------
    base.ColumnReference | object
        The column or value contained in the expression.
    """
    if isinstance(expression, expr.StaticValueExpression):
        return expression.value
    elif isinstance(expression, expr.ColumnExpression):
        return expression.column
    else:
        raise ValueError("Cannot unwrap expression " + str(expression))


UnwrappedFilter = tuple[base.ColumnReference, expr.LogicalSqlOperators, object]
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
    def column(self) -> base.ColumnReference:
        """Get the filtered column.

        Returns
        -------
        base.ColumnReference
            The column
        """
        return self._column

    @property
    def operation(self) -> expr.LogicalSqlOperators:
        """Get the SQL operation that is used for the filter (e.g. ``IN`` or ``<>``).

        Returns
        -------
        expr.LogicalSqlOperators
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
    def filters_for(self, table: base.TableReference) -> Optional[AbstractPredicate]:
        """Provides all filter predicates that reference a specific table.

        If multiple individual filter predicates are specified in the query, they will be combined in one large
        conjunction.

        The determination of matching filter predicates is the same as for the `filters()` method.

        Parameters
        ----------
        table : base.TableReference
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
    def joins_for(self, table: base.TableReference) -> Collection[AbstractPredicate]:
        """Provides all join predicates that reference a specific table.

        Each entry in the resulting collection is a join predicate between the given table and a (set of) partner tables, such
        that the partner tables in different entries in the collection are also different. If multiple join predicates are
        specified between the given table and a specific (set of) partner tables, these predicates are aggregated into one
        large conjunction.

        The determination of matching join predicates is the same as for the `joins()` method.

        Parameters
        ----------
        table : base.TableReference
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
        distinct_joins: dict[frozenset[base.TableReference], list[AbstractPredicate]] = collections.defaultdict(list)
        for join_predicate in applicable_joins:
            partners = {column.table for column in join_predicate.join_partners_of(table)}
            distinct_joins[frozenset(partners)].append(join_predicate)

        aggregated_predicates = []
        for join_group in distinct_joins.values():
            aggregated_predicates.append(CompoundPredicate.create_and(join_group))
        return aggregated_predicates

    def joins_between(self, first_table: base.TableReference | Iterable[base.TableReference],
                      second_table: base.TableReference | Iterable[base.TableReference], *,
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
        first_table : base.TableReference | Iterable[base.TableReference]
            The (set of) tables to join
        second_table : base.TableReference | Iterable[base.TableReference]
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

    def joins_tables(self, tables: base.TableReference | Iterable[base.TableReference],
                     *more_tables: base.TableReference) -> bool:
        """Checks, whether specific tables are all joined with each other.

        This does not mean that there has to be a join predicate between each pair of tables, but rather that all pairs
        of tables must at least be connected through a sequence of other tables. From a graph-theory centric point of
        view this means that the join (sub-) graph induced by the given tables is connected.

        Parameters
        ----------
        tables : base.TableReference | Iterable[base.TableReference]
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
            table = collection_utils.simplify(tables)
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
    def _join_tables_check(self, tables: frozenset[base.TableReference]) -> bool:
        """Constructs the join graph for the given tables and checks, whether it is connected.

        Parameters
        ----------
        tables : frozenset[base.TableReference]
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
        errors.StateError
            If there is no root predicate
        """
        if self._root is None:
            raise errors.StateError("No query predicates!")

    def _init_join_predicate_map(self) -> dict[frozenset[base.TableReference], AbstractPredicate]:
        """Generates the necessary mapping for `_map_based_joins_between`.

        This is a static data structure and hence can be pre-computed.

        Returns
        -------
        dict[frozenset[base.TableReference], AbstractPredicate]
            A mapping from a set of tables to the join predicate that is specified between those tables. If a set of tables
            does not appear in the dictionary, there is no join predicate between the specific tables.
        """
        if self.is_empty():
            return {}

        predicate_map: dict[frozenset[base.TableReference], AbstractPredicate] = {}
        for table in self._root.tables():
            join_partners = self.joins_for(table)
            for join_predicate in join_partners:
                partner_tables = {partner.table for partner in join_predicate.join_partners_of(table)}
                map_key = frozenset(partner_tables | {table})
                if map_key in predicate_map:
                    continue
                predicate_map[map_key] = join_predicate
        return predicate_map

    def _legacy_joins_between(self, first_table: base.TableReference | Iterable[base.TableReference],
                              second_table: base.TableReference | Iterable[base.TableReference]
                              ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined using the legacy recursive structure.

        .. deprecated::
            There is no real advantage of using this method, other than slightly easier debugging. Should be removed at some
            later point in time (in which case the old ``"legacy"`` strategy key will be re-mapped to a different) strategy.

        Parameters
        ----------
        first_table : base.TableReference | Iterable[base.TableReference]
            The (set of) tables to join
        second_table : base.TableReference | Iterable[base.TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, ``None`` is returned.
        """
        if isinstance(first_table, base.TableReference) and isinstance(second_table, base.TableReference):
            if first_table == second_table:
                return None
            first_joins: Collection[AbstractPredicate] = self.joins_for(first_table)
            matching_joins = {join for join in first_joins if join.joins_table(second_table)}
            return CompoundPredicate.create_and(matching_joins) if matching_joins else None

        matching_joins = set()
        first_table, second_table = collection_utils.enlist(first_table), collection_utils.enlist(second_table)
        for first in frozenset(first_table):
            for second in frozenset(second_table):
                join_predicate = self.joins_between(first, second)
                if not join_predicate:
                    continue
                matching_joins.add(join_predicate)
        return CompoundPredicate.create_and(matching_joins) if matching_joins else None

    def _graph_based_joins_between(self, first_table: base.TableReference | Iterable[base.TableReference],
                                   second_table: base.TableReference | Iterable[base.TableReference]
                                   ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined using a graph-based approach.

        Parameters
        ----------
        first_table : base.TableReference | Iterable[base.TableReference]
            The (set of) tables to join
        second_table : base.TableReference | Iterable[base.TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, ``None`` is returned.

        """
        join_graph = self.join_graph()
        first_table, second_table = collection_utils.enlist(first_table), collection_utils.enlist(second_table)

        if len(first_table) > 1:
            first_first_table, *remaining_first_tables = first_table
            for remaining_first_table in remaining_first_tables:
                join_graph = nx.contracted_nodes(join_graph, first_first_table, remaining_first_table)
            first_hook = first_first_table
        else:
            first_hook = collection_utils.simplify(first_table)

        if len(second_table) > 1:
            first_second_table, *remaining_second_tables = second_table
            for remaining_second_table in remaining_second_tables:
                join_graph = nx.contracted_nodes(join_graph, first_second_table, remaining_second_table)
            second_hook = first_second_table
        else:
            second_hook = collection_utils.simplify(second_table)

        if (first_hook, second_hook) not in join_graph.edges:
            return None
        return join_graph.edges[first_hook, second_hook]["predicate"]

    def _map_based_joins_between(self, first_table: base.TableReference | Iterable[base.TableReference],
                                 second_table: base.TableReference | Iterable[base.TableReference]
                                 ) -> Optional[AbstractPredicate]:
        """Determines how two (sets of) tables can be joined together using a map-based approach.

        This method is the preferred way of inferring the join predicate from the two candidate sets. It is based on static
        map data that was precomputed as part of `_init_join_predicate_map`.

        Parameters
        ----------
        first_table : base.TableReference | Iterable[base.TableReference]
            The (set of) tables to join
        second_table : base.TableReference | Iterable[base.TableReference]
            The (set of) join partners for `first_table`.

        Returns
        -------
        Optional[AbstractPredicate]
            A conjunction of all the individual join predicates between the two sets of candidate tables. If there is no join
            predicate between any of the tables, ``None`` is returned.

        """
        join_predicates = set()
        first_table, second_table = collection_utils.enlist(first_table), collection_utils.enlist(second_table)
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
