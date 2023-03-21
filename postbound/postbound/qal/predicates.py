"""Contains those parts of the `qal`, that are dedicated to representing predicates of SQL queries."""

from __future__ import annotations

import abc
import functools
import itertools
from collections.abc import Collection
from typing import Iterable, Iterator, Union

from postbound.qal import base, expressions as expr
from postbound.util import errors, collections as collection_utils


def _normalize_join_pair(columns: tuple[base.ColumnReference, base.ColumnReference]
                         ) -> tuple[base.ColumnReference, base.ColumnReference]:
    first_col, second_col = columns
    return (second_col, first_col) if second_col < first_col else columns


class NoJoinPredicateError(errors.StateError):
    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


class NoFilterPredicateError(errors.StateError):
    def __init__(self, predicate: AbstractPredicate | None = None) -> None:
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


BaseExpression = Union[expr.ColumnExpression, expr.StaticValueExpression, expr.SubqueryExpression]


def _collect_base_expressions(expression: expr.SqlExpression) -> Iterable[BaseExpression]:
    if isinstance(expression, (expr.ColumnExpression, expr.StaticValueExpression, expr.SubqueryExpression)):
        return [expression]
    return collection_utils.flatten(_collect_base_expressions(child_expr) for child_expr in expression.iterchildren())


def _collect_subquery_expressions(expression: expr.SqlExpression) -> Iterable[expr.SubqueryExpression]:
    return [child_expr for child_expr in _collect_base_expressions(expression)
            if isinstance(child_expr, expr.SubqueryExpression)]


def _collect_column_expression_columns(expression: expr.SqlExpression) -> set[base.ColumnReference]:
    return collection_utils.set_union(base_expr.columns() for base_expr in _collect_base_expressions(expression)
                                      if isinstance(base_expr, expr.ColumnExpression))


def _collect_column_expression_tables(expression: expr.SqlExpression) -> set[base.TableReference]:
    return {column.table for column in _collect_column_expression_columns(expression) if column.is_bound()}


def _generate_join_pairs(first_columns: Iterable[base.ColumnReference],
                         second_columns: Iterable[base.ColumnReference]
                         ) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
    return {_normalize_join_pair((first_col, second_col)) for first_col, second_col
            in itertools.product(first_columns, second_columns) if first_col.table != second_col.table}


class AbstractPredicate(abc.ABC):
    def __init__(self) -> None:
        pass

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

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class BasePredicate(AbstractPredicate, abc.ABC):
    def __init__(self, operation: expr.SqlOperator) -> None:
        super().__init__()
        self.operation = operation

    def is_compound(self) -> bool:
        return False

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]


class BinaryPredicate(BasePredicate):
    def __init__(self, operation: expr.SqlOperator, first_argument: expr.SqlExpression,
                 second_argument: expr.SqlExpression) -> None:
        super().__init__(operation)
        self.first_argument = first_argument
        self.second_argument = second_argument

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

    def __hash__(self) -> int:
        return hash(tuple([self.operation, self.first_argument, self.second_argument]))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.operation == other.operation
                and self.first_argument == other.first_argument
                and self.second_argument == other.second_argument)

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        return f"{self.first_argument} {self.operation.value} {self.second_argument}"


class BetweenPredicate(BasePredicate):
    def __init__(self, column: expr.SqlExpression, interval: tuple[expr.SqlExpression, expr.SqlExpression]) -> None:
        super().__init__(expr.LogicalSqlOperators.Between)
        self.column = column
        self.interval = interval
        self.interval_start, self.interval_end = self.interval

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

    def __hash__(self) -> int:
        return hash(tuple(["BETWEEN", self.column, self.interval]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.column == other.column and self.interval == other.interval

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        interval_start, interval_end = self.interval
        return f"{self.column} BETWEEN {interval_start} AND {interval_end}"


class InPredicate(BasePredicate):
    def __init__(self, column: expr.SqlExpression, values: list[expr.SqlExpression]) -> None:
        super().__init__(expr.LogicalSqlOperators.In)
        self.column = column
        self.values = values

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
        return [self.column] + self.values

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        self._assert_join_predicate()
        predicate_columns = _collect_column_expression_columns(self.column)

        partners = _generate_join_pairs(predicate_columns, predicate_columns)
        for value in self.values:
            value_columns = _collect_column_expression_columns(value)
            partners |= _generate_join_pairs(predicate_columns, value_columns)
        return partners

    def __hash__(self) -> int:
        return hash(tuple(["IN", self.column, tuple(self.values)]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.column == other.column and set(self.values) == set(other.values)

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        if len(self.values) == 1:
            value = collection_utils.simplify(self.values)
            vals = str(value) if isinstance(value, expr.SubqueryExpression) else f"({value})"
        else:
            vals = "(" + ", ".join(str(val) for val in self.values) + ")"
        return f"{self.column} IN {vals}"


class UnaryPredicate(BasePredicate):
    def __init__(self, operation: expr.SqlOperator, column: expr.SqlExpression):
        super().__init__(operation)
        self.operation = operation
        self.column = column

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

    def __hash__(self) -> int:
        return hash(tuple([self.operation, self.column]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.operation == other.operation and self.column == other.column

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
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
    @staticmethod
    def create_and(parts: Iterable[AbstractPredicate]) -> AbstractPredicate:
        parts = list(parts)
        if len(parts) == 1:
            return parts[0]
        return CompoundPredicate(expr.LogicalSqlCompoundOperators.And, list(parts))

    def __init__(self, operation: expr.LogicalSqlCompoundOperators,
                 children: AbstractPredicate | list[AbstractPredicate]):
        super().__init__()
        self.operation = operation
        self.children = collection_utils.enlist(children)
        if not self.children:
            raise ValueError("Child predicates can not be empty")

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self.children)

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(child.columns() for child in self.children)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(child.itercolumns() for child in self.children)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return collection_utils.flatten(child.iterexpressions() for child in self.children)

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        return collection_utils.set_union(child.join_partners() for child in self.children)

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return collection_utils.set_union(set(child.base_predicates()) for child in self.children)

    def __hash__(self) -> int:
        return hash(tuple([self.operation, tuple(self.children)]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.operation == other.operation and self.children == other.children

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        if self.operation == expr.LogicalSqlCompoundOperators.Not:
            return f"NOT {self.children[0]}"
        elif self.operation == expr.LogicalSqlCompoundOperators.Or:
            return "(" + " OR ".join(str(child) for child in self.children) + ")"
        elif self.operation == expr.LogicalSqlCompoundOperators.And:
            return " AND ".join(str(child) for child in self.children)
        else:
            raise ValueError(f"Unknown operation: '{self.operation}'")


def _collect_filter_predicates(predicate: AbstractPredicate) -> set[AbstractPredicate]:
    if predicate.is_base():
        return set() if predicate.is_join() else {predicate}
    else:
        if not isinstance(predicate, CompoundPredicate):
            raise ValueError(f"Predicate claims to be compound but is not instance of CompoundPredicate: {predicate}")
        compound_pred: CompoundPredicate = predicate
        if compound_pred.operation == expr.LogicalSqlCompoundOperators.Or:
            or_filter_children = [child_pred for child_pred in compound_pred.children if child_pred.is_filter()]
            if not or_filter_children:
                return set()
            or_filters = CompoundPredicate(expr.LogicalSqlCompoundOperators.Or, or_filter_children)
            return {or_filters}
        elif compound_pred.operation == expr.LogicalSqlCompoundOperators.Not:
            not_filter_children = compound_pred.children[0] if compound_pred.children[0].is_filter() else None
            if not not_filter_children:
                return set()
            return {compound_pred}
        elif compound_pred.operation == expr.LogicalSqlCompoundOperators.And:
            return collection_utils.set_union([_collect_filter_predicates(child) for child in compound_pred.children])
        else:
            raise ValueError(f"Unknown operation: '{compound_pred.operation}'")


def _collect_join_predicates(predicate: AbstractPredicate) -> set[AbstractPredicate]:
    if predicate.is_base():
        return {predicate} if predicate.is_join() else set()
    else:
        if not isinstance(predicate, CompoundPredicate):
            raise ValueError(f"Predicate claims to be compound but is not instance of CompoundPredicate: {predicate}")
        compound_pred: CompoundPredicate = predicate
        if compound_pred.operation == expr.LogicalSqlCompoundOperators.Or:
            or_join_children = [child_pred for child_pred in compound_pred.children if child_pred.is_join()]
            if not or_join_children:
                return set()
            or_joins = CompoundPredicate(expr.LogicalSqlCompoundOperators.Or, or_join_children)
            return {or_joins}
        elif compound_pred.operation == expr.LogicalSqlCompoundOperators.Not:
            not_join_children = compound_pred.children[0] if compound_pred.children[0].is_join() else None
            if not not_join_children:
                return set()
            return {compound_pred}
        elif compound_pred.operation == expr.LogicalSqlCompoundOperators.And:
            return collection_utils.set_union([_collect_join_predicates(child) for child in compound_pred.children])
        else:
            raise ValueError(f"Unknown operation: '{compound_pred.operation}'")


class QueryPredicates:

    @staticmethod
    def empty_predicate() -> QueryPredicates:
        return QueryPredicates(None)

    def __init__(self, root: AbstractPredicate | None):
        self._root = root

    def is_empty(self) -> bool:
        return self._root is None

    def root(self) -> AbstractPredicate:
        self._assert_not_empty()
        return self._root

    @functools.cache
    def filters(self) -> Collection[AbstractPredicate]:
        self._assert_not_empty()
        return _collect_filter_predicates(self._root)

    @functools.cache
    def joins(self) -> Collection[AbstractPredicate]:
        self._assert_not_empty()
        return _collect_join_predicates(self._root)

    def filters_for(self, table: base.TableReference) -> Collection[AbstractPredicate]:
        self._assert_not_empty()
        return [filter_pred for filter_pred in self.filters() if table in filter_pred.tables()]

    def joins_for(self, table: base.TableReference) -> Collection[AbstractPredicate]:
        self._assert_not_empty()
        return [join_pred for join_pred in self.joins() if table in join_pred.tables()]

    def and_(self, other_predicate: QueryPredicates | AbstractPredicate) -> QueryPredicates:
        other_predicate = other_predicate._root if isinstance(other_predicate, QueryPredicates) else other_predicate
        if self._root is None:
            return QueryPredicates(other_predicate)

        merged_predicate = CompoundPredicate(expr.LogicalSqlCompoundOperators.And, [self._root, other_predicate])
        return QueryPredicates(merged_predicate)

    def __iter__(self) -> Iterator[AbstractPredicate]:
        return (list(self.filters()) + list(self.joins())).__iter__()

    def _assert_not_empty(self) -> None:
        if self._root is None:
            raise errors.StateError("No query predicates!")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._root)
