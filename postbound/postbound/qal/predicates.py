"""Contains those parts of the `qal`, that are dedicated to representing predicates of SQL queries."""

from __future__ import annotations

import abc
import itertools
from typing import Iterable

from postbound.qal import base, expressions as expr
from postbound.util import errors, dicts as dict_utils, collections as collection_utils

_ReflexiveOps = ["=", "!=", "<>"]


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
        """Checks, whether this predicate describes a join between two tables."""
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

    def tables(self) -> set[base.ColumnReference]:
        """Provides all tables that are accessed by this predicate."""
        return {attribute.table for attribute in self.columns()}

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
    def __init__(self, mosp_data: dict) -> None:
        super().__init__()

        self._operation = dict_utils.key(mosp_data)

    def is_compound(self) -> bool:
        return False

    def base_predicates(self) -> Iterable[AbstractPredicate]:
        return [self]


_MospOperatorsSQL = {"eq": "=", "neq": "<>",
                     "lt": "<", "le": "<=", "lte": "<=",
                     "gt": ">", "ge": ">=", "gte": ">=",
                     "like": "LIKE", "not_like": "NOT LIKE",
                     "ilike": "ILIKE", "not_ilike": "NOT ILIKE",
                     "in": "IN", "between": "BETWEEN",
                     "and": "AND", "or": "OR", "not": "NOT",
                     "add": "+", "sub": "-", "neg": "-", "mul": "*", "div": "/", "mod": "%",
                     "exists": "EXISTS", "missing": "MISSING"}


class BinaryPredicate(BasePredicate):
    def __init__(self, operation: str, first_argument: expr.SqlExpression,
                 second_argument: expr.SqlExpression,
                 mosp_data: dict) -> None:
        super().__init__(mosp_data)
        self.operation = operation
        self.first_argument = first_argument
        self.second_argument = second_argument

    def is_join(self) -> bool:
        first_tables = self.first_argument.tables()
        second_tables = self.second_argument.tables()
        return len(first_tables) > 1 or len(second_tables) > 1 or len(first_tables ^ second_tables) > 0

    def columns(self) -> set[base.ColumnReference]:
        return self.first_argument.columns() | self.second_argument.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return list(self.first_argument.itercolumns()) + list(self.second_argument.itercolumns())

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        self._assert_join_predicate()
        partners = []
        first_columns = self.first_argument.columns()
        second_columns = self.second_argument.columns()

        partners.extend(_normalize_join_pair(join) for join in collection_utils.pairs(first_columns))
        partners.extend(_normalize_join_pair(join) for join in collection_utils.pairs(second_columns))

        for first_col, second_col in itertools.product(first_columns, second_columns):
            if first_col.table == second_col.table:
                continue
            partners.append(_normalize_join_pair((first_col, second_col)))

        return set(partners)

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
        operation_str = _MospOperatorsSQL.get(self.operation, self.operation)
        return f"{self.first_argument} {operation_str} {self.second_argument}"


class BetweenPredicate(BasePredicate):
    def __init__(self, column: expr.SqlExpression, interval: tuple[expr.SqlExpression, expr.SqlExpression],
                 mosp_data: dict) -> None:
        super().__init__(mosp_data)
        self.column = column
        self.interval = interval
        self.interval_start, self.interval_end = self.interval

    def is_join(self) -> bool:
        column_tables = self.column.tables()
        interval_start_tables = self.interval_start.tables()
        interval_end_tables = self.interval_end.tables()
        if any(len(tabs) > 1 for tabs in [column_tables, interval_start_tables, interval_end_tables]):
            return True
        return len(column_tables ^ interval_start_tables ^ interval_end_tables) > 0

    def columns(self) -> set[base.ColumnReference]:
        return self.column.columns() | self.interval_start.columns() | self.interval_end.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return (list(self.column.itercolumns())
                + list(self.interval_start.itercolumns())
                + list(self.interval_end.itercolumns()))

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        self._assert_join_predicate()
        partners = []
        predicate_columns = self.column.columns()
        start_columns = self.interval_start.columns()
        end_columns = self.interval_end.columns()

        partners.extend(_normalize_join_pair(join) for join in collection_utils.pairs(predicate_columns))
        partners.extend(_normalize_join_pair(join) for join in collection_utils.pairs(start_columns))
        partners.extend(_normalize_join_pair(join) for join in collection_utils.pairs(end_columns))

        for first_col, second_col in itertools.product(predicate_columns, start_columns):
            if first_col.table == second_col.table:
                continue
            partners.append(_normalize_join_pair((first_col, second_col)))
        for first_col, second_col in itertools.product(predicate_columns, end_columns):
            if first_col.table == second_col.table:
                continue
            partners.append(_normalize_join_pair((first_col, second_col)))

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
    def __init__(self, column: expr.SqlExpression, values: list[expr.SqlExpression], mosp_data: dict) -> None:
        super().__init__(mosp_data)
        self.column = column
        self.values = values

    def is_join(self) -> bool:
        column_tables = self.column.tables()
        value_tables = [val.tables() for val in self.values]
        if any(len(tabs) > 1 for tabs in [column_tables] + value_tables):
            return True
        return len(column_tables ^ collection_utils.set_union(value_tables)) > 0

    def columns(self) -> set[base.ColumnReference]:
        all_columns = self.column.columns()
        for val in self.values:
            all_columns |= val.columns()
        return all_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return list(self.column.itercolumns()) + collection_utils.flatten(val.itercolumns() for val in self.values)

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        self._assert_join_predicate()
        partners = []
        predicate_columns = self.column.columns()
        value_columns = [val.columns() for val in self.values]

        partners.extend(_normalize_join_pair(join) for join in collection_utils.pairs(predicate_columns))
        for cols in value_columns:
            partners.extend(_normalize_join_pair(join) for join in collection_utils.pairs(cols))

            for first_col, second_col in itertools.product(predicate_columns, cols):
                if first_col.table == second_col.table:
                    continue
                partners.append(_normalize_join_pair((first_col, second_col)))

        return set(partners)

    def __hash__(self) -> int:
        return hash(tuple(["IN", self.column, tuple(self.values)]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.column == other.column and set(self.values) == set(other.values)

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        vals = ", ".join(str(val) for val in self.values)
        return f"{self.column} IN ({vals})"


class UnaryPredicate(BasePredicate):
    def __init__(self, operation: str, column: expr.SqlExpression, mosp_data: dict):
        super().__init__(mosp_data)
        self.operation = operation
        self.column = column

    def is_join(self) -> bool:
        return len(self.column.tables()) > 1

    def columns(self) -> set[base.ColumnReference]:
        return self.column.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.column.itercolumns()

    def join_partners(self) -> set[tuple[base.ColumnReference, base.ColumnReference]]:
        return set(collection_utils.pairs(self.column.columns()))

    def __hash__(self) -> int:
        return hash(tuple([self.operation, self.column]))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.operation == other.operation and self.column == other.column

    def __repr__(self) -> str:
        return super().__repr__()

    def __str__(self) -> str:
        operator_str = _MospOperatorsSQL.get(self.operation, self.operation)
        if isinstance(self.column, expr.SubqueryExpression):
            return f"{operator_str} {self.column}"

        if self.operation == "exists":
            return f"{self.column} IS NOT NULL"
        elif self.operation == "missing":
            return f"{self.column} IS NULL"
        return f"{operator_str}{self.column}"


class CompoundPredicate(AbstractPredicate):
    def __init__(self, operation: str, children: AbstractPredicate | list[AbstractPredicate]):
        super().__init__()
        self.operation = operation
        self.children = collection_utils.enlist(children)

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self.children)

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(child.columns() for child in self.children)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(child.itercolumns() for child in self.children)

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
        if self.operation == "not":
            return f"NOT {self.children[0]}"
        elif self.operation == "or":
            return "(" + " OR ".join(str(child) for child in self.children) + ")"
        elif self.operation == "and":
            return " AND ".join(str(child) for child in self.children)
        else:
            raise ValueError(f"Unknown operation: '{self.operation}'")


def _collect_filter_predicates(predicate: AbstractPredicate) -> set[AbstractPredicate]:
    if predicate.is_base():
        if predicate.is_join():
            return set()
        else:
            return {predicate}
    else:
        if not isinstance(predicate, CompoundPredicate):
            raise ValueError(f"Predicate claims to be compound but is not instance of CompoundPredicate: {predicate}")
        compound_pred: CompoundPredicate = predicate
        if compound_pred.operation == "or":
            or_filter_children = [child_pred for child_pred in compound_pred.children if child_pred.is_filter()]
            if not or_filter_children:
                return set()
            or_filters = CompoundPredicate("or", or_filter_children)
            return {or_filters}
        elif compound_pred.operation == "not":
            not_filter_children = compound_pred.children[0] if compound_pred.children[0].is_filter() else None
            if not not_filter_children:
                return set()
            return {compound_pred}
        elif compound_pred.operation == "and":
            return collection_utils.set_union([_collect_filter_predicates(child) for child in compound_pred.children])
        else:
            raise ValueError(f"Unknown operation: '{compound_pred.operation}'")


def _collect_join_predicates(predicate: AbstractPredicate) -> set[AbstractPredicate]:
    if predicate.is_base():
        if predicate.is_join():
            return {predicate}
        else:
            return set()
    else:
        if not isinstance(predicate, CompoundPredicate):
            raise ValueError(f"Predicate claims to be compound but is not instance of CompoundPredicate: {predicate}")
        compound_pred: CompoundPredicate = predicate
        if compound_pred.operation == "or":
            or_join_children = [child_pred for child_pred in compound_pred.children if child_pred.is_join()]
            if not or_join_children:
                return set()
            or_joins = CompoundPredicate("or", or_join_children)
            return {or_joins}
        elif compound_pred.operation == "not":
            not_join_children = compound_pred.children[0] if compound_pred.children[0].is_join() else None
            if not not_join_children:
                return set()
            return {compound_pred}
        elif compound_pred.operation == "and":
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

    def filters(self) -> Iterable[AbstractPredicate]:
        self._assert_not_empty()
        return _collect_filter_predicates(self._root)

    def joins(self) -> Iterable[AbstractPredicate]:
        self._assert_not_empty()
        return _collect_join_predicates(self._root)

    def and_(self, other_predicate: QueryPredicates | AbstractPredicate) -> QueryPredicates:
        other_predicate = other_predicate._root if isinstance(other_predicate, QueryPredicates) else other_predicate
        if self._root is None:
            return QueryPredicates(other_predicate)

        merged_predicate = CompoundPredicate("and", [self._root, other_predicate])
        return QueryPredicates(merged_predicate)

    def _assert_not_empty(self) -> None:
        if self._root is None:
            raise errors.StateError("No query predicates!")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self._root)
