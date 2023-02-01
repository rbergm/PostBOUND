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
    def __init__(self, predicate: AbstractPredicate | None = None):
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


class NoFilterPredicateError(errors.StateError):
    def __init__(self, predicate: AbstractPredicate | None = None):
        super().__init__(f"For predicate {predicate}" if predicate else "")
        self.predicate = predicate


class AbstractPredicate(abc.ABC):
    def __init__(self, mosp_data: dict) -> None:
        self.mosp_data = mosp_data

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
    def base_predicates(self) -> Iterable["AbstractPredicate"]:
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
    def __eq__(self, __o: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class BasePredicate(AbstractPredicate, abc.ABC):
    def __init__(self, mosp_data: dict) -> None:
        super().__init__(mosp_data)

        self._operation = dict_utils.key(mosp_data)

    def is_compound(self) -> bool:
        return False

    def base_predicates(self) -> Iterable["AbstractPredicate"]:
        return [self]


_MospOperatorsSQL = {"eq": "==", "neq": "<>",
                     "lt": "<", "le": "<=",
                     "gt": ">", "ge": ">=",
                     "like": "LIKE", "not_like": "NOT LIKE",
                     "ilike": "ILIKE", "not_ilike": "NOT ILIKE",
                     "in": "IN", "between": "BETWEEN",
                     "and": "AND", "or": "OR", "not": "NOT"}


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

    def __eq__(self, __o: object) -> bool:
        return (isinstance(__o, type(self))
                and self.operation == __o.operation
                and self.first_argument == __o.first_argument
                and self.second_argument == __o.second_argument)

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


class CompoundPredicate(AbstractPredicate):
    def __init__(self, operation: str, children: AbstractPredicate | list[AbstractPredicate], mosp_data: dict):
        super().__init__(mosp_data)
        self.operation = operation
        self.children = children


class QueryPredicates:

    @staticmethod
    def empty_predicate() -> "QueryPredicates":
        return QueryPredicates(None)

    def __init__(self, root: AbstractPredicate | None):
        self._root = root

    def root(self) -> AbstractPredicate:
        self._assert_not_empty()
        return self._root

    def filters(self) -> Iterable[AbstractPredicate]:
        self._assert_not_empty()
        pass

    def joins(self) -> Iterable[AbstractPredicate]:
        self._assert_not_empty()
        pass

    def and_(self, other_predicate: QueryPredicates | AbstractPredicate):
        other_predicate = other_predicate._root if isinstance(other_predicate, QueryPredicates) else other_predicate
        if self._root is None:
            return QueryPredicates(other_predicate)

        merged_predicate = CompoundPredicate("and", [self._root, other_predicate],
                                             {"and": [self._root.mosp_data, other_predicate.mosp_data]})
        return QueryPredicates(merged_predicate)

    def _assert_not_empty(self) -> None:
        if self._root is None:
            raise errors.StateError("No query predicates!")
