"""Contains those parts of the `qal`, that are dedicated to representing predicates of SQL queries."""

import abc
from typing import Set, Tuple, Iterable

from postbound.qal import base
from postbound.util import errors, dicts as dict_utils


__ReflexiveOps = ["=", "!=", "<>"]
__MospCompoundOperations = {"and", "or", "not"}


class NoJoinPredicateError(errors.StateError):
    def __init__(self, msg: str = ""):
        super().__init__(msg)


class NoFilterPredicateError(errors.StateError):
    def __init__(self, msg: str = ""):
        super().__init__(msg)


class AbstractPredicate(abc.ABC):
    def __init__(self, mosp_data: dict) -> None:
        self._mosp_data = mosp_data

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
    def collect_columns(self) -> Set[base.ColumnReference]:
        """Provides all columns that are referenced by this predicate."""
        raise NotImplementedError

    def collect_tables(self) -> Set[base.ColumnReference]:
        """Provides all tables that are accessed by this predicate."""
        return {attribute.table for attribute in self.collect_columns()}

    def contains_table(self, table: base.TableReference) -> bool:
        """Checks, whether this predicate filters or joins a column of the given table."""
        return any(table == tab for tab in self.collect_tables())

    def joins_table(self, table: base.TableReference) -> bool:
        """Checks, whether this predicate describes a join and one of the join partners is the given table."""
        return self.is_join() and self.contains_table(table)

    def columns_of(self, table: base.TableReference) -> Set[base.ColumnReference]:
        """Retrieves all columns of the given table that are referenced by this predicate."""
        return {attr for attr in self.collect_columns() if attr.table == table}

    @abc.abstractmethod
    def join_partners_of(self, table: base.TableReference) -> Set[base.ColumnReference]:
        """Retrieves all columns that are joined with the given table.

        If this predicate is not a join, an error will be raised.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def join_partners(self) -> Set[Tuple[base.ColumnReference, base.ColumnReference]]:
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
            raise NoJoinPredicateError

    def _assert_filter_predicate(self) -> None:
        """Raises a `NoFilterPredicateError` if `self` is not a filter."""
        if not self.is_filter():
            raise NoFilterPredicateError

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


class BinaryPredicate(BasePredicate):
    pass


class UnaryPredicate(BasePredicate):
    pass


class CompoundPredicate(AbstractPredicate):
    pass


class QueryPredicates:
    def get(self) -> AbstractPredicate:
        pass

    def filters(self) -> Iterable[AbstractPredicate]:
        pass

    def joins(self) -> Iterable[AbstractPredicate]:
        pass
