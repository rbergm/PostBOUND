"""Pre-checks are used to determine whether an input query can be optimized or if it contains unsupported features."""
from __future__ import annotations

import abc
from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx

from postbound.db import db
from postbound.qal import qal, expressions, predicates
from postbound.util import collections as collection_utils, errors

FAILURE_EXPLICIT_FROM_CLAUSE = "EXPLICIT_FROM_CLAUSE"
FAILURE_NON_EQUI_JOIN = "NON_EQUI_JOIN"
FAILURE_NON_CONJUNCTIVE_JOIN = "NON_CONJUNCTIVE_JOIN"
FAILURE_DEPENDENT_SUBQUERY = "DEPENDENT_SUBQUERY"
FAILURE_HAS_CROSS_PRODUCT = "CROSS_PRODUCT"
FAILURE_VIRTUAL_TABLES = "VIRTUAL_TABLES"
FAILURE_JOIN_PREDICATE = "BAD_JOIN_PREDICATE"


@dataclass
class PreCheckResult:
    """Wrapper for a validation result.

    `passed` indicates whether the query can be optimized by the specific strategy. If the query is not supported,
    `failure_reason` provides descriptions of what went wrong.
    """
    passed: bool = True
    failure_reason: str | list[str] = ""

    @staticmethod
    def with_all_passed() -> PreCheckResult:
        return PreCheckResult()

    def ensure_all_passed(self, context: qal.SqlQuery | db.Database | None = None) -> None:
        if self.passed:
            return
        if context is None:
            raise errors.StateError(f"Pre check failed {self._generate_failure_str()}")
        elif isinstance(context, qal.SqlQuery):
            raise UnsupportedQueryError(context, self.failure_reason)
        elif isinstance(context, db.Database):
            raise UnsupportedSystemError(context, self.failure_reason)

    def _generate_failure_str(self) -> str:
        if not self.failure_reason:
            return ""
        elif isinstance(self.failure_reason, str):
            inner_contents = self.failure_reason
        elif isinstance(self.failure_reason, Iterable):
            inner_contents = " | ".join(reason for reason in self.failure_reason)
        else:
            raise ValueError("Unexpected failure reason type: " + str(self.failure_reason))
        return f"[{inner_contents}]"


class OptimizationPreCheck(abc.ABC):
    """Basic interface for a pre-check.

    Most importantly, the `check_supported_query` method validates an input query.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        """Performs the actual validation to check, whether the query contains unsupported features."""
        raise NotImplementedError

    def check_supported_database_system(self, database_instance: db.Database) -> PreCheckResult:
        return PreCheckResult.with_all_passed()

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides the description for this check. This should include the query features that are validated."""
        raise NotImplementedError

    def __contains__(self, item: object) -> bool:
        return item == self

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.name == other.name

    def __repr__(self) -> str:
        return f"OptimizationPreCheck [{self.name}]"

    def __str__(self) -> str:
        return self.name


class CompoundCheck(OptimizationPreCheck):
    """A compound check combines an arbitrary number of base checks and asserts that all of them are satisfied."""

    def __init__(self, checks: Iterable[OptimizationPreCheck]) -> None:
        super().__init__("Compound check")
        checks = collection_utils.flatten([check.checks if isinstance(check, CompoundCheck) else [check]
                                           for check in checks if not isinstance(check, EmptyPreCheck)])
        self.checks = [check for check in checks if not isinstance(check, EmptyPreCheck)]

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        check_results = [check.check_supported_query(query) for check in self.checks]
        aggregated_passed = all(check_result.passed for check_result in check_results)
        aggregated_failures = (collection_utils.flatten(check_result.failure_reason for check_result in check_results)
                               if not aggregated_passed else [])
        return PreCheckResult(aggregated_passed, aggregated_failures)

    def describe(self) -> dict:
        return {"multiple_checks": [check.describe() for check in self.checks]}

    def __contains__(self, item: object) -> bool:
        return super().__contains__(item) or any(item in child_check for child_check in self.checks)

    def __hash__(self) -> int:
        return hash(tuple(self.checks))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.checks == other.checks

    def __str__(self) -> str:
        child_checks_str = "|".join(str(child_check) for child_check in self.checks)
        return f"CompoundCheck [{child_checks_str}]"


def merge_checks(checks: OptimizationPreCheck | Iterable[OptimizationPreCheck], *more_checks) -> OptimizationPreCheck:
    """Combines all of the supplied checks into one compound check. Duplicate checks are eliminated as far as possible.

    If there is only a single (unique) check, this is returned directly.
    """
    if not checks:
        return EmptyPreCheck()
    all_checks = ({checks} if isinstance(checks, OptimizationPreCheck) else set(checks)) | set(more_checks)
    compound_checks = [check for check in all_checks if isinstance(check, CompoundCheck)]
    atomic_checks = {check for check in all_checks if not isinstance(check, CompoundCheck)}
    compound_check_children = collection_utils.set_union(set(check.checks) for check in compound_checks)
    merged_checks = atomic_checks | compound_check_children
    merged_checks = {check for check in merged_checks if not isinstance(check, EmptyPreCheck)}
    if not merged_checks:
        return EmptyPreCheck()
    return CompoundCheck(merged_checks) if len(merged_checks) > 1 else collection_utils.simplify(merged_checks)


class EmptyPreCheck(OptimizationPreCheck):
    """Dummy check that does not actually validate anything."""

    def __init__(self) -> None:
        super().__init__("empty")

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        return PreCheckResult.with_all_passed()

    def describe(self) -> dict:
        return {"name": "no_check"}


class CrossProductPreCheck(OptimizationPreCheck):
    def __init__(self) -> None:
        super().__init__("no-cross-products")

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        no_cross_products = nx.is_connected(query.predicates().join_graph())
        failure_reason = "" if no_cross_products else FAILURE_HAS_CROSS_PRODUCT
        return PreCheckResult(no_cross_products, failure_reason)

    def describe(self) -> dict:
        return {"name": "no_cross_products"}


class VirtualTablesPreCheck(OptimizationPreCheck):
    def __init__(self) -> None:
        super().__init__("no-virtual-tables")

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        no_virtual_tables = all(not table.virtual for table in query.tables())
        failure_reason = "" if no_virtual_tables else FAILURE_VIRTUAL_TABLES
        return PreCheckResult(no_virtual_tables, failure_reason)

    def describe(self) -> dict:
        return {"name": "no_virtual_tables"}


class EquiJoinPreCheck(OptimizationPreCheck):
    def __init__(self, *, allow_conjunctions: bool = False, allow_nesting: bool = False) -> None:
        super().__init__("equi-joins-only")
        self._allow_conjunctions = allow_conjunctions
        self._allow_nesting = allow_nesting

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        join_predicates = query.predicates().joins()
        all_passed = all(self._perform_predicate_check(join_pred) for join_pred in join_predicates)
        failure_reason = "" if all_passed else FAILURE_JOIN_PREDICATE
        return PreCheckResult(all_passed, failure_reason)

    def describe(self) -> dict:
        return {
            "name": "equi_joins_only",
            "allow_conjunctions": self._allow_conjunctions,
            "allow_nesting": self._allow_nesting
        }

    def _perform_predicate_check(self, predicate: predicates.AbstractPredicate) -> bool:
        if isinstance(predicate, predicates.BasePredicate):
            return self._perform_base_predicate_check(predicate)
        elif isinstance(predicate, predicates.CompoundPredicate):
            return self._perform_compound_predicate_check(predicate)
        else:
            return False

    def _perform_base_predicate_check(self, predicate: predicates.BasePredicate) -> bool:
        if not isinstance(predicate, predicates.BinaryPredicate) or len(predicate.columns()) != 2:
            return False
        if predicate.operation != expressions.LogicalSqlOperators.Equal:
            return False

        if self._allow_nesting:
            return True
        first_is_col = isinstance(predicate.first_argument, expressions.ColumnExpression)
        second_is_col = isinstance(predicate.second_argument, expressions.ColumnExpression)
        return first_is_col and second_is_col

    def _perform_compound_predicate_check(self, predicate: predicates.CompoundPredicate) -> bool:
        if not self._allow_conjunctions:
            return False
        elif predicate.operation != expressions.LogicalSqlCompoundOperators.And:
            return False
        return all(self._perform_predicate_check(child_pred) for child_pred in predicate.children)

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._allow_conjunctions == other._allow_conjunctions
                and self._allow_nesting == other._allow_nesting)

    def __hash__(self) -> int:
        return hash((self.name, self._allow_conjunctions, self._allow_nesting))


class UnsupportedQueryError(RuntimeError):
    """Error to indicate that the specified query cannot be optimized by the selected algorithms.

    The violated restrictions are provided in the `features` attribute.
    """

    def __init__(self, query: qal.SqlQuery, features: str | list[str] = "") -> None:
        if isinstance(features, list):
            features = ", ".join(features)
        features_str = f" [{features}]" if features else ""

        super().__init__(f"Query contains unsupported features{features_str}: {query}")
        self.query = query
        self.features = features


class UnsupportedSystemError(RuntimeError):
    """Error to indicate that the selected query plan cannot be enforced on the target system.

    For example, this error can be raised if a join algorithm was chosen that the database system does not support.
    """

    def __init__(self, db_instance: db.Database, reason: str = "") -> None:
        error_msg = f"Unsupported database system: {db_instance}"
        if reason:
            error_msg += f" ({reason})"
        super().__init__(error_msg)
        self.db_system = db_instance
        self.reason = reason
