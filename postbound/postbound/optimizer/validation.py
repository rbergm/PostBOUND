"""Pre-checks are used to determine whether an input query can be optimized or if it contains unsupported features."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable

from postbound.db import db
from postbound.qal import qal, predicates, expressions as expr
from postbound.util import collections as collection_utils

FAILURE_EXPLICIT_FROM_CLAUSE = "EXPLICIT_FROM_CLAUSE"
FAILURE_NON_EQUI_JOIN = "NON_EQUI_JOIN"
FAILURE_NON_CONJUNCTIVE_JOIN = "NON_CONJUNCTIVE_JOIN"
FAILURE_DEPENDENT_SUBQUERY = "DEPENDENT_SUBQUERY"


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


def _detect_dependent_subqueries(expression: expr.SqlExpression) -> bool:
    if isinstance(expression, expr.SubqueryExpression):
        return expression.query.is_dependent()

    return any(_detect_dependent_subqueries(child_expr) for child_expr in expression.iterchildren())


class UESOptimizationPreCheck(OptimizationPreCheck):
    """Asserts that the provided query can be optimized by UES.

    It may not contain dependent subqueries and all joins have to be equi-joins or conjunctions of equi-joins.
    """

    def __init__(self) -> None:
        super().__init__("UES check")

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        failures = set()
        if query.is_explicit():
            failures.add(FAILURE_EXPLICIT_FROM_CLAUSE)

        if not query.predicates():
            return PreCheckResult(not failures, list(failures))

        for join_predicate in query.predicates().joins():
            if not isinstance(join_predicate, predicates.BasePredicate):
                failures.add(FAILURE_NON_CONJUNCTIVE_JOIN)

            if join_predicate.operation != expr.LogicalSqlOperators.Equal:
                failures.add(FAILURE_NON_EQUI_JOIN)

        for predicate in query.predicates():
            for base_predicate in predicate.base_predicates():
                if any(_detect_dependent_subqueries(expression) for expression in base_predicate.iterexpressions()):
                    failures.add(FAILURE_DEPENDENT_SUBQUERY)
                    break
            if FAILURE_DEPENDENT_SUBQUERY in failures:
                break

        return PreCheckResult(not failures, list(failures))

    def describe(self) -> dict:
        return {"name": "ues", "features": [FAILURE_EXPLICIT_FROM_CLAUSE, FAILURE_DEPENDENT_SUBQUERY,
                                            FAILURE_NON_CONJUNCTIVE_JOIN, FAILURE_NON_EQUI_JOIN]}


class EmptyPreCheck(OptimizationPreCheck):
    """Dummy check that does not actually validate anything."""

    def __init__(self) -> None:
        super().__init__("empty")

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        return PreCheckResult.with_all_passed()

    def describe(self) -> dict:
        return {"name": "no_check"}


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
