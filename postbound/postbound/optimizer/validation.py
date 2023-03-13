""""""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable

from postbound.db.systems import systems
from postbound.qal import qal, predicates, expressions as expr
from postbound.util import collections as collection_utils


@dataclass
class PreCheckResult:
    passed: bool
    failure_reason: str | list[str]


class OptimizationPreCheck(abc.ABC):

    @abc.abstractmethod
    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        raise NotImplementedError


class CompoundCheck(OptimizationPreCheck):
    def __init__(self, checks: Iterable[OptimizationPreCheck]) -> None:
        super().__init__()
        self.checks = checks

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        check_results = [check.check_supported_query(query) for check in self.checks]
        aggregated_passed = all(check_result.passed for check_result in check_results)
        aggregated_failures = (collection_utils.flatten(check_result.failure_reason for check_result in check_results)
                               if not aggregated_passed else [])
        return PreCheckResult(aggregated_passed, aggregated_failures)

    def describe(self) -> dict:
        return {"multiple_checks": [check.describe() for check in self.checks]}


def merge_checks(checks: OptimizationPreCheck | Iterable[OptimizationPreCheck], *more_checks) -> OptimizationPreCheck:
    checks = collection_utils.enlist(checks)
    all_checks = checks + list(more_checks)
    return CompoundCheck(all_checks)


def _detect_dependent_subqueries(expression: expr.SqlExpression) -> bool:
    if isinstance(expression, expr.SubqueryExpression):
        return expression.query.is_dependent()

    return any(_detect_dependent_subqueries(child_expr) for child_expr in expression.iterchildren())


class UESOptimizationPreCheck(OptimizationPreCheck):

    def check_supported_query(self, query: qal.SqlQuery) -> PreCheckResult:
        failures = set()
        if query.is_explicit():
            failures.add("IMPL_QUERY")

        if not query.predicates():
            return PreCheckResult(not failures, list(failures))

        for join_predicate in query.predicates().joins():
            if not join_predicate.is_base():
                failures.add("COMP_JOIN_PRED")

            if not isinstance(join_predicate, predicates.BasePredicate):
                failures.add("EQUI_JOIN")

            if join_predicate.operation != expr.LogicalSqlOperators.Equal:
                failures.add("EQUI_JOIN")

        for predicate in query.predicates():
            for base_predicate in predicate.base_predicates():
                if any(_detect_dependent_subqueries(expression) for expression in base_predicate.iterexpressions()):
                    failures.add("DEPEND_SUBQUERY")
                    break
            if "DEPEND_SUBQUERY" in failures:
                break

        return PreCheckResult(not failures, list(failures))

    def describe(self) -> dict:
        return {"name": "ues"}


class EmptyPreCheck(OptimizationPreCheck):
    def check_supported_query(self, query: qal.SqlQuery) -> tuple[bool, str | list[str]]:
        return True, ""

    def describe(self) -> dict:
        return {"name": "no_check"}


class UnsupportedQueryError(RuntimeError):
    def __init__(self, query: qal.SqlQuery, features: str | list[str] = "") -> None:
        if isinstance(features, list):
            features = ", ".join(features)
        features_str = f" [{features}]" if features else ""

        super().__init__(f"Query contains unsupported features{features_str}: {query}")
        self.query = query
        self.features = features


class UnsupportedSystemError(RuntimeError):
    def __init__(self, db_system: systems.DatabaseSystem, reason: str = "") -> None:
        error_msg = f"Unsupported database system: {db_system}"
        if reason:
            error_msg += f" ({reason})"
        super().__init__(error_msg)
        self.db_system = db_system
        self.reason = reason
