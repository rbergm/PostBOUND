"""Pre-checks make sure that optimization strategies and input can be optimized as indicated.

These checks should prevent the optimization of queries that contain features that the optimization algorithm does not support,
as well as the usage of optimization algorithms that make decisions that the target database cannot enforce.

The `OptimizationPreCheck` defines the abstract interface that all checks should adhere to.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import networkx as nx

from .. import util
from .._core import PhysicalOperator
from .._hints import HintType
from .._stages import EmptyPreCheck, OptimizationPreCheck, PreCheckResult
from ..db._db import Database
from ..qal._qal import (
    AbstractPredicate,
    BasePredicate,
    BinaryPredicate,
    ColumnExpression,
    CompoundOperator,
    CompoundPredicate,
    DirectTableSource,
    ExplicitFromClause,
    From,
    ImplicitFromClause,
    ImplicitSqlQuery,
    JoinTableSource,
    LogicalOperator,
    SqlQuery,
    SubqueryTableSource,
    TableSource,
    ValuesTableSource,
)

ImplicitFromClauseFailure = "NO_IMPLICIT_FROM_CLAUSE"
EquiJoinFailure = "NON_EQUI_JOIN"
InnerJoinFailure = "NON_INNER_JOIN"
ConjunctiveJoinFailure = "NON_CONJUNCTIVE_JOIN"
SubqueryFailure = "SUBQUERY"
DependentSubqueryFailure = "DEPENDENT_SUBQUERY"
CrossProductFailure = "CROSS_PRODUCT"
VirtualTablesFailure = "VIRTUAL_TABLES"
JoinPredicateFailure = "BAD_JOIN_PREDICATE"


class CompoundCheck(OptimizationPreCheck):
    """A compound check combines an arbitrary number of base checks and asserts that all of them are satisfied.

    If multiple checks fail, the `failure_reason` of the result contains all individual failure reasons.

    Parameters
    ----------
    checks : Iterable[OptimizationPreCheck]
        The checks that must all be passed.
    """

    def __init__(self, checks: Iterable[OptimizationPreCheck]) -> None:
        super().__init__("compound-check")
        checks = util.flatten(
            [
                check.checks if isinstance(check, CompoundCheck) else [check]
                for check in checks
                if not isinstance(check, EmptyPreCheck)
            ]
        )
        self.checks = [
            check for check in checks if not isinstance(check, EmptyPreCheck)
        ]

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        check_results = [check.check_supported_query(query) for check in self.checks]
        aggregated_passed = all(check_result.passed for check_result in check_results)
        aggregated_failures = (
            util.flatten(check_result.failure_reason for check_result in check_results)
            if not aggregated_passed
            else []
        )
        return PreCheckResult(aggregated_passed, aggregated_failures)

    def describe(self) -> dict:
        return {"multiple_checks": [check.describe() for check in self.checks]}

    def __contains__(self, item: object) -> bool:
        return super().__contains__(item) or any(
            item in child_check for child_check in self.checks
        )

    def __hash__(self) -> int:
        return hash(tuple(self.checks))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.checks == other.checks

    def __str__(self) -> str:
        child_checks_str = "|".join(str(child_check) for child_check in self.checks)
        return f"CompoundCheck [{child_checks_str}]"


def merge_checks(
    checks: OptimizationPreCheck | Iterable[OptimizationPreCheck], *more_checks
) -> OptimizationPreCheck:
    """Combines all of the supplied checks into one compound check.

    This method is smarter than creating a compound check directly. It eliminates duplicate checks as far as possible and
    ignores empty checks.

    If there is only a single (unique) check, this is returned directly

    Parameters
    ----------
    checks : OptimizationPreCheck | Iterable[OptimizationPreCheck]
        The checks to combine
    *more_checks
        Additional checks that should also be included

    Returns
    -------
    OptimizationPreCheck
        A check that combines all of the given checks.
    """
    if not checks:
        return EmptyPreCheck()
    all_checks = (
        {checks} if isinstance(checks, OptimizationPreCheck) else set(checks)
    ) | set(more_checks)
    all_checks = {check for check in all_checks if check}
    compound_checks = [
        check for check in all_checks if isinstance(check, CompoundCheck)
    ]
    atomic_checks = {
        check for check in all_checks if not isinstance(check, CompoundCheck)
    }
    compound_check_children = util.set_union(
        set(check.checks) for check in compound_checks
    )
    merged_checks = atomic_checks | compound_check_children
    merged_checks = {
        check for check in merged_checks if not isinstance(check, EmptyPreCheck)
    }
    if not merged_checks:
        return EmptyPreCheck()
    return (
        CompoundCheck(merged_checks)
        if len(merged_checks) > 1
        else util.simplify(merged_checks)
    )


class ImplicitQueryPreCheck(OptimizationPreCheck):
    """Check to assert that an input query is a `ImplicitSqlQuery`."""

    def __init__(self) -> None:
        super().__init__("implicit-query")

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        passed = isinstance(query, ImplicitSqlQuery)
        failure_reason = "" if passed else ImplicitFromClauseFailure
        return PreCheckResult(passed, failure_reason)

    def describe(self) -> dict:
        return {"name": "implicit_query"}


class CrossProductPreCheck(OptimizationPreCheck):
    """Check to assert that a query does not contain any cross products."""

    def __init__(self) -> None:
        super().__init__("no-cross-products")

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        no_cross_products = nx.is_connected(query.predicates().join_graph())
        failure_reason = "" if no_cross_products else CrossProductFailure
        return PreCheckResult(no_cross_products, failure_reason)

    def describe(self) -> dict:
        return {"name": "no_cross_products"}


class VirtualTablesPreCheck(OptimizationPreCheck):
    """Check to assert that a query does not contain any virtual tables."""

    def __init__(self) -> None:
        super().__init__("no-virtual-tables")

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        no_virtual_tables = all(not table.virtual for table in query.tables())
        failure_reason = "" if no_virtual_tables else VirtualTablesFailure
        return PreCheckResult(no_virtual_tables, failure_reason)

    def describe(self) -> dict:
        return {"name": "no_virtual_tables"}


class EquiJoinPreCheck(OptimizationPreCheck):
    """Check to assert that a query only contains equi-joins.

    This does not restrict the filters in any way. The determination of joins is based on `QueryPredicates.joins`.
    """

    def __init__(
        self, *, allow_conjunctions: bool = False, allow_nesting: bool = False
    ) -> None:
        super().__init__("equi-joins-only")
        self._allow_conjunctions = allow_conjunctions
        self._allow_nesting = allow_nesting

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        join_predicates = query.predicates().joins()
        all_passed = all(
            self._perform_predicate_check(join_pred) for join_pred in join_predicates
        )
        failure_reason = "" if all_passed else EquiJoinFailure
        return PreCheckResult(all_passed, failure_reason)

    def describe(self) -> dict:
        return {
            "name": "equi_joins_only",
            "allow_conjunctions": self._allow_conjunctions,
            "allow_nesting": self._allow_nesting,
        }

    def _perform_predicate_check(self, predicate: AbstractPredicate) -> bool:
        """Handler method to dispatch to the appropriate check utility depending on the predicate type.

        Parameters
        ----------
        predicate : AbstractPredicate
            The predicate to check

        Returns
        -------
        bool
            Whether the predicate passed the check
        """
        if isinstance(predicate, BasePredicate):
            return self._perform_base_predicate_check(predicate)
        elif isinstance(predicate, CompoundPredicate):
            return self._perform_compound_predicate_check(predicate)
        else:
            return False

    def _perform_base_predicate_check(self, predicate: BasePredicate) -> bool:
        """Handler method to check a single base predicate.

        Parameters
        ----------
        predicate : BasePredicate
            The predicate to check

        Returns
        -------
        bool
            Whether the predicate passed the check
        """
        if not isinstance(predicate, BinaryPredicate) or len(predicate.columns()) != 2:
            return False
        if predicate.operation != LogicalOperator.Equal:
            return False

        if self._allow_nesting:
            return True
        first_is_col = isinstance(predicate.first_argument, ColumnExpression)
        second_is_col = isinstance(predicate.second_argument, ColumnExpression)
        return first_is_col and second_is_col

    def _perform_compound_predicate_check(self, predicate: CompoundPredicate) -> bool:
        """Handler method to check a compound predicate.

        Parameters
        ----------
        predicate : CompoundPredicate
            The predicate to check

        Returns
        -------
        bool
            Whether the predicate passed the check
        """
        if not self._allow_conjunctions:
            return False
        elif predicate.operation != CompoundOperator.And:
            return False
        return all(
            self._perform_predicate_check(child_pred)
            for child_pred in predicate.children
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._allow_conjunctions == other._allow_conjunctions
            and self._allow_nesting == other._allow_nesting
        )

    def __hash__(self) -> int:
        return hash((self.name, self._allow_conjunctions, self._allow_nesting))


class InnerJoinPreCheck(OptimizationPreCheck):
    """Check to assert that a query only contains inner joins."""

    def __init__(self) -> None:
        super().__init__("inner-joins-only")

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        if not query.from_clause:
            return PreCheckResult.with_all_passed()

        match query.from_clause:
            case ImplicitFromClause():
                return PreCheckResult.with_all_passed()
            case ExplicitFromClause(join):
                return self._check_table_source(join)
            case From(items):
                checks = [self._check_table_source(entry) for entry in items]
                return PreCheckResult.merge(checks)
            case _:
                raise ValueError(f"Unknown FROM clause type: {query.from_clause}")

    def describe(self) -> dict:
        return {"name": "inner_joins_only"}

    def _check_table_source(self, source: TableSource) -> PreCheckResult:
        """Handler method to check a single table source."""
        match source:
            case DirectTableSource() | ValuesTableSource():
                return PreCheckResult.with_all_passed()
            case SubqueryTableSource(subquery):
                return self.check_supported_query(subquery)
            case JoinTableSource(left, right, _, join_type):
                checks = (
                    [PreCheckResult.with_failure(InnerJoinFailure)]
                    if join_type != "INNER"
                    else []
                )
                checks.extend(
                    [self._check_table_source(left), self._check_table_source(right)]
                )
                return PreCheckResult.merge(checks)
            case _:
                raise ValueError(f"Unknown table source type: {source}")


class SubqueryPreCheck(OptimizationPreCheck):
    """Check to assert that a query does not contain any subqueries."""

    def __init__(self) -> None:
        super().__init__("no-subqueries")

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        return (
            PreCheckResult.with_all_passed()
            if not query.subqueries()
            else PreCheckResult.with_failure(SubqueryFailure)
        )

    def describe(self) -> dict:
        return {"name": "no_subqueries"}


class DependentSubqueryPreCheck(OptimizationPreCheck):
    """Check to assert that a query does not contain any dependent subqueries."""

    def __init__(self) -> None:
        super().__init__("no-dependent-subquery")

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        passed = not any(subquery.is_dependent() for subquery in query.subqueries())
        failure_reason = "" if passed else DependentSubqueryFailure
        return PreCheckResult(passed, failure_reason)

    def describe(self) -> dict:
        return {"name": "no_dependent_subquery"}


class SetOperationsPreCheck(OptimizationPreCheck):
    """Check to assert that a query does not contain any set operations (**UNION**, **EXCEPT**, etc.)."""

    def __init__(self) -> None:
        super().__init__("no-set-operations")

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        passed = not query.is_set_query()
        failure_reason = "" if passed else "SET_OPERATION"
        return PreCheckResult(passed, failure_reason)

    def describe(self) -> dict:
        return {"name": "no_set_operations"}


class SupportedHintCheck(OptimizationPreCheck):
    """Check to assert that a number of operators are supported by a database system.

    Parameters
    ----------
    hints : HintType | PhysicalOperator | Iterable[HintType | PhysicalOperator]
        The operators and hints that have to be supported by the database system. Can be either a single hint, or an iterable
        of hints.

    See Also
    --------
    HintService.supports_hint
    """

    def __init__(
        self, hints: HintType | PhysicalOperator | Iterable[HintType | PhysicalOperator]
    ) -> None:
        super().__init__("database-check")
        self._features = util.enlist(hints)

    def check_supported_database_system(
        self, database_instance: Database
    ) -> PreCheckResult:
        failures = [
            hint
            for hint in self._features
            if not database_instance.hinting().supports_hint(hint)
        ]
        passed = not failures
        return PreCheckResult(passed, failures)

    def describe(self) -> dict:
        return {"name": "database_operator_support", "features": self._features}


class CustomCheck(OptimizationPreCheck):
    """Check to quickly implement arbitrary one-off checks.

    The custom check somewhat clashes with directly implementing the `OptimizationPreCheck` interface. The latter is generally
    preferred since it is more readable and easier to understand. However, the custom check can be useful for checks that
    will not be used in multiple places and are not worth the effort of creating a separate class.

    Parameters
    ----------
    name : str, optional
        The name of the check. It is heavily recommended to supply a descriptive name, even though a default value exists.
    query_check : Optional[Callable[[SqlQuery], PreCheckResult]], optional
        Check to apply to each query
    db_check : Optional[Callable[[Database], PreCheckResult]], optional
        Check to apply to the database
    """

    def __init__(
        self,
        name: str = "custom-check",
        *,
        query_check: Optional[Callable[[SqlQuery], PreCheckResult]] = None,
        db_check: Optional[Callable[[Database], PreCheckResult]] = None,
    ) -> None:
        super().__init__(name)
        self._query_check = query_check
        self._db_check = db_check

    def check_supported_query(self, query: SqlQuery) -> PreCheckResult:
        if self._query_check is None:
            return PreCheckResult.with_all_passed()
        return self._query_check(query)

    def check_supported_database_system(
        self, database_instance: Database
    ) -> PreCheckResult:
        if self._db_check is None:
            return PreCheckResult.with_all_passed()
        return self._db_check(database_instance)
