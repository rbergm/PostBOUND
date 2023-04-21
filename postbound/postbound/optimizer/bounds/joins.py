"""Basic interface to determine cardinality estimates or upper bounds for (intermediate) joins."""
from __future__ import annotations

import abc
import math
from typing import Optional

import numpy as np

from postbound.qal import base, qal, predicates
from postbound.optimizer import data, validation
from postbound.optimizer.bounds import stats
from postbound.util import collections as collection_utils


class JoinBoundCardinalityEstimator(abc.ABC):
    """The join cardinality estimator calculates cardinality estimates for arbitrary n-ary joins.

    This can be considered a meta-strategy that is used by the actual join enumerator or operator selector.

    Implementations could for example use direct computation based on advanced statistics, sampling strategies or
    machine learning-based approaches.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def setup_for_query(self, query: qal.SqlQuery, stats_container: stats.StatisticsContainer) -> None:
        """
        The setup functionality allows the estimator to prepare internal data structures such that the input query can
        be optimized.

        The statistics container provides access to the underlying statistical data. This should be adjusted such that
        the cardinality estimator can work with the provided statistics by the optimization algorithm that uses the
        cardinality estimates.
        """
        pass

    @abc.abstractmethod
    def estimate_for(self, join_edge: predicates.AbstractPredicate, join_graph: data.JoinGraph) -> int:
        """Calculates the cardinality estimate for the given join predicate, given the current state in the join graph.

        How the join predicate should be interpreted is completely up to the estimator implementation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the selected cardinality estimation strategy."""
        raise NotImplementedError

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        """Provides requirements that an input query has to satisfy in order for the estimator to work properly."""
        return None


class UESJoinBoundEstimator(JoinBoundCardinalityEstimator):
    """Join cardinality estimator that produces upper bounds according to the formula described in the UES paper.

    This requires maximum frequency statistics over the join columns in order to work properly.

    See Hertzschuch et al.: "Simplicity Done Right for Join Ordering", CIDR'2021 for details.
    """

    def __init__(self) -> None:
        super().__init__("UES join estimator")
        self.query: qal.ImplicitSqlQuery | None = None
        self.stats_container: stats.StatisticsContainer[stats.MaxFrequency] | None = None

    def setup_for_query(self, query: qal.SqlQuery,
                        stats_container: stats.StatisticsContainer[stats.MaxFrequency]) -> None:
        self.query = query
        self.stats_container = stats_container

    def estimate_for(self, join_edge: predicates.AbstractPredicate, join_graph: data.JoinGraph) -> int:
        current_min_bound = np.inf

        for base_predicate in join_edge.base_predicates():
            first_col, second_col = collection_utils.simplify(base_predicate.join_partners())
            if join_graph.is_pk_fk_join(first_col.table, second_col.table):
                join_bound = self._estimate_pk_fk_join(first_col, second_col)
            elif join_graph.is_pk_fk_join(second_col.table, first_col.table):
                join_bound = self._estimate_pk_fk_join(second_col, first_col)
            else:
                join_bound = self._estimate_n_m_join(first_col, second_col)

            if join_bound < current_min_bound:
                current_min_bound = join_bound

        return current_min_bound

    def describe(self) -> dict:
        return {"name": "ues"}

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        # TODO: the UES check is slightly too restrictive here.
        # It suffices to check that there are only conjunctive equi joins.
        return validation.UESOptimizationPreCheck()

    def _estimate_pk_fk_join(self, fk_column: base.ColumnReference, pk_column: base.ColumnReference) -> int:
        """Estimation formula for primary key/foreign key joins."""
        pk_cardinality = self.stats_container.base_table_estimates[pk_column.table]
        fk_frequency = self.stats_container.attribute_frequencies[fk_column]
        return math.ceil(fk_frequency * pk_cardinality)

    def _estimate_n_m_join(self, first_column: base.ColumnReference, second_column: base.ColumnReference) -> int:
        """Estimation formula for n:m joins."""
        first_bound, second_bound = self._fetch_bound(first_column), self._fetch_bound(second_column)
        first_freq = self.stats_container.attribute_frequencies[first_column]
        second_freq = self.stats_container.attribute_frequencies[second_column]

        if any(var == 0 for var in [first_bound, second_bound, first_freq, second_freq]):
            return 0

        first_distinct_vals = first_bound / first_freq
        second_distinct_vals = second_bound / second_freq

        n_m_bound = min(first_distinct_vals, second_distinct_vals) * first_freq * second_freq
        return math.ceil(n_m_bound)

    def _fetch_bound(self, column: base.ColumnReference) -> int:
        """Provides the appropriate table bound (based on upper bound or base table estimate) for the given column."""
        table = column.table
        return (self.stats_container.upper_bounds[table] if table in self.stats_container.upper_bounds
                else self.stats_container.base_table_estimates[table])
