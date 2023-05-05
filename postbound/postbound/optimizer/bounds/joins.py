"""Basic interface to determine cardinality estimates or upper bounds for (intermediate) joins."""
from __future__ import annotations

import abc
from typing import Optional

from postbound.qal import qal, predicates
from postbound.optimizer import data, validation


class JoinBoundCardinalityEstimator(abc.ABC):
    """The join cardinality estimator calculates cardinality estimates for arbitrary n-ary joins.

    This can be considered a meta-strategy that is used by the actual join enumerator or operator selector.

    Implementations could for example use direct computation based on advanced statistics, sampling strategies or
    machine learning-based approaches.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def setup_for_query(self, query: qal.SqlQuery) -> None:
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
