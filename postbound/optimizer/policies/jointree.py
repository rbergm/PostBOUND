"""Contains interfaces to influence the structure of a join tree."""
from __future__ import annotations

import abc

from postbound.qal import qal, predicates
from .. import joingraph, validation


class BranchGenerationPolicy(abc.ABC):
    """This policy influences the creation of branches in the join tree in contrast to linear join paths.

    The terminology used in this policy treats branches in the join tree and subqueries as synonymous.

    If an implementation of this policy requires additional information to work properly, this information should be supplied
    via custom setup methods.

    Parameters
    ----------
    name : str
        The name of the actual branching strategy.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.SqlQuery) -> None:
        """Enables the policy to setup of internal data structures.

        Parameters
        ----------
        query : qal.SqlQuery
            The query that should be optimized next
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: joingraph.JoinGraph) -> bool:
        """Decides whether the given join should be executed in a subquery.

        Parameters
        ----------
        join : predicates.AbstractPredicate
            The join that should be executed **within the subquery**. This is not the predicate that should be used to combine
            the results of two intermediate relations.
        join_graph : joingraph.JoinGraph
            The current optimization state, providing information about joined relations and the join types (e.g. primary
            key/foreign key or n:m joins).

        Returns
        -------
        bool
            Whether a branch should be created for the join
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a JSON-serializable representation of the selected branching strategy.

        Returns
        -------
        dict
            The representation
        """
        raise NotImplementedError

    def pre_check(self) -> validation.OptimizationPreCheck:
        """Provides requirements that an input query has to satisfy in order for the policy to work properly.

        Returns
        -------
        validation.OptimizationPreCheck
            The requirements check
        """
        return validation.EmptyPreCheck()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"SubqueryGenerationStrategy[{self.name}]"


class LinearJoinTreeGenerationPolicy(BranchGenerationPolicy):
    """Branching strategy that leaves all join paths linear, and therefore does not generate subqueries at all."""

    def __init__(self):
        super().__init__("Linear subquery policy")

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        pass

    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: joingraph.JoinGraph) -> bool:
        return False

    def describe(self) -> dict:
        return {"name": "linear"}
