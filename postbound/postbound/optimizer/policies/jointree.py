"""Contains interfaces to influence the structure of a join tree."""
from __future__ import annotations

import abc
from typing import Optional

from postbound.qal import qal, predicates
from postbound.optimizer import joingraph, validation


class BranchGenerationPolicy(abc.ABC):
    """This policy allows to adapt the structure of the resulting join tree during join order optimization.

    It is intended to customize when branches should be created in a join tree and to compare different such
    strategies.

    Although the terminology here is subquery-related, the policy is actually more general and concerned with branches
    in the join tree. Whether these are achieved via subqueries during query execution is merely a technical detail
    (albeit a useful analogy).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def setup_for_query(self, query: qal.SqlQuery) -> None:
        """Enables the setup of internal data structures to enable decisions about the given query."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: joingraph.JoinGraph) -> bool:
        """Decides whether the given join should be executed in a subquery.

        Notice that the supplied join predicate really is the one that should be executed in a subquery, it is not the
        predicate that should be used to merge the result of two subqueries.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def describe(self) -> dict:
        """Provides a representation of the branching strategy."""
        raise NotImplementedError

    def pre_check(self) -> Optional[validation.OptimizationPreCheck]:
        """Provides requirements that an input query has to satisfy in order for the policy to work properly."""
        return None


class LinearJoinTreeGenerationPolicy(BranchGenerationPolicy):
    """Subquery strategy that leaves all join paths linear, i.e. does not generate subqueries at all."""

    def __init__(self):
        super().__init__("Linear subquery policy")

    def setup_for_query(self, query: qal.SqlQuery) -> None:
        pass

    def generate_subquery_for(self, join: predicates.AbstractPredicate, join_graph: joingraph.JoinGraph) -> bool:
        return False

    def describe(self) -> dict:
        return {"name": "linear"}
