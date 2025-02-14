"""Provides an implementation of join trees to model join orders that can be annotated with arbitrary data.

The join tree implementation follows a traditional composite structure. It uses `IntermediateJoinNode` to represent
inner nodes of the composite. These correspond to joins in the join tree. `BaseTableNode` represents leaf nodes in the
composite and correspond to scans of base relations. In addition to this structural information, each node can be
annotated by metadata objects (`BaseMetadata` for base table nodes and `JoinMetadata` for intermediate nodes). These
act as storage for additional information about the joins and scans. For example, they provide estimates on the number
of tuples that is emitted on each node. By creating subclasses of these metadata containers, arbitrary data can be
included in the join trees.

This strategy is applied for two pre-defined kinds of join trees: `LogicalJoinTree` is akin to join orders that only
focus on the sequence in which base tables should be combined. In contrast, `PhysicalQueryPlan` also includes the
physical operators that should be used to execute individual joins and scans. A physical query plan can capture all
optimization decisions made by PostBOUND in one single structure. However, this requires the optimization algorithms to
be capable of making decisions on a per-operator basis and in one integrated process. If that is not the case, the models
for partial decisions can be used (see the `physops` and `planparams` modules).

Join trees are designed as immutable data objects. In order to modify them, new join trees have to be created. This
aligns with the design principle of the query abstraction layer. Any forced modifications will break the entire join
tree model and lead to unpredictable behaviour.
"""
from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import math
import typing
from collections.abc import Callable, Collection, Container, Iterable, Sequence
from typing import Generic, Literal, Optional, Union

import Levenshtein

from ._hints import (
    PhysicalOperator,
    PhysicalOperatorAssignment, ScanOperatorAssignment, JoinOperatorAssignment, DirectionalJoinOperatorAssignment,
    PlanParameterization,
    read_operator_json
)
from .. import qal, util
from .._core import Cost, Cardinality, QueryExecutionPlan
from .._qep import SortKey
from ..qal import parser, TableReference, ColumnReference, SqlExpression
from ..util import jsondict, StateError
from ..util.typing import Lazy, LazyVal


VisitorResult = typing.TypeVar("VisitorResult")
"""Result type of visitor processes."""


class JoinTreeVisitor(abc.ABC, Generic[VisitorResult]):
    """Basic visitor to operator on arbitrary join trees.

    See Also
    --------
    JoinTree

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_intermediate_node(self, node: IntermediateJoinNode) -> VisitorResult:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_base_table_node(self, node: BaseTableNode) -> VisitorResult:
        raise NotImplementedError

    def visit_auxiliary_node(self, node: AuxiliaryNode) -> VisitorResult:
        return node.input_node.accept(self)


AnnotationType = typing.TypeVar("AnnotationType")

NestedTableSequence = Union[Sequence["NestedTableSequence"], TableReference]
"""Type alias for a convenient format to notate join trees.

The notation is composed of nested lists. These lists can either contain more lists, or references to base tables.
Each list correponds to a branch in the join tree and the each table reference to a leaf.

Examples
--------

The nested sequence ``[[S, T], R]`` corresponds to the following tree:

::

    ⨝
    ├── ⨝
    │   ├── S
    │   └── T
    └── R

In this example, tables are simply denoted by their full name.
"""


def parse_nested_table_sequence(sequence: list[dict | list]) -> NestedTableSequence:
    """Loads the table sequence that is encoded by JSON-representation of the base tables.

    This is the inverse operation to writing a proper nested table sequence to a JSON object.

    Parameters
    ----------
    sequence : list[dict  |  list]
        The (parsed) JSON data. Each table is represented as a dictionary/nested JSON object.

    Returns
    -------
    NestedTableSequence
        The corresponding table sequence

    Raises
    ------
    TypeError
        If the list contains something other than more lists and dictionaries.
    """
    if isinstance(sequence, list):
        return [parse_nested_table_sequence(item) for item in sequence]
    elif isinstance(sequence, dict):
        table_name, alias = sequence["full_name"], sequence.get("alias", "")
        return TableReference(table_name, alias)
    else:
        raise TypeError(f"Unknown list element: {sequence}")


class JoinTree(Container[TableReference], Generic[AnnotationType]):

    @staticmethod
    def scan(table: TableReference, *, annotation: AnnotationType) -> JoinTree[AnnotationType]:
        """Creates a new join tree with a single base table.

        Parameters
        ----------
        table : TableReference
            The base table to scan
        annotation : AnnotationType
            The annotation to attach to the base table node

        Returns
        -------
        JoinTree[AnnotationType]
            The new join tree
        """
        return JoinTree(table=table, annotation=annotation)

    @staticmethod
    def join(outer: JoinTree[AnnotationType], inner: JoinTree[AnnotationType], *,
             annotation: AnnotationType) -> JoinTree[AnnotationType]:
        """Creates a new join tree by combining two existing join trees.

        Parameters
        ----------
        outer : JoinTree[AnnotationType]
            The outer join tree
        inner : JoinTree[AnnotationType]
            The inner join tree
        annotation : AnnotationType
            The annotation to attach to the intermediate join node

        Returns
        -------
        JoinTree[AnnotationType]
            The new join tree
        """
        return JoinTree(outer=outer, inner=inner, annotation=annotation)

    @staticmethod
    def empty() -> JoinTree[AnnotationType]:
        """Creates an empty join tree.

        Returns
        -------
        JoinTree[AnnotationType]
            The empty join tree
        """
        return JoinTree()

    def __init__(self, *, table: TableReference | None = None,
                 outer: JoinTree[AnnotationType] | None, inner: JoinTree[AnnotationType] | None = None,
                 annotation: AnnotationType | None = None) -> None:
        self._table = table
        self._outer = outer
        self._inner = inner
        self._annotation = annotation
        self._hash_val = hash((table, outer, inner))

    @property
    def base_table(self) -> TableReference:
        if not self._table:
            raise StateError("This join tree does not represent a base table.")
        return self._table

    @property
    def outer(self) -> JoinTree[AnnotationType]:
        if not self._outer:
            raise StateError("This join tree does not represent an intermediate node.")
        return self._outer

    @property
    def inner(self) -> JoinTree[AnnotationType]:
        if not self._inner:
            raise StateError("This join tree does not represent an intermediate node.")
        return self._inner

    @property
    def children(self) -> tuple[JoinTree[AnnotationType], JoinTree[AnnotationType]]:
        if self.is_empty():
            raise StateError("This join tree is empty.")
        if self.is_scan():
            return ()
        return self._outer, self._inner

    @property
    def annotation(self) -> AnnotationType:
        if self.is_empty():
            raise StateError("Join tree is empty.")
        return self._annotation

    def is_empty(self) -> bool:
        return self._table is None and (self._outer is None or self._inner is None)

    def is_join(self) -> bool:
        return self._table is None

    def is_scan(self) -> bool:
        return self._table is not None

    def is_linear(self) -> bool:
        if self.is_scan():
            return True
        return self._outer.is_scan() or self._inner.is_scan()

    def is_bushy(self) -> bool:
        return not self.is_linear()

    def tables(self) -> set[TableReference]:
        if self.is_scan():
            return {self._table}
        return self._outer.tables() | self._inner.tables()

    def plan_depth(self) -> int:
        if self.is_scan():
            return 1
        return 1 + max(self._outer.plan_depth(), self._inner.plan_depth())

    def lookup(self, table: TableReference | Iterable[TableReference]) -> Optional[JoinTree[AnnotationType]]:
        needle: set[TableReference] = set(util.enlist(table))
        candidates = self.tables()

        if needle == candidates:
            return self
        if not needle.issubset(candidates):
            return None

        for child in self.children:
            result = child.lookup(needle)
            if result is not None:
                return result

        return None

    def inspect(self) -> str:
        return _inspectify(self)

    def iternodes(self) -> Iterable[JoinTree[AnnotationType]]:
        if self.is_empty():
            return []
        if self.is_scan():
            return [self]
        return [self] + self._outer.iternodes() + self._inner.iternodes()

    def __json__(self) -> jsondict:
        if self.is_scan():
            return {"table": self._table, "annotation": self._annotation}
        return {"outer": self._outer, "inner": self._inner, "annotation": self._annotation}

    def __contains__(self, x: object) -> bool:
        return self.lookup(x)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._table == other._table
                and self._outer == other._outer
                and self._inner == other._inner)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        if self.is_scan():
            return str(self._table)
        return f"({self._outer} ⋈ {self._inner})"


LogicalJoinTree = JoinTree[Cardinality]


def _inspectify(join_tree: JoinTree[AnnotationType], *, indentation: int = 0) -> str:
    padding = " " * indentation
    prefix = "<- " if padding else ""

    if join_tree.is_scan():
        return f"{padding}{prefix}{join_tree.base_table} ({join_tree.annotation})"

    join_node = f"{padding}{prefix}⨝ ({join_tree.annotation})"
    child_inspections = [_inspectify(child, indentation=indentation + 2) for child in join_tree.children]
    return f"{join_node}\n" + "\n".join(child_inspections)


def top_down_similarity(a: JoinTree | AbstractJoinTreeNode, b: JoinTree | AbstractJoinTreeNode, *,
                        symmetric: bool = False, gamma: float = 1.1) -> float:
    """Computes the similarity of two join trees using a top-down approach.

    Parameters
    ----------
    a : JoinTree | AbstractJoinTreeNode
        The first join tree
    b : JoinTree | AbstractJoinTreeNode
        The second join tree
    symmetric : bool, optional
        Whether the calculation should be symmetric. If true, the occurence of joins in different branches is not
        penalized. See Notes for details.
    gamma : float, optional
        The reinforcement factor to prioritize similarity of earlier (i.e. deeper) joins. The higher the value, the
        stronger the amplification, by default 1.1

    Returns
    -------
    float
        An artificial similarity score in [0, 1]. Higher values indicate larger similarity.

    Notes
    -----
    TODO: add discussion of the algorithm
    """
    tables_a, tables_b = a.tables(), b.tables()
    total_n_tables = len(tables_a | tables_b)
    normalization_factor = 1 / total_n_tables

    # similarity between two leaf nodes
    if len(tables_a) == 1 and len(tables_b) == 1:
        return 1 if tables_a == tables_b else 0

    a = a.root if isinstance(a, JoinTree) else a
    b = b.root if isinstance(b, JoinTree) else b

    # similarity between leaf node and intermediate node
    if len(tables_a) == 1 or len(tables_b) == 1:
        leaf_tree = a if len(tables_a) == 1 else b
        intermediate_tree = b if leaf_tree == a else a
        assert isinstance(intermediate_tree, IntermediateJoinNode)

        left_score = util.jaccard(leaf_tree.tables(), intermediate_tree.left_child.tables())
        right_score = util.jaccard(leaf_tree.tables(), intermediate_tree.right_child.tables())

        return normalization_factor * max(left_score, right_score)

    assert isinstance(a, IntermediateJoinNode) and isinstance(b, IntermediateJoinNode)

    # similarity between two intermediate nodes
    a_left, a_right = a.left_child, a.right_child
    b_left, b_right = b.left_child, b.right_child

    symmetric_score = (util.jaccard(a_left.tables(), b_left.tables())
                       + util.jaccard(a_right.tables(), b_right.tables()))
    crossover_score = (util.jaccard(a_left.tables(), b_right.tables())
                       + util.jaccard(a_right.tables(), b_left.tables())
                       if symmetric else 0)
    node_score = normalization_factor * max(symmetric_score, crossover_score)

    if symmetric and crossover_score > symmetric_score:
        child_score = (top_down_similarity(a_left, b_right, symmetric=symmetric, gamma=gamma)
                       + top_down_similarity(a_right, b_left, symmetric=symmetric, gamma=gamma))
    else:
        child_score = (top_down_similarity(a_left, b_left, symmetric=symmetric, gamma=gamma)
                       + top_down_similarity(a_right, b_right, symmetric=symmetric, gamma=gamma))

    return node_score + gamma * child_score


def bottom_up_similarity(a: JoinTree, b: JoinTree) -> float:
    """Computes the similarity of two join trees based on a bottom-up approach.

    Parameters
    ----------
    a : JoinTree
        The first join tree to compare
    b : JoinTree
        The second join tree to compare

    Returns
    -------
    float
        An artificial similarity score in [0, 1]. Higher values indicate larger similarity.

    Notes
    -----
    TODO: add discussion of the algorithm
    """
    a_subtrees = {join.tables() for join in a.join_sequence()}
    b_subtrees = {join.tables() for join in b.join_sequence()}
    return util.jaccard(a_subtrees, b_subtrees)


def linearized_levenshtein_distance(a: JoinTree, b: JoinTree) -> int:
    """Computes the levenshtein distance of the table sequences of two join trees.

    Parameters
    ----------
    a : JoinTree
        The first join tree to compare
    b : JoinTree
        The second join tree to compare

    Returns
    -------
    int
        The distance score. Higher values indicate larger distance.

    References
    ----------

    .. Levenshtein distance: https://en.wikipedia.org/wiki/Levenshtein_distance
    """
    return Levenshtein.distance(a.table_sequence(), b.table_sequence())


_DepthState = collections.namedtuple("_DepthState", ["current_level", "depths"])
"""Keeps track of the current calculated depths of different base tables."""


def _traverse_join_tree_depth(current_node: AbstractJoinTreeNode, current_depth: _DepthState) -> _DepthState:
    """Calculates a new depth state for the current join tree node based on the current depth.

    This is the handler method for `join_depth`.

    Depending on the specific node, different calculations are applied:

    - for base tables, a new entry of depth one is inserted into the depth state
    - for intermediate nodes, the children are visited to integrate their depth states. Afterwards, their depth is
      increase to incoporate the join

    Parameters
    ----------
    current_node : AbstractJoinTreeNode
        The node whose depth information should be integrated
    current_depth : _DepthState
        The current depth state

    Returns
    -------
    _DepthState
        The updated depth state

    Raises
    ------
    TypeError
        If the node is neither a base table node, nor an intermediate join node. This indicates that the class
        hierarchy of join tree nodes was expanded, and this method was not updated properly.
    """
    if isinstance(current_node, BaseTableNode):
        return _DepthState(1, current_depth.depths | {current_node.table: 1})

    if not isinstance(current_node, IntermediateJoinNode):
        raise TypeError("Unknown current node type: " + str(current_node))

    left_child, right_child = current_node.left_child, current_node.right_child
    if isinstance(left_child, BaseTableNode) and isinstance(right_child, BaseTableNode):
        return _DepthState(1, current_depth.depths | {left_child.table: 1, right_child.table: 1})
    elif isinstance(left_child, BaseTableNode):
        right_depth = _traverse_join_tree_depth(right_child, current_depth)
        updated_depth = right_depth.current_level + 1
        return _DepthState(updated_depth, right_depth.depths | {left_child.table: updated_depth})
    elif isinstance(right_child, BaseTableNode):
        left_depth = _traverse_join_tree_depth(left_child, current_depth)
        updated_depth = left_depth.current_level + 1
        return _DepthState(updated_depth, left_depth.depths | {right_child.table: updated_depth})
    else:
        left_depth = _traverse_join_tree_depth(left_child, current_depth)
        right_depth = _traverse_join_tree_depth(right_child, current_depth)
        updated_depth = max(left_depth.current_level, right_depth.current_level) + 1
        return _DepthState(updated_depth, left_depth.depths | right_depth.depths)


def join_depth(join_tree: JoinTree) -> dict[TableReference, int]:
    """Calculates for each base table in a join tree the join index when it was integrated into an intermediate result.

    For joins of two base tables, the depth value is 1. If a table is joined with the intermediate result of the base
    table join, its depth is 2. Generally speaking, the depth of each table is 1 plus the maximum depth of any table
    in the intermediate result that the new table is joined with.

    Parameters
    ----------
    join_tree : JoinTree
        The join tree for which the depths should be calculated.

    Returns
    -------
    dict[TableReference, int]
        A mapping from tables to their depth values.

    Examples
    --------
    TODO add examples
    """
    if join_tree.is_empty():
        return {}
    return _traverse_join_tree_depth(join_tree.root, _DepthState(0, {})).depths


@dataclasses.dataclass
class JointreeChangeEntry:
    """Models a single diff between two join trees.

    The compared join trees are referred two as the left tree and the right tree, respectively.

    Attributes
    ----------
    change_type : Literal["tree-structure", "join-direction", "physical-op", "card-est"]
        Describes the precise difference between the trees. *tree-structure* indicates that the two trees are fundamentally
        different. This occurs when the join orders are not the same. *join-direction* means that albeit the join orders are
        the same, the roles in a specific join are reversed: the inner relation of one tree acts as the outer relation in the
        other one and vice-versa. *physical-op* means that two structurally identical nodes (i.e. same join or base table)
        differ in the assigned physical operator. *card-est* indicates that two structurally identifcal nodes (i.e. same join
        or base table) differ in the estimated cardinality, while *cost-est* does the same, just for the estimated cost.
    left_state : frozenset[TableReference] | PhysicalOperator | float
        Depending on the `change_type` this attribute describes the left tree. For example, for different tree structures,
        these are the tables in the left subtree, for different physical operators, this is the operator assigned to the node
        in the left tree and so on. For different join directions, this is the entire join node
    right_state : frozenset[TableReference] | PhysicalOperator | float
        Equivalent attribute to `left_state`, just for the right tree.
    context : Optional[frozenset[TableReference]], optional
        For different physical operators or cardinality estimates, this describes the intermediate that is different. This
        attribute is unset by default.
    """

    change_type: Literal["tree-structure", "join-direction", "physical-op", "card-est", "cost-est"]
    left_state: frozenset[TableReference] | PhysicalOperator | float
    right_state: frozenset[TableReference] | PhysicalOperator | float
    context: Optional[frozenset[TableReference]] = None

    def inspect(self) -> str:
        """Provides a human-readable string of the diff.

        Returns
        -------
        str
            The diff
        """
        match self.change_type:
            case "tree-structure":
                left_str = [tab.identifier() for tab in self.left_state]
                right_str = [tab.identifier() for tab in self.right_state]
                return f"Different subtrees: left={left_str} right={right_str}"
            case "join-direction":
                left_str = [tab.identifier() for tab in self.left_state]
                right_str = [tab.identifier() for tab in self.right_state]
                return f"Swapped join direction: left={left_str} right={right_str}"
            case "physical-op":
                return f"Different physical operators on node {self.context}: left={self.left_state} right={self.right_state}"
            case "card-est":
                return (f"Different cardinality estimates on node {self.context}: "
                        f"left={self.left_state} right={self.right_state}")
            case "cost-est":
                return (f"Different cost estimates on node {self.context}: "
                        f"left={self.left_state} right={self.right_state}")
            case _:
                raise StateError(f"Unknown change type '{self.change_type}'")


@dataclasses.dataclass
class JointreeChangeset:
    """Captures an arbitrary amount of join tree diffs.

    Attributes
    ----------
    changes : Collection[JointreeChangeEntry]
        The diffs
    """

    changes: Collection[JointreeChangeEntry]

    def inspect(self) -> str:
        """Provides a human-readable string of the entire diff.

        The diff will typically contain newlines to separate individual entries.

        Returns
        -------
        str
            The diff
        """
        return "\n".join(entry.inspect() for entry in self.changes)


def _extract_card_from_annotation(node: AbstractJoinTreeNode | None) -> float:
    """Provides the cardinality estimate from a join tree node if there is any.

    Parameters
    ----------
    node : AbstractJoinTreeNode | None
        The node to extract from

    Returns
    -------
    Cardinality
        The node's cardinality. Can be *NaN* if either the node is undefined or does not contain an annotated cardinality.
    """
    if node is None:
        return math.nan

    annot = node.annotation
    if isinstance(annot, BaseTableMetadata):
        return annot.cardinality
    elif isinstance(annot, JoinMetadata):
        return annot.cardinality

    return math.nan


def _extract_cost_from_annotation(node: AbstractJoinTreeNode | None) -> float:
    """Provides the cost estiamte froma join tree node if there is any.

    Parameters
    ----------
    node : AbstractJoinTreeNode | None
        The node to extract from

    Returns
    -------
    float
        The node's cost. Can be *NaN* if either the node is undefined, does not support cost estimates (e.g. for logical
        nodes), or does not contain an annotated cost.
    """
    if node is None:
        return math.nan

    if isinstance(node.annotation, (PhysicalBaseTableMetadata, PhysicalJoinMetadata)):
        return node.annotation.cost
    return math.nan


def _extract_operator_from_annotation(node: AbstractJoinTreeNode | None) -> Optional[PhysicalOperator]:
    """Provides the physical operator of a join tree node if there is one.

    Parameters
    ----------
    node : AbstractJoinTreeNode | None
        The node to extract from

    Returns
    -------
    Cardinality
        The node's operator. Can be *None* if either the node is undefined or does not contain an annotated cardinality.
    """
    if node is None:
        return None

    annot = node.annotation
    if isinstance(annot, PhysicalBaseTableMetadata):
        return annot.operator
    elif isinstance(annot, PhysicalJoinMetadata):
        return annot.operator

    return None


def compare_jointrees(left: JoinTree | AbstractJoinTreeNode,
                      right: JoinTree | AbstractJoinTreeNode) -> JointreeChangeset:
    """Computes differences between two join tree instances.

    Parameters
    ----------
    left : JoinTree | AbstractJoinTreeNode
        The first join tree to compare
    right : JoinTree | AbstractJoinTreeNode
        The second join tree to compare

    Returns
    -------
    JointreeChangeset
        A diff between the two join trees
    """
    if left.tables() != right.tables():
        changeset = [JointreeChangeEntry("tree-structure", left_state=left.tables(), right_state=right.tables())]
        return JointreeChangeset(changeset)

    left: AbstractJoinTreeNode = left.root if isinstance(left, JoinTree) else left
    right: AbstractJoinTreeNode = right.root if isinstance(right, JoinTree) else right
    changes: list[JointreeChangeEntry] = []

    left_card, right_card = _extract_card_from_annotation(left), _extract_card_from_annotation(right)
    left_cost, right_cost = _extract_cost_from_annotation(left), _extract_cost_from_annotation(right)
    if left_card != right_card and not (math.isnan(left_card) and math.isnan(right_card)):
        changes.append(JointreeChangeEntry("card-est", left_state=left_card, right_state=right_card, context=left.tables()))
    if left_cost != right_cost and not (math.isnan(left_cost) and math.isnan(left_cost)):
        changes.append(JointreeChangeEntry("cost-est", left_state=left_cost, right_state=right_cost, context=left.tables()))

    left_op, right_op = _extract_operator_from_annotation(left), _extract_operator_from_annotation(right)
    if left_op != right_op:
        changes.append(JointreeChangeEntry("physical-op", left_state=left_op, right_state=right_op, context=left.tables()))

    if isinstance(left, IntermediateJoinNode):
        left_intermediate: IntermediateJoinNode = left
        # we can also assume that right is an intermediate node since we know both nodes have the same tables and the left tree
        # is an intermediate node
        right_intermediate: IntermediateJoinNode = right

        join_direction_swap = left_intermediate.left_child.tables() == right_intermediate.right_child.tables()
        if join_direction_swap:
            changes.append(JointreeChangeEntry("join-direction", left_state=left_intermediate, right_state=right_intermediate))
            changes.extend(compare_jointrees(left_intermediate.left_child, right_intermediate.right_child).changes)
            changes.extend(compare_jointrees(left_intermediate.right_child, right_intermediate.left_child).changes)
        else:
            changes.extend(compare_jointrees(left_intermediate.left_child, right_intermediate.left_child).changes)
            changes.extend(compare_jointrees(left_intermediate.right_child, right_intermediate.right_child).changes)

    return JointreeChangeset(changes)
