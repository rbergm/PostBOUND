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

import math
import typing
from collections.abc import Container, Iterable
from typing import Generic, Literal, Optional, Union, TypeAlias

from ._hints import PhysicalOperatorAssignment, PlanParameterization
from .. import util
from .._core import Cardinality, ScanOperator, JoinOperator
from .._qep import JoinDirection, QueryPlan
from ..qal import parser, TableReference, SqlQuery
from ..util import jsondict, StateError


AnnotationType = typing.TypeVar("AnnotationType")

NestedTableSequence = Union[tuple["NestedTableSequence", "NestedTableSequence"], TableReference]
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
    def scan(table: TableReference, *, annotation: Optional[AnnotationType] = None) -> JoinTree[AnnotationType]:
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
             annotation: Optional[AnnotationType] = None) -> JoinTree[AnnotationType]:
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
    def outer_child(self) -> JoinTree[AnnotationType]:
        if not self._outer:
            raise StateError("This join tree does not represent an intermediate node.")
        return self._outer

    @property
    def inner_child(self) -> JoinTree[AnnotationType]:
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

    def is_base_join(self) -> bool:
        return self.is_join() and self._outer.is_scan() and self._inner.is_scan()

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

    def update_annotation(self, new_annotation: AnnotationType) -> JoinTree[AnnotationType]:
        if self.is_empty():
            raise StateError("Cannot update annotation of an empty join tree.")
        return JoinTree(table=self._table, outer=self._outer, inner=self._inner, annotation=new_annotation)

    def join_with(self, partner: JoinTree[AnnotationType] | TableReference, *, annotation: Optional[AnnotationType] = None,
                  partner_annotation: AnnotationType | None = None,
                  partner_direction: JoinDirection = "inner") -> JoinTree[AnnotationType]:
        if isinstance(partner, JoinTree) and partner.is_empty():
            return self
        if self.is_empty():
            return self._init_empty_join_tree(partner, annotation=partner_annotation)

        if isinstance(partner, JoinTree) and partner_annotation is not None:
            partner = partner.update_annotation(partner_annotation)
        elif isinstance(partner, TableReference):
            partner = JoinTree.scan(partner, annotation=partner_annotation)

        outer, inner = self, partner if partner_direction == "inner" else partner, self
        return JoinTree.join(outer, inner, annotation=annotation)

    def inspect(self) -> str:
        return _inspectify(self)

    def iternodes(self) -> Iterable[JoinTree[AnnotationType]]:
        if self.is_empty():
            return []
        if self.is_scan():
            return [self]
        return [self] + self._outer.iternodes() + self._inner.iternodes()

    def itertables(self) -> Iterable[TableReference]:
        if self.is_empty():
            return []
        if self.is_scan():
            return [self._table]
        return self._outer.itertables() + self._inner.itertables()

    def iterjoins(self) -> Iterable[JoinTree[AnnotationType]]:
        if self.is_empty() or self.is_scan():
            return []
        return self._outer.iterjoins() + self._inner.iterjoins() + [self]

    def _init_empty_join_tree(self, partner: JoinTree[AnnotationType] | TableReference, *,
                              annotation: Optional[AnnotationType] = None) -> JoinTree[AnnotationType]:
        if isinstance(partner, TableReference):
            return JoinTree.scan(partner, annotation=annotation)

        if annotation is not None:
            partner = partner.update_annotation(annotation)
        return partner

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


LogicalJoinTree: TypeAlias = JoinTree[Cardinality]


def _make_simple_plan(join_tree: JoinTree, *, scan_op: ScanOperator, join_op: JoinOperator,
                      query: Optional[SqlQuery] = None, plan_params: Optional[PlanParameterization] = None) -> QueryPlan:
    tables = frozenset(join_tree.tables())
    if plan_params and plan_params.cardinality_hints.get(tables, None):
        cardinality = plan_params.cardinality_hints[tables]
    elif isinstance(join_tree.annotation, Cardinality):
        cardinality = join_tree.annotation
    else:
        cardinality = math.nan

    if join_tree.is_join():
        operator = join_op
        outer_plan = _make_simple_plan(join_tree.outer_child, scan_op=scan_op, join_op=join_op, query=query,
                                       plan_params=plan_params)
        inner_plan = _make_simple_plan(join_tree.inner_child, scan_op=scan_op, join_op=join_op, query=query,
                                       plan_params=plan_params)
        children = (outer_plan, inner_plan)
    else:
        operator = scan_op
        children = []

    if query is None:
        return QueryPlan(operator, children=children, estimated_cardinality=cardinality)

    predicates = query.predicates()
    filter_condition = (predicates.joins_between(join_tree.outer_child.tables(), join_tree.inner_child.tables())
                        if join_tree.is_join() else predicates.filters_for(join_tree.base_table))
    return QueryPlan(operator, children=children, estimated_cardinality=cardinality, filter_condition=filter_condition)


def _make_custom_plan(join_tree: JoinTree, *, physical_ops: PhysicalOperatorAssignment,
                      query: Optional[SqlQuery] = None, plan_params: Optional[PlanParameterization] = None) -> QueryPlan:
    tables = frozenset(join_tree.tables())
    if plan_params and plan_params.cardinality_hints.get(tables, None):
        cardinality = plan_params.cardinality_hints[tables]
    elif isinstance(join_tree.annotation, Cardinality):
        cardinality = join_tree.annotation
    else:
        cardinality = math.nan

    par_workers = plan_params.parallel_worker_hints.get(tables, None) if plan_params else None

    operator = physical_ops[tables]
    if join_tree.is_join():
        outer_plan = _make_simple_plan(join_tree.outer_child, physical_ops=physical_ops, plan_params=plan_params)
        inner_plan = _make_simple_plan(join_tree.inner_child, physical_ops=physical_ops, plan_params=plan_params)
        children = (outer_plan, inner_plan)
    else:
        children = []

    if query is None:
        return QueryPlan(operator, children=children, estimated_cardinality=cardinality, parallel_workers=par_workers)

    predicates = query.predicates()
    filter_condition = (predicates.joins_between(join_tree.outer_child.tables(), join_tree.inner_child.tables())
                        if join_tree.is_join() else predicates.filters_for(join_tree.base_table))
    return QueryPlan(operator, children=children, estimated_cardinality=cardinality, filter_condition=filter_condition,
                     parallel_workers=par_workers)


def to_query_plan(join_tree: JoinTree, *, query: Optional[SqlQuery] = None,
                  physical_ops: Optional[PhysicalOperatorAssignment] = None,
                  plan_params: Optional[PlanParameterization] = None,
                  scan_op: Optional[ScanOperator] = None, join_op: Optional[JoinOperator] = None) -> QueryPlan:
    if physical_ops:
        return _make_custom_plan(join_tree, physical_ops=physical_ops, query=query, plan_params=plan_params)
    elif scan_op is not None and join_op is not None:
        return _make_simple_plan(join_tree, scan_op=scan_op, join_op=join_op, query=query, plan_params=plan_params)
    else:
        raise ValueError("Either operator assignment or default operators must be provided")


def jointree_from_plan(plan: QueryPlan, *, card_source: Literal["estimates", "actual"] = "estimates") -> LogicalJoinTree:
    card = plan.estimated_cardinality if card_source == "estimates" else plan.actual_cardinality
    if plan.is_scan():
        return JoinTree.scan(plan.base_table, annotation=card)
    elif plan.is_join():
        outer = jointree_from_plan(plan.outer_child, card_source=card_source)
        inner = jointree_from_plan(plan.inner_child, card_source=card_source)
        return JoinTree.join(outer, inner, annotation=card)
    else:
        # auxiliary node handler
        return jointree_from_plan(plan.input_node, card_source=card_source)


def jointree_from_sequence(sequence: NestedTableSequence) -> JoinTree[None]:
    if isinstance(sequence, TableReference):
        return JoinTree(table=sequence)

    outer, inner = sequence
    return JoinTree.join(jointree_from_sequence(outer), jointree_from_sequence(inner))


def read_jointree_json(json_data: dict) -> JoinTree:
    annotation = json_data.get("annotation", None)

    table_json = json_data.get("table", None)
    if table_json:
        base_table = parser.load_table_json(table_json)
        return JoinTree.scan(base_table, annotation=annotation)

    outer_child = read_jointree_json(json_data["outer"])
    inner_child = read_jointree_json(json_data["inner"])
    return JoinTree.join(outer_child, inner_child, annotation=annotation)


def _inspectify(join_tree: JoinTree[AnnotationType], *, indentation: int = 0) -> str:
    padding = " " * indentation
    prefix = "<- " if padding else ""

    if join_tree.is_scan():
        return f"{padding}{prefix}{join_tree.base_table} ({join_tree.annotation})"

    join_node = f"{padding}{prefix}⨝ ({join_tree.annotation})"
    child_inspections = [_inspectify(child, indentation=indentation + 2) for child in join_tree.children]
    return f"{join_node}\n" + "\n".join(child_inspections)


def parameters_from_plan(query_plan: QueryPlan | LogicalJoinTree, *,
                         target_cardinality: Literal["estimated", "actual"] = "estimated") -> PlanParameterization:
    params = PlanParameterization()

    if isinstance(query_plan, LogicalJoinTree):
        card = query_plan.annotation
        parallel_workers = None
    else:
        card = query_plan.estimated_cardinality if target_cardinality == "estimated" else query_plan.actual_cardinality
        parallel_workers = query_plan.params.parallel_workers

    if not math.isnan(card):
        params.add_cardinality_hint(query_plan.tables(), card)
    if parallel_workers is not None:
        params.add_parallelization_hint(query_plan.tables(), parallel_workers)

    for child in query_plan.children:
        child_params = parameters_from_plan(child, target_cardinality=target_cardinality)
        params = params.merge_with(child_params)

    return params
