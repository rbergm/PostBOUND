from __future__ import annotations

import json
import math
import typing
import warnings
from collections.abc import Container, Iterable
from typing import Generic, Literal, Optional, Union

from .. import util
from .._core import (
    Cardinality,
    IntermediateOperator,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
)
from .._qep import (
    JoinDirection,
    PlanEstimates,
    PlanMeasures,
    PlanParams,
    QueryPlan,
    SortKey,
    Subplan,
)
from ..qal import SqlQuery, TableReference, parser
from ..util import StateError, jsondict
from ._hints import (
    PhysicalOperatorAssignment,
    PlanParameterization,
    operators_from_plan,
    read_operator_json,
)

AnnotationType = typing.TypeVar("AnnotationType")
"""The concrete annotation used to augment information stored in the join tree."""

NestedTableSequence = Union[
    tuple["NestedTableSequence", "NestedTableSequence"], TableReference
]
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
    """A join tree models the sequence in which joins should be performed in a query plan.

    A join tree is a composite structure that contains base tables at its leaves and joins as inner nodes. Each node can
    optionally be annotated with arbitrary metadata (`annotation` property). While a join tree does usually not contain any
    information regarding physical operators to execute its joins or scans, we do distinguish between inner and outer relations
    at the join level.

    Each join tree instance is immutable. To expand the join tree, either use the `join_with` member method or create a new
    join tree, for example using the `join` factory method. The metadata can be updated using the `update_annotation` method.

    Regular join trees
    -------------------

    Depending on the specific node, different attributes are available. For leaf nodes, this is just the `base_table`
    property. For joins, the `outer_child` and `inner_child` properties are available. The specific node type can be checked
    using the `is_scan` and `is_join` methods respectively. Notice that these methods are "binary": ``is_join() = False``
    implies ``is_scan() = True`` and vice versa.
    No matter the specific node type, the `children` property always provides iteration support for the input nodes of the
    current node (which in case of base tables is just an empty iterable). Likewise, the `annotation` property is always
    available, but its value is entirely up to the user.

    Empty join trees
    ----------------

    An empty join tree is a special case that can be created using the `empty` factory method or by calling the constructor
    without any arguments. Empty join trees should only be used when starting the construction of a join tree and never be
    returned as a result of the optimization process. Clients are not required to check for emptiness and empty join trees
    also violate some of the invariants of proper join trees. Consider them syntactic sugar to simplify the construction, but
    only use them sparingly. If you decide to work with empty join trees, use the `is_empty` method to check for emptiness.

    Parameters
    ----------
    base_table : TableReference, optional
        The base table being scanned. Accessing this property on join nodes raises an error.
    outer_child : JoinTree[AnnotationType] | None, optional
        The left child of the join. Accessing this property on base tables raises an error.
    inner_child : JoinTree[AnnotationType] | None, optional
        The right child of the join. Accessing this property on base tables raises
    annotation : AnnotationType | None, optional
        The annotation for the node. This can be used to store arbitrary data.
    """

    # Note for maintainers: if you add new methods that return a join tree, make sure to add similar methods with the same
    # signature to the LogicalJoinTree (and a return type of LogicalJoinTree) to keep the two classes in sync.
    # Likewise, some methods deliberately have the same signatures as the QueryPlan class to allow for easy duck-typed usage.
    # These methods should also be kept in sync.

    @staticmethod
    def scan(
        table: TableReference, *, annotation: Optional[AnnotationType] = None
    ) -> JoinTree[AnnotationType]:
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
        return JoinTree(base_table=table, annotation=annotation)

    @staticmethod
    def join(
        outer: JoinTree[AnnotationType],
        inner: JoinTree[AnnotationType],
        *,
        annotation: Optional[AnnotationType] = None,
    ) -> JoinTree[AnnotationType]:
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
        return JoinTree(outer_child=outer, inner_child=inner, annotation=annotation)

    @staticmethod
    def empty() -> JoinTree[AnnotationType]:
        """Creates an empty join tree.

        Returns
        -------
        JoinTree[AnnotationType]
            The empty join tree
        """
        return JoinTree()

    def __init__(
        self,
        *,
        base_table: TableReference | None = None,
        outer_child: JoinTree[AnnotationType] | None = None,
        inner_child: JoinTree[AnnotationType] | None = None,
        annotation: AnnotationType | None = None,
    ) -> None:
        self._table = base_table
        self._outer = outer_child
        self._inner = inner_child
        self._annotation = annotation
        self._hash_val = hash((base_table, outer_child, inner_child))

    @property
    def base_table(self) -> TableReference:
        """Get the base table for join tree leaves.

        Accessing this property on a join node raises an error.
        """
        if not self._table:
            raise StateError("This join tree does not represent a base table.")
        return self._table

    @property
    def outer_child(self) -> JoinTree[AnnotationType]:
        """Get the left child of the join node.

        Accessing this property on a base table raises an error.
        """
        if not self._outer:
            raise StateError("This join tree does not represent an intermediate node.")
        return self._outer

    @property
    def inner_child(self) -> JoinTree[AnnotationType]:
        """Get the right child of the join node.

        Accessing this property on a base table raises an error.
        """
        if not self._inner:
            raise StateError("This join tree does not represent an intermediate node.")
        return self._inner

    @property
    def children(self) -> tuple[JoinTree[AnnotationType], JoinTree[AnnotationType]]:
        """Get the children of the current node.

        For base tables, this is an empty tuple. For join nodes, this is a tuple of the outer and inner child.
        """
        if self.is_empty():
            raise StateError("This join tree is empty.")
        if self.is_scan():
            return ()
        return self._outer, self._inner

    @property
    def annotation(self) -> AnnotationType:
        """Get the annotation of the current node."""
        if self.is_empty():
            raise StateError("Join tree is empty.")
        return self._annotation

    def is_empty(self) -> bool:
        """Check, whether the current join tree is an empty one."""
        return self._table is None and (self._outer is None or self._inner is None)

    def is_join(self) -> bool:
        """Check, whether the current join tree node is an intermediate."""
        return self._table is None

    def is_scan(self) -> bool:
        """Check, whether the current join tree node is a leaf node."""
        return self._table is not None

    def is_linear(self) -> bool:
        """Checks, whether the join tree encodes a linear join sequence.

        In a linear join tree each join node is always a join between a base table and another join node or another base table.
        As a special case, this implies that join trees that only constist of a single node are also considered to be linear.

        The opposite of linear join trees are bushy join trees. There also exists a `is_base_join` method to check whether a
        join node joins two base tables directly.

        See Also
        --------
        is_bushy
        """
        if self.is_empty():
            raise StateError("An empty join tree does not have a shape.")
        if self.is_scan():
            return True
        return self._outer.is_scan() or self._inner.is_scan()

    def is_bushy(self) -> bool:
        """Checks, whether the join tree encodes a bushy join sequence.

        In a bushy join tree, at least one join node is a join between two other join nodes. This implies that the join tree is
        not linear.

        See Also
        --------
        is_linear
        """
        return not self.is_linear()

    def is_base_join(self) -> bool:
        """Checks, whether the current join node joins two base tables directly."""
        return self.is_join() and self._outer.is_scan() and self._inner.is_scan()

    def tables(self) -> set[TableReference]:
        """Provides all tables that are scanned in the join tree.

        Notice that this does not consider tables that might be stored in the annotation of the join tree nodes.
        """
        if self.is_empty():
            return set()
        if self.is_scan():
            return {self._table}
        return self._outer.tables() | self._inner.tables()

    def plan_depth(self) -> int:
        """Calculates the depth of the join tree.

        The depth of a join tree is the length of the longest path from the root to a leaf node. The depth of an empty join
        is defined to be 0, while the depth of a join tree with a single node is 1.
        """
        if self.is_empty():
            return 0
        if self.is_scan():
            return 1
        return 1 + max(self._outer.plan_depth(), self._inner.plan_depth())

    def lookup(
        self, table: TableReference | Iterable[TableReference]
    ) -> Optional[JoinTree[AnnotationType]]:
        """Traverses the join tree to find a specific (intermediate) node.

        Parameters
        ----------
        table : TableReference | Iterable[TableReference]
            The tables that should be contained in the intermediate. If a single table is provided (either as-is or as a
            singleton iterable), the correponding leaf node will be returned. If multiple tables are provided, the join node
            that calculates the intermediate *exactly* is returned.

        Returns
        -------
        Optional[JoinTree[AnnotationType]]
            The join tree node that contains the specified tables. If no such node exists, *None* is returned.
        """
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

    def update_annotation(
        self, new_annotation: AnnotationType
    ) -> JoinTree[AnnotationType]:
        """Creates a new join tree with the same structure, but a different annotation.

        The original join tree is not modified.
        """
        if self.is_empty():
            raise StateError("Cannot update annotation of an empty join tree.")
        return JoinTree(
            base_table=self._table,
            outer_child=self._outer,
            inner_child=self._inner,
            annotation=new_annotation,
        )

    def join_with(
        self,
        partner: JoinTree[AnnotationType] | TableReference,
        *,
        annotation: Optional[AnnotationType] = None,
        partner_annotation: AnnotationType | None = None,
        partner_direction: JoinDirection = "inner",
    ) -> JoinTree[AnnotationType]:
        """Creates a new join tree by combining the current join tree with another one.

        Both input join trees are not modified. If one of the join trees is empty, the other one is returned as-is. As a
        special case, joining two empty join trees results once again in an empty join tree.

        Parameters
        ----------
        partner : JoinTree[AnnotationType] | TableReference
            The join tree to join with the current tree. This can also be a base table, in which case it is treated as a scan
            node of the table. The scan can be further described with the `partner_annotation` parameter.
        annotation : Optional[AnnotationType], optional
            The annotation of the new join node.
        partner_annotation : AnnotationType | None, optional
            If the join partner is given as a plain table, this annotation is used to describe the corresponding scan node.
            Otherwise it is ignored.
        partner_direction : JoinDirection, optional
            Which role the partner node should play in the new join. Defaults to "inner", which means that the current node
            becomes the outer node of the new join and the partner becomes the inner child. If set to "outer", the roles are
            reversed.

        Returns
        -------
        JoinTree[AnnotationType]
            The resulting join tree
        """
        if isinstance(partner, JoinTree) and partner.is_empty():
            return self
        if self.is_empty():
            return self._init_empty_join_tree(partner, annotation=partner_annotation)

        if isinstance(partner, JoinTree) and partner_annotation is not None:
            partner = partner.update_annotation(partner_annotation)
        elif isinstance(partner, TableReference):
            partner = JoinTree.scan(partner, annotation=partner_annotation)

        outer, inner = (
            (self, partner) if partner_direction == "inner" else (partner, self)
        )
        return JoinTree.join(outer, inner, annotation=annotation)

    def inspect(self) -> str:
        """Provides a pretty-printed an human-readable representation of the join tree."""
        return _inspectify(self)

    def iternodes(self) -> Iterable[JoinTree[AnnotationType]]:
        """Provides all nodes in the join tree, with outer nodes coming first."""
        if self.is_empty():
            return []
        if self.is_scan():
            return [self]
        return [self] + self._outer.iternodes() + self._inner.iternodes()

    def itertables(self) -> Iterable[TableReference]:
        """Provides all tables that are scanned in the join tree. Outer tables appear first."""
        if self.is_empty():
            return []
        if self.is_scan():
            return [self._table]
        return self._outer.itertables() + self._inner.itertables()

    def iterjoins(self) -> Iterable[JoinTree[AnnotationType]]:
        """Provides all join nodes in the join tree, with outer nodes coming first."""
        if self.is_empty() or self.is_scan():
            return []
        return self._outer.iterjoins() + self._inner.iterjoins() + [self]

    def _init_empty_join_tree(
        self,
        partner: JoinTree[AnnotationType] | TableReference,
        *,
        annotation: Optional[AnnotationType] = None,
    ) -> JoinTree[AnnotationType]:
        """Handler method to create a new join tree when the current tree is empty."""
        if isinstance(partner, TableReference):
            return JoinTree.scan(partner, annotation=annotation)

        if annotation is not None:
            partner = partner.update_annotation(annotation)
        return partner

    def __json__(self) -> jsondict:
        if self.is_scan():
            return {
                "type": "join_tree_generic",
                "table": self._table,
                "annotation": self._annotation,
            }
        return {
            "type": "join_tree_generic",
            "outer": self._outer,
            "inner": self._inner,
            "annotation": self._annotation,
        }

    def __contains__(self, x: object) -> bool:
        return self.lookup(x)

    def __len__(self) -> int:
        return len(self.tables())

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self._table == other._table
            and self._outer == other._outer
            and self._inner == other._inner
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self):
        if self.is_scan():
            return self._table.identifier()
        return f"({self._outer} ⋈ {self._inner})"


class LogicalJoinTree(JoinTree[Cardinality]):
    """A logical join tree is a special kind of join tree that has cardinality estimates attached to each node.

    Other than the annotation type, it behaves exactly like a regular `JoinTree`. The cardinality estimates can be directly
    accessed using the `cardinality` property.
    """

    @staticmethod
    def scan(
        table: TableReference, *, annotation: Optional[Cardinality] = None
    ) -> LogicalJoinTree:
        return LogicalJoinTree(table=table, annotation=annotation)

    @staticmethod
    def join(
        outer: LogicalJoinTree,
        inner: LogicalJoinTree,
        *,
        annotation: Optional[Cardinality] = None,
    ) -> LogicalJoinTree:
        return LogicalJoinTree(outer=outer, inner=inner, annotation=annotation)

    @staticmethod
    def empty() -> LogicalJoinTree:
        return LogicalJoinTree()

    def __init__(
        self,
        *,
        table: TableReference | None = None,
        outer: LogicalJoinTree | None = None,
        inner: LogicalJoinTree | None = None,
        annotation: Cardinality | None = None,
    ) -> None:
        super().__init__(
            base_table=table,
            outer_child=outer,
            inner_child=inner,
            annotation=annotation,
        )

    @property
    def cardinality(self) -> Cardinality:
        return self.annotation

    @property
    def outer_child(self) -> LogicalJoinTree:
        return super().outer_child

    @property
    def inner_child(self) -> LogicalJoinTree:
        return super().inner_child

    @property
    def children(self) -> tuple[LogicalJoinTree, LogicalJoinTree]:
        return super().children

    def lookup(
        self, table: TableReference | Iterable[TableReference]
    ) -> Optional[LogicalJoinTree]:
        return super().lookup(table)

    def update_annotation(self, new_annotation: Cardinality) -> LogicalJoinTree:
        return super().update_annotation(new_annotation)

    def join_with(
        self,
        partner: LogicalJoinTree | TableReference,
        *,
        annotation: Optional[Cardinality] = None,
        partner_annotation: Cardinality | None = None,
        partner_direction: JoinDirection = "inner",
    ) -> LogicalJoinTree:
        return super().join_with(
            partner,
            annotation=annotation,
            partner_annotation=partner_annotation,
            partner_direction=partner_direction,
        )

    def iternodes(self) -> Iterable[LogicalJoinTree]:
        return super().iternodes()

    def iterjoins(self) -> Iterable[LogicalJoinTree]:
        return super().iterjoins()

    def __json__(self) -> jsondict:
        if self.is_scan():
            return {
                "type": "join_tree_logical",
                "table": self._table,
                "annotation": self._annotation,
            }
        return {
            "type": "join_tree_logical",
            "outer": self._outer,
            "inner": self._inner,
            "annotation": self._annotation,
        }


def _make_simple_plan(
    join_tree: JoinTree,
    *,
    scan_op: ScanOperator,
    join_op: JoinOperator,
    query: Optional[SqlQuery] = None,
    plan_params: Optional[PlanParameterization] = None,
) -> QueryPlan:
    """Handler function to create a query plan with default operators.

    (Estimated) cardinalities can still be customized accroding to the plan parameters. However, parallel workers are ignored.
    """
    tables = frozenset(join_tree.tables())
    if plan_params and plan_params.cardinalities.get(tables, None):
        cardinality = plan_params.cardinalities[tables]
    elif isinstance(join_tree, LogicalJoinTree):
        cardinality = join_tree.annotation
    else:
        cardinality = math.nan

    if join_tree.is_join():
        operator = join_op
        outer_plan = _make_simple_plan(
            join_tree.outer_child,
            scan_op=scan_op,
            join_op=join_op,
            query=query,
            plan_params=plan_params,
        )
        inner_plan = _make_simple_plan(
            join_tree.inner_child,
            scan_op=scan_op,
            join_op=join_op,
            query=query,
            plan_params=plan_params,
        )
        children = (outer_plan, inner_plan)
    else:
        operator = scan_op
        children = []

    if query is None:
        return QueryPlan(operator, children=children, estimated_cardinality=cardinality)

    predicates = query.predicates()
    filter_condition = (
        predicates.joins_between(
            join_tree.outer_child.tables(), join_tree.inner_child.tables()
        )
        if join_tree.is_join()
        else predicates.filters_for(join_tree.base_table)
    )
    return QueryPlan(
        operator,
        children=children,
        estimated_cardinality=cardinality,
        filter_condition=filter_condition,
    )


def _make_custom_plan(
    join_tree: JoinTree,
    *,
    physical_ops: PhysicalOperatorAssignment,
    query: Optional[SqlQuery] = None,
    plan_params: Optional[PlanParameterization] = None,
    fallback_scan_op: Optional[ScanOperator] = None,
    fallback_join_op: Optional[JoinOperator] = None,
) -> QueryPlan:
    """Handler function to create a query plan with a dynamic assignment of physical operators.

    If an operator is not contained in the assignment, the fallback operators are used. If these are also not available,
    this is an error.

    In addition to the operators, the estimated cardinalities as well as the parallel workers can be customized using the plan
    parameters. As a fallback, cardinalities from the join tree annotations are used.
    """
    tables = frozenset(join_tree.tables())
    if plan_params and plan_params.cardinalities.get(tables, None):
        cardinality = plan_params.cardinalities[tables]
    elif isinstance(join_tree, LogicalJoinTree):
        cardinality = join_tree.annotation
    else:
        cardinality = math.nan

    par_workers = (
        plan_params.parallel_workers.get(tables, None) if plan_params else None
    )

    operator = physical_ops.get(tables)
    if not operator and len(tables) == 1:
        operator = fallback_scan_op
    elif not operator and len(tables) > 1:
        operator = fallback_join_op
    if not operator:
        raise ValueError("No operator assignment found for join: " + str(tables))

    if join_tree.is_join():
        outer_plan = _make_simple_plan(
            join_tree.outer_child, physical_ops=physical_ops, plan_params=plan_params
        )
        inner_plan = _make_simple_plan(
            join_tree.inner_child, physical_ops=physical_ops, plan_params=plan_params
        )
        children = (outer_plan, inner_plan)
    else:
        children = []

    if query is None:
        plan = QueryPlan(
            operator,
            children=children,
            estimated_cardinality=cardinality,
            parallel_workers=par_workers,
        )
    else:
        predicates = query.predicates()
        filter_condition = (
            predicates.joins_between(
                join_tree.outer_child.tables(), join_tree.inner_child.tables()
            )
            if join_tree.is_join()
            else predicates.filters_for(join_tree.base_table)
        )
        plan = QueryPlan(
            operator,
            children=children,
            estimated_cardinality=cardinality,
            filter_condition=filter_condition,
            parallel_workers=par_workers,
        )

    intermediate_op = physical_ops.intermediate_operators.get(frozenset(plan.tables()))
    if not intermediate_op:
        return plan
    if intermediate_op in {IntermediateOperator.Sort, IntermediateOperator.Memoize}:
        warnings.warn(
            "Ignoring intermediate operator for sort/memoize. These require additional information to be inserted."
        )
        return plan

    plan = QueryPlan(intermediate_op, children=plan, estimated_cardinality=cardinality)
    return plan


def to_query_plan(
    join_tree: JoinTree,
    *,
    query: Optional[SqlQuery] = None,
    physical_ops: Optional[PhysicalOperatorAssignment] = None,
    plan_params: Optional[PlanParameterization] = None,
    scan_op: Optional[ScanOperator] = None,
    join_op: Optional[JoinOperator] = None,
) -> QueryPlan:
    """Creates a query plan from a join tree.

    This function operates in two different modes: physical operators can either be assigned to each node of the join tree
    individually using the `physical_ops`, or the same operator can be assigned to all scans and joins using the `scan_op` and
    `join_op` parameters. If the former approach is used, fallback/default operators can be provided to compensate missing
    operators in the assignment.
    Furthermore, `plan_params` can be used to inject custom cardinality estimates and parallel workers to the nodes.

    If the supplied `join_tree` is a `LogicalJoinTree`, its cardinality estimates are used as a fallback if no estimate from
    the plan parameters is available.

    Notice that the resulting query plan does not contain any DB-specific features. For example, assigning a hash join to
    an intermediate does not also insert a hash operator, as is done by some database systems.

    Parameters
    ----------
    join_tree : JoinTree
        The join order to use for the query plan. If this is a logical join tree, the cardinality estimates can be added to the
        query plan if no more specific estimates are available through the `plan_params`.
    query : Optional[SqlQuery], optional
        The query that is computed by the query plan. If this is supplied, it is used to compute join predicates and filters
        that can be computed at the various nodes of the query plan.
    physical_ops : Optional[PhysicalOperatorAssignment], optional
        The physical operators that should be used for individual nodes of the join tree. If this is supplied, the `scan_op`
        and `join_op` parameters are used as a fallback if no assignment exists for a specific intermediate. Notice that
        parallel workers contained in the operator assignments are never used since this information should be made available
        through the `plan_params`.
    plan_params : Optional[PlanParameterization], optional
        Optional cardinality estimates and parallelization info for the nodes of the join tree. If this is not supplied,
        cardinality estimates are inferred from a logical join tree or left as NaN otherwise.
    scan_op : Optional[ScanOperator], optional
        The operator to assign to all scans in the query plan. If no `physical_ops` are given, this parameter has to be
        specified. If `physical_ops` are indeed given, this parameter is used as a fallback if no assignment exists for a
        specific scan.
    join_op : Optional[JoinOperator], optional
        The operator to assign to all joins in the query plan. If no `physical_ops` are given, this parameter has to be
        specified. If `physical_ops` are indeed given, this parameter is used as a fallback if no assignment exists for a
        specific join.

    Returns
    -------
    QueryPlan
        The resulting query plan
    """
    if physical_ops:
        return _make_custom_plan(
            join_tree,
            physical_ops=physical_ops,
            query=query,
            plan_params=plan_params,
            fallback_scan_op=scan_op,
            fallback_join_op=join_op,
        )
    elif scan_op is not None and join_op is not None:
        return _make_simple_plan(
            join_tree,
            scan_op=scan_op,
            join_op=join_op,
            query=query,
            plan_params=plan_params,
        )
    else:
        raise ValueError(
            "Either operator assignment or default operators must be provided"
        )


def read_query_plan_json(json_data: dict | str) -> QueryPlan:
    """Reads a query plan from its JSON representation.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    QueryPlan
        The corresponding query plan
    """
    from ..qal import parser  # local import to prevent circular imports

    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    node_type: str = json_data["node_type"]
    operator: PhysicalOperator = read_operator_json(json_data.get("operator"))
    children = [read_query_plan_json(child) for child in json_data.get("children", [])]

    params_json: dict = json_data.get("plan_params", {})
    base_table_json: dict | None = params_json.get("base_table")
    base_table = parser.load_table_json(base_table_json) if base_table_json else None

    predicate_json: dict | None = params_json.get("filter_predicate")
    filter_predicate = (
        parser.load_predicate_json(predicate_json) if predicate_json else None
    )

    sort_keys: list[SortKey] = []
    for sort_key_json in params_json.get("sort_keys", []):
        sort_column = [
            parser.load_expression_json(col)
            for col in sort_key_json.get("equivalence_class", [])
        ]
        ascending = sort_key_json["ascending"]
        sort_keys.append(SortKey.of(sort_column, ascending))

    index = params_json.get("index", "")
    additional_params = {
        key: value
        for key, value in params_json.items()
        if key not in {"base_table", "filter_predicate", "sort_keys", "index"}
    }

    plan_params = PlanParams(
        base_table=base_table,
        filter_predicate=filter_predicate,
        sort_keys=sort_keys,
        index=index,
        **additional_params,
    )

    estimates_json: dict = json_data.get("estimates", {})
    cardinality = estimates_json.get("cardinality", math.nan)
    cost = estimates_json.get("cost", math.nan)
    additional_estimates = {
        key: value
        for key, value in estimates_json.items()
        if key not in {"cardinality", "cost"}
    }
    estimates = PlanEstimates(
        cardinality=cardinality, cost=cost, **additional_estimates
    )

    measures_json: dict = json_data.get("measures", {})
    cardinality = measures_json.get("cardinality", math.nan)
    exec_time = measures_json.get("execution_time", math.nan)
    cache_hits = measures_json.get("cache_hits")
    cache_misses = measures_json.get("cache_misses")
    additional_measures = {
        key: value
        for key, value in measures_json.items()
        if key not in {"cardinality", "execution_time", "cache_hits", "cache_misses"}
    }
    measures = PlanMeasures(
        cardinality=cardinality,
        execution_time=exec_time,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        **additional_measures,
    )

    subplan_json: dict = json_data.get("subplan", {})
    if subplan_json:
        subplan_root = parser.parse_query(subplan_json["root"])
        subplan_target = subplan_json.get("target_name", "")
        subplan = Subplan(root=subplan_root, target_name=subplan_target)
    else:
        subplan = None

    return QueryPlan(
        node_type,
        operator=operator,
        children=children,
        plan_params=plan_params,
        estimates=estimates,
        measures=measures,
        subplan=subplan,
    )


def jointree_from_plan(
    plan: QueryPlan, *, card_source: Literal["estimates", "actual"] = "estimates"
) -> LogicalJoinTree:
    """Extracts the join tree encoded in a query plan.

    The cardinality estimates of the join tree can be inferred from either the estimated cardinalities or from the measured
    actual cardinalities of the query plan.
    """
    card = (
        plan.estimated_cardinality
        if card_source == "estimates"
        else plan.actual_cardinality
    )
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
    """Creates a raw join tree from a table sequence.

    The table sequence encodes the join structure using nested lists, see `NestedTableSequence` for details.
    """
    if isinstance(sequence, TableReference):
        return JoinTree(base_table=sequence)

    outer, inner = sequence
    return JoinTree.join(jointree_from_sequence(outer), jointree_from_sequence(inner))


def read_jointree_json(json_data: dict | str) -> JoinTree:
    """Loads a jointree from its JSON representations.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    JoinTree
        The corresponding join tree
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data

    annotation = json_data.get("annotation", None)

    table_json = json_data.get("table", None)
    if table_json:
        base_table = parser.load_table_json(table_json)
        return JoinTree.scan(base_table, annotation=annotation)

    outer_child = read_jointree_json(json_data["outer"])
    inner_child = read_jointree_json(json_data["inner"])
    return JoinTree.join(outer_child, inner_child, annotation=annotation)


def parameters_from_plan(
    query_plan: QueryPlan | LogicalJoinTree,
    *,
    target_cardinality: Literal["estimated", "actual"] = "estimated",
) -> PlanParameterization:
    """Extracts the cardinality estimates from a join tree.

    The join tree can be either a logical representation, in which case the cardinalities are extracted directly. Or, it can be
    a full query plan, in which case the cardinalities are extracted from the estimates or actual measurements. The cardinality
    source depends on the `target_cardinality` setting.
    """
    params = PlanParameterization()

    if isinstance(query_plan, LogicalJoinTree):
        card = query_plan.annotation
        parallel_workers = None
    else:
        card = (
            query_plan.estimated_cardinality
            if target_cardinality == "estimated"
            else query_plan.actual_cardinality
        )
        parallel_workers = query_plan.params.parallel_workers

    if not math.isnan(card):
        params.add_cardinality(query_plan.tables(), card)
    if parallel_workers:
        params.set_workers(query_plan.tables(), parallel_workers)

    for child in query_plan.children:
        child_params = parameters_from_plan(
            child, target_cardinality=target_cardinality
        )
        params = params.merge_with(child_params)

    return params


def explode_query_plan(
    query_plan: QueryPlan, *, card_source: Literal["estimated", "actual"] = "estimated"
) -> tuple[LogicalJoinTree, PhysicalOperatorAssignment, PlanParameterization]:
    """Extracts the join tree, physical operators, and plan parameters from a query plan.

    Parameters
    ----------
    query_plan : QueryPlan
        The query plan to extract the information from
    card_source : Literal["estimated", "actual"], optional
        Which cardinalities to use in the join tree and the plan parameters. Defaults to the estimated cardinalities.

    Returns
    -------
    tuple[LogicalJoinTree, PhysicalOperatorAssignment, PlanParameterization]
        The different components of the query plan
    """
    return (
        jointree_from_plan(query_plan, card_source=card_source),
        operators_from_plan(query_plan),
        parameters_from_plan(query_plan, target_cardinality=card_source),
    )


def _inspectify(join_tree: JoinTree[AnnotationType], *, indentation: int = 0) -> str:
    """Handler method to generate a human-readable string representation of a join tree."""
    padding = " " * indentation
    prefix = "<- " if padding else ""

    if join_tree.is_scan():
        return f"{padding}{prefix}{join_tree.base_table} ({join_tree.annotation})"

    join_node = f"{padding}{prefix}⨝ ({join_tree.annotation})"
    child_inspections = [
        _inspectify(child, indentation=indentation + 2) for child in join_tree.children
    ]
    return f"{join_node}\n" + "\n".join(child_inspections)
