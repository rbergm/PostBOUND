from __future__ import annotations

import abc
import typing
from collections.abc import Container, Iterable, Sequence
from typing import Callable, Generic, Optional, Union, Iterator

import numpy as np

from postbound.qal import base, predicates
from postbound.optimizer.physops import operators as physops
from postbound.optimizer.planmeta import hints as params
from postbound.util import collections as collection_utils, errors


class BaseMetadata(abc.ABC):
    def __init__(self, upper_bound: float = np.nan) -> None:
        self._upper_bound = upper_bound

    @property
    def upper_bound(self) -> float:
        return self._upper_bound


class JoinMetadata(BaseMetadata, abc.ABC):
    def __init__(self, predicate: Optional[predicates.AbstractPredicate] = None, upper_bound: float = np.nan) -> None:
        super().__init__(upper_bound)
        self._join_predicate = predicate

    @property
    def join_predicate(self) -> Optional[predicates.AbstractPredicate]:
        return self._join_predicate

    def inspect(self) -> str:
        return f"JOIN ON {self.join_predicate}" if self.join_predicate else "CROSS JOIN"


class LogicalJoinMetadata(JoinMetadata):
    def __init__(self, predicate: Optional[predicates.AbstractPredicate] = None, upper_bound: float = np.nan) -> None:
        super().__init__(predicate, upper_bound)
        self._hash_val = hash((predicate, upper_bound))

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.join_predicate == __value.join_predicate
                and self.upper_bound == __value.upper_bound)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"predicate={self.join_predicate}, upper bound={self.upper_bound}"


class PhysicalJoinMetadata(JoinMetadata):
    def __init__(self, predicate: Optional[predicates.AbstractPredicate] = None, upper_bound: float = np.nan,
                 join_info: Optional[physops.JoinOperatorAssignment] = None) -> None:
        super().__init__(predicate, upper_bound)
        self._operator_assignment = join_info
        self._hash_val = hash((predicate, upper_bound, join_info))

    @property
    def operator(self) -> physops.JoinOperatorAssignment:
        return self._operator_assignment

    def inspect(self) -> str:
        base_inspection = super().inspect()
        return f"{base_inspection} {self.operator}" if self._operator_assignment else base_inspection

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.join_predicate == __value.join_predicate
                and self.upper_bound == __value.upper_bound
                and self.operator == __value.operator)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"predicate={self.join_predicate}, upper bound={self.upper_bound}, operator={self.operator}"


class BaseTableMetadata(BaseMetadata, abc.ABC):
    def __init__(self, filter_predicate: Optional[predicates.AbstractPredicate], upper_bound: float = np.nan) -> None:
        super().__init__(upper_bound)
        self._filter_predicate = filter_predicate

    @property
    def filter_predicate(self) -> Optional[predicates.AbstractPredicate]:
        return self._filter_predicate

    def inspect(self) -> str:
        return f"FILTER {self.filter_predicate}" if self._filter_predicate else ""


class LogicalBaseTableMetadata(BaseTableMetadata):
    def __init__(self, filter_predicate: Optional[predicates.AbstractPredicate], upper_bound: float = np.nan) -> None:
        super().__init__(filter_predicate, upper_bound)
        self._hash_val = hash((filter_predicate, upper_bound))

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.filter_predicate == __value.filter_predicate
                and self.upper_bound == __value.upper_bound)


class PhysicalBaseTableMetadata(BaseTableMetadata):
    def __init__(self, filter_predicate: Optional[predicates.AbstractPredicate], upper_bound: float = np.nan,
                 scan_info: Optional[physops.ScanOperatorAssignment] = None) -> None:
        super().__init__(filter_predicate, upper_bound)
        self._operator_assignment = scan_info
        self._hash_val = hash((filter_predicate, upper_bound, scan_info))

    @property
    def operator(self) -> Optional[physops.ScanOperatorAssignment]:
        return self._operator_assignment

    def inspect(self) -> str:
        base_inspection = super().inspect()
        return f"{base_inspection} {self.operator}" if self._operator_assignment else base_inspection

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.filter_predicate == __value.filter_predicate
                and self.upper_bound == __value.upper_bound
                and self.operator == __value.operator)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"predicate={self.filter_predicate}, upper bound={self.upper_bound}, operator={self.operator}"


JoinMetadataType = typing.TypeVar("JoinMetadataType", bound=JoinMetadata)
BaseTableMetadataType = typing.TypeVar("BaseTableMetadataType", bound=BaseTableMetadata)
NestedTableSequence = typing.NewType("NestedTableSequence", Union[Sequence["NestedTableSequence"], base.TableReference])


class AbstractJoinTreeNode(abc.ABC, Container[base.TableReference], Generic[JoinMetadataType, BaseTableMetadataType]):
    """The fundamental type to construct a join tree. This node contains the actual entries/data.

    A join tree distinguishes between two types of nodes: join nodes which act as intermediate nodes that join
    together their child nodes and base table nodes that represent a scan over a base table. These nodes act as leaves
    in the join tree.

    The `JoinTreeNode` describes the behavior that is common to both node types.
    """

    @property
    def annotation(self) -> Optional[BaseMetadata]:
        raise NotImplementedError

    @property
    def upper_bound(self) -> float:
        return self.annotation.upper_bound if self.annotation else np.nan

    @abc.abstractmethod
    def is_join_node(self) -> bool:
        """Checks, whether this node is a join node."""
        raise NotImplementedError

    def is_base_table_node(self) -> bool:
        """Checks, whether this node is a base table node."""
        return not self.is_join_node()

    @abc.abstractmethod
    def tables(self) -> frozenset[base.TableReference]:
        """Provides all tables that are present in the subtree under and including this node."""
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> frozenset[base.ColumnReference]:
        """Provides all columns that are present in the subtree under and including this node."""
        raise NotImplementedError

    @abc.abstractmethod
    def join_sequence(self) -> Sequence[IntermediateJoinNode[JoinMetadataType]]:
        """Provides all joins under and including this node in a right-deep manner.

        If this node is a base table node, the returned container will be empty.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def table_sequence(self) -> Sequence[BaseTableNode[BaseTableMetadataType]]:
        """Provides all base tables under and including this node in a right-deep manner."""
        raise NotImplementedError

    @abc.abstractmethod
    def as_list(self) -> NestedTableSequence:
        """Provides the selected join order as a nested list.

        The table of each base table node will be contained directly in the join order. Each join node will be
        represented as a list of the list-representations of its child nodes.

        For example, the join order R ⋈ (S ⋈ T) will be represented as `[R, [S, T]]`.
        """
        raise NotImplementedError

    def as_join_tree(self) -> JoinTree:
        """Creates a new join tree with this node as root and all children as sub-nodes."""
        return JoinTree(self)

    @abc.abstractmethod
    def count_cross_product_joins(self) -> int:
        """Counts the number of joins below and including this node that do not have an attached join predicate."""
        raise NotImplementedError

    @abc.abstractmethod
    def homomorphic_hash(self) -> int:
        """
        Calculates a hash value that is independent of join directions, i.e. R ⋈ S and S ⋈ R have the same hash value.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def base_table(self, traverse_right: bool = True) -> base.TableReference:
        """Provides the left-most or right-most table in the join tree."""
        raise NotImplementedError

    @abc.abstractmethod
    def inspect(self, *, indentation: int = 0) -> str:
        """Produces a human-readable structure that describes the structure of this join tree."""
        raise NotImplementedError

    @abc.abstractmethod
    def __contains__(self, item) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, __value: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class IntermediateJoinNode(AbstractJoinTreeNode, Generic[JoinMetadataType]):
    def __init__(self, left_child: AbstractJoinTreeNode, right_child: AbstractJoinTreeNode,
                 annotation: Optional[JoinMetadataType]) -> None:
        if not left_child or not right_child:
            raise ValueError("Left child and right child are required")
        self._left_child = left_child
        self._right_child = right_child
        self._annotation = annotation
        self._hash_val = hash((left_child, right_child))

    @property
    def left_child(self) -> AbstractJoinTreeNode:
        return self._left_child

    @property
    def right_child(self) -> AbstractJoinTreeNode:
        return self._right_child

    @property
    def annotation(self) -> Optional[JoinMetadataType]:
        return self._annotation

    def is_join_node(self) -> bool:
        return True

    def tables(self) -> frozenset[base.TableReference]:
        return frozenset(self._left_child.tables() | self._right_child.tables())

    def columns(self) -> frozenset[base.ColumnReference]:
        predicate_columns = (self.annotation.join_predicate.columns()
                             if self.annotation and self.annotation.join_predicate else set())
        return frozenset(self._left_child.columns() | self._right_child.columns() | predicate_columns)

    def join_sequence(self) -> Sequence[IntermediateJoinNode[JoinMetadataType]]:
        is_final_join = not self.left_child.is_join_node() and not self.right_child.is_join_node()
        if is_final_join:
            return [self]

        sequence = []
        if self.right_child.is_join_node():
            sequence.extend(self.right_child.join_sequence())
        if self.left_child.is_join_node():
            sequence.extend(self.left_child.join_sequence())
        sequence.append(self)
        return sequence

    def table_sequence(self) -> Sequence[BaseTableNode[BaseTableMetadataType]]:
        return list(self.right_child.table_sequence()) + list(self.left_child.table_sequence())

    def as_list(self) -> NestedTableSequence:
        return [self.left_child.as_list(), self.right_child.as_list()]

    def count_cross_product_joins(self) -> int:
        self_cross_product_value = 1 if not self.annotation or not self.annotation.join_predicate else 0
        return (self_cross_product_value
                + self.left_child.count_cross_product_joins()
                + self.right_child.count_cross_product_joins())

    def homomorphic_hash(self) -> int:
        left_hash = self.left_child.homomorphic_hash()
        right_hash = self.right_child.homomorphic_hash()
        return hash((left_hash, right_hash)) if left_hash < right_hash else hash((right_hash, left_hash))

    def base_table(self, traverse_right: bool = True) -> base.TableReference:
        return self.right_child.base_table(True) if traverse_right else self.left_child.base_table(False)

    def inspect(self, *, indentation: int = 0) -> str:
        padding = " " * indentation
        prefix = f"{padding}<- " if padding else ""
        own_inspection = f"{prefix}{self.annotation.inspect()}" if self.annotation else f"{prefix}CROSS JOIN"
        left_inspection = self.left_child.inspect(indentation=indentation + 2)
        right_inspection = self.right_child.inspect(indentation=indentation + 2)
        return "\n".join([own_inspection, left_inspection, right_inspection])

    def __contains__(self, item) -> bool:
        if self.annotation and self.annotation.join_predicate == item:
            return True
        return item in self.left_child or item in self.right_child

    def __len__(self) -> int:
        return len(self.left_child) + len(self.right_child)

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.left_child == __value.left_child
                and self.right_child == __value.right_child)

    def __str__(self) -> str:
        left_str = f"({self.left_child})" if self.left_child.is_join_node() else str(self.left_child)
        right_str = str(self.right_child)
        return f"{right_str} ⋈ {left_str}"


class BaseTableNode(AbstractJoinTreeNode, Generic[BaseTableMetadataType]):
    def __init__(self, table: base.TableReference, annotation: Optional[BaseTableMetadataType]):
        if not table:
            raise ValueError("Table is required")
        self._table = table
        self._annotation = annotation
        self._hash_val = hash(table)

    @property
    def table(self) -> base.TableReference:
        return self._table

    @property
    def annotation(self) -> Optional[BaseTableMetadataType]:
        return self._annotation

    def is_join_node(self) -> bool:
        return False

    def tables(self) -> frozenset[base.TableReference]:
        return frozenset({self._table})

    def columns(self) -> frozenset[base.ColumnReference]:
        return frozenset(self.annotation.filter_predicate.columns()
                         if self.annotation and self.annotation.filter_predicate else set())

    def join_sequence(self) -> Sequence[IntermediateJoinNode[JoinMetadataType]]:
        return []

    def table_sequence(self) -> Sequence[BaseTableNode[BaseTableMetadataType]]:
        return [self]

    def as_list(self) -> NestedTableSequence:
        return self._table

    def count_cross_product_joins(self) -> int:
        return 0

    def homomorphic_hash(self) -> int:
        return hash(self)

    def base_table(self, traverse_right: bool = True) -> base.TableReference:
        return self._table

    def inspect(self, *, indentation: int = 0) -> str:
        padding = " " * indentation
        prefix = f"{padding} <- " if padding else ""
        annotation_str = f" {self.annotation.inspect()}" if self.annotation else ""
        return f"{prefix} SCAN :: {self.table}{annotation_str}"

    def __contains__(self, item) -> bool:
        if isinstance(item, predicates.AbstractPredicate):
            return self.annotation.filter_predicate == item if self.annotation else False
        return item == self._table

    def __len__(self) -> int:
        return 1

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, type(self)) and self._table == __value._table

    def __str__(self) -> str:
        return str(self._table)


JoinTreeType = typing.TypeVar("JoinTreeType", bound="JoinTree")


class JoinTree(Container[base.TableReference], Iterable[IntermediateJoinNode[JoinMetadataType]],
               Generic[JoinMetadataType, BaseTableMetadataType]):

    @staticmethod
    def cross_product_of(*trees: JoinTree[JoinMetadataType, BaseTableMetadataType],
                         annotation_supplier: Optional[Callable[[BaseMetadata, BaseMetadata], BaseMetadata]] = None
                         ) -> JoinTree[JoinMetadataType, BaseTableMetadataType]:
        """Generates a new join tree with by applying a cross product over the given join trees."""
        if not trees:
            raise ValueError("No trees given")
        elif len(trees) == 1:
            return trees[0]
        first_tree, *additional_trees = trees

        current_root = first_tree.root
        for additional_tree in additional_trees:
            merged_annotation = (annotation_supplier(current_root.annotation, additional_tree.annotation)
                                 if annotation_supplier else None)
            current_root = IntermediateJoinNode(additional_tree.root, current_root, merged_annotation)

        return JoinTree(current_root)

    @staticmethod
    def for_base_table(table: base.TableReference,
                       base_annotation: Optional[BaseTableMetadataType] = None
                       ) -> JoinTree[JoinMetadataType, BaseTableMetadataType]:
        """Generates a new join tree that for now only includes the given base table."""
        root = BaseTableNode(table, base_annotation)
        return JoinTree(root)

    @staticmethod
    def joining(left_tree: JoinTree[JoinMetadataType, BaseTableMetadataType],
                right_tree: JoinTree[JoinMetadataType, BaseTableMetadataType],
                join_annotation: Optional[JoinMetadataType] = None) -> JoinTree:
        """Constructs a new join tree that joins the two input trees."""
        if left_tree.is_empty():
            return right_tree
        if right_tree.is_empty():
            return left_tree
        join_node = IntermediateJoinNode(left_tree.root, right_tree.root, join_annotation)
        return JoinTree(join_node)

    def __init__(self: JoinTreeType, root: Optional[AbstractJoinTreeNode] = None):
        self._root = root

    @property
    def root(self) -> Optional[AbstractJoinTreeNode]:
        """Get the root element/node of this tree if there is any."""
        return self._root

    @property
    def upper_bound(self) -> float:
        """Provides the current upper bound or cardinality estimate of this join tree (i.e. its root node)."""
        if self.is_empty():
            return np.nan
        return self._root.annotation.upper_bound if self._root.annotation else np.nan

    def is_empty(self) -> bool:
        """Checks, whether there is at least one table in the join tree."""
        return self._root is None

    def traverse(self) -> Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]:
        """Accesses the root element of this join tree if there is any."""
        return self._root

    def tables(self) -> frozenset[base.TableReference]:
        """Provides all tables that are currently contained in the join tree."""
        if self.is_empty():
            return frozenset()
        return self._root.tables()

    def columns(self) -> frozenset[base.ColumnReference]:
        """Provides all columns that are currently referenced by the join and filter predicates of this join tree."""
        if self.is_empty():
            return frozenset()
        return self._root.columns()

    def join_columns(self) -> frozenset[base.ColumnReference]:
        """Provides all columns that are referenced by the join predicates in this join tree."""
        return frozenset(collection_utils.set_union(join_node.annotation.join_predicate.columns() for join_node
                                                    in self.join_sequence()
                                                    if join_node.annotation and join_node.annotation.join_predicate))

    def join_sequence(self) -> Sequence[IntermediateJoinNode[JoinMetadataType]]:
        """Provides a right-deep iteration of all join nodes in the join tree. See `JoinTreeNode`."""
        if self.is_empty():
            return []
        return self._root.join_sequence()

    def table_sequence(self) -> Sequence[BaseTableNode[BaseTableMetadataType]]:
        """Provides a right-deep iteration of all base tables in the join tree. See `JoinTreeNode`."""
        if self.is_empty():
            return []
        return self._root.table_sequence()

    def as_list(self) -> NestedTableSequence:
        """Provides the selected join order as a nested list.

        The table of each base table node will be contained directly in the join order. Each join node will be
        represented as a list of the list-representations of its child nodes. The "order" of left and right children
        is preserved.

        For example, the join order R ⋈ (S ⋈ T) will be represented as `[R, [S, T]]`.
        """
        if self.is_empty():
            return []
        return self._root.as_list()

    def base_table(self, direction: str = "right") -> base.TableReference:
        """Provides the left-most or right-most table in the join tree."""
        if direction not in {"right", "left"}:
            raise ValueError(f"Direction must be either 'left' or 'right', not '{direction}'")
        self._assert_not_empty()
        return self._root.base_table(direction == "right")

    def count_cross_product_joins(self) -> int:
        """Counts the number of joins in this tree that do not have an attached join predicate."""
        return 0 if self.is_empty() else self.root.count_cross_product_joins()

    def homomorphic_hash(self) -> int:
        """
        Calculates a hash value that is independent of join directions, i.e. R ⋈ S and S ⋈ R have the same hash value.
        """
        return self.root.homomorphic_hash() if self.root else hash(self.root)

    def inspect(self) -> str:
        """Produces a human-readable structure that describes the structure of this join tree."""
        if self.is_empty():
            return "[EMPTY JOIN TREE]"
        return self.root.inspect()

    def join_with_base_table(self, table: base.TableReference, base_annotation: Optional[BaseTableMetadataType] = None,
                             join_annotation: Optional[JoinMetadataType] = None, *,
                             insert_left: bool = True) -> JoinTreeType:
        if self.is_empty() and join_annotation:
            raise ValueError("Cannot use join_annotation for join with empty JoinTree")

        base_table_node = BaseTableNode(table, base_annotation)
        if self.is_empty():
            return JoinTree(base_table_node)
        left, right = (base_table_node, self.root) if insert_left else (self.root, base_table_node)
        join_node = IntermediateJoinNode(left, right, join_annotation)
        return JoinTree(join_node)

    def join_with_subtree(self, subtree: JoinTreeType, annotation: Optional[JoinMetadataType] = None, *,
                          insert_left: bool = True) -> JoinTreeType:
        if subtree.is_empty():
            raise ValueError("Cannot join with empty join tree")
        if self.is_empty() and annotation:
            raise ValueError("Cannot use annotation for join with empty join tree")

        if self.is_empty():
            return subtree
        left, right = (subtree, self.root) if insert_left else (self.root, subtree)
        join_node = IntermediateJoinNode(left, right, annotation)
        return JoinTree(join_node)

    def _assert_not_empty(self) -> None:
        """Raises an error if this tree is empty."""
        if self.is_empty():
            raise errors.StateError("Empty join tree")

    def __contains__(self, __x: object) -> bool:
        return __x in self._root if self._root else False

    def __len__(self) -> int:
        return 0 if self.is_empty() else len(self._root)

    def __iter__(self) -> Iterator[IntermediateJoinNode[JoinMetadataType]]:
        return iter(self.join_sequence())

    def __hash__(self) -> int:
        return hash(self._root)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, type(self)) and self._root == __value._root

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_empty():
            return "[EMPTY JOIN TREE]"
        return str(self._root)


LogicalTreeMetadata = typing.Union[LogicalBaseTableMetadata, LogicalJoinMetadata]


class LogicalJoinTree(JoinTree[LogicalJoinMetadata, LogicalBaseTableMetadata]):

    @staticmethod
    def cross_product_of(*trees: LogicalJoinTree,
                         annotation_supplier:
                         Optional[Callable[[LogicalTreeMetadata, LogicalTreeMetadata], LogicalTreeMetadata]] = None
                         ) -> LogicalJoinTree:
        if not trees:
            raise ValueError("No trees given")
        elif len(trees) == 1:
            return trees[0]
        first_tree, *additional_trees = trees

        current_root = first_tree.root
        for additional_tree in additional_trees:
            merged_annotation = (annotation_supplier(current_root.annotation, additional_tree.annotation)
                                 if annotation_supplier else None)
            current_root = IntermediateJoinNode(additional_tree.root, current_root, merged_annotation)

        return LogicalJoinTree(current_root)

    @staticmethod
    def for_base_table(table: base.TableReference,
                       base_annotation: Optional[LogicalBaseTableMetadata] = None) -> LogicalJoinTree:
        root = BaseTableNode(table, base_annotation)
        return LogicalJoinTree(root)

    @staticmethod
    def joining(left_tree: LogicalJoinTree, right_tree: LogicalJoinTree,
                join_annotation: Optional[LogicalJoinMetadata] = None) -> LogicalJoinTree:
        if left_tree.is_empty():
            return right_tree
        if right_tree.is_empty():
            return left_tree
        join_node = IntermediateJoinNode(left_tree.root, right_tree.root, join_annotation)
        return LogicalJoinTree(join_node)

    def __init__(self, root: Optional[AbstractJoinTreeNode[LogicalJoinMetadata, LogicalBaseTableMetadata]] = None
                 ) -> None:
        super().__init__(root)


def logical_join_tree_annotation_merger(first_annotation: Optional[LogicalTreeMetadata],
                                        second_annotation: Optional[LogicalTreeMetadata]) -> LogicalJoinMetadata:
    if not first_annotation or not second_annotation:
        return LogicalJoinMetadata()
    return LogicalJoinMetadata(upper_bound=first_annotation.upper_bound * second_annotation.upper_bound)


PhysicalPlanMetadata = typing.Union[PhysicalBaseTableMetadata, PhysicalJoinMetadata]


class PhysicalQueryPlan(JoinTree[PhysicalJoinMetadata, PhysicalBaseTableMetadata]):

    @staticmethod
    def cross_product_of(*trees: PhysicalQueryPlan,
                         annotation_supplier:
                         Optional[Callable[[PhysicalPlanMetadata, PhysicalPlanMetadata], PhysicalPlanMetadata]] = None
                         ) -> PhysicalQueryPlan:
        if not trees:
            raise ValueError("No trees given")
        elif len(trees) == 1:
            return trees[0]
        first_tree, *additional_trees = trees

        current_root = first_tree.root
        for additional_tree in additional_trees:
            merged_annotation = (annotation_supplier(current_root.annotation, additional_tree.annotation)
                                 if annotation_supplier else None)
            current_root = IntermediateJoinNode(additional_tree.root, current_root, merged_annotation)

        return PhysicalQueryPlan(current_root)

    @staticmethod
    def for_base_table(table: base.TableReference,
                       base_annotation: Optional[PhysicalBaseTableMetadata] = None) -> PhysicalQueryPlan:
        root = BaseTableNode(table, base_annotation)
        return PhysicalQueryPlan(root)

    @staticmethod
    def joining(left_tree: PhysicalQueryPlan, right_tree: PhysicalQueryPlan,
                join_annotation: Optional[PhysicalJoinMetadata] = None) -> PhysicalQueryPlan:
        if left_tree.is_empty():
            return right_tree
        if right_tree.is_empty():
            return left_tree
        join_node = IntermediateJoinNode(left_tree.root, right_tree.root, join_annotation)
        return PhysicalQueryPlan(join_node)

    def __init__(self, root: Optional[AbstractJoinTreeNode[PhysicalJoinMetadata, PhysicalBaseTableMetadata]] = None, *,
                 global_operator_settings: Optional[physops.PhysicalOperatorAssignment] = None) -> None:
        super().__init__(root)
        self._global_settings = (global_operator_settings.global_settings_only() if global_operator_settings
                                 else physops.PhysicalOperatorAssignment())

    @property
    def global_settings(self) -> physops.PhysicalOperatorAssignment:
        return self._global_settings

    def physical_operators(self) -> physops.PhysicalOperatorAssignment:
        assignment = self._global_settings.clone()

        for base_table in self.table_sequence():
            if not base_table.annotation or not base_table.annotation.operator:
                continue
            assignment.set_scan_operator(base_table.annotation.operator)

        for join in self.join_sequence():
            if not join.annotation or not join.annotation.operator:
                continue
            assignment.set_join_operator(join.annotation.operator)

        return assignment

    def plan_parameters(self) -> params.PlanParameterization:
        parameters = params.PlanParameterization()

        for base_table in self.table_sequence():
            if np.isnan(base_table.upper_bound):
                continue
            parameters.add_cardinality_hint(base_table.tables(), base_table.upper_bound)

        for join in self.join_sequence():
            if np.isnan(join.upper_bound):
                continue
            parameters.add_cardinality_hint(join.tables(), join.upper_bound)

        return parameters

    def join_with_base_table(self, table: base.TableReference, base_annotation: Optional[BaseTableMetadataType] = None,
                             join_annotation: Optional[JoinMetadataType] = None, *,
                             insert_left: bool = True) -> PhysicalQueryPlan:
        if self.is_empty() and join_annotation:
            raise ValueError("Cannot use join_annotation for join with empty JoinTree")

        base_table_node = BaseTableNode(table, base_annotation)
        if self.is_empty():
            return PhysicalQueryPlan(base_table_node)
        left, right = (base_table_node, self.root) if insert_left else (self.root, base_table_node)
        join_node = IntermediateJoinNode(left, right, join_annotation)
        return PhysicalQueryPlan(join_node)

    def join_with_subtree(self, subtree: PhysicalQueryPlan, annotation: Optional[JoinMetadataType] = None, *,
                          insert_left: bool = True) -> PhysicalQueryPlan:
        if subtree.is_empty():
            raise ValueError("Cannot join with empty join tree")
        if self.is_empty() and annotation:
            raise ValueError("Cannot use annotation for join with empty join tree")

        if self.is_empty():
            return subtree
        left, right = (subtree, self.root) if insert_left else (self.root, subtree)
        join_node = IntermediateJoinNode(left, right, annotation)
        merged_global_settings = self.global_settings.merge_with(subtree.global_settings)
        return PhysicalQueryPlan(join_node, global_operator_settings=merged_global_settings)


def physical_join_tree_annotation_merger(first_annotation: Optional[PhysicalPlanMetadata],
                                         second_annotation: Optional[PhysicalPlanMetadata]) -> PhysicalJoinMetadata:
    if not first_annotation or not second_annotation:
        return PhysicalJoinMetadata()
    return PhysicalJoinMetadata(upper_bound=first_annotation.upper_bound * second_annotation.upper_bound)
