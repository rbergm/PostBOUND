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

from postbound.qal import base, predicates, parser, qal
from postbound.db import db
from postbound.optimizer import physops, planparams
from postbound.util import collections as collection_utils, errors, jsonize, stats


class BaseMetadata(abc.ABC, jsonize.Jsonizable):
    """Common metadata information that is present on all join and base table nodes.

    The interpretation of the data can vary depending on the context in which the join tree is being used. For example,
    the cardinality to could either be a traditional cardinality estimate, a guaranteed upper bound, or the true
    cardinality of a join or scan.

    Parameters
    ----------
    cardinality : float, optional
        An indicator of the number of tuples that are *produced* by the node. If such a value is not available, the
        default value of ``math.nan`` can be used.
    """

    def __init__(self, cardinality: float = math.nan) -> None:
        self._cardinality = cardinality

    @property
    def cardinality(self) -> float:
        """Get the cardinality that is produced by the current node.

        Depending on the context, this cardinality can be interpreted in different ways, including as a traditional
        cardinality estimate, as an upper bound on the actual cardinality, or as a precise count of the number of
        tuples.

        Returns
        -------
        float
            The estimate. Can be ``math.nan`` to indicate that no value is available.
        """
        return self._cardinality

    @abc.abstractmethod
    def __json__(self) -> object:
        raise NotImplementedError


class JoinMetadata(BaseMetadata, abc.ABC):
    """Metadata that is specific to intermediate join nodes.

    In addition to the cardinality value, the join predicate that corresponds to the current node can be provided.

    Parameters
    ----------
    predicate : Optional[predicates.AbstractPredicate], optional
        The join condition that should be used to combine the child relations of the join node. Defaults to ``None`` if
        no condition is available, or if the join corresponds to a cross-product.
    cardinality : float, optional
        An indicator of the number of tuples that are *produced* by the node. If such a value is not available, the
        default value of ``math.nan`` can be used.
    """

    def __init__(self, predicate: Optional[predicates.AbstractPredicate] = None,
                 cardinality: float = math.nan) -> None:
        super().__init__(cardinality)
        self._join_predicate = predicate

    @property
    def join_predicate(self) -> Optional[predicates.AbstractPredicate]:
        """Get the condition that should be used to combine the child relations of the join node.

        Returns
        -------
        Optional[predicates.AbstractPredicate]
            The predicate or ``None`` if this is either unknown, or if the join node denotes a cross-product.
        """
        return self._join_predicate

    def inspect(self) -> str:
        """Provides the contents of the join node as a natural string.

        Returns
        -------
        str
            Information about the current join node
        """
        return f"JOIN ON {self.join_predicate}" if self.join_predicate else "CROSS JOIN"

    def __json__(self) -> object:
        return {"predicate": self._join_predicate, "cardinality": self._cardinality}


class LogicalJoinMetadata(JoinMetadata):
    """Models the join metadata that is specific to logical joins.

    Logical joins are only concerned with the relations that should be combined, but make no assumptions about how the
    joins should actually be executed.

    Parameters
    ----------
    predicate : Optional[predicates.AbstractPredicate], optional
        The join condition that should be used to combine the child relations of the join node. Defaults to ``None`` if
        no condition is available, or if the join corresponds to a cross-product.
    cardinality : float, optional
        An indicator of the number of tuples that are *produced* by the node. If such a value is not available, the
        default value of ``math.nan`` can be used.
    """

    def __init__(self, predicate: Optional[predicates.AbstractPredicate] = None,
                 cardinality: float = math.nan) -> None:
        super().__init__(predicate, cardinality)
        self._hash_val = hash((predicate, cardinality))

    __json__ = JoinMetadata.__json__

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.join_predicate == __value.join_predicate
                and self.cardinality == __value.cardinality)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"predicate={self.join_predicate}, cardinality={self.cardinality}"


class PhysicalJoinMetadata(JoinMetadata):
    """Models the join metadata that is specific to joins in physical query plans.

    Physical query plans do not only describe the order in which joins should be carried out, but also specify the
    physical operators that should be used to do so.

    Parameters
    ----------
    predicate : Optional[predicates.AbstractPredicate], optional
        The join condition that should be used to combine the child relations of the join node. Defaults to ``None`` if
        no condition is available, or if the join corresponds to a cross-product.
    cardinality : float, optional
        An indicator of the number of tuples that are *produced* by the node. If such a value is not available, the
        default value of ``math.nan`` can be used.
    join_info : Optional[physops.JoinOperatorAssignment], optional
        A description of the join operator that should be used for the join. If this is not known, the default value of
        ``None`` can be used. Notice however, that the entire purpose of the query execution plan is in having
        precisely this information. Therefore, cases were the `join_info` actually is ``None`` should be rare. If they
        are frequent, an investigation of other representations of the query plan are recommended.
    """
    def __init__(self, predicate: Optional[predicates.AbstractPredicate] = None, cardinality: float = math.nan,
                 join_info: Optional[physops.JoinOperatorAssignment] = None) -> None:
        super().__init__(predicate, cardinality)
        self._operator_assignment = join_info
        self._hash_val = hash((predicate, cardinality, join_info))

    @property
    def operator(self) -> Optional[physops.JoinOperatorAssignment]:
        """Get the physical operator that should be used to execute the join.

        Returns
        -------
        Optional[physops.JoinOperatorAssignment]
            The operator or ``None`` if it is unknown. Such situations should be quite rare however.
        """
        return self._operator_assignment

    def inspect(self) -> str:
        """Provides the contents of the join node as a natural string.

        Returns
        -------
        str
            Information about the current join node
        """
        base_inspection = super().inspect()
        return f"{base_inspection} USING {self.operator.operator.value}" if self._operator_assignment else base_inspection

    def __json__(self) -> object:
        return super().__json__() | {"operator": self._operator_assignment}

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.join_predicate == __value.join_predicate
                and self.cardinality == __value.cardinality
                and self.operator == __value.operator)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"predicate={self.join_predicate}, cardinality={self.cardinality}, operator={self.operator.operator}"


class BaseTableMetadata(BaseMetadata, abc.ABC):
    """Metadata that is specific to leaf nodes for scans.

    In addition to the cardinality value, a filter predicate for the base table can be provided.

    Parameters
    ----------
    filter_predicate : Optional[predicates.AbstractPredicate], optional
        The filter condition that restricts the allowed tuples from the base table. Defaults to ``None`` if
        no condition is available, or if the base table is not filtered.
    cardinality : float, optional
        An indicator of the number of tuples that are *produced* by the node. If such a value is not available, the
        default value of ``math.nan`` can be used.
    """
    def __init__(self, filter_predicate: Optional[predicates.AbstractPredicate],
                 cardinality: float = math.nan) -> None:
        super().__init__(cardinality)
        self._filter_predicate = filter_predicate

    @property
    def filter_predicate(self) -> Optional[predicates.AbstractPredicate]:
        """Get the condition that should be used to filter the base table.

        Returns
        -------
        Optional[predicates.AbstractPredicate]
            The predicate or ``None`` if this is either unknown, or if the base table is unfiltered.
        """
        return self._filter_predicate

    def inspect(self) -> str:
        """Provides the contents of the scan node as a natural string.

        Returns
        -------
        str
            Information about the current scan node
        """
        return f"FILTER {self.filter_predicate}" if self._filter_predicate else ""

    def __json__(self) -> object:
        return {"predicate": self._filter_predicate, "cardinality": self._cardinality}


class LogicalBaseTableMetadata(BaseTableMetadata):
    """Models the scan metadata that is specific to logical join orders.

    Logical joins are only concerned with the relations that should be combined, but make no assumptions about how the
    joins should actually be executed. The same holds for the logical base table nodes: only the existence of a scan
    operator is assumed, not how the scan is actually achieved.

    Parameters
    ----------
    filter_predicate : Optional[predicates.AbstractPredicate], optional
        The filter condition that restricts the allowed tuples from the base table. Defaults to ``None`` if
        no condition is available, or if the base table is not filtered.
    cardinality : float, optional
        An indicator of the number of tuples that are *produced* by the node. If such a value is not available, the
        default value of ``math.nan`` can be used.
    """
    def __init__(self, filter_predicate: Optional[predicates.AbstractPredicate],
                 cardinality: float = math.nan) -> None:
        super().__init__(filter_predicate, cardinality)
        self._hash_val = hash((filter_predicate, cardinality))

    __json__ = BaseTableMetadata.__json__

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.filter_predicate == __value.filter_predicate
                and self.cardinality == __value.cardinality)


class PhysicalBaseTableMetadata(BaseTableMetadata):
    """Models the scan metadata that is specific to scans in physical query plans.

    Physical query plans do not only describe the order in which tables should be scanned, but also specify the
    physical operators that should be used to do so.

    Parameters
    ----------
    filter_predicate : Optional[predicates.AbstractPredicate], optional
        The filter condition that restricts the allowed tuples from the base table. Defaults to ``None`` if
        no condition is available, or if the base table is not filtered.
    cardinality : float, optional
        An indicator of the number of tuples that are *produced* by the node. If such a value is not available, the
        default value of ``math.nan`` can be used.
    scan_info : Optional[physops.ScanOperatorAssignment], optional
        A description of the scan operator that should be used for the base table. If this is not known, the default
        value of ``None`` can be used. Notice however, that the entire purpose of the query execution plan is in having
        precisely this information. Therefore, cases were the `scan_info` actually is ``None`` should be rare. If they
        are frequent, an investigation of other representations of the query plan are recommended.
    """

    def __init__(self, filter_predicate: Optional[predicates.AbstractPredicate], cardinality: float = math.nan,
                 scan_info: Optional[physops.ScanOperatorAssignment] = None) -> None:
        super().__init__(filter_predicate, cardinality)
        self._operator_assignment = scan_info
        self._hash_val = hash((filter_predicate, cardinality, scan_info))

    @property
    def operator(self) -> Optional[physops.ScanOperatorAssignment]:
        """Get the physical operator that should be used to execute the scan.

        Returns
        -------
        Optional[physops.ScanOperatorAssignment]
            The operator or ``None`` if it is unknown. Such situations should be quite rare however.
        """
        return self._operator_assignment

    def inspect(self) -> str:
        """Provides the contents of the scan node as a natural string.

        Returns
        -------
        str
            Information about the current scan node
        """
        base_inspection = super().inspect()
        return f"{base_inspection} USING {self.operator.operator.value}" if self._operator_assignment else base_inspection

    def __json__(self) -> object:
        return super().__json__() | {"operator": self._operator_assignment}

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self.filter_predicate == __value.filter_predicate
                and self.cardinality == __value.cardinality
                and self.operator == __value.operator)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"predicate={self.filter_predicate}, upper bound={self.cardinality}, operator={self.operator.operator}"


class JoinTreeVisitor(abc.ABC):
    """Basic visitor to operator on arbitrary join trees.

    See Also
    --------
    JoinTree

    References
    ----------

    .. Visitor pattern: https://en.wikipedia.org/wiki/Visitor_pattern
    """

    @abc.abstractmethod
    def visit_intermediate_node(self, node: IntermediateJoinNode) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def visit_base_table_node(self, node: BaseTableNode) -> None:
        raise NotImplementedError


JoinMetadataType = typing.TypeVar("JoinMetadataType", bound=JoinMetadata)
"""Generic type that is bound to the actual metadata type that is used for the joins in a join tree."""

BaseTableMetadataType = typing.TypeVar("BaseTableMetadataType", bound=BaseTableMetadata)
"""Generic type that is bound to the actual metadata type that is used for the scans in a join tree."""

NestedTableSequence = Union[Sequence["NestedTableSequence"], base.TableReference]
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
        return base.TableReference(table_name, alias)
    else:
        raise TypeError(f"Unknown list element: {sequence}")


def _read_metadata_json(json_data: dict, base_table: bool, *, include_cardinalities: bool) -> BaseTableMetadata | JoinMetadata:
    """Generates the appropriate metadata object from a JSON representation.

    Parameters
    ----------
    json_data : dict
        The JSON data that should be parsed.
    base_table : bool
        Whether the metadata represents a scan over a base table or a join between two relations.
    include_cardinalities : bool
        Whether cardinality information should be included in the metadata.

    Returns
    -------
    BaseTableMetadata | JoinMetadata
        The parsed metadata object. Whether this is a physical metadata object (i.e. containing information about physical
        operators), or a logical metadata object is inferred from the JSON data.
    """
    json_parser = parser.JsonParser()
    cardinality: float = json_data.get("cardinality", math.nan) if include_cardinalities else math.nan
    if base_table:
        filter_predicate = json_parser.load_predicate(json_data["predicate"]) if "predicate" in json_data else None
        if "operator" in json_data:
            scan_assignment = physops.read_operator_json(json_data["operator"])
            return PhysicalBaseTableMetadata(filter_predicate, cardinality, scan_assignment)
        return LogicalBaseTableMetadata(filter_predicate, cardinality)
    else:
        join_predicate = json_parser.load_predicate(json_data["predicate"]) if "predicate" in json_data else None
        if "operator" in json_data:
            join_assignment = physops.read_operator_json(json_data["operator"])
            return PhysicalJoinMetadata(join_predicate, cardinality, join_assignment)
        return LogicalJoinMetadata(join_predicate, cardinality)


def read_from_json(json_data: dict, *, include_cardinalities: bool = True) -> LogicalJoinTree | PhysicalQueryPlan:
    """Reads a join tree from its JSON representation.

    This acts as the reverse operation to the *jsonize* protocol.

    Parameters
    ----------
    json_data : dict
        The JSON data that should be parsed.
    include_cardinalities : bool, optional
        Whether cardinality information should be included in the metadata. By default this is the case.

    Returns
    -------
    LogicalJoinTree | PhysicalQueryPlan
        A join tree that corresponds to the JSON data. Whether this is a logical join tree or a physical query plan is inferred
        based on the JSON data.
    """
    json_parser = parser.JsonParser()
    if "table" in json_data:
        base_table = json_parser.load_table(json_data["table"])
        metadata = _read_metadata_json(json_data["metadata"], base_table=True, include_cardinalities=include_cardinalities)
        base_table_node = BaseTableNode(base_table, metadata)
        if isinstance(metadata, PhysicalBaseTableMetadata):
            return PhysicalQueryPlan(base_table_node)
        return LogicalJoinTree(base_table_node)
    elif "left" in json_data and "right" in json_data:
        left_tree = read_from_json(json_data["left"], include_cardinalities=include_cardinalities)
        right_tree = read_from_json(json_data["right"], include_cardinalities=include_cardinalities)
        metadata = _read_metadata_json(json_data["metadata"], base_table=False, include_cardinalities=include_cardinalities)
        if isinstance(metadata, PhysicalJoinMetadata):
            return PhysicalQueryPlan.joining(left_tree, right_tree, metadata)
        return LogicalJoinTree.joining(left_tree, right_tree, metadata)
    else:
        raise ValueError("Malformed json data")


class AbstractJoinTreeNode(jsonize.Jsonizable, abc.ABC, Container[base.TableReference],
                           Generic[JoinMetadataType, BaseTableMetadataType]):
    """The fundamental type to construct a join tree. This node contains the actual entries/data.

    A join tree distinguishes between two types of nodes: join nodes which act as intermediate nodes that join
    together their child nodes and base table nodes that represent a scan over a base table. These nodes act as leaves
    in the join tree. This class describes the behavior that is common to both node types. The `IntermediateJoinNode`
    models the first node type and the `BaseTableNode` models the second node type.

    Nodes support containment checks for tables. Therefore, the ``in`` syntax can be used to check, whether a specific
    table is scanned in the current branch of the join tree, such as ``my_table in current_node``.
    """

    @property
    def annotation(self) -> Optional[JoinMetadataType | BaseTableMetadataType]:
        """Get the annotation that is added to the current node.

        Depending on the specific node type, this will either be join metadata or base table metadata. Although this
        value is optional, it should be present on the nodes of almost all join trees.

        Returns
        -------
        Optional[JoinMetadataType | BaseTableMetadataType]
            The annotation if it exists, ``None`` otherwise. For `BaseTableNode` this should be an instance of
            `BaseTableMetadata` and for `IntermediateJoinNode` this should be an instance of `JoinMetadata`.
        """
        raise NotImplementedError

    @property
    def cardinality(self) -> float:
        """Get the cardinality that is produced by the current node.

        Depending on the context, this cardinality can be interpreted in different ways, including as a traditional
        cardinality estimate, as an upper bound on the actual cardinality, or as a precise count of the number of
        tuples.

        Returns
        -------
        float
            The estimate. Can be ``math.nan`` to indicate that no value is available.
        """
        return self.annotation.cardinality if self.annotation else math.nan

    @abc.abstractmethod
    def is_join_node(self) -> bool:
        """Checks, whether this node is a join node.

        This is a more idiomatic check instead of using ``isinstance`` directly. The result of both checks should be
        the same however.

        Returns
        -------
        bool
            Whether this node is an intermediate node. If it is, it should have child nodes.
        """
        raise NotImplementedError

    def is_base_table_node(self) -> bool:
        """Checks, whether this node is a base table node.

        This is the inverse of `is_join_node`. Each node will either be a join node, or a base table node.

        Returns
        -------
        bool
            Whether this node is a leaf node. If it is, it should not have any more children.
        """
        return not self.is_join_node()

    @abc.abstractmethod
    def is_left_deep(self) -> bool:
        """Checks, whether the branch below and including the current node is left-deep.

        A left-deep tree always contains leaf nodes as the right children. It only grows to the left. As a special
        case, each leaf node is left deep. For all other nodes however, a left-deep node will not be right-deep or
        bushy.


        Returns
        -------
        bool
            Whether the join tree branch is left-deep.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_right_deep(self) -> bool:
        """Checks, whether the branch below and including the current node is right-deep.

        A right-deep tree always contains leaf nodes as the left children. It only grows to the right. As a special
        case, each leaf node is right deep. For all other nodes however, a right-deep node will not be left-deep or
        bushy.


        Returns
        -------
        bool
            Whether the join tree branch is right-deep.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_zigzag(self) -> bool:
        """Checks, whether the branch below and including the current node is a zig-zag tree.

        The intermediate nodes of a zig-zag tree always contain at least one leaf node as direct child. This is the
        opposite of a bushy tree and a superclass for left-deep and right-deep trees. Zig-zag trees and bushy trees are
        considered two mutually exclusive cases in PostBOUND. See the warning in `is_bushy` for more details.

        Zig-zag trees are also called linear trees.

        Returns
        -------
        bool
            Whether the join tree branch is a zig-zag tree.
        """
        raise NotImplementedError

    def is_linear(self) -> bool:
        """Checks, whether the branch below and including below the current node is a linear tree.

        This is a more idiomatic alias for `is_zigzag`. The intermediate nodes of a linear tree always contain at least
        one direct child node that is a leaf node.

        Returns
        -------
        bool
            Whether the join tree is a zig-zag/linear tree.
        """
        return self.is_zigzag()

    def is_bushy(self) -> bool:
        """Checks, whether the branch below and including the current node is a bushy tree.

        Bushy trees are trees where an intermediate node can have child nodes that are in turn intermediate nodes. As
        an additional restriction, we require at least one node of the current branch to actually make use of this
        property. Therefore, each bushy (sub-)tree is guaranteed to have at least one intermediate node with two
        intermediate nodes as direct children. Therefore, leaf nodes are never bushy.

        Returns
        -------
        bool
            Whether the join tree is a bushy tree.

        Warnings
        --------
        The terminology of linear and bushy trees is not strictly standardized. Some authors define bushy trees as a
        superclass of linear trees that also allow intermediate nodes with two intermediate node children. However,
        they also consider linear trees as (special cases of) bushy trees. We do not follow this definition in
        PostBOUND. This is because according to the definition all join trees would be bushy trees. This is not a
        particularly interesting finding. Instead, we want to be able to detect non-linear treest quickly and with
        idiomatic methods. Contrasting bushy trees against linear trees gives us precisely this possibility.
        """
        return not self.is_linear()

    @abc.abstractmethod
    def lookup(self, tables: set[base.TableReference]
               ) -> Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]:
        """Traverse the join tree until a branch that exactly contains a specific set of tables is found.

        Parameters
        ----------
        tables : set[base.TableReference]
            The tables for which to search.

        Returns
        -------
        Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]
            The deepest node that exactly contains the given tables. If no node contains the tables, ``None`` is
            returned.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tree_depth(self) -> int:
        """Provides the maximum number of nodes that need to be passed until a base table node is reached.

        This is equivalent to performing a depth-first search and determining the length of the returned path. An
        alternative take on this number is "the largest amount of hops that need to be passed until a base table is
        guaranteed to be reached".

        The depth treats the base table as a node that needs to be passed as well, i.e. calling `tree_depth` on
        a base table node will return `1`.

        Returns
        -------
        int
            The depth of the join tree under and including the current node.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def tables(self) -> frozenset[base.TableReference]:
        """Provides all tables that are scanned in the subtree under and including this node.

        Notice that this only checks the tables that are directly referenced by the leaf nodes. If join or filter
        predicates reference additional tables, these are not included in the set.

        Returns
        -------
        frozenset[base.TableReference]
            The tables in the leaf nodes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> frozenset[base.ColumnReference]:
        """Provides all columns that are present in the subtree under and including this node.

        In contrast to the `tables` method, this method actually checks the join and filter conditions of the nodes
        (otherwise, there simply would not be any columns). Therefore, the referenced tables in the `columns` method
        can be a superset of the tables in the `tables` method.

        Returns
        -------
        frozenset[base.ColumnReference]
            The columns that are referenced in the current branch of the join tree.

        Notes
        -----
        In a special case the set from the `tables` method might contain tables that are not found as referenced tables
        in the columns of the `columns` method. This occurs if a table is scanned without a filter condition and only
        joined as a cross-product afterwards.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def join_sequence(self) -> Sequence[IntermediateJoinNode[JoinMetadataType, BaseTableMetadataType]]:
        """Provides all joins under and including this node in depth-first order.

        The deepest join will be the first item in the sequence, the second deepest join the second item and so on.
        This means that children are located before their parents. In case of equally deep joins, left children will be
        returned before right children.

        Returns
        -------
        Sequence[IntermediateJoinNode[JoinMetadataType, BaseTableMetadataType]]
            All joins that are performed under and including this node. If this node is a base table node, the sequence
            is empty.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def table_sequence(self) -> Sequence[BaseTableNode[JoinMetadataType, BaseTableMetadataType]]:
        """Provides all base tables under and including this node in depth-first order.

        The base table that is located deepest in the current branch will be the first item in the sequence. In case of
        equally deep tables (i.e. tables that are scanned on the same level), left children are returned before right
        children.

        Returns
        -------
        Sequence[BaseTableNode[JoinMetadataType, BaseTableMetadataType]]
            All base tables that are scanned under and including this node. If this node is a base table node already,
            its table will be the only entry in the sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def as_list(self) -> NestedTableSequence:
        """Provides the join order of the current branch as a nested list.

        The table of each base table node will be contained directly in the join order. Each join node will provide the
        representation of its child nodes in a list.

        For example, the join order R ⋈ (S ⋈ T) will be represented as ``[R, [S, T]]``.

        Returns
        -------
        NestedTableSequence
            The sequence in which the joins are performed.
        """
        raise NotImplementedError

    def as_join_tree(self) -> JoinTree[JoinMetadataType, BaseTableMetadataType]:
        """Creates a new join tree with this node as root and all children as sub-nodes.

        Returns
        -------
        JoinTree[JoinMetadataType, BaseTableMetadataType]
            The join tree, anchored at the current node.
        """
        return JoinTree(self)

    @abc.abstractmethod
    def count_cross_product_joins(self) -> int:
        """Counts the number of joins below and including this node that do not have an attached join predicate.

        Returns
        -------
        int
            The number of cross products that appear in the current branch.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def homomorphic_hash(self) -> int:
        """Calculates a hash value of the current branch that is independent of join directions.

        The homorphic hash values of two join trees will be the same, if they join the same tables at the same time.
        It does not matter, whether relations are joined as a left child or as a right child, only that the join is
        calculated. For example, the homomorphic hash of R ⋈ S and S ⋈ R will be the same value.

        Returns
        -------
        int
            The homomorphic hash, which is a hash value that is independent of the assignment of the child nodes to
            left or right children.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def base_table(self, traverse_right: bool = True) -> base.TableReference:
        """Provides the base table that is located farthest down in one direction of the tree.

        The join tree can be traversed either in right-deep or left-deep manner and the direction will stay the same
        for the entire traversal. The traversal terminates as soon as a base table node has been found.

        Parameters
        ----------
        traverse_right : bool, optional
            Whether the traversal should decent into the right children of intermediate nodes, by default true.

        Returns
        -------
        base.TableReference
            The table that is scanned in the right-most or left-most base table node of the current branch

        See Also
        --------
        `table_sequence` : using `table_sequence()[0]` the deepest base table can be calculated
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inspect(self, *, _indentation: int = 0) -> str:
        """Produces a human-readable string that describes the structure of this join tree.

        Parameters
        ----------
        indentation : int, optional
            Internal parameter that denotes the indentation level that should be used to align branches nicely. This
            should not be set by the user. Defaults to 0 to not indent the root node.

        Returns
        -------
        str
            A pretty string reprensentation of the current join branch.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def accept_visitor(self, visitor: JoinTreeVisitor) -> None:
        """Enables processing of the current node by a join tree visitor.

        Parameters
        ----------
        visitor : JoinTreeVisitor
            The visitor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __json__(self) -> object:
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


class IntermediateJoinNode(AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType],
                           Generic[JoinMetadataType, BaseTableMetadataType]):
    """An intermediate join node represents a join between two child nodes (base table leaves or intermediate joins).

    Each join node is guaranteed to have exactly two child nodes. The join node distinguishes between the left child
    and the right child. However, it is up to the user and the specific type of join tree whether this distinction
    actually means anything. If the fact that the assignment of left and right children is not important should be
    emphasized, the `children` property can be used. This property provides the children in an arbitrary order.

    Parameters
    ----------
    left_child : AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]
        The left input relation of the join
    right_child : AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]
        The right input relation of the join
    annotation : Optional[JoinMetadataType]
        Additional metadata that describes the join. If no such information exists, ``None`` can be passed.
    """

    def __init__(self, left_child: AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType],
                 right_child: AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType],
                 annotation: Optional[JoinMetadataType]) -> None:
        if not left_child or not right_child:
            raise ValueError("Left child and right child are required")
        self._left_child = left_child
        self._right_child = right_child
        self._annotation = annotation
        self._hash_val = hash((left_child, right_child))

    @property
    def left_child(self) -> AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]:
        """Get the left child of the join.

        What the precise role of the left child in contrast to the right child is depends on the specific type of join
        tree as well as the usage context.

        Returns
        -------
        AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]
            The child node
        """
        return self._left_child

    @property
    def right_child(self) -> AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]:
        """Get the right child of the join.

        What the precise role of the right child in contrast to the left child is depends on the specific type of join
        tree as well as the usage context.

        Returns
        -------
        AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]
            The child node
        """
        return self._right_child

    @property
    def children(self) -> tuple[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType],
                                AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]:
        """Get the input relations of the join.

        Returns
        -------
        tuple[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType],
              AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]
            The left child and the right child in an arbitrary order
        """
        return self._left_child, self._right_child

    @property
    def annotation(self) -> Optional[JoinMetadataType]:
        """Get the additional metadata that describes this join.

        Returns
        -------
        Optional[JoinMetadataType]
            The annotation or ``None`` if no extra information exists.
        """
        return self._annotation

    def is_join_node(self) -> bool:
        return True

    def is_base_join(self) -> bool:
        """Checks, whether this join node is a join of two base table nodes.

        Returns
        -------
        bool
            Whether this join joins two base tables
        """
        return self.left_child.is_base_table_node() and self.right_child.is_base_table_node()

    def is_bushy_join(self) -> bool:
        """Checks, whether this join node is a join of two intermediate results.

        Returns
        -------
        bool
            Whether this join joins two intermediate results, i.e. relations that are themselves composed of joins of
            other intermediates and/or base tables.
        """
        return self.left_child.is_join_node() and self.right_child.is_join_node()

    def is_left_deep(self) -> bool:
        if not self.right_child.is_base_table_node():
            return False
        return self.left_child.is_left_deep()

    def is_right_deep(self) -> bool:
        if not self.left_child.is_base_table_node():
            return False
        return self.right_child.is_right_deep()

    def is_zigzag(self) -> bool:
        if self.left_child.is_join_node() and self.right_child.is_join_node():
            return False
        return self.left_child.is_zigzag() and self.right_child.is_zigzag()

    def lookup(self, tables: set[base.TableReference]) -> Optional[base.TableReference]:
        own_tables = self.tables()
        if len(own_tables) < len(tables):
            return None

        if len(own_tables) == len(tables):
            return self if own_tables == tables else None

        left_tables = self.left_child.tables()
        if tables <= left_tables:
            return self.left_child.lookup(tables)

        right_tables = self.right_child.tables()
        if tables <= right_tables:
            return self.right_child.lookup(tables)

        return None

    @functools.cache
    def tree_depth(self) -> int:
        return 1 + max(self.left_child.tree_depth(), self.right_child.tree_depth())

    def tables(self) -> frozenset[base.TableReference]:
        return frozenset(self._left_child.tables() | self._right_child.tables())

    def columns(self) -> frozenset[base.ColumnReference]:
        predicate_columns = (self.annotation.join_predicate.columns()
                             if self.annotation and self.annotation.join_predicate else set())
        return frozenset(self._left_child.columns() | self._right_child.columns() | predicate_columns)

    def children_by_depth(self) -> tuple[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType],
                                         AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]:
        """Provides the children of this join node, ordered by their depth.

        The depth refers to the largest number of joins that are located between the current node and the any of the
        base tables that are scanned in the current branch.

        Returns
        -------
        tuple[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType],
              AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]
            The input relations of this join, starting with the deepest node. In case of ties, the left child is
            put first.

        See Also
        --------
        tree_depth
        """
        return ((self.left_child, self.right_child) if self.left_child.tree_depth() >= self.right_child.tree_depth()
                else (self.right_child, self.left_child))

    def join_sequence(self) -> Sequence[IntermediateJoinNode[JoinMetadataType, BaseTableMetadataType]]:
        if self.is_base_join():
            return [self]

        deep_child, flat_child = self.children_by_depth()
        return list(deep_child.join_sequence()) + list(flat_child.join_sequence()) + [self]

    def table_sequence(self) -> Sequence[BaseTableNode[JoinMetadataType, BaseTableMetadataType]]:
        if self.is_base_join():
            assert isinstance(self.left_child, BaseTableNode) and isinstance(self.right_child, BaseTableNode)
            return [self.left_child, self.right_child]

        deep_child, flat_child = self.children_by_depth()
        return list(deep_child.table_sequence()) + list(flat_child.table_sequence())

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

    def inspect(self, *, _indentation: int = 0) -> str:
        padding = " " * _indentation
        prefix = f"{padding}<- " if padding else ""
        own_inspection = f"{prefix}{self.annotation.inspect()}" if self.annotation else f"{prefix}CROSS JOIN"
        left_inspection = self.left_child.inspect(_indentation=_indentation + 2)
        right_inspection = self.right_child.inspect(_indentation=_indentation + 2)
        return "\n".join([own_inspection, left_inspection, right_inspection])

    def accept_visitor(self, visitor: JoinTreeVisitor) -> None:
        visitor.visit_intermediate_node(self)

    def __json__(self) -> object:
        return {"left": self.left_child, "right": self.right_child, "metadata": self.annotation}

    def __contains__(self, item) -> bool:
        if item == self:
            return True
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
        left_str = str(self.left_child)
        if self.left_child.is_join_node():
            left_str = f"({left_str})"

        right_str = str(self.right_child)
        if self.right_child.is_join_node():
            right_str = f"({right_str})"

        return f"{left_str} ⋈ {right_str}"


class BaseTableNode(AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType],
                    Generic[JoinMetadataType, BaseTableMetadataType]):
    """A base table node represents the scan of table.

    Such nodes form the leaves in the join tree and their relation is typically further processed in join nodes.

    Parameters
    ----------
    table : base.TableReference
        The table that is scanned in this node
    annotation : Optional[BaseTableMetadataType]
        Additional metadata that describes the scan. If no such information exists, ``None`` can be passed.
    """

    def __init__(self, table: base.TableReference, annotation: Optional[BaseTableMetadataType]):
        if not table:
            raise ValueError("Table is required")
        self._table = table
        self._annotation = annotation
        self._hash_val = hash(table)

    @property
    def table(self) -> base.TableReference:
        """Get the table that is scanned in this node.

        Returns
        -------
        base.TableReference
            The table
        """
        return self._table

    @property
    def annotation(self) -> Optional[BaseTableMetadataType]:
        """Get the additional metadata that describes this scan.

        Returns
        -------
        Optional[BaseTableMetadataType]
            The annotation or ``None`` if no extra information exists.
        """
        return self._annotation

    def is_join_node(self) -> bool:
        return False

    def is_left_deep(self) -> bool:
        return True

    def is_right_deep(self) -> bool:
        return True

    def is_zigzag(self) -> bool:
        return True

    def lookup(self, tables: set[base.TableReference]) -> Optional[base.TableReference]:
        if len(tables) == 1 and self.table in tables:
            return self.table
        return None

    def tree_depth(self) -> int:
        return 1

    def tables(self) -> frozenset[base.TableReference]:
        return frozenset({self._table})

    def columns(self) -> frozenset[base.ColumnReference]:
        return frozenset(self.annotation.filter_predicate.columns()
                         if self.annotation and self.annotation.filter_predicate else set())

    def join_sequence(self) -> Sequence[IntermediateJoinNode[JoinMetadataType, BaseTableMetadataType]]:
        return []

    def table_sequence(self) -> Sequence[BaseTableNode[JoinMetadataType, BaseTableMetadataType]]:
        return [self]

    def as_list(self) -> NestedTableSequence:
        return self._table

    def count_cross_product_joins(self) -> int:
        return 0

    def homomorphic_hash(self) -> int:
        return hash(self)

    def base_table(self, traverse_right: bool = True) -> base.TableReference:
        return self._table

    def inspect(self, *, _indentation: int = 0) -> str:
        padding = " " * _indentation
        prefix = f"{padding}<- " if padding else ""
        annotation_str = f" {self.annotation.inspect()}" if self.annotation else ""
        return f"{prefix} SCAN :: {self.table}{annotation_str}"

    def accept_visitor(self, visitor: JoinTreeVisitor) -> None:
        visitor.visit_base_table_node(self)

    def __json__(self) -> object:
        return {"table": self.table, "metadata": self.annotation}

    def __contains__(self, item) -> bool:
        if item == self:
            return True
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
        return self._table.identifier()


JoinTreeType = typing.TypeVar("JoinTreeType", bound="JoinTree")
"""Join tree type used by generic methods that accept arbitrary join trees and produces trees of the same type."""

AnnotationMerger = Optional[Callable[[Optional[BaseMetadata], Optional[BaseMetadata]], Optional[BaseMetadata]]]
"""Type alias for methods that can combine different metadata objects."""


class JoinTree(jsonize.Jsonizable, Container[base.TableReference], Generic[JoinMetadataType, BaseTableMetadataType]):
    """The join tree captures the order in which base tables as well as intermediate results are joined together.

    Each join tree maintains the root of a composite tree structure consisting of `BaseTableNode` as leaves and
    `IntermediateJoinNode` as intermediate nodes. The interface it defines combines both access to the underlying
    tree structure, as well as some additional methods to modify and expand join trees.

    Each join tree is designed as an immutable object. Therefore, the modification of a join tree always results in a
    new tree instance.

    Parameters
    ----------
    root : Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]], optional
        The root node of the tree structure that should be maintained by this join tree. Can be ``None``, in which
        case the join tree remain empty.

    Notes
    -----
    In addition to the methods that are provided with the join tree the following standard Python methods work on
    join tree instances:

    - join trees support `bool()` calls and can be used directly in truth value testing. They return ``False`` if the
      join tree is empty and ``True`` otherwise.
    - join trees can be used for `in` checks with table references, i.e. ``table_ref in join_tree``. This check
      succeeds if the join tree contains the given table reference as a base table node
    - join trees support `len()` calls, which provide the number of base table scans in the tree

    Hashing is based on the join order only, i.e. specific annotations of individual nodes are not considered.
    """

    @staticmethod
    def cross_product_of(*trees: JoinTreeType, annotation_supplier: Optional[AnnotationMerger] = None) -> JoinTreeType:
        """Constructs a join tree that is the cross product of specific join trees.

        If more than two trees are given, the join trees are joined in the order in which they appear.

        Parameters
        ----------
        *trees :
            The join trees that should be combined via a cross product. The individual trees are combined via
            intermediate join nodes according to the order in which they are given. That is, the first two trees are
            combined first, than the resulting intermediate is combined with the third tree and so on.
        annotation_supplier : Optional[AnnotationMerger], optional
            Handler method to create `JoinMetadata` instances for the intermediate nodes. It receives the annotation
            of the current (intermediate) join tree and the annotation of the next join tree to join via a cross
            product as input and generates the annotation of the intermediate node for the next cross product. By
            default this is ``None``, which does not create any annotations for the intermediate nodes.

        Returns
        -------
        JoinTreeType
            The join tree that contains cross product joins for all given join trees. If just a single tree is given,
            that tree is returned directly.

        Raises
        ------
        ValueError
            If no trees are given
        """
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
                       base_annotation: Optional[BaseTableMetadataType] = None) -> JoinTreeType:
        """Generates a new join tree that only scans a specific base table.

        This join tree can than be used as a basis for constructing successively larger and more complicated join
        trees.

        Parameters
        ----------
        table : base.TableReference
            The table that should be scanned
        base_annotation : Optional[BaseTableMetadataType], optional
            The annotation for the base table. By default ``None`` is supplied, which indicates that there is no
            annotation.

        Returns
        -------
        JoinTreeType
            The join tree wrapping a single base table node for the given table.
        """
        root = BaseTableNode(table, base_annotation)
        return JoinTree(root)

    @staticmethod
    def joining(left_tree: JoinTreeType, right_tree: JoinTreeType,
                join_annotation: Optional[JoinMetadataType] = None) -> JoinTreeType:
        """Constructs a new join tree that joins two input trees.

        This method combines the two input trees in a new intermediate join and uses this join as the root of the tree.

        Parameters
        ----------
        left_tree : JoinTreeType
            The left child of the new tree root
        right_tree : JoinTreeType
            The right child of the new tree root
        join_annotation : Optional[JoinMetadataType], optional
            The annotation for the join. By default ``None`` is supplied, which indicates that there is no annotation.

        Returns
        -------
        JoinTreeType
            The join tree wrapping the join of the given input trees
        """
        if left_tree.is_empty():
            return right_tree
        if right_tree.is_empty():
            return left_tree
        join_node = IntermediateJoinNode(left_tree.root, right_tree.root, join_annotation)
        return JoinTree(join_node)

    def __init__(self: JoinTreeType,
                 root: Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]] = None) -> None:
        self._root = root

    @property
    def root(self) -> Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]:
        """Get the root node of this tree.

        Returns
        -------
        Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]
            The root node if it exists, or ``None`` if the join tree is empty.
        """
        return self._root

    @property
    def cardinality(self) -> float:
        """Get the cardinality that is associated with the root node of this tree.

        Returns
        -------
        float
            The cardinality or ``math.nan`` if the tree is empty
        """
        if self.is_empty():
            return math.nan
        return self._root.annotation.cardinality if self._root.annotation else math.nan

    def is_empty(self) -> bool:
        """Checks, whether there is at least one table in the join tree.

        Returns
        -------
        bool
            Whether the join tree has a valid root node
        """
        return self._root is None

    def is_right_deep(self) -> bool:
        """Checks, whether the root node induces a right-deep tree.

        Empty trees pass this check.

        See Also
        --------
        AbstractJoinNode.is_right_deep
        """
        return self.is_empty() or self.root.is_right_deep()

    def is_left_deep(self) -> bool:
        """Checks, whether the root node induces a left-deep tree.

        Empty trees pass this check.

        See Also
        --------
        AbstractJoinNode.is_left_deep
        """
        return self.is_empty() or self.root.is_left_deep()

    def is_zigzag(self) -> bool:
        """Checks, whether the root node induces a zig-zag tree.

        Empty trees pass this check.

        See Also
        --------
        AbstractJoinNode.is_zigzag
        """
        return self.is_empty() or self.root.is_zigzag()

    def is_linear(self) -> bool:
        """Checks, whether the root node induces a linear tree.

        Empty trees pass this check.

        See Also
        --------
        AbstractJoinNode.is_linear
        """
        return self.is_empty() or self.root.is_linear()

    def is_bushy(self) -> bool:
        """Checks, whether the root node induces a bushy tree.

        Empty trees pass this check.

        See Also
        --------
        AbstractJoinNode.is_bushy
        """
        return self.is_empty() or self.root.is_bushy()

    def traverse(self) -> Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]:
        """Provides access to the root node of this tree.

        This is just an alias for the `root` property that can be more idiomatic in some cases.

        Returns
        -------
        Optional[AbstractJoinTreeNode[JoinMetadataType, BaseTableMetadataType]]
            The root node if exists, or ``None`` if the tree is empty.
        """
        return self._root

    def lookup(self, tables: Iterable[base.TableReference]) -> Optional[AbstractJoinTreeNode]:
        """Traverses the join tree until a branch has been found that provides a specific set of tables exactly.

        See Also
        --------
        AbstractJoinNode.traverse
        """
        if self.is_empty():
            return None
        tables = set(tables)
        if not tables:
            return None
        return self._root.lookup(tables)

    def tables(self) -> frozenset[base.TableReference]:
        """Provides all tables that are contained in the join tree.

        See Also
        --------
        AbstractJoinNode.tables
        """
        if self.is_empty():
            return frozenset()
        return self._root.tables()

    def columns(self) -> frozenset[base.ColumnReference]:
        """Provides all columns that are referenced by the join and filter predicates of this join tree.

        See Also
        --------
        AbstractJoinNode.columns
        """
        if self.is_empty():
            return frozenset()
        return self._root.columns()

    def join_columns(self) -> frozenset[base.ColumnReference]:
        """Provides all columns that are referenced by the join predicates in this join tree.

        In contrast to the `columns` method, this excludes all columns that are only accessed as part of filter
        predicates.

        Returns
        -------
        frozenset[base.ColumnReference]
            The columns that appear in jion predicates.
        """
        return frozenset(collection_utils.set_union(join_node.annotation.join_predicate.columns() for join_node
                                                    in self.join_sequence()
                                                    if join_node.annotation and join_node.annotation.join_predicate))

    def join_sequence(self) -> Sequence[IntermediateJoinNode[JoinMetadataType, BaseTableMetadataType]]:
        """Provides all join nodes in the join tree in depth-first order.

        Returns
        -------
        Sequence[IntermediateJoinNode[JoinMetadataType, BaseTableMetadataType]]
            All join nodes of the join tree. For empty join trees the sequence will also be empty

        See Also
        --------
        AbstractJoinNode.join_sequence
        """
        if self.is_empty():
            return []
        return self._root.join_sequence()

    def table_sequence(self) -> Sequence[BaseTableNode[JoinMetadataType, BaseTableMetadataType]]:
        """Provides all base tables that are scanned in the join tree in depth-first order.

        Returns
        -------
        Sequence[BaseTableNode[JoinMetadataType, BaseTableMetadataType]]
            All base tables of the join tree. For empty join trees the sequence will also be empty

        See Also
        --------
        AbstractJoinNode.table_sequence
        """
        if self.is_empty():
            return []
        return self._root.table_sequence()

    def as_list(self) -> NestedTableSequence:
        """Provides the join order as a nested list.

        Returns
        -------
        NestedTableSequence
            The join order. For empty join trees the sequence will also be empty.

        See Also
        --------
        AbstractJoinNode.as_list
        """
        if self.is_empty():
            return []
        return self._root.as_list()

    def base_table(self, direction: Literal["left", "right"] = "right") -> base.TableReference:
        """Provides the base table is located the farthest down a direction in the join tree.

        Parameters
        ----------
        direction : str, optional
            In which direction the join tree should be traversed, by default "right" to follow the right child nodes.

        Returns
        -------
        base.TableReference
            The base table

        Raises
        ------
        errors.StateError
            If the join tree is empty

        See Also
        --------
        AbstractJoinNode.base_table
        """
        if direction not in {"right", "left"}:
            raise ValueError(f"Direction must be either 'left' or 'right', not '{direction}'")
        self._assert_not_empty()
        return self._root.base_table(direction == "right")

    def count_cross_product_joins(self) -> int:
        """Counts the number of joins in this tree that do not have an attached join predicate.

        Returns
        -------
        int
            The number of cross product join nodes in the join tree. For empty join trees 0 is returned.
        """
        return 0 if self.is_empty() else self.root.count_cross_product_joins()

    def homomorphic_hash(self) -> int:
        """Calculates a hash value for the join tree that is independent of the join directions

        Returns
        -------
        int
            The hash value

        See Also
        --------
        AbstractJoinNode.homomorphic_hash
        """
        return hash(None) if self.is_empty else self.root.homomorphic_hash()

    def inspect(self) -> str:
        """Produces a human-readable string that describes the structure of this join tree.

        Returns
        -------
        str
            A pretty string representation of the join tree
        """
        if self.is_empty():
            return "[EMPTY JOIN TREE]"
        return self.root.inspect()

    def join_with_base_table(self, table: base.TableReference, base_annotation: Optional[BaseTableMetadataType] = None,
                             join_annotation: Optional[JoinMetadataType] = None, *,
                             insert_left: bool = False) -> JoinTreeType:
        """Adds a new join with a base table to the current join tree.

        The given table will become the last join in the join tree and an intermediate join node is created as the new
        root node. For empty join trees the given table is used as the root node of the new tree directly.

        Parameters
        ----------
        table : base.TableReference
            The table that should be joined
        base_annotation : Optional[BaseTableMetadataType], optional
            Additional metadata about the table and its scan. Can be ``None`` if no metadata exists
        join_annotation : Optional[JoinMetadataType], optional
            Additional metadata about the join. Can be ``None`` if no metadata exists
        insert_left : bool, optional
            Whether the table should be inserted as the left child of the new root node. Defaults to ``False`` which
            inserts it to the right. By chaining `join_with_base_table` method calls and keeping the default parameter
            value a left-deep join tree can be constructed.

        Returns
        -------
        JoinTreeType
            The new join tree. It contains a new root node that combines the root of the current join tree with the
            new base table node.

        Raises
        ------
        ValueError
            If the current join tree is empty but a `join_annotation` is given. This might not always be an error, but
            better safe than sorry.
        """
        if self.is_empty() and join_annotation:
            raise ValueError("Cannot use join_annotation for join with empty JoinTree")

        base_table_node = BaseTableNode(table, base_annotation)
        if self.is_empty():
            return JoinTree(base_table_node)
        left, right = (base_table_node, self.root) if insert_left else (self.root, base_table_node)
        join_node = IntermediateJoinNode(left, right, join_annotation)
        return JoinTree(join_node)

    def join_with_subtree(self, subtree: JoinTreeType, annotation: Optional[JoinMetadataType] = None, *,
                          insert_left: bool = False) -> JoinTreeType:
        """Adds a new join with an entire subtree to the current join tree.

        The given subtree will become the last join in the join tree and an intermediate join node is created as the
        new root node. For empty join trees the given subtree is used as the root node of the new tree directly.

        Parameters
        ----------
        subtree : JoinTreeType
            The subtree to join
        annotation : Optional[JoinMetadataType], optional
            Additional metadata about the new join. Can be ``None`` if no metadata exists.
        insert_left : bool, optional
            Whether the subtree should be inserted as the left child of the new root node. Defaults to ``False`` which
            inserts it to the right.

        Returns
        -------
        JoinTreeType
            The new join tree. It contains a new root node that combines the root of the current join tree with the
            root node of the given subtree.

        Raises
        ------
        ValueError
            If the subtree is empty
        ValueError
            If the current join tree is empty but a `join_annotation` is given. This might not always be an error, but
            better safe than sorry.
        """
        if subtree.is_empty():
            raise ValueError("Cannot join with empty join tree")
        if self.is_empty() and annotation:
            raise ValueError("Cannot use annotation for join with empty join tree")

        if self.is_empty():
            return subtree
        left, right = (subtree.root, self.root) if insert_left else (self.root, subtree.root)
        join_node = IntermediateJoinNode(left, right, annotation)
        return JoinTree(join_node)

    def accept_visitor(self, visitor: JoinTreeVisitor) -> None:
        """Enables an arbitrary algorithm to be executed on the join tree.

        Parameters
        ----------
        visitor : JoinTreeVisitor
            The visitor algorithm to use
        """
        self._assert_not_empty()
        self._root.accept_visitor(visitor)

    def _assert_not_empty(self) -> None:
        """Raises an error if this tree is empty.

        Raises
        ------
        errors.StateError
            If the join tree is empty
        """
        if self.is_empty():
            raise errors.StateError("Empty join tree")

    def __json__(self) -> object:
        return self._root

    def __bool__(self) -> bool:
        return not self.is_empty()

    def __contains__(self, __x: object) -> bool:
        return __x in self._root if self._root else False

    def __len__(self) -> int:
        return 0 if self.is_empty() else len(self._root)

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
"""Type alias for the different metadata types that logical join trees use."""

LogicalAnnotationMerger = Optional[
    Callable[[Optional[LogicalTreeMetadata], Optional[LogicalTreeMetadata]], Optional[LogicalTreeMetadata]]]
"""Type alias for methods that can combine different metadata objects for logical join trees."""


class LogicalJoinTree(JoinTree[LogicalJoinMetadata, LogicalBaseTableMetadata]):
    """The logical join tree is a join tree that only focuses on the order in which joins should be carried out.

    It does not restrict the operators that should be used to actually execute the joins, nor does it care about those.

    In addition to the static factory methods that are also present on the normal join tree class, logical join trees
    provide the following extra factories:

    - `load_from_join_sequence`
    - `load_from_list` to generae a join tree directly from a `NestedTableSequence`
    - `load_from_query_plan` to generate a join tree directly from the output of the native optimizer of some database
       system

    Parameters
    ----------
    root : Optional[AbstractJoinTreeNode[LogicalJoinMetadata, LogicalBaseTableMetadata]], optional
            The root node of the tree structure that should be maintained by this join tree. Can be ``None``, in which
            case the join tree remain empty.
    """

    @staticmethod
    def cross_product_of(*trees: LogicalJoinTree,
                         annotation_supplier: Optional[LogicalAnnotationMerger] = None) -> LogicalJoinTree:
        """Constructs a join tree that is the cross product of specific join trees.

        See Also
        --------
        JoinTree.cross_product_of
        """
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
        """Generates a new join tree that only scans a specific base table.

        See Also
        --------
        JoinTree.for_base_table
        """
        root = BaseTableNode(table, base_annotation)
        return LogicalJoinTree(root)

    @staticmethod
    def joining(left_tree: LogicalJoinTree, right_tree: LogicalJoinTree,
                join_annotation: Optional[LogicalJoinMetadata] = None) -> LogicalJoinTree:
        """Constructs a new join tree that joins two input trees.

        See Also
        --------
        JoinTree.joining
        """
        if left_tree.is_empty():
            return right_tree
        if right_tree.is_empty():
            return left_tree
        join_node = IntermediateJoinNode(left_tree.root, right_tree.root, join_annotation)
        return LogicalJoinTree(join_node)

    @staticmethod
    def load_from_join_sequence(join_order: NestedTableSequence) -> LogicalJoinTree:
        """Creates a join tree as encoded by a join sequence.

        This is the inverse method to calling `join_sequence` on a join tree instance. Notice however, that no
        annotations are recovered.

        Parameters
        ----------
        join_order : NestedTableSequence
            The join sequence that should be loaded.

        Returns
        -------
        LogicalJoinTree
            The corresponding join tree

        See Also
        --------
        JoinTree.join_sequence
        """
        current_join_tree = LogicalJoinTree()
        for join in join_order:
            if isinstance(join, Sequence):
                subtree = LogicalJoinTree.load_from_join_sequence(join)
                current_join_tree = current_join_tree.join_with_subtree(subtree)
            else:
                current_join_tree = current_join_tree.join_with_base_table(join)
        return current_join_tree

    @staticmethod
    def load_from_list(join_order: NestedTableSequence) -> LogicalJoinTree:
        """Creates a join tree from its list representation.

        This is the inverse method to calling `as_list` on a join tree instance. Notice however, that no annotations
        are recovered.

        Parameters
        ----------
        join_order : NestedTableSequence
            A list encoding of a join tree

        Returns
        -------
        LogicalJoinTree
            The corresponding join tree

        See Also
        --------
        JoinTree.as_list
        """
        current_join_tree = LogicalJoinTree()
        for join in join_order:
            if isinstance(join, Sequence):
                subtree = LogicalJoinTree.load_from_list(join)
                current_join_tree = current_join_tree.join_with_subtree(subtree, insert_left=False)
            else:
                current_join_tree = current_join_tree.join_with_base_table(join, insert_left=False)
        return current_join_tree

    @staticmethod
    def load_from_query_plan(query_plan: db.QueryExecutionPlan,
                             query: Optional[qal.SqlQuery] = None) -> LogicalJoinTree:
        """Creates a join tree from a query plan.

        The join order used in the query plan will become the join order of the join tree. However, no physical
        operators are loaded. This is what the `PhysicalQueryPlan` is for.

        Inner children of the query plan nodes are inserted as right children in the corresponding intermediate join node.

        Parameters
        ----------
        query_plan : db.QueryExecutionPlan
            The query plan that was emitted by the optimizer of a database system. The plan will be used in two
            different ways: the join order will be used to generate the join tree itself. Furthermore, the cardinality
            estimates will be used as annotations for the nodes in the join tree.
        query : Optional[qal.SqlQuery], optional
            The query that was planned. If this is specified, the query will be used to inflate some of the annotations
            of the nodes in the join tree.

        Returns
        -------
        LogicalJoinTree
            The join tree that was used in the query plan

        Raises
        ------
        ValueError
            If any of the scan nodes does not have an associated table
        ValueError
            If a join nodes does not have exactly two child nodes
        ValueError
            If an intermediate node (i.e. non-join and non-scan) does not have exactly one child
        """
        if query_plan.is_scan:
            current_join_tree = LogicalJoinTree()
            table = query_plan.table
            if not table:
                raise ValueError(f"Scan nodes must have an associated table: {query_plan}")
            filter_predicate = query.predicates().filters_for(table) if query else None
            cardinality = query_plan.true_cardinality if query_plan.is_analyze() else query_plan.estimated_cardinality
            table_annotation = LogicalBaseTableMetadata(filter_predicate, cardinality)
            return current_join_tree.join_with_base_table(table, table_annotation)
        elif query_plan.is_join:
            if len(query_plan.children) != 2:
                raise ValueError(f"Join nodes must have exactly two child nodes: {query_plan}")
            if query_plan.inner_child is None:
                outer_child, inner_child = query_plan.children
            else:
                outer_child, inner_child = query_plan.outer_child, query_plan.inner_child
            outer_tree = LogicalJoinTree.load_from_query_plan(outer_child, query)
            inner_tree = LogicalJoinTree.load_from_query_plan(inner_child, query)
            join_predicate = (query.predicates().joins_between(outer_tree.tables(), inner_tree.tables())
                              if query else None)
            cardinality = query_plan.true_cardinality if query_plan.is_analyze() else query_plan.estimated_cardinality
            join_annotation = LogicalJoinMetadata(join_predicate, cardinality)
            return outer_tree.join_with_subtree(inner_tree, join_annotation, insert_left=False)

        if len(query_plan.children) != 1:
            raise ValueError(f"Non join/scan nodes must have exactly one child: {query_plan}")
        return LogicalJoinTree.load_from_query_plan(query_plan.children[0], query)

    def __init__(self: LogicalJoinTree,
                 root: Optional[AbstractJoinTreeNode[LogicalJoinMetadata, LogicalBaseTableMetadata]] = None) -> None:
        super().__init__(root)


def logical_join_tree_annotation_merger(first_annotation: Optional[LogicalTreeMetadata],
                                        second_annotation: Optional[LogicalTreeMetadata]) -> LogicalJoinMetadata:
    """Default handler to combine two logical annotations.

    Since logical join annotations are only composed of a join predicate and a cardinality, this task is quite simple:

    - If both annotations specify a join predicate, those predicates are combined in a conjunction. Otherwise valid
      predicates are retained or ``None`` is used.
    - The cardinality of the merged annotation is derived from the product of the cardinalities of the source
      cardinalities. This implies that as soon as one of the cardinalities is NaN, the entire cardinality becomes NaN

    Parameters
    ----------
    first_annotation : Optional[LogicalTreeMetadata]
        The first annotation to merge
    second_annotation : Optional[LogicalTreeMetadata]
        The second annotation to merge

    Returns
    -------
    LogicalJoinMetadata
        The merged annotation
    """
    if not first_annotation or not second_annotation:
        return LogicalJoinMetadata()
    merged_predicate = first_annotation.join_predicate
    if merged_predicate is None:
        merged_predicate = second_annotation.join_predicate
    elif second_annotation.join_predicate is not None:
        merged_predicate = predicates.CompoundPredicate.create_and([merged_predicate,
                                                                    second_annotation.join_predicate])

    return LogicalJoinMetadata(merged_predicate,
                               cardinality=first_annotation.cardinality * second_annotation.cardinality)


PhysicalPlanMetadata = typing.Union[PhysicalBaseTableMetadata, PhysicalJoinMetadata]
"""Type alias for the different metadata types that physical query plans use."""

PhysicalAnnotationMerger = Optional[
    Callable[[Optional[PhysicalPlanMetadata], Optional[PhysicalPlanMetadata]], Optional[PhysicalPlanMetadata]]]
"""Type alias for methods that can combine different metadata objects for physical query plans."""


def _physical_to_logical(physical_node: AbstractJoinTreeNode[PhysicalJoinMetadata, PhysicalBaseTableMetadata]
                         ) -> AbstractJoinTreeNode[LogicalJoinMetadata, LogicalBaseTableMetadata]:
    """Converts a physical join tree node into a logical node.

    Parameters
    ----------
    physical_node : AbstractJoinTreeNode[PhysicalJoinMetadata, PhysicalBaseTableMetadata]
        The physical node to convert

    Returns
    -------
    AbstractJoinTreeNode[LogicalJoinMetadata, LogicalBaseTableMetadata]
        The equivalent logical node, stripped of all physical information
    """
    if isinstance(physical_node, BaseTableNode):
        physical_annotation = physical_node.annotation
        logical_annotation = (LogicalBaseTableMetadata(physical_annotation.filter_predicate,
                                                       physical_annotation.cardinality)
                              if physical_annotation else None)
        return BaseTableNode(physical_node.table, logical_annotation)

    assert isinstance(physical_node, IntermediateJoinNode)
    left_logical = _physical_to_logical(physical_node.left_child)
    right_logical = _physical_to_logical(physical_node.right_child)
    physical_annotation = physical_node.annotation
    logical_annotation = (LogicalJoinMetadata(physical_annotation.join_predicate, physical_annotation.cardinality)
                          if physical_annotation else None)
    return IntermediateJoinNode(left_logical, right_logical, logical_annotation)


class PhysicalQueryPlan(JoinTree[PhysicalJoinMetadata, PhysicalBaseTableMetadata]):
    """The physical query plan is a join tree that adds physical operators to the join order.

    The intermediate join nodes do no longer just denote which tables should be combined, but also which join
    algorithms, etc. should be used. Likewise, base table nodes do not just describe the table that should be scanned,
    but also how the scan should take place.

    In addition to the static factory methods that are also present on the normal join tree class, physical query plans
    provide an additional `load_from_query_plan` factory method that can be used to create new join tree instances from
    a query plan that was emitted by the optimizer of an actual database system.

    Parameters
    ----------
    root : Optional[AbstractJoinTreeNode[PhysicalJoinMetadata, PhysicalBaseTableMetadata]], optional
        The root node of the tree structure that should be maintained by this join tree. Can be ``None``, in which
        case the join tree remain empty.
    global_operator_settings : Optional[physops.PhysicalOperatorAssignment], optional
        Settings that apply to the join tree as a whole, rather than to individual joins or scans. Defaults to ``None``
        which indicates that there are no such settings. If the assignment contain per-operator choices, these are
        simply ignored.
    """

    @staticmethod
    def cross_product_of(*trees: PhysicalQueryPlan,
                         annotation_supplier: PhysicalAnnotationMerger = None) -> PhysicalQueryPlan:
        """Constructs a join tree that is the cross product of specific join trees.

        See Also
        --------
        JoinTree.cross_product_of
        """
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
        """Generates a new join tree that only scans a specific base table.

        See Also
        --------
        JoinTree.for_base_table
        """
        root = BaseTableNode(table, base_annotation)
        return PhysicalQueryPlan(root)

    @staticmethod
    def joining(left_tree: PhysicalQueryPlan, right_tree: PhysicalQueryPlan,
                join_annotation: Optional[PhysicalJoinMetadata] = None) -> PhysicalQueryPlan:
        """Constructs a new join tree that joins two input trees.

        See Also
        --------
        JoinTree.joining
        """
        if left_tree.is_empty():
            return right_tree
        if right_tree.is_empty():
            return left_tree
        join_node = IntermediateJoinNode(left_tree.root, right_tree.root, join_annotation)
        return PhysicalQueryPlan(join_node)

    @staticmethod
    def load_from_query_plan(query_plan: db.QueryExecutionPlan, query: Optional[qal.SqlQuery] = None, *,
                             operators_only: bool = False) -> PhysicalQueryPlan:
        """Creates a join tree from a query plan.

        The join order used in the query plan will become the join order of the join tree. Furthermore, physical
        operators and additional parameters (e.g. cardinality estimates) are derived from the plan.

        Inner children of the query plan will become the right children of the corresponding intermediate join nodes.

        Parameters
        ----------
        query_plan : db.QueryExecutionPlan
            The query plan that was emitted by the optimizer of a database system. The plan will be used in two
            different ways: the join order will be used to generate the join tree itself. Furthermore, the physical
            operators, cardinality estimates, etc. will be used as annotations for the nodes in the join tree.
        query : Optional[qal.SqlQuery], optional
            The query that was planned. If this is specified, the query will be used to inflate some of the annotations
            of the nodes in the join tree.
        operators_only : bool, optional
            Whether only the physical operators should extracted from the query plan. If enabled, no cardinalities or parallel
            workers are loaded. Disabled by default.

        Returns
        -------
        PhysicalQueryPlan
            The join tree that was used in the query plan

        Raises
        ------
        ValueError
            If any of the scan nodes does not have an associated table
        ValueError
            If a join nodes does not have exactly two child nodes
        ValueError
            If an intermediate node (i.e. non-join and non-scan) does not have exactly one child
        """
        if query_plan.is_scan:
            table = query_plan.table
            if not table:
                raise ValueError(f"Scan nodes must have an associated table: {query_plan}")

            filter_predicate = query.predicates().filters_for(table) if query else None
            if operators_only:
                scan_info = physops.ScanOperatorAssignment(query_plan.physical_operator, table)
                cardinality = math.nan
            else:
                cardinality = query_plan.true_cardinality if query_plan.is_analyze() else query_plan.estimated_cardinality
                scan_info = (physops.ScanOperatorAssignment(query_plan.physical_operator, table,
                                                            query_plan.parallel_workers)
                             if query_plan.physical_operator else None)

            table_annotation = PhysicalBaseTableMetadata(filter_predicate, cardinality, scan_info)
            return PhysicalQueryPlan.for_base_table(table, table_annotation)
        elif query_plan.is_join:
            if len(query_plan.children) != 2:
                raise ValueError(f"Join nodes must have exactly two child nodes: {query_plan}")
            if query_plan.inner_child:
                outer_child, inner_child = query_plan.outer_child, query_plan.inner_child
            else:
                outer_child, inner_child = query_plan.children
            outer_tree = PhysicalQueryPlan.load_from_query_plan(outer_child, query, operators_only=operators_only)
            inner_tree = PhysicalQueryPlan.load_from_query_plan(inner_child, query, operators_only=operators_only)

            join_predicate = (query.predicates().joins_between(outer_tree.tables(), inner_tree.tables())
                              if query else None)
            if operators_only:
                join_info = physops.DirectionalJoinOperatorAssignment(query_plan.physical_operator,
                                                                      outer=outer_child.tables(), inner=inner_child.tables())
                cardinality = math.nan
            else:
                join_info = (physops.DirectionalJoinOperatorAssignment(query_plan.physical_operator,
                                                                       outer=outer_child.tables(),
                                                                       inner=inner_child.tables(),
                                                                       parallel_workers=query_plan.parallel_workers)
                             if query_plan.physical_operator else None)
                cardinality = query_plan.true_cardinality if query_plan.is_analyze() else query_plan.estimated_cardinality

            join_annotation = PhysicalJoinMetadata(join_predicate, cardinality, join_info)
            return outer_tree.join_with_subtree(inner_tree, join_annotation, insert_left=False)

        if len(query_plan.children) != 1:
            raise ValueError(f"Non join/scan nodes must have exactly one child: {query_plan}")
        return PhysicalQueryPlan.load_from_query_plan(query_plan.children[0], query, operators_only=operators_only)

    @staticmethod
    def load_from_logical_order(logical_order: LogicalJoinTree,
                                operators: Optional[physops.PhysicalOperatorAssignment] = None,
                                metadata: Optional[planparams.PlanParameterization] = None) -> PhysicalQueryPlan:
        """Expands a logical join order to a full physical plan.

        Parameters
        ----------
        logical_order : LogicalJoinTree
            The logical join order to use. This defines the basic structure of the physical plan.
        operators : Optional[physops.PhysicalOperatorAssignment], optional
            Information about the scan and join operators to use. If there are no associated operators for parts of the join
            order, no operators will be added for that part.
        metadata : Optional[planparams.PlanParameterization], optional
            Information about the cardinalities of different operators. Parallelization information is ignored, since this
            should already be supplied by the operator hints. If there is no cardinality hint for parts of the join order
            contained in the metadata, the cardinality from the `logical_order` will be re-used. Otherwise, that cardinality
            is overwritten.

        Returns
        -------
        PhysicalQueryPlan
            The logical plan expanded by the additional physical information.

        Raises
        ------
        TypeError
            If the logical join tree is malformed. Most likely this indicates that the model of the join tree was changed and
            this method was not updated properly.
        """
        if not logical_order:
            return PhysicalQueryPlan(global_operator_settings=operators)
        root_node = logical_order.root
        if isinstance(root_node, IntermediateJoinNode):
            left_child, right_child = root_node.left_child, root_node.right_child
            physical_left_child = PhysicalQueryPlan.load_from_logical_order(left_child.as_join_tree(), operators, metadata)
            physical_right_child = PhysicalQueryPlan.load_from_logical_order(right_child.as_join_tree(), operators, metadata)

            join_predicate = root_node.annotation.join_predicate if root_node.annotation else None
            join_cardinality = (metadata.cardinality_hints.get(root_node.tables(), math.nan) if metadata
                                else (root_node.annotation.cardinality if root_node.annotation else math.nan))
            join_operator = operators[root_node.tables()] if operators else None
            join_annotation = PhysicalJoinMetadata(join_predicate, join_cardinality, join_operator)

            physical_join_node = IntermediateJoinNode(physical_left_child.root, physical_right_child.root, join_annotation)
            return PhysicalQueryPlan(physical_join_node)
        elif isinstance(root_node, BaseTableNode):
            filter_predicate = root_node.annotation.filter_predicate if root_node.annotation else None
            scan_cardinality = (metadata.cardinality_hints.get(root_node.tables(), math.nan) if metadata
                                else (root_node.annotation.cardinality if root_node.annotation else math.nan))
            scan_operator = operators[root_node.table] if operators else None
            scan_annotation = PhysicalBaseTableMetadata(filter_predicate, scan_cardinality, scan_operator)
            physical_scan_node = BaseTableNode(root_node.table, scan_annotation)
            return PhysicalQueryPlan(physical_scan_node)
        else:
            raise TypeError("Unexpected node type (should be either IntermediateJoinNode or BaseTableNode): " + str(root_node))

    def __init__(self: PhysicalQueryPlan,
                 root: Optional[AbstractJoinTreeNode[PhysicalJoinMetadata, PhysicalBaseTableMetadata]] = None, *,
                 global_operator_settings: Optional[physops.PhysicalOperatorAssignment] = None) -> None:
        super().__init__(root)
        self._global_settings = (global_operator_settings.global_settings_only() if global_operator_settings
                                 else physops.PhysicalOperatorAssignment())

    @property
    def global_settings(self) -> physops.PhysicalOperatorAssignment:
        """Get the global settings of the query plan.

        Global settings are settings that apply to the plan as a whole, rather than just to individual operators. For
        example, such settings can restrict the choice of join operators for all nodes.

        Returns
        -------
        physops.PhysicalOperatorAssignment
            The global settings. Although the assignment could contain per-operator settings as well, these are not
            set.
        """
        return self._global_settings

    def physical_operators(self) -> physops.PhysicalOperatorAssignment:
        """Provides the physical operator assignment that is induced by this query plan.

        Returns
        -------
        physops.PhysicalOperatorAssignment
            An assignment of all scans and joins in the join tree, if the respective nodes contain assigned operators.
        """
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

    def plan_parameters(self) -> planparams.PlanParameterization:
        """Provides the plan parameterization that is induced by this query plan.

        Returns
        -------
        planparams.PlanParameterization
            An assignment of all parameters for all nodes in the join tree, if they are parameterized properly.
        """
        parameters = planparams.PlanParameterization()

        for base_table in self.table_sequence():
            if math.isnan(base_table.cardinality):
                continue
            parameters.add_cardinality_hint(base_table.tables(), base_table.cardinality)

        for join in self.join_sequence():
            if math.isnan(join.cardinality):
                continue
            parameters.add_cardinality_hint(join.tables(), join.cardinality)

        return parameters

    def join_with_base_table(self, table: base.TableReference, base_annotation: Optional[BaseTableMetadataType] = None,
                             join_annotation: Optional[JoinMetadataType] = None, *,
                             insert_left: bool = True) -> PhysicalQueryPlan:
        """Adds a new join with a base table to the current join tree.

        See Also
        --------
        JoinTree.join_with_base_table
        """
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
        """Adds a new join with an entire subtree to the current join tree.

        See Also
        --------
        JoinTree.join_with_subtree

        """
        if subtree.is_empty():
            raise ValueError("Cannot join with empty join tree")
        if self.is_empty() and annotation:
            raise ValueError("Cannot use annotation for join with empty join tree")

        if self.is_empty():
            return subtree
        left, right = (subtree.root, self.root) if insert_left else (self.root, subtree.root)
        join_node = IntermediateJoinNode(left, right, annotation)
        merged_global_settings = self.global_settings.merge_with(subtree.global_settings)
        return PhysicalQueryPlan(join_node, global_operator_settings=merged_global_settings)

    @functools.cache
    def as_logical_join_tree(self) -> LogicalJoinTree:
        """Transforms the physical query plan into an equivalent logical join tree.

        This strips the physical plan of all operator assignments.

        Returns
        -------
        LogicalJoinTree
            The logical join tree
        """
        if self.is_empty():
            return LogicalJoinTree()
        return LogicalJoinTree(_physical_to_logical(self.root))

    def plan_hash(self, *, exclude_predicates: bool = False, exclude_cardinalities: bool = True) -> int:
        """Calculates a hash value that considers the join order as well as the assigned physical operators.

        This method differs from the default hash method because join trees do only consider the structure of the tree, but
        no additional information. In order to ensure correct substitution properties, we retain the default hashing
        behavior in physical plans as well and use this method to obtain the full hash of a physical query plan.

        Parameters
        ----------
        exclude_predicates : bool, optional
            Whether the hash should ignore the join and filter predicates. This is off by default.
        exclude_cardinalities : bool, optional
            Whether the hash should ignore the cardinality estimates. This is on by default.

        Returns
        -------
        int
            The hash value.
        """
        original_hash = hash(self)
        join_hash_components = []
        for join in self.join_sequence():
            annotation = join.annotation
            if not annotation:
                join_hash_components.append((None,))
                continue
            current_annotation = [annotation.operator]
            if not exclude_predicates:
                current_annotation.append(annotation.join_predicate)
            if not exclude_cardinalities:
                current_annotation.append(annotation.cardinality)
            join_hash_components.append(tuple(current_annotation))

        scan_hash_components = []
        for scan in self.table_sequence():
            annotation = scan.annotation
            if not annotation:
                scan_hash_components.append((None,))
                continue
            current_annotation = [annotation.operator]
            if not exclude_predicates:
                current_annotation.append(annotation.filter_predicate)
            if not exclude_cardinalities:
                current_annotation.append(annotation.cardinality)
            scan_hash_components.append(tuple(current_annotation))

        join_hash_values = hash(tuple(join_hash_components))
        scan_hash_values = hash(tuple(scan_hash_components))
        return hash((original_hash, join_hash_values, scan_hash_values))


def physical_join_tree_annotation_merger(first_annotation: Optional[PhysicalPlanMetadata],
                                         second_annotation: Optional[PhysicalPlanMetadata]) -> PhysicalJoinMetadata:
    """Default handler to combine two physical annotations.

    There is no meaningful way to infer the operator of a merged annotation, even based on the operators of the
    input annotation. Therefore, this handler completely ignores the operator annotation. However, predicate and
    cardinality are handled in the following way (similar to the `logical_join_tree_annotation_merger`):

    - If both annotations specify a join predicate, those predicates are combined in a conjunction. Otherwise valid
      predicates are retained or ``None`` is used.
    - The cardinality of the merged annotation is derived from the product of the cardinalities of the source
      cardinalities. This implies that as soon as one of the cardinalities is NaN, the entire cardinality becomes NaN

    Parameters
    ----------
    first_annotation : Optional[PhysicalPlanMetadata]
        The first annotation to merge
    second_annotation : Optional[PhysicalPlanMetadata]
        The second annotation to merge

    Returns
    -------
    PhysicalJoinMetadata
        The merged annotations
    """
    if not first_annotation or not second_annotation:
        return PhysicalJoinMetadata()
    merged_predicate = first_annotation.join_predicate
    if merged_predicate is None:
        merged_predicate = second_annotation.join_predicate
    elif second_annotation.join_predicate is not None:
        merged_predicate = predicates.CompoundPredicate.create_and([merged_predicate,
                                                                    second_annotation.join_predicate])
    return PhysicalJoinMetadata(merged_predicate,
                                cardinality=first_annotation.cardinality * second_annotation.cardinality)


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

        left_score = stats.jaccard(leaf_tree.tables(), intermediate_tree.left_child.tables())
        right_score = stats.jaccard(leaf_tree.tables(), intermediate_tree.right_child.tables())

        return normalization_factor * max(left_score, right_score)

    assert isinstance(a, IntermediateJoinNode) and isinstance(b, IntermediateJoinNode)

    # similarity between two intermediate nodes
    a_left, a_right = a.left_child, a.right_child
    b_left, b_right = b.left_child, b.right_child

    symmetric_score = (stats.jaccard(a_left.tables(), b_left.tables())
                       + stats.jaccard(a_right.tables(), b_right.tables()))
    crossover_score = (stats.jaccard(a_left.tables(), b_right.tables())
                       + stats.jaccard(a_right.tables(), b_left.tables())
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
    return stats.jaccard(a_subtrees, b_subtrees)


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


def join_depth(join_tree: JoinTree) -> dict[base.TableReference, int]:
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
    dict[base.TableReference, int]
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
        or base table) differ in the estimated cardinality.
    left_state : frozenset[base.TableReference] | physops.PhysicalOperator | float
        Depending on the `change_type` this attribute describes the left tree. For example, for different tree structures,
        these are the tables in the left subtree, for different physical operators, this is the operator assigned to the node
        in the left tree and so on. For different join directions, this is the entire join node
    right_state : frozenset[base.TableReference] | physops.PhysicalOperator | float
        Equivalent attribute to `left_state`, just for the right tree.
    context : Optional[frozenset[base.TableReference]], optional
        For different physical operators or cardinality estimates, this describes the intermediate that is different. This
        attribute is unset by default.
    """

    change_type: Literal["tree-structure", "join-direction", "physical-op", "card-est"]
    left_state: frozenset[base.TableReference] | physops.PhysicalOperator | float
    right_state: frozenset[base.TableReference] | physops.PhysicalOperator | float
    context: Optional[frozenset[base.TableReference]] = None

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
            case _:
                raise errors.StateError(f"Unknown change type '{self.change_type}'")


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
    float
        The node's cardinality. Can be *NaN* if either the node is undefined or does not contain an annotated cardinality.
    """
    if node is None:
        return math.nan

    if isinstance(node, BaseTableMetadata):
        return node.cardinality
    elif isinstance(node, JoinMetadata):
        return node.cardinality

    return math.nan


def _extract_operator_from_annotation(node: AbstractJoinTreeNode | None) -> Optional[physops.PhysicalOperator]:
    """Provides the physical operator of a join tree node if there is one.

    Parameters
    ----------
    node : AbstractJoinTreeNode | None
        The node to extract from

    Returns
    -------
    float
        The node's operator. Can be *None* if either the node is undefined or does not contain an annotated cardinality.
    """
    if node is None:
        return None

    if isinstance(node, PhysicalBaseTableMetadata):
        return node.operator
    elif isinstance(node, PhysicalJoinMetadata):
        return node.operator

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
    if left_card != right_card and not (math.isnan(left_card) and math.isnan(right_card)):
        changes.append(JointreeChangeEntry("card-est", left_state=left_card, right_state=right_card, context=left.tables()))

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
