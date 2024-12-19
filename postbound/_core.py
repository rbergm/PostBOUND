
from __future__ import annotations

import collections
import math
import numbers
import warnings
from enum import Enum
from typing import Callable, Iterable, Literal, Optional, Sequence, Union

from . import util
from .util.errors import StateError


Cost = float
"""Type alias for a cost estimate."""

Cardinality = int
"""Type alias for a cardinality estimate."""


class ScanOperators(Enum):
    """The scan operators supported by PostBOUND.

    These can differ from the scan operators that are actually available in the selected target database system. The individual
    operators are chosen because they are supported by a wide variety of database systems and they are sufficiently different
    from each other.
    """
    SequentialScan = "Seq. Scan"
    IndexScan = "Idx. Scan"
    IndexOnlyScan = "Idx-only Scan"
    BitmapScan = "Bitmap Scan"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.value < other.value


class JoinOperators(Enum):
    """The join operators supported by PostBOUND.

    These can differ from the join operators that are actually available in the selected target database system. The individual
    operators are chosen because they are supported by a wide variety of database systems and they are sufficiently different
    from each other.
    """
    NestedLoopJoin = "NLJ"
    HashJoin = "Hash Join"
    SortMergeJoin = "Sort-Merge Join"
    IndexNestedLoopJoin = "Idx. NLJ"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.value < other.value


PhysicalOperator = Union[ScanOperators, JoinOperators]
"""Supertype to model all physical operators supported by PostBOUND.

These can differ from the operators that are actually available in the selected target database system.
"""


class TableReference:
    """A table reference represents a database table.

    It can either be a physical table, a CTE, or an entirely virtual query created via subqueries. Note that a table
    reference is indeed just a reference and not a 1:1 "representation" since each table can be sourced multiple times
    in a query. Therefore, in addition to the table name, each instance can optionally also contain an alias to
    distinguish between different references to the same table. In case of virtual tables, the full name will usually be empty
    and only the alias set. An exception are table references that refer to CTEs: their full name is set to the CTE name, the
    alias to the alias from the FROM clause (if present) and the table is still treated as virtual.

    Table references can be sorted lexicographically. All instances should be treated as immutable objects.

    Parameters
    ----------
    full_name : str
        The name of the table, corresponding to the name of a physical database table (or a view)
    alias : str, optional
        Alternative name that is in queries to refer to the table, or to refer to the same table multiple times.
        Defaults to an empty string
    virtual : bool, optional
        Whether the table is virtual or not. As a rule of thumb, virtual tables cannot be part of a ``FROM`` clause on
        their own, but need some sort of context. For example, the alias of a subquery is typically represented as a
        virtual table in PostBOUND. One cannot directly reference that alias in a ``FROM`` clause, without also
        specifying the subquery. Defaults to ``False`` since most tables will have direct physical counterparts.

    Raises
    ------
    ValueError
        If neither full name nor an alias are provided
    """

    @staticmethod
    def create_virtual(alias: str, *, full_name: str = "") -> TableReference:
        """Generates a new virtual table reference with the given alias.

        Parameters
        ----------
        alias : str
            The alias of the virtual table. Cannot be ``None``.
        full_name : str, optional
            An optional full name for the entire table. This is mostly used to create references to CTE tables.

        Returns
        -------
        TableReference
            The virtual table reference
        """
        return TableReference(full_name, alias, True)

    def __init__(self, full_name: str, alias: str = "", virtual: bool = False) -> None:
        if not full_name and not alias:
            raise ValueError("Full name or alias required")
        self._full_name = full_name if full_name else ""
        self._alias = alias if alias else ""
        self._virtual = virtual
        self._hash_val = hash((full_name, alias))

    @property
    def full_name(self) -> str:
        """Get the full name of this table. If empty, alias is guaranteed to be set.

        Returns
        -------
        str
            The name of the table
        """
        return self._full_name

    @property
    def alias(self) -> str:
        """Get the alias of this table. If empty, the full name is guaranteed to be set.

        The precise semantics of alias usage differ from database system to system. For example, in Postgres an alias
        shadows the original table name, i.e. once an alias is specified, it *must* be used to reference to the table
        and its columns.

        Returns
        -------
        str
            The alias of the table
        """
        return self._alias

    @property
    def virtual(self) -> bool:
        """Checks whether this table is virtual. In this case, only the alias and not the full name is set.

        Returns
        -------
        bool
            Whether this reference describes a virtual table
        """
        return self._virtual

    def identifier(self) -> str:
        """Provides a shorthand key that columns can use to refer to this table reference.

        For example, a table reference for ``movie_companies AS mc`` would have ``mc`` as its identifier (i.e. the
        alias), whereas a table reference without an alias such as ``company_type`` would provide the full table name
        as its identifier, i.e. ``company_type``.

        Returns
        -------
        str
            The shorthand
        """
        return self.alias if self.alias else self.full_name

    def drop_alias(self) -> TableReference:
        """Removes the alias from the current table if there is one. Returns the tabel as-is otherwise.

        Returns
        -------
        TableReference
            This table, but without an alias. Since table references are immutable, the original reference is not
            modified

        Raises
        ------
        StateError
            If this table is a virtual table, since virtual tables only have an alias and no full name.
        """
        if self.virtual:
            raise StateError("An alias cannot be dropped from a virtual table!")
        return TableReference(self.full_name)

    def with_alias(self, alias: str) -> TableReference:
        """Creates a new table reference for the same table but with a different alias.

        Parameters
        ----------
        alias : str
            The new alias

        Returns
        -------
        TableReference
            The updated table reference

        Raises
        ------
        StateError
            If the current table does not have a full name.
        """
        if not self.full_name:
            raise StateError("Cannot add an alias to a table without full name")
        return TableReference(self.full_name, alias, self.virtual)

    def __json__(self) -> object:
        return {"full_name": self._full_name, "alias": self._alias}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TableReference):
            return NotImplemented
        return self.identifier() < other.identifier()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self._full_name == __value._full_name
                and self._alias == __value._alias)

    def __repr__(self) -> str:
        return f"TableReference(full_name='{self.full_name}', alias='{self.alias}', virtual={self.virtual})"

    def __str__(self) -> str:
        if self.full_name and self.alias:
            return f"{self.full_name} AS {self.alias}"
        elif self.alias:
            return self.alias
        elif self.full_name:
            return self.full_name
        else:
            return "[UNKNOWN TABLE]"


class QueryExecutionPlan:
    """Heavily simplified system-independent model of physical query execution plans.

    A plan is a tree structure of `QueryExecutionPlan` objects. Each plan is a node in the tree and can have additional
    child nodes. These represent sub-operators that provide the input for the current plan.

    Since the information contained in physical query execution plans varies heavily between different database
    systems, this class models the smallest common denominator between all such plans. It focuses focuses on joins,
    scans and intermediate nodes and does not restrict the specific node types. Generally speaking, this model enforces
    very little constraints on the query plans. It is up to the users of the query plans to accomodate for deviations
    from the expected data and to the producers of query plan instances to be as strict as possible with the produced
    plans.

    In addition to the unified data storage in the different attributes, a number of introspection methods are also
    defined.

    All parameters to the `__init__` method are available as attributes on the new object. However, they should be
    considered read-only, even though this is not enforced.

    Comparisons between two query plans and hashing of a plan is based on the node type, the children and the table of each
    node. Other attributes are not considered for these operations such as cost estimates.


    Parameters
    ----------
    node_type : str
        The name of the operator/node
    is_join : bool
        Whether the operator represents a join. This is usually but not always mutually exclusive to the `is_scan`
        attribute.
    is_scan : bool
        Whether the operator represents a scan. This is usually but not always mutually exclusive to the `is_join`
        attribute.
    table : Optional[TableReference], optional
        For scan operators this can denote the table being scanned. Defaults to ``None`` if not applicable.
    children : Optional[Iterable[QueryExecutionPlan]], optional
        The sub-operators that provide input for the current operator. Can be ``None`` leaf nodes (most probably scans)
    parallel_workers : float, optional
        The total number of parallel workers that are used by the current operator. Defaults to ``math.nan`` if unknown
        or not applicable (e.g. if the operator does not support parallelization).
    cost : float, optional
        The cost of the operator, according to some cost model. This can be the cost as estimated by the native query
        optimizer of the database system. Alternatively, this can also represent costs from a custom cost model or even
        the real cost of the operator, depending on the context. Defaults to ``math.nan`` if unknown or not important
        in the current context.
    estimated_cardinality : float, optional
        The estimated number of rows that are produced by the operator. This estimate can be provided by the database
        system, or by some other estimation algorithm depending on context. Defaults to ``math.nan`` if unknown or not
        important in the current context.
    true_cardinality : float, optional
        The actual number of rows that are produced by the operator. This number is usually obtained by running the
        query and counting the output tuples. Notice that this number might only be an approximation of the true true
        cardinality (although usually a pretty precise one at that). Defaults to ``math.nan`` if the query execution
        plan is really only a plan and therefore does not contain any execution knowledge, or if the cardinality is
        simply not relevant in the current context.
    execution_time : float, optional
        The time in seconds it took to execute the operator. For intermediate nodes, this includes the execution time
        of all child operators as well. Usually, this denotes the wall time, but can be a more precise (e.g.
        tick-based) or coarse measurement, depending on the context. Defaults to  ``math.nan`` if the query execution
        plan is really only a plan and therefore does not contain any execution knowledge, or if the execution time is
        simply not relevant in the current context.
    cached_pages : float, optional
        The number of pages that were processed by this node and could be retrieved from cache (including child nodes).
        Defaults to ``math.nan`` if this is unknown or not applicable (e.g. for non-analyze plans). Notice that *cache* in this
        context means a database system-level cache (also called buffer pool, etc.), not the OS cache or a hardware cache.
    scanned_pages : float, optional
        The number of pages that were processed by this node and had to be read from disk and could not be retrieved from
        cache (including child nodes). Default to ``math.nan`` if this unknown or not applicable (e.g. for non-analyze plans).
        See `cached_pages` for more details.
    physical_operator : Optional[PhysicalOperator], optional
        The physical operator that corresponds to the current node if such an operator exists. Defaults to ``None``
        if no such operator exists or it could not be derived by the producer of the query plan.
    inner_child : Optional[QueryExecutionPlan], optional
        For intermediate operators that contain an inner and an outer relation, this denotes the inner relation. This
        assumes that the operator as exactly two children and allows to infer the outer children. Defaults to ``None``
        if the concept of inner/outer children is not applicable to the database system, not relevant to the current
        context, or if the relation could not be inferred.
    subplan_input : Optional[QueryExecutionPlan], optional
        Input node that was computed as part of a nested query. Such nodes will typically be used within the filter predicates
        of the node.
    subplan_root : bool, optional
        Whether the current node is the root of a subplan that computes a nested query. See `subplan_input` for more details.
    subplan_name : str, optional
        For subplan roots, this is the name of the subplan being exported. For nodes with subplan input, this is the name of
        the subplan being imported.

    Notes
    -----
    In case of parallel execution, all measures should be thought of "meaningful totals", i.e. the cardinality
    numbers are the total number of tuples produced by all workers. The execution time should denote the wall time, it
    took to execute the entire operator (which just happened to include parallel processing), **not** an average of the
    worker execution time or some other measure.
    """
    def __init__(self, node_type: str, is_join: bool, is_scan: bool, *, table: Optional[TableReference] = None,
                 children: Optional[Iterable[QueryExecutionPlan]] = None,
                 parallel_workers: float = math.nan, cost: float = math.nan,
                 estimated_cardinality: float = math.nan, true_cardinality: float = math.nan,
                 execution_time: float = math.nan,
                 cached_pages: float = math.nan, scanned_pages: float = math.nan,
                 physical_operator: Optional[PhysicalOperator] = None,
                 inner_child: Optional[QueryExecutionPlan] = None,
                 subplan_input: Optional[QueryExecutionPlan] = None, is_subplan_root: bool = False,
                 subplan_name: str = "") -> None:
        self.node_type = node_type
        self.physical_operator = physical_operator
        self.is_join = is_join
        self.is_scan = is_scan
        if is_scan and not isinstance(physical_operator, ScanOperators):
            warnings.warn("Supplied operator is scan operator but node is created as non-scan node")
        if is_join and not isinstance(physical_operator, JoinOperators):
            warnings.warn("Supplied operator is join operator but node is created as non-join node")

        self.parallel_workers = parallel_workers
        self.children: Sequence[QueryExecutionPlan] = tuple(children) if children else ()
        self.inner_child = inner_child
        self.outer_child: Optional[QueryExecutionPlan] = None
        if self.inner_child and len(self.children) == 2:
            first_child, second_child = self.children
            self.outer_child = first_child if self.inner_child == second_child else second_child

        self.table = table

        self.cost = cost
        self.estimated_cardinality = estimated_cardinality
        self.true_cardinality = true_cardinality
        self.execution_time = execution_time

        self.cached_pages = cached_pages
        self.scanned_pages = scanned_pages

        self.subplan_input = subplan_input
        if self.subplan_input:
            self.subplan_input.is_subplan_root = True
        self.is_subplan_root = is_subplan_root
        self.subplan_name = subplan_name

    @property
    def total_accessed_pages(self) -> float:
        """The total number of pages that where processed in this node, as well as all child nodes.

        This includes pages that were fetched from the DB cache, as well as pages that had to be read from disk.

        Returns
        -------
        float
            The number of pages. Can be ``NaN`` if this number cannot be inferred, e.g. for non-analyze plans.
        """
        return self.cached_pages + self.scanned_pages

    @property
    def cache_hit_ratio(self) -> float:
        """The share of pages that could be fetched from cache compared to the total number of processed pages.

        Returns
        -------
        float
            The hit ratio. Can be ``NaN`` if the ratio cannot be inferred, e.g. for non-analyze plans.
        """
        return self.cached_pages / self.total_accessed_pages

    def is_analyze(self) -> bool:
        """Checks, whether the plan contains runtime information.

        If that is the case, the plan most likely contains additional information about the execution time of the
        operator as well as the actual cardinality that was produced.

        Returns
        -------
        bool
            Whether the plan corresponds to an ANALYZE plan.
        """
        return not math.isnan(self.true_cardinality) or not math.isnan(self.execution_time)

    def tables(self) -> frozenset[TableReference]:
        """Collects all tables that are referenced by this operator as well as all child operators.

        Most likely this corresponds to all tables that were scanned below the current operator.

        Returns
        -------
        frozenset[TableReference]
            The tables
        """
        own_table = [self.table] if self.table else []
        return frozenset(own_table + util.flatten(child.tables() for child in self.children))

    def is_base_join(self) -> bool:
        """Checks, whether this operator is a join and only contains base table children.

        This does not necessarily mean that the children will already be scan operators, but it means that this
        operator does not consume intermediate relations that have been joined themselves. For example, a hash join
        can have one child that is scanned directly, and another child that is an intermediate hash node. That hash
        node takes care of creating the hash table for its children, which in turn is a scan of a base table.

        Returns
        -------
        bool
            Whether this operator corresponds to a join of base table relations
        """
        return self.is_join and all(child.is_scan_branch() for child in self.children)

    def is_bushy_join(self) -> bool:
        """Checks, whether this operator is a join of two intermediate tables.

        An intermediate table is a table that is itself composed of a join of base tables. The children can be
        arbitrary other operators, but each of the children will at some point have a join operator as a child node.

        Returns
        -------
        bool
            Whether this operator correponds to a join of intermediate relations
        """
        return self.is_join and all(child.is_join_branch() for child in self.children)

    def is_scan_branch(self) -> bool:
        """Checks, whether this branch of the query plan leads to a scan eventually.

        Most importantly, a *scan branch* does not contain any joins and is itself not a join. Typically, it will end
        with a scan leaf and only contain intermediate nodes (such as a Hash node in PostgreSQL) afterwards.

        Returns
        -------
        bool
            Whether this node does not contain any joins below.
        """
        return self.is_scan or (len(self.children) == 1 and self.children[0].is_scan_branch())

    def is_join_branch(self) -> bool:
        """Checks, whether this branch of the query plan leads to a join eventually.

        In contrast to a scan branch, a join branch is guaranteed to contain at least one more child (potentially
        transitively) that is a join node. Alternatively, the current node might be a join node itself.

        Returns
        -------
        bool
            Whether this node does contain at least one join children (or is itself a join)
        """
        return self.is_join or (len(self.children) == 1 and self.children[0].is_join_branch())

    def fetch_base_table(self) -> Optional[TableReference]:
        """Provides the base table that is associated with this scan branch.

        This method basically traverses the current branch of the query execution plan until a node with an associated
        `table` is found.

        Returns
        -------
        Optional[TableReference]
            The associated table of the highest child node that has a valid `table` attribute. As a special case this
            might be the table of this very plan node. If none of the child nodes contain a valid table returns
            ``None``.

        Raises
        ------
        ValueError
            _description_

        See Also
        --------
        is_scan_branch
        """
        if not self.is_scan_branch():
            raise ValueError("No unique base table for non-scan branches!")
        if self.table:
            return self.table
        return self.children[0].fetch_base_table()

    def total_processed_rows(self) -> float:
        """Counts the sum of all rows that have been processed at each node below and including this plan node.

        This basically calculates the value of the *C_out* cost model for the current branch. Calling this method on
        the root node of the query plan provides the *C_out* value of the entire plan.

        Returns
        -------
        float
            The *C_out* value

        Notes
        -----
        *C_out* is defined as

        .. math ::
            C_{out}(T) = \\begin{cases}
                |T|, & \\text{if T is a single relation} \\\\
                |T| + C_{out}(T_1) + C_{out}(T_2), & \\text{if } T = T_1 \\bowtie T_2
            \\end{cases}

        Notice that there are variations of the *C_out* value that do not include any cost for single relations and
        only measure the cost of joins.
        """
        if not self.is_analyze():
            return math.nan
        return self.true_cardinality + sum(child.total_processed_rows() for child in self.children)

    def qerror(self) -> float:
        """Calculates the q-error of the current node.

        Returns
        -------
        float
            The q-error

        Notes
        -----
        The q-error can only be calculate for analyze nodes. We use the following formula:

        .. math ::
            qerror(e, a) = \\frac{max(e, a) + 1}{min(e, a) + 1}

        where *e* is the estimated cardinality of the node and *a* is the actual cardinality. Notice that we add 1 to both the
        numerator as well as the denominator to prevent infinite errors for nodes that do not process any rows (e.g. due to
        pruning).
        """
        if math.isnan(self.true_cardinality):
            return math.nan
        # we add 1 to our q-error to prevent probelms with 0 cardinalities
        return ((max(self.estimated_cardinality, self.true_cardinality) + 1)
                / (min(self.estimated_cardinality, self.true_cardinality) + 1))

    def scan_nodes(self) -> frozenset[QueryExecutionPlan]:
        """Provides all scan nodes under and including this node.

        Returns
        -------
        frozenset[QueryExecutionPlan]
            The scan nodes
        """
        own_node = [self] if self.is_scan else []
        child_scans = util.flatten(child.scan_nodes() for child in self.children)
        return frozenset(own_node + child_scans)

    def join_nodes(self) -> frozenset[QueryExecutionPlan]:
        """Provides all join nodes under and including this node.

        Returns
        -------
        frozenset[QueryExecutionPlan]
            The join nodes
        """
        own_node = [self] if self.is_join else []
        child_joins = util.flatten(child.join_nodes() for child in self.children)
        return frozenset(own_node + child_joins)

    def iternodes(self) -> Iterable[QueryExecutionPlan]:
        """Provides all nodes in the plan, in breadth-first order.

        The current node is also included in the output.

        Returns
        -------
        Iterable[QueryExecutionPlan]
            The nodes.
        """
        nodes = [self]
        for child_node in self.children:
            nodes.extend(child_node.iternodes())
        return nodes

    def plan_depth(self) -> int:
        """Calculates the maximum path length from this node to a leaf node.

        Since the current node is included in the calculation, the minimum value is 1 (if this node already is a leaf
        node).

        Returns
        -------
        int
            The path length
        """
        return 1 + max((child.plan_depth() for child in self.children), default=0)

    def find_first_node(self, predicate: Callable[[QueryExecutionPlan], bool], *,
                        traversal: Literal["left", "right", "inner", "outer"] = "left") -> Optional[QueryExecutionPlan]:
        """Recursively searches for the first node that matches a specific predicate.

        Parameters
        ----------
        predicate : Callable[[QueryExecutionPlan], bool]
            The predicate to check. The predicate is called on each node in the tree and should return ``True`` if the node
            matches the desired search criteria.
        traversal : Literal["left", "right", "inner", "outer"], optional
            The traversal strategy to use. It indicates which child node should be checked first if the `predicate` is not
            satisfied by the current node. Defaults to ``"left"`` which means that the left child is checked first.

        Returns
        -------
        Optional[QueryExecutionPlan]
            The first node that matches the predicate. If no such node exists, ``None`` is returned.
        """
        if predicate(self):
            return self
        if not self.children:
            return None

        if len(self.children) == 1:
            return self.children[0].find_first_node(predicate, traversal=traversal)
        if len(self.children) != 2:
            raise ValueError("Cannot traverse plan nodes with more than two children")

        if traversal == "inner" or traversal == "outer":
            first_child_to_check = self.inner_child if traversal == "inner" else self.outer_child
            second_child_to_check = self.outer_child if traversal == "inner" else self.inner_child
        else:
            first_child_to_check = self.children[0] if traversal == "left" else self.children[1]
            second_child_to_check = self.children[1] if traversal == "left" else self.children[0]

        first_result = first_child_to_check.find_first_node(predicate, traversal=traversal)
        if first_result is not None:
            return first_result
        return second_child_to_check.find_first_node(predicate, traversal=traversal)

    def find_all_nodes(self, predicate: Callable[[QueryExecutionPlan], bool]) -> Sequence[QueryExecutionPlan]:
        """Recursively searches for all nodes that match a specific predicate.

        Parameters
        ----------
        predicate : Callable[[QueryExecutionPlan], bool]
            The predicate to check. The predicate is called on each node in the tree and should return ``True`` if the node
            matches the desired search criteria.

        Returns
        -------
        Sequence[QueryExecutionPlan]
            All nodes that match the predicate. If no such node exists, an empty sequence is returned. Matches are returned in
            depth-first order, i.e. nodes higher up in the plan are returned before their matching child nodes.
        """
        def _handler(node: QueryExecutionPlan, predicate: Callable[[QueryExecutionPlan], bool]) -> list[QueryExecutionPlan]:
            matches = [node] if predicate(node) else []
            for child in node.children:
                matches.extend(_handler(child, predicate))
            return matches
        return _handler(self, predicate)

    def simplify(self) -> Optional[QueryExecutionPlan]:
        """Provides a query execution plan that is stripped of all non-join and non-scan nodes.

        Notice that this operation can break certain assumptions of mathematical relation between parent and child
        nodes, e.g. plan costs might no longer add up correctly.

        Returns
        -------
        Optional[QueryExecutionPlan]
            The simplified plan. If this method is called on a non-scan and non-join node that does not have any more
            children, ``None`` is returned. In all other cases, the final join or the only scan node will be returned.
        """
        if not self.is_join and not self.is_scan:
            if len(self.children) != 1:
                return None
            return self.children[0].simplify()

        simplified_children = [child.simplify() for child in self.children] if not self.is_scan else []
        simplified_inner = self.inner_child.simplify() if not self.is_scan and self.inner_child else None
        return QueryExecutionPlan(self.node_type, self.is_join, self.is_scan,
                                  table=self.table,
                                  children=simplified_children,
                                  parallel_workers=self.parallel_workers,
                                  cost=self.cost,
                                  estimated_cardinality=self.estimated_cardinality,
                                  true_cardinality=self.true_cardinality,
                                  execution_time=self.execution_time,
                                  physical_operator=self.physical_operator,
                                  inner_child=simplified_inner)

    def inspect(self, *, fields: Optional[Iterable[str]] = None, skip_intermediates: bool = False,
                _current_indentation: int = 0) -> str:
        """Provides a nice hierarchical string representation of the plan structure.

        The representation typically spans multiple lines and uses indentation to separate parent nodes from their
        children.

        Parameters
        ----------
        fields : Optional[Iterable[str]], optional
            The attributes of the nodes that should be included in the output. Can be set to any number of the available
            attributes. If no fields are given, a default configuration inspired by Postgres' **EXPLAIN ANALYZE** output is
            used.
        skip_intermediates : bool, optional
            Whether non-scan and non-join nodes should be excluded from the representation. Defaults to ``False``.
        _current_indentation : int, optional
            Internal parameter to the `_inspect` function. Should not be modified by the user. Denotes how deeply
            recursed we are in the plan tree. This enables the correct calculation of the current indentation level.
            Defaults to 0 for the root node.

        Returns
        -------
        str
            A string representation of the query plan
        """
        # TODO: include subplan_input in the inspection
        padding = " " * _current_indentation
        prefix = f"{padding}<- " if padding else ""

        if not fields:
            own_inspection = [prefix + self._explain_text()]
        else:
            attr_values = {attr: getattr(self, attr) for attr in fields}
            pretty_attrs = {attr: round(val, 3) if isinstance(val, numbers.Number) else val
                            for attr, val in attr_values.items()}
            attr_str = " ".join(f"{attr}={val}" for attr, val in pretty_attrs.items())
            own_inspection = [prefix + f"{self.node_type} ({attr_str})"]

        child_inspections = [
            child.inspect(fields=fields, skip_intermediates=skip_intermediates, _current_indentation=_current_indentation + 2)
            for child in self.children]
        if self.subplan_input:
            subplan_name = self.subplan_input.subplan_name if self.subplan_input.subplan_name else ""
            child_inspections.append(f"{padding}SubPlan: {subplan_name}")
            child_inspections.append(self.subplan_input.inspect(fields=fields, skip_intermediates=skip_intermediates,
                                                                _current_indentation=_current_indentation + 2))

        if not skip_intermediates or self.is_join or self.is_scan or not _current_indentation:
            return "\n".join(own_inspection + child_inspections)
        else:
            return "\n".join(child_inspections)

    def plan_summary(self) -> dict[str, object]:
        """Generates a short summary about some node statistics.

        Currently, the following information is reported:

        - The *C_out* value of the subtree
        - The maximum and average q-error of all nodes in the subtree (including the current node)
        - A usage count of all physical operators used in the subtree (including the current node)

        Returns
        -------
        dict[str, object]
            The node summary
        """
        all_nodes = list(self.iternodes())
        summary: dict[str, object] = {}
        summary["estimated_card"] = round(self.estimated_cardinality, 3)
        summary["actual_card"] = round(self.true_cardinality, 3)
        summary["estimated_cost"] = round(self.cost, 3)
        summary["c_out"] = self.total_processed_rows()
        summary["max_qerror"] = round(max(node.qerror() for node in all_nodes), 3)
        summary["avg_qerror"] = round(sum(node.qerror() for node in all_nodes) / len(all_nodes), 3)
        summary["phys_ops"] = collections.Counter(child.node_type for child in all_nodes)
        return summary

    def _explain_text(self) -> str:
        """Generates an EXPLAIN-like text representation of the current node.

        Returns
        -------
        str
            A textual description of the node
        """
        if self.is_analyze():
            exec_time = round(self.execution_time, 3)
            true_card = round(self.true_cardinality, 3)
            analyze_str = f" (execution time={exec_time} true cardinality={true_card})"
        else:
            analyze_str = ""

        if self.table and (not self.is_subplan_root and self.subplan_name):
            table_str = f" :: {self.table}, {self.subplan_name}"
        elif self.table:
            table_str = f" :: {self.table}" if self.table else ""
        elif not self.is_subplan_root and self.subplan_name:
            table_str = f" :: {self.subplan_name}"
        else:
            table_str = ""
        cost = round(self.cost, 3)
        estimated_card = round(self.estimated_cardinality, 3)
        plan_str = f" (cost={cost} estimated cardinality={estimated_card})"
        return "".join((self.node_type, table_str, plan_str, analyze_str))

    def __json__(self) -> dict:
        return vars(self)

    def __hash__(self) -> int:
        return hash((self.node_type, self.table, tuple(self.children)))

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.node_type == other.node_type and self.table == other.table
                and self.children == other.children)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        normalized_node_type = self.node_type.replace(" ", "")
        if self.table:
            return f"{normalized_node_type}({self.table.identifier()})"
        child_texts = ", ".join(str(child) for child in self.children)
        return f"{normalized_node_type}({child_texts})"
