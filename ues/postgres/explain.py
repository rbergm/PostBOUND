
import enum
import json
import re
import warnings
from typing import Any, Callable, List, Tuple, Union

import numpy as np

from transform import db, mosp, util


class QueryNode(enum.Enum):
    @staticmethod
    def parse(node: str) -> "QueryNode":
        target_node = next(filter(lambda node_type: node_type.value in node, list(QueryNode)), None)
        if not target_node:
            raise ValueError("Unknown node type: {}".format(node))
        return target_node

    @staticmethod
    def is_join_node(node: str) -> bool:
        try:
            return QueryNode.parse(node).is_join()
        except ValueError:
            return False

    @staticmethod
    def is_scan_node(node: str) -> bool:
        try:
            return QueryNode.parse(node).is_scan()
        except ValueError:
            return False

    HASH_JOIN = "Hash Join"
    NESTED_LOOP = "Nested Loop"
    MERGE_JOIN = "Merge Join"
    SEQ_SCAN = "Seq Scan"
    BMP_IDX_SCAN = "Bitmap Index Scan"
    BMP_HEAP_SCAN = "Bitmap Heap Scan"
    BMP_SCAN = "Bitmap Scan (virtual)"
    IDX_ONLY_SCAN = "Index Only Scan"
    IDX_SCAN = "Index Scan"

    def is_join(self) -> bool:
        return self in [QueryNode.HASH_JOIN, QueryNode.NESTED_LOOP, QueryNode.MERGE_JOIN]

    def is_scan(self) -> bool:
        return self in [QueryNode.SEQ_SCAN, QueryNode.IDX_ONLY_SCAN, QueryNode.IDX_SCAN,
                        QueryNode.BMP_IDX_SCAN, QueryNode.BMP_SCAN]

    def is_idxscan(self) -> bool:
        return self == QueryNode.IDX_ONLY_SCAN or self == QueryNode.IDX_SCAN


class PlanNode:
    def __init__(self, node: "QueryNode", *, pruned: bool = False, join_pred: str = "", filter_pred: str = "",
                 source_table: str = "", alias_name: str = "", index_name: str = "",
                 exec_time: np.double = np.nan,
                 proc_rows: np.double = np.nan, planned_rows: np.double = np.nan, filtered_rows: np.double = np.nan,
                 children: List = None, subquery: bool = False, associated_query: "mosp.MospQuery" = None):
        self.node = node
        self.pruned = pruned

        self.subquery = subquery
        self.join_pred = join_pred
        self.filter_pred = filter_pred
        self.source_table = source_table
        self.alias_name = alias_name
        self.index_name = index_name

        self.exec_time = exec_time
        self.proc_rows = proc_rows
        self.planned_rows = planned_rows
        self.filtered_rows = filtered_rows

        self.parent, self.left, self.right = None, None, None
        self.children = children if children else []
        for child in self.children:
            child.parent = self
        if len(self.children) == 2:
            self.left, self.right = self.children

        self.associated_query = associated_query

    def is_subquery(self):
        return self.subquery

    def is_join(self):
        return self.node.is_join()

    def is_scan(self):
        return self.node.is_scan()

    def extract_subqueries(self, *, simplify=False) -> List["PlanNode"]:
        subqueries = []
        if self.is_subquery():
            subqueries.append(self)
        for child in self.children:
            subqueries.extend(child.extract_subqueries())

        if simplify and len(subqueries) == 1:
            return subqueries[0]
        else:
            return subqueries

    def lookup_subquery(self, join_filter: str) -> mosp.MospQuery:
        subqueries = [sq.subquery for sq in self.associated_query.subqueries()]
        return _lookup_join_predicate(join_filter, subqueries)

    def lookup_scan(self, table: db.TableRef) -> "PlanNode":
        if not self.is_scan():
            for child in self.children:
                lookup_res = child.lookup_scan(table)
                if lookup_res:
                    return lookup_res
            return None
        if self.source_table == table.full_name and self.alias_name == table.alias:
            return self
        else:
            return None

    def lookup_join(self, filter_cond: str) -> "PlanNode":
        if not self.is_join():
            return None
        if self.join_pred == filter_cond:
            return self
        for child in self.children:
            lookup_res = child.lookup_join(filter_cond)
            if lookup_res:
                return lookup_res
        return None

    def leaf_join(self) -> "PlanNode":
        leaf, __ = self._traverse_leaf_join()
        return leaf

    def depth(self, *, _curr_depth=1) -> int:
        if not self.is_join():
            return _curr_depth
        return max(child.depth(_curr_depth=_curr_depth+1) for child in self.children)

    def incoming_rows(self, *, fallback_live: bool = False,
                      fallback_live_idxscan: bool = False, fallback_live_seqscan: bool = False) -> int:
        """Counts the number of rows the operator receives.

        For basic (i.e. scan) nodes, the tuple count can optionally be retrieved from a live database.
        This is usefull for Index scans, since the EXPLAIN ANALYZE output usually does not provide this number if an
        Index filter condition is present.

        Parameters
        ----------
        fallback_live : bool, optional
            If set to `true`, will always retrieve the tuple count for scan operators from the live database. By
            default False.
        fallback_live_idxscan : bool, optional
            Use tuple count from live database for Index scans (and Index Only scans), by default False
        fallback_live_seqscan : bool, optional
            Use tuple count from live database for Sequential scans, by default False

        Returns
        -------
        int
            Number of tuples the operator received. For Index Scans, the size of the base table.
        """
        trigger_idx_fallback = fallback_live or (self.node.is_idxscan() and fallback_live_idxscan)
        trigger_seq_fallback = fallback_live or (self.node == QueryNode.SEQ_SCAN and fallback_live_seqscan)
        if trigger_idx_fallback or trigger_seq_fallback:
            dbschema = db.DBSchema.get_instance()
            return dbschema.count_tuples(db.TableRef(self.source_table))
        filter = self.filtered_rows if not np.isnan(self.filtered_rows) else 0
        return self.proc_rows + filter

    def traverse(self, fn):
        fn(self)
        for child in self.children:
            child.traverse(fn)

    def any_pruned(self, *, exclude_subqueries=False):
        if exclude_subqueries and self.subquery:
            return False

        if self.pruned:
            return True
        return any(child.any_pruned() for child in self.children)

    def inspect_node(self) -> str:
        node_label = f"{self.node.value}"
        if self.is_scan():
            node_label += f" :: {self.source_table} {self.alias_name}"
        if self.node.is_idxscan():
            node_label += f" ({self.index_name})"

        if self.pruned:
            node_label += " [PRUNED]"

        join_cond   = "  Join  : {}".format(self.join_pred if self.join_pred else "/")  # noqa: E221
        filter_cond = "  Filter: {}".format(self.filter_pred if self.filter_pred else "/")

        if not self.is_join():
            in_rows = "  Incoming Rows: {}".format(self.incoming_rows())
        else:
            in_rows =  "  Incoming Rows (left) : {}\n".format(self.left.proc_rows)  # noqa: E222
            in_rows += "  Incoming Rows (right): {}".format(self.right.proc_rows)

        filter_rows = "  Filtered Rows: {}".format(self.filtered_rows if self.filtered_rows else "/")
        out_rows    = "  Outgoing Rows: {}".format(self.proc_rows)  # noqa: E221

        runtime     = "  Execution Time: {} ms".format(self.exec_time)  # noqa: E221

        sep = "-" * max(len(line) for line in [node_label,
                                               join_cond, filter_cond,
                                               filter_rows, out_rows,
                                               runtime])

        return "\n".join([node_label, sep, join_cond, filter_cond, sep, in_rows, filter_rows, out_rows, sep, runtime])

    def pretty_print(self, *, include_filter=False, indent=0):
        indent_str = " " * indent
        if indent:
            indent_str += "<- "

        if self.is_join():
            node_label = f"{self.node.value} {self.join_pred}" if self.join_pred else self.node.value
        elif self.is_scan():
            node_label = f"{self.node.value} :: {self.source_table}"
            if include_filter and self.filter_pred:
                node_label += f" ({self.filter_pred})"
            if include_filter and self.index_name:
                node_label += f" (Idx: {self.index_name})"
        else:
            node_label = self.node.value

        if self.pruned:
            node_label = "[PRUNED] " + node_label

        if self.is_subquery():
            node_label = "[SQ] " + node_label
        node_label = indent_str + node_label + "\n"

        child_labels = []
        for child in self.children:
            child_labels.append(child.pretty_print(include_filter=include_filter, indent=indent+2))
        child_content = "".join(child_labels)
        return node_label + child_content

    def _traverse_leaf_join(self, *, curr_depth=0) -> Tuple["PlanNode", int]:
        if not self.left.is_join() and not self.right.is_join():
            return self, curr_depth

        if self.left.is_join():
            left_leaf, left_depth = self.left._traverse_leaf_join(curr_depth=curr_depth+1)
        else:
            left_leaf, left_depth = None, -1

        if self.right.is_join():
            right_leaf, right_depth = self.right._traverse_leaf_join(curr_depth=curr_depth+1)
        else:
            right_leaf, right_depth = None, -1

        return (left_leaf, left_depth) if left_depth > right_depth else (right_leaf, right_depth)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        node_name = f"~{self.node.value}~" if self.pruned else self.node.value
        if self.is_join():
            node_label = f"{node_name} {self.join_pred}" if self.join_pred else self.node.value
        elif self.is_scan():
            node_label = f"{node_name} :: {self.source_table}"
        else:
            node_label = node_name
        return f"{node_label} <- {self.children}" if self.children else node_label


def _simplify_plan_tree(plans: List[Any]) -> Union[Any, List[Any]]:
    while isinstance(plans, list) and len(plans) == 1:
        plans = plans[0]
    return plans


EXPLAIN_PREDICATE_FORMAT = re.compile(r"\(?(?P<left>[\w\.]+) (?P<op>[<>=!]+) (?P<right>[\w\.]+)\)?")


def _lookup_join_predicate(join_filter_needle: str, mosp_query_haystack: List[mosp.MospQuery]) -> mosp.MospQuery:
    for query in mosp_query_haystack:
        predicates = [join.predicate() for join in query.joins()]
        if _matches_any_predicate(join_filter_needle, predicates):
            return query
    return None


def _matches_any_predicate(explain_filter_needle: str, mosp_predicate_haystack: List[Any]) -> bool:
    match = _search_matching_predicate(explain_filter_needle, mosp_predicate_haystack)
    return True if match else False


def _search_matching_predicate(explain_filter_needle: str, mosp_predicate_haystack: List[Any]) -> Any:
    parsed_candidates = util.flatten([mosp.MospPredicate.break_compound(pred)
                                      for pred in mosp_predicate_haystack], recursive=True)
    explain_pred_match = EXPLAIN_PREDICATE_FORMAT.match(explain_filter_needle)
    if not explain_pred_match:
        raise ValueError("Unkown filter format: {}".format(explain_pred_match))
    left, op, right = explain_pred_match.groupdict().values()

    # We need to include some of the hacks from `compare_predicate_strs` here as well. In theory, we could delegate
    # our work to that function as well. However, we would need to convert the MOSP predicate back to a string.
    # This includes correctly formatting literal values such as Strings, having the correct escaping, etc. Therefore it
    # is probably easier to just duplicate the comparison logic for our special case here.

    for candidate in parsed_candidates:
        left_uneq_op = op in ["<>", "!="]
        right_uneq_op = candidate.pretty_operation() in ["<>", "!="]
        operations_match = (op == candidate.pretty_operation()) or (left_uneq_op and right_uneq_op)
        if not operations_match:
            continue

        reflexive_operator = op in ["=", "!=", "<>"]
        direct_operand_match = candidate.left_op() == left and candidate.right_op() == right
        if reflexive_operator:
            reversed_operand_match = candidate.right_op() == left and candidate.left_op() == right
        else:
            reversed_operand_match = False
        operands_match = direct_operand_match or reversed_operand_match

        if operands_match:
            return candidate

    return None


def compare_predicate_strs(first_pred: str, second_pred: str) -> bool:
    first_match = EXPLAIN_PREDICATE_FORMAT.match(first_pred)
    second_match = EXPLAIN_PREDICATE_FORMAT.match(second_pred)
    first_left, first_op, first_right = first_match.groupdict().values()
    second_left, second_op, second_right = second_match.groupdict().values()

    # if we compare operations, we need to be extra careful: a != b may also be expressed as a <> b !
    first_uneq_op = first_op in ["<>", "!="]
    second_uneq_op = second_op in ["<>", "!="]
    operations_match = (first_op == second_op) or (first_uneq_op and second_uneq_op)
    if not operations_match:
        return False

    # at this point we have established that both operations are equal, so we may use any of them to represent the
    # predicate operation
    reflexive_operator = first_op in ["=", "!=", "<>"]

    direct_operand_match = first_left == second_left and first_right == second_right
    if reflexive_operator:
        reversed_operand_match = first_left == second_right and first_right == second_left
    else:
        reversed_operand_match = False

    operands_match = direct_operand_match or reversed_operand_match

    return operands_match


def parse_explain_analyze(orig_query: "mosp.MospQuery", plan, *,
                          with_subqueries: bool = True, inner: bool = False) -> "PlanNode":
    # unwrap plan content if necessary
    if isinstance(plan, list):
        plan = plan[0]["Plan"]

    node_type = plan.get("Node Type", "")

    exec_time = plan["Actual Total Time"] / 1000  # convert ms -> s

    # The multiplication by Actual Loops is necessary b/c Actual Rows is actually a per-loop value.
    # The aggregated/total rows are obtained only by considering the number of loops.
    # see https://www.postgresql.org/docs/current/using-explain.html#USING-EXPLAIN-ANALYZE for more details.
    # FIXME: actually, this seems to not always be the case. In some (yet unknown) cases, Actual Rows already considers
    # the total number of rows and multiplication by Actual Loops over-estimates this number. One of these situations
    # is definitely related to inner nodes (which is already caught by the `inner` parameter). However, some other
    # cases still remain.
    loop_row_count_multiplier = 1 if inner else plan["Actual Loops"]
    proc_rows = plan["Actual Rows"] * loop_row_count_multiplier

    planned_rows = plan["Plan Rows"]
    filter_pred = plan.get("Filter", "")
    filtered_rows = (plan.get("Rows Removed by Filter", 0)
                     + plan.get("Rows Removed by Index Recheck", 0)
                     + plan.get("Rows Removed by Filter", 0)) * loop_row_count_multiplier

    if QueryNode.is_join_node(node_type):
        node = QueryNode.parse(node_type)

        left, right = plan["Plans"]
        left_parsed = parse_explain_analyze(orig_query, left, with_subqueries=with_subqueries)
        right_parsed = parse_explain_analyze(orig_query, right, with_subqueries=with_subqueries, inner=True)
        children = [_simplify_plan_tree(left_parsed), _simplify_plan_tree(right_parsed)]

        if node == QueryNode.HASH_JOIN:
            join_pred = plan["Hash Cond"]
        elif node == QueryNode.NESTED_LOOP:
            join_pred = plan.get("Join Filter", "")

            # Postgres sometimes does something interesting with NLJs: instead of actually executing an NLJ with a join
            # predicate, it will run the NLJ without any predicate. This obviously produces a cross product of the
            # incoming relations. However, one of these relations will be an Index Scan. The Index Condition of this
            # scan is set in a way to only retrieve tuples with a matching join partner in the other relation. In the
            # end, this once again emulates a full NLJ with better performance. However, this neat optimization breaks
            # our algorithm because we now need to consider the child nodes of the NLJ to re-construct the join
            # predicate that is actually applied in the NLJ.
            # I am unsure, whether this optimization only applies if the Index Scan is a direct child of the NLJ,
            # but for the sake of simplicity pulling the full predicate only works in that case.
            # Marking this as TODO for now.

            if not join_pred:
                scan_child = (left_parsed if left_parsed.is_scan() and left_parsed.join_pred
                              else right_parsed)
                scan_condition = scan_child.join_pred
                join_col, join_op, target_col = EXPLAIN_PREDICATE_FORMAT.match(scan_condition).groupdict().values()
                reconstructed_join_condition = f"({scan_child.alias_name}.{join_col} {join_op} {target_col})"
                join_pred = reconstructed_join_condition
                scan_child.join_pred = ""
        elif node == QueryNode.MERGE_JOIN:
            join_pred = plan.get("Merge Cond", "")
        else:
            warnings.warn("Could not determine join condition for join '{}'".format(node_type))
            join_pred = ""

        if with_subqueries and join_pred:
            subquery_joins = [sq.subquery.joins()[-1] for sq in orig_query.subqueries()]
            subquery_predicates = [join.predicate() for join in util.flatten(subquery_joins)]
            is_subquery = _matches_any_predicate(join_pred, subquery_predicates)
        else:
            is_subquery = False

        if plan["Actual Loops"] == 0:
            pruned = True
        else:
            pruned = False

        return PlanNode(node, pruned=pruned, join_pred=join_pred, filter_pred=filter_pred,
                        exec_time=exec_time,
                        proc_rows=proc_rows, planned_rows=planned_rows, filtered_rows=filtered_rows,
                        children=children, subquery=is_subquery, associated_query=orig_query)
    elif QueryNode.is_scan_node(node_type):
        node = QueryNode.parse(node_type)
        join_pred = plan.get("Index Cond", "")

        source_tab = plan.get("Relation Name", "")
        index_name = plan.get("Index Name", "")
        alias = plan.get("Alias", "")

        if plan["Actual Loops"] == 0:
            pruned = True
        else:
            pruned = False

        return PlanNode(node, pruned=pruned, join_pred=join_pred, filter_pred=filter_pred,
                        source_table=source_tab, alias_name=alias, index_name=index_name,
                        exec_time=exec_time,
                        proc_rows=proc_rows, planned_rows=planned_rows, filtered_rows=filtered_rows,
                        associated_query=orig_query)
    else:
        parsed_children = [parse_explain_analyze(orig_query, child_plan, with_subqueries=with_subqueries, inner=inner)
                           for child_plan in plan.get("Plans", [])]

        # Dirty workaround/fix: A bitmap scan has elements of both Index Scan and Sequential Scan. It is also split up
        # among two nodes: The Bitmap Heap Scan and the Bitmap Index Scan. The relevant attributes are scattered among
        # them. To integrate this two-step scan into our structure, we dissolve the Bitmap Heap Scan part and push
        # important attributes down to the Index Scan. Finally, the Index part is replaced by a virtual Bitmap Scan
        # node.
        if node_type == "Bitmap Heap Scan" and len(parsed_children) == 1:
            bmp_idx_scan = parsed_children[0]

            bmp_idx_scan.node = QueryNode.BMP_SCAN
            bmp_idx_scan.exec_time = exec_time

            bmp_idx_scan.source_table = plan.get("Relation Name", "")
            bmp_idx_scan.alias_name = plan.get("Alias", "")

            bmp_idx_scan.planned_rows = planned_rows
            bmp_idx_scan.filtered_rows += filtered_rows
            return bmp_idx_scan
        else:
            warnings.warn("Unknown node type: {}".format(node_type))
            return _simplify_plan_tree(parsed_children)


def prune_pg_plan(plan):
    if isinstance(plan, list):
        plan = plan[0]["Plan"]

    plan = dict(plan)

    # prune superfluous attributes
    superfluous_attrs = ["Parallel Aware", "Async Capable", "Startup Cost", "Total Cost", "Plan width"]
    for attr in superfluous_attrs:
        plan.pop(attr, None)

    node_type = plan.get("Node Type", "")
    if QueryNode.is_join_node(node_type):
        left, right = plan["Plans"]
        left_parsed = prune_pg_plan(left)
        right_parsed = prune_pg_plan(right)
        pruned_children = [_simplify_plan_tree(left_parsed), _simplify_plan_tree(right_parsed)]
        plan["Plans"] = pruned_children
        return plan
    elif QueryNode.is_scan_node(node_type):
        return plan
    else:
        child_nodes = plan.get("Plans", [])
        pruned_children = [prune_pg_plan(child) for child in child_nodes]
        return _simplify_plan_tree(pruned_children)


def traverse_pg_plan_until(plan: dict, condition: Callable[[dict], bool]) -> Union[dict, None]:
    """Performs a depth-first search on the query plan until the first node passes the condition test.

    Parameters
    ----------
    plan : dict
        the plan tree
    condition : Callable[[dict], bool]
        a function which accepts a plan node and checks whether it matches the termination condition

    Returns
    -------
    Union[dict, None]
        a plan node matching the condition, or None if no such node was found
    """
    if isinstance(plan, list):
        plan = plan[0]["Plan"]

    if condition(plan):
        return plan

    for child in plan.get("Plans", []):
        child_res = traverse_pg_plan_until(child, condition)
        if child_res:
            return child_res
    return None


def print_pg_plan(plan: dict):
    print(json.dumps(plan, indent=2))
