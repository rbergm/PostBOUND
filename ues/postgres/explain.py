
import enum
import re
import warnings
from typing import Any, List, Union

import numpy as np

from transform import mosp, util


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
    IDX_ONLY_SCAN = "Index Only Scan"
    IDX_SCAN = "Index Scan"

    def is_join(self) -> bool:
        return self in [QueryNode.HASH_JOIN, QueryNode.NESTED_LOOP, QueryNode.MERGE_JOIN]

    def is_scan(self) -> bool:
        return self in [QueryNode.SEQ_SCAN, QueryNode.IDX_ONLY_SCAN, QueryNode.IDX_SCAN]


class PlanNode:
    def __init__(self, node: "QueryNode", *, pruned: bool = False, join_pred: str = "", filter_pred: str = "",
                 source_table: str = "", alias_name: str = "", index_name: str = "",
                 exec_time: np.double = np.nan, proc_rows: np.double = np.nan, planned_rows: np.double = np.nan,
                 children: List = None, subquery: bool = False, associated_query: "mosp.MospQuery" = None):
        self.node = node
        self.pruned = pruned

        self.subquery = subquery
        self.join_pred = join_pred
        self.filter_pred = filter_pred
        self.source_table = source_table
        self.alias_name = alias_name
        self.index_name = index_name

        self.exec_time = exec_time,
        self.proc_rows = proc_rows
        self.planned_rows = planned_rows

        self.parent, self.left, self.right = None, None, None
        self.children = children if children else []
        for child in self.children:
            child.parent = self
        if len(self.children) == 2:
            self.left, self.right = self.children

        self.associated_query = associated_query

    def is_subquery(self):
        return self.subquery

    def extract_subqueries(self):
        subqueries = []
        if self.is_subquery():
            subqueries.append(self)
        for child in self.children:
            subqueries.extend(child.extract_subqueries())
        return subqueries

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

    def pretty_print(self, *, include_filter=False, indent=0):
        indent_str = " " * indent
        if indent:
            indent_str += "<- "

        if self.node.is_join():
            node_label = f"{self.node.value} {self.join_pred}" if self.join_pred else self.node.value
        elif self.node.is_scan():
            node_label = f"{self.node.value} :: {self.source_table}"
            if include_filter and self.filter_pred:
                node_label += f" ({self.filter_pred})"
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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        node_name = f"~{self.node.value}~" if self.pruned else self.node.value
        if self.node.is_join():
            node_label = f"{node_name} {self.join_pred}" if self.join_pred else self.node.value
        elif self.node.is_scan():
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


def parse_explain_analyze(orig_query: "mosp.MospQuery", plan, *, with_subqueries=True) -> "PlanNode":
    # unwrap plan content if necessary
    if isinstance(plan, list):
        plan = plan[0]["Plan"]

    node_type = plan.get("Node Type", "")
    exec_time = plan["Actual Total Time"]
    proc_rows = plan["Actual Rows"]
    planned_rows = plan["Plan Rows"]
    filter_pred = plan.get("Filter", "")

    if QueryNode.is_join_node(node_type):
        node = QueryNode.parse(node_type)

        left, right = plan["Plans"]
        left_parsed = parse_explain_analyze(orig_query, left, with_subqueries=with_subqueries)
        right_parsed = parse_explain_analyze(orig_query, right, with_subqueries=with_subqueries)
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
                scan_child = (left_parsed if left_parsed.node == QueryNode.IDX_SCAN
                              or left_parsed.node == QueryNode.IDX_ONLY_SCAN
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
            subquery_joins = [sq.subquery.joins() for sq in orig_query.subqueries()]
            subquery_predicates = [join.predicate() for join in util.flatten(subquery_joins)]
            is_subquery = _matches_any_predicate(join_pred, subquery_predicates)
        else:
            is_subquery = False

        if float(plan.get("Actual Startup Time", "-1")) == 0 and float(plan.get("Actual Total Time", "-1")) == 0:
            pruned = True
        else:
            pruned = False

        return PlanNode(node, pruned=pruned, join_pred=join_pred, filter_pred=filter_pred,
                        exec_time=exec_time, proc_rows=proc_rows, planned_rows=planned_rows,
                        children=children, subquery=is_subquery, associated_query=orig_query)
    elif QueryNode.is_scan_node(node_type):
        node = QueryNode.parse(node_type)
        join_pred = plan.get("Index Cond", "")

        source_tab = plan["Relation Name"]
        index_name = plan.get("Index Name", "")
        alias = plan.get("Alias", "")

        if float(plan.get("Actual Startup Time", "-1")) == 0 and float(plan.get("Actual Total Time", "-1")) == 0:
            pruned = True
        else:
            pruned = False

        return PlanNode(node, pruned=pruned, join_pred=join_pred, filter_pred=filter_pred,
                        source_table=source_tab, alias_name=alias, index_name=index_name,
                        exec_time=exec_time, proc_rows=proc_rows, planned_rows=planned_rows,
                        associated_query=orig_query)
    else:
        warnings.warn("Unknown node type: {}".format(node_type))
        parsed_children = [parse_explain_analyze(orig_query, child_plan, with_subqueries=with_subqueries)
                           for child_plan in plan.get("Plans", [])]
        return _simplify_plan_tree(parsed_children)
