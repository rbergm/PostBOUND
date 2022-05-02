
import enum
import re
import warnings
from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
import psycopg2

from transform import mosp, util


class TableRef:
    @staticmethod
    def virtual(alias: str) -> "TableRef":
        return TableRef(None, alias, True)

    def __init__(self, full_name: str, alias: str, virtual: bool = False):
        self.full_name = full_name
        self.alias = alias
        self.is_virtual = virtual

    def has_attr(self, attr_name) -> bool:
        if not isinstance(attr_name, str):
            warnings.warn("Treating non-string attribute as false: " + str(attr_name))
            return False

        table_qualifier = self.alias + "."
        return attr_name.startswith(table_qualifier)

    def bind_attribute(self, attr_name) -> str:
        return f"{self.alias}.{attr_name}"

    def to_mosp(self):
        if self.is_virtual:
            raise ValueError("Can not convert virtual tables")
        return {"value": self.full_name, "name": self.alias}

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_virtual:
            return f"{self.alias} (virtual)"
        return f"{self.full_name} AS {self.alias}"


@dataclass
class AttributeRef:
    src_table: TableRef
    attribute: str

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.src_table.alias}.{self.attribute}"


class DBSchema:
    def __init__(self, cursor: "psycopg2.cursor"):
        self.cursor = cursor

    def lookup_attribute(self, attribute_name: str, candidate_tables: List[TableRef]):
        for table in [tab for tab in candidate_tables if not tab.is_virtual]:
            columns = self._fetch_columns(table.full_name)
            if attribute_name in columns:
                return table
        raise KeyError(f"Attribute not found: {attribute_name} in candidates {candidate_tables}")

    def _fetch_columns(self, table_name):
        base_query = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        self.cursor.execute(base_query, (table_name,))
        result_set = self.cursor.fetchall()
        return [col[0] for col in result_set]


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
    def __init__(self, node: "QueryNode", *, join_pred: str = "", filter_pred: str = "",
                 source_table: str = "", alias_name: str = "", index_name: str = "",
                 exec_time: np.double = np.nan, proc_rows: np.double = np.nan, planned_rows: np.double = np.nan,
                 children: List = None, subquery: bool = False):
        self.node = node
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

    def is_subquery(self):
        return self.subquery

    def traverse(self, fn):
        fn(self)
        for child in self.children:
            child.traverse(fn)

    def pretty_print(self, *, indent=0):
        indent_str = " " * indent
        if indent:
            indent_str += "<- "

        if self.node.is_join():
            node_label = f"{self.node.value} {self.join_pred}" if self.join_pred else self.node.value
        elif self.node.is_scan():
            node_label = f"{self.node.value} :: {self.source_table}"
        else:
            node_label = self.node.value

        if self.is_subquery():
            node_label = "[SQ] " + node_label
        node_label = indent_str + node_label + "\n"

        child_labels = []
        for child in self.children:
            child_labels.append(child.pretty_print(indent=indent+2))
        child_content = "".join(child_labels)
        return node_label + child_content

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.node.is_join():
            node_label = f"{self.node.value} {self.join_pred}" if self.join_pred else self.node.value
        elif self.node.is_scan():
            node_label = f"{self.node.value} :: {self.source_table}"
        else:
            node_label = self.node.value
        return f"{node_label} <- {self.children}" if self.children else node_label


def _simplify_plan_tree(plans: List[Any]) -> Union[Any, List[Any]]:
    while isinstance(plans, list) and len(plans) == 1:
        plans = plans[0]
    return plans


EXPLAIN_PREDICATE_FORMAT = re.compile(r"\(?(?P<left>[\w\.]+) (?P<op>[<>=!]+) (?P<right>[\w\.]+)\)?")


def _matches_any_predicate(explain_filter_needle: str, mosp_predicate_haystack: List[Any]) -> bool:
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
            return False

        reflexive_operator = op in ["=", "!=", "<>"]
        direct_operand_match = candidate.left_op() == left and candidate.right_op() == right
        if reflexive_operator:
            reversed_operand_match = candidate.right_op() == left and candidate.left_op() == right
        else:
            reversed_operand_match = False
        operands_match = direct_operand_match or reversed_operand_match

        if operands_match:
            return True

    return False


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

        return PlanNode(node, join_pred=join_pred, filter_pred=filter_pred,
                        exec_time=exec_time, proc_rows=proc_rows, planned_rows=planned_rows,
                        children=children, subquery=is_subquery)
    elif QueryNode.is_scan_node(node_type):
        node = QueryNode.parse(node_type)
        join_pred = plan.get("Index Cond", "")

        source_tab = plan["Relation Name"]
        index_name = plan.get("Index Name", "")
        alias = plan.get("Alias", "")

        return PlanNode(node, join_pred=join_pred, filter_pred=filter_pred,
                        source_table=source_tab, alias_name=alias, index_name=index_name,
                        exec_time=exec_time, proc_rows=proc_rows, planned_rows=planned_rows)
    else:
        warnings.warn("Unknown node type: {}".format(node_type))
        parsed_children = [parse_explain_analyze(orig_query, child_plan, with_subqueries=with_subqueries)
                           for child_plan in plan.get("Plans", [])]
        return _simplify_plan_tree(parsed_children)
