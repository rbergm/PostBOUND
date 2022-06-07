
import re
import warnings
from typing import Any, List, Dict, Union, Tuple

import mo_sql_parsing as mosp

from transform import db, util


def extract_tableref(mosp_data) -> db.TableRef:
    return db.TableRef(mosp_data.get("value", ""), mosp_data.get("name", ""))


_EXPLAIN_PREDICATE_FORMAT = re.compile(r"\(?(?P<predicate>(?P<left>[\w\.]+) (?P<op>[<>=!]+) (?P<right>[\w\.]+))\)?")
_REFLEXIVE_OPS = ["=", "!=", "<>"]


class MospQuery:
    """Provides accessors to work more comfortably with MOSP parse trees."""
    @staticmethod
    def parse(query: str) -> "MospQuery":
        return MospQuery(mosp.parse(query))

    def __init__(self, mosp_data):
        self.query = mosp_data
        self.alias_map = None

    def from_clause(self):
        return self.query["from"]

    def where_clause(self):
        return self.query["where"]

    def base_table(self) -> "db.TableRef":
        tab = next(tab for tab in self.from_clause() if "value" in tab)
        return db.TableRef(tab["value"], tab["name"])

    def collect_tables(self) -> List["db.TableRef"]:
        tables = [db.TableRef(tab["value"], tab["name"]) for tab in self.from_clause() if "value" in tab]
        for join in self.joins():
            tables.extend(join.collect_tables())
        return tables

    def joins(self, simplify=False) -> List["MospJoin"]:
        joins = [MospJoin(tab) for tab in self.from_clause() if "join" in tab]
        if simplify and len(joins) == 1:
            return joins[0]
        else:
            return joins

    def predicates(self) -> List[Union["MospPredicate", "CompoundMospFilterPredicate"]]:
        return CompoundMospFilterPredicate.parse(self.where_clause(), skip_initial_level=True,
                                                 alias_map=self._build_alias_map())

    def subqueries(self, simplify=False) -> List["MospJoin"]:
        subqueries = [sq for sq in self.joins() if sq.is_subquery()]
        if simplify and len(subqueries) == 1:
            return subqueries[0]
        else:
            return subqueries

    def text(self) -> str:
        return str(self)

    def lookup_subquery(self, join_predicate: str) -> Union["MospJoin", None]:
        predicate_match = _EXPLAIN_PREDICATE_FORMAT.match(join_predicate)
        if not predicate_match:
            raise ValueError("Malformed join predicate: '{}'".format(join_predicate))
        predicate_components = predicate_match.groupdict()
        pretty_predicate = predicate_components["predicate"]

        # make sure that we match both a = b, as well as b = a
        # FIXME: the hack to prevent non-reflexive predicates suddenly matching is incredibly dirty..
        left, op, right = predicate_components['right'], predicate_components['op'], predicate_components['left']
        if op in _REFLEXIVE_OPS:
            swapped_pretty_predicate = f"{left} {op} {right}"
        else:
            swapped_pretty_predicate = pretty_predicate

        for subquery in self.subqueries():
            subquery_join_predicates = [str(join.parse_predicate()) for join in subquery.subquery.joins()]
            if pretty_predicate in subquery_join_predicates or swapped_pretty_predicate in subquery_join_predicates:
                return subquery
        return None

    def count_result_tuples(self) -> int:
        count_query = dict(self.query)
        count_query["select"] = {"value": {"count": "*"}}
        query_str = mosp.format(count_query)
        n_tuples = db.DBSchema.get_instance().execute_query(query_str)[0][0]
        return n_tuples

    def _build_alias_map(self) -> Dict[str, db.TableRef]:
        if self.alias_map:
            return self.alias_map
        self.alias_map = {}
        for tab in self.collect_tables():
            self.alias_map[tab.alias] = tab
        return self.alias_map

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return mosp.format(self.query)


class MospJoin:
    @staticmethod
    def build(base_table: "db.TableRef", predicate) -> "MospJoin":
        mosp_data = {
            "join": {"value": base_table.full_name, "name": base_table.alias},
            "on": predicate
        }
        return MospJoin(mosp_data)

    def __init__(self, mosp_data):
        self.mosp_data = mosp_data
        self.join_data = self.mosp_data["join"]
        self.join_predicate = self.mosp_data["on"]

        join_value = self.mosp_data["join"]["value"]
        if isinstance(join_value, dict) and "select" in join_value:
            self.subquery = MospQuery(join_value)
        elif isinstance(join_value, str):
            self.subquery = False
        else:
            warnings.warn("Unknown join structure. Assuming not a subquery: " + join_value)
            self.subquery = False

    def base_table(self):
        if self.is_subquery():
            return self.subquery.base_table()
        else:
            return db.TableRef(self.join_data["value"], self.join_data["name"])

    def is_subquery(self):
        return self.subquery

    def predicate(self):
        return self.join_predicate

    def parse_predicate(self):
        all_predicates = MospPredicate.break_compound(self.join_predicate)
        if isinstance(all_predicates, MospPredicate):
            return all_predicates
        join_predicates = [pred for pred in all_predicates if not pred.has_literal_op()]
        if len(join_predicates) == 0:
            return None
        elif len(join_predicates) == 1:
            return join_predicates[0]
        else:
            return join_predicates

    def name(self) -> str:
        return self.join_data["name"]

    def collect_tables(self) -> List["db.TableRef"]:
        return self.subquery.collect_tables() if self.is_subquery() else [self.base_table()]

    def __hash__(self) -> int:
        return hash(frozenset(self.collect_tables()))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        join_body = f"({self.subquery})" if self.subquery else extract_tableref(self.join_data)
        return f"JOIN {join_body} ON {self.parse_predicate()}"


_OperationPrinting = {
    "eq": "=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "like": "LIKE",
    "or": "OR",
    "and": "AND"
}

CompoundOperations = {
    "and", "or", "not"
}


def _expand_predicate_to_mosp_query(base_table: db.TableRef, mosp_data):
    return {
        "select": "*",
        "from": {"value": base_table.full_name, "name": base_table.alias},
        "where": mosp_data
    }


class MospPredicate:
    @staticmethod
    def break_compound(mosp_data: dict, *, alias_map: dict = None) -> List["MospPredicate"]:
        operation = util.dict_key(mosp_data)
        if operation not in CompoundOperations:
            return MospPredicate(mosp_data, alias_map=alias_map)
        return util.flatten([MospPredicate.break_compound(sub_predicate, alias_map=alias_map)
                             for sub_predicate in mosp_data[operation]])

    def __init__(self, mosp_data, *, alias_map=None):
        self.mosp_data = mosp_data
        self.alias_map = alias_map
        if not isinstance(mosp_data, dict):
            raise TypeError("Predicate type not supported: " + str(mosp_data))
        self.operation = util.dict_key(mosp_data)
        if self.operation in CompoundOperations:
            raise ValueError("Predicate may not be compound: " + str(mosp_data))
        self.left, *self.right = util.dict_value(mosp_data)
        if len(self.right) == 1:
            self.right = self.right[0]
        elif self.operation == "exists" or self.operation == "missing":
            self.left = self.left + "".join(self.right)
            self.right = ""

    def has_literal_op(self) -> bool:
        if self.right is None or not isinstance(self.right, str):
            return True
        if self.operation == "like" or self.operation == "exists" or self.operation == "missing":
            return True
        # FIXME: this heuristic is incomplete: a predicate like a.date (25, b.date) fails the tests
        return False

    def is_join_predicate(self) -> bool:
        return not self.has_literal_op()

    def is_compound(self) -> bool:
        return False

    def left_op(self) -> str:
        return self.left

    def left_table(self) -> str:
        return self._extract_table(self.left)

    def left_attribute(self) -> str:
        return self._extract_attribute(self.left)

    def parse_left_attribute(self) -> db.AttributeRef:
        if not self.alias_map:
            raise ValueError("Cannot parse without alias map")
        table = self.alias_map[self.left_table()]
        return db.AttributeRef(table, self.left_attribute())

    def right_op(self) -> str:
        if self.has_literal_op():
            return util.dict_value(self.right) if isinstance(self.right, dict) else self.right
        return self.right

    def right_table(self) -> str:
        return None if self.has_literal_op() else self._extract_table(self.right)

    def right_attribute(self) -> str:
        return None if self.has_literal_op() else self._extract_attribute(self.right)

    def parse_right_attribute(self) -> db.AttributeRef:
        if self.has_literal_op():
            raise ValueError("Can only parse attributes, not literal values")
        if not self.alias_map:
            raise ValueError("Cannot parse without alias map")
        table = self.alias_map[self.right_table()]
        return db.AttributeRef(table, self.right_attribute())

    def operands(self) -> Tuple[str, Union[str, Any]]:
        return (self.left, self.right)

    def tables(self) -> Tuple[str, Union[str, Any]]:
        return (self.left_table(), self.right_table())

    def attributes(self) -> Tuple[str, Union[str, Any]]:
        return (self.left_attribute(), self.right_attribute())

    def pretty_operation(self) -> str:
        return _OperationPrinting.get(self.operation, self.operation)

    def to_mosp(self):
        if self.operation == "between":
            return {"between": [self.left, *self.right]}
        elif self.operation == "exists" or self.operation == "missing":
            return {self.operation: self.left}
        return {self.operation: [self.left, self.right]}

    def _extract_table(self, op: str) -> str:
        return op.split(".")[0]

    def _extract_attribute(self, op: str) -> str:
        return ".".join(op.split(".")[1:])

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MospPredicate):
            return False
        return str(self) == str(other)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.operation == "exists":
            return self.left + " IS NOT NULL"
        elif self.operation == "missing":
            return self.left + " IS NULL"

        op_str = _OperationPrinting.get(self.operation, self.operation)
        right = self.right_op()

        right_is_str_value = not isinstance(right, list) and not util.represents_number(right)
        if self.has_literal_op() and right_is_str_value:
            right = f"'{right}'"

        return f"{self.left} {op_str} {right}"


class CompoundMospFilterPredicate:
    @staticmethod
    def is_compound_predicate(mosp_data) -> bool:
        operation = util.dict_key(mosp_data)
        return operation in CompoundOperations

    @staticmethod
    def parse(mosp_data, *, skip_initial_level: bool = False,
              alias_map: dict = None) -> Union["MospPredicate", "CompoundMospFilterPredicate"]:
        operation = util.dict_key(mosp_data)
        if not CompoundMospFilterPredicate.is_compound_predicate(mosp_data):
            return MospPredicate(mosp_data, alias_map=alias_map)
        parsed_children = [CompoundMospFilterPredicate.parse(child, alias_map=alias_map)
                           for child in mosp_data[operation]]
        if skip_initial_level:
            return parsed_children
        return CompoundMospFilterPredicate(parsed_children, operation)

    @staticmethod
    def build_and_predicate(children: List[Union["MospPredicate", "CompoundMospFilterPredicate"]]
                            ) -> Union["MospPredicate", "CompoundMospFilterPredicate"]:
        if len(children) == 1:
            return children[0]

        return CompoundMospFilterPredicate(children, "and")

    def __init__(self, children: List[Union["MospPredicate", "CompoundMospFilterPredicate"]], operation: str):
        if not children:
            raise ValueError("Empty child list")
        for child in children:
            if isinstance(child, MospPredicate) and child.is_join_predicate():
                raise ValueError("CompoundFILTERPredicate can only be built over filters, not join '{}'".format(child))
        self.children = children
        self.operation = operation

    def is_compound(self) -> bool:
        return True

    def is_join_predicate(self) -> bool:
        return False

    def parse_left_attribute(self) -> db.AttributeRef:
        left_attributes = set(child.parse_left_attribute() for child in self.children)
        if len(left_attributes) != 1:
            raise ValueError("Left is undefined for compound predicates over multiple attributes.")
        return list(left_attributes)[0]

    def base_table(self) -> db.TableRef:
        # we know this is not a join so there may only be one base table
        return self.children[0].parse_left_attribute().table

    def to_mosp(self):
        return {self.operation: [child.to_mosp() for child in self.children]}

    def estimate_result_rows(self, *, dbs: db.DBSchema = None) -> int:
        # TODO: sampling variant
        dbs = db.DBSchema.get_instance() if not dbs else dbs
        mosp_query = _expand_predicate_to_mosp_query(self.base_table(), self.to_mosp())
        return dbs.pg_estimate(mosp.format(mosp_query))

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompoundMospFilterPredicate):
            return str(self) == str(other)

    def __repr__(self):
        return str(self)

    def __str__(self):
        op_str = _OperationPrinting.get(self.operation, self.operation)
        return f" {op_str} ".join(str(child) for child in self.children)


def parse(query):
    return mosp.parse(query)


def format(query):
    return mosp.format(query)
