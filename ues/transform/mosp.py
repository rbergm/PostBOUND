
from typing import Any, List, Union, Tuple
import warnings

import mo_sql_parsing as mosp

from transform import db, util


def extract_tableref(mosp_data) -> db.TableRef:
    return db.TableRef(mosp_data.get("value", ""), mosp_data.get("name", ""))


class MospQuery:
    """Provides accessors to work more comfortably with MOSP parse trees."""
    @staticmethod
    def parse(query: str) -> "MospQuery":
        return MospQuery(mosp.parse(query))

    def __init__(self, mosp_data):
        self.query = mosp_data

    def from_clause(self):
        return self.query["from"]

    def base_table(self) -> "db.TableRef":
        tab = next(tab for tab in self.from_clause() if "value" in tab)
        return db.TableRef(tab["value"], tab["name"])

    def collect_tables(self) -> List["db.TableRef"]:
        tables = [self.base_table()]
        for join in self.joins():
            tables.extend(join.collect_tables())
        return tables

    def joins(self, simplify=False) -> List["MospJoin"]:
        joins = [MospJoin(tab) for tab in self.from_clause() if "join" in tab]
        if simplify and len(joins) == 1:
            return joins[0]
        else:
            return joins

    def subqueries(self, simplify=False) -> List["MospJoin"]:
        subqueries = [sq for sq in self.joins() if sq.is_subquery()]
        if simplify and len(subqueries) == 1:
            return subqueries[0]
        else:
            return subqueries

    def text(self) -> str:
        return str(self)

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

    def name(self) -> str:
        return self.join_data["name"]

    def collect_tables(self) -> List["db.TableRef"]:
        return self.subquery.collect_tables() if self.is_subquery() else [self.base_table()]

    def __hash__(self) -> int:
        return hash(frozenset(self.collect_tables()))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{extract_tableref(self.join_data)} ON {self.join_predicate}"


_OperationPrinting = {
    "eq": "=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
    "like": "LIKE"
}

CompoundOperations = {
    "and", "or", "not"
}


class MospPredicate:
    @staticmethod
    def break_compound(mosp_data) -> List["MospPredicate"]:
        operation = util.dict_key(mosp_data)
        if operation not in CompoundOperations:
            return MospPredicate(mosp_data)
        return [MospPredicate.break_compound(sub_predicate) for sub_predicate in mosp_data[operation]]

    def __init__(self, mosp_data):
        self.mosp_data = mosp_data
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

    def left_op(self) -> str:
        return self.left

    def left_table(self) -> str:
        return self._extract_table(self.left)

    def left_attribute(self) -> str:
        return self._extract_attribute(self.left)

    def right_op(self) -> str:
        if self.has_literal_op():
            return util.dict_value(self.right) if isinstance(self.right, dict) else self.right
        return self.right

    def right_table(self) -> str:
        return None if self.has_literal_op() else self._extract_table(self.right)

    def right_attribute(self) -> str:
        return None if self.has_literal_op() else self._extract_attribute(self.right)

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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        op_str = _OperationPrinting.get(self.operation, self.operation)
        right = self.right_op()
        return f"{self.left} {op_str} {right}"


def parse(query):
    return mosp.parse(query)


def format(query):
    return mosp.format(query)
