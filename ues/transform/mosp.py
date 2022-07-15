
import abc
from ast import alias
import re
import warnings
from typing import Any, List, Dict, Set, Union, Tuple

import mo_sql_parsing as mosp

from transform import db, util


def extract_tableref(mosp_data) -> db.TableRef:
    return db.TableRef(mosp_data.get("value", ""), mosp_data.get("name", ""))


def tableref_to_mosp(table: db.TableRef) -> Dict[str, str]:
    return {"value": table.full_name, "name": table.alias}


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

    def select_clause(self):
        return self.query["select"]

    def from_clause(self):
        return self.query.get("from", [])

    def where_clause(self):
        return self.query.get("where", {})

    def base_table(self) -> "db.TableRef":
        tab = next(tab for tab in self.from_clause() if "value" in tab)
        return db.TableRef(tab["value"], tab["name"])

    def collect_tables(self) -> List["db.TableRef"]:
        tables = [db.TableRef(tab["value"], tab["name"]) for tab in util.enlist(self.from_clause()) if "value" in tab]
        for join in self.joins():
            tables.extend(join.collect_tables())
        return tables

    def projection(self) -> "MospProjection":
        return MospProjection(self.select_clause(), table_alias_map=self._build_alias_map())

    def joins(self, simplify=False) -> List["MospJoin"]:
        joins = [MospJoin(tab) for tab in self.from_clause() if "join" in tab]
        if simplify and len(joins) == 1:
            return joins[0]
        else:
            return joins

    def parse_where_clause(self):
        return MospWhereClause(self.where_clause(), alias_map=self._build_alias_map())

    # def predicates(self, *,
    #                include_joins: bool = False) -> List[Union["MospPredicate", "CompoundMospFilterPredicate"]]:
    #     if include_joins:
    #         join_predicates = []
    #         for join in self.joins():
    #             if join.is_subquery():
    #                 join_predicates.extend(join.subquery.predicates(include_joins=True))
    #             join_predicates.extend(MospPredicate.break_compound(join.join_predicate))
    #         return join_predicates + self.predicates(include_joins=False)

    #     if not self.where_clause():
    #         return []

    #     return util.enlist(CompoundMospFilterPredicate.parse(self.where_clause(), skip_initial_level=True,
    #                                                          alias_map=self._build_alias_map()))

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

    def join_path(self) -> str:
        path = [f"({join.subquery.join_path()})" if join.is_subquery() else str(join.base_table())
                for join in self.joins()]
        path.insert(0, str(self.base_table()))
        return " ⋈ ".join(path)

    def count_result_tuples(self) -> int:
        count_query = dict(self.query)
        count_query["select"] = {"value": {"count": "*"}}
        query_str = mosp.format(count_query)
        n_tuples = db.DBSchema.get_instance().execute_query(query_str)
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


class MospProjection:
    def __init__(self, mosp_data, *, table_alias_map):
        self.mosp_data = mosp_data
        self.table_alias_map = table_alias_map
        self.alias_map = dict()
        self.attributes = dict()

        self._inflate_alias_map(self.mosp_data)

    def resolve(self, attribute_alias: str) -> db.AttributeRef:
        if attribute_alias in self.alias_map:
            return self.alias_map[attribute_alias]
        return self.attributes[attribute_alias]

    def _inflate_alias_map(self, mosp_data):
        if isinstance(mosp_data, list):
            for mosp_attribute in mosp_data:
                self._inflate_alias_map(mosp_attribute)
        elif isinstance(mosp_data, dict):
            attribute = mosp_data["value"]
            parsed_attribute = db.AttributeRef.parse(attribute, alias_map=self.table_alias_map)
            alias = mosp_data["name"]
            self.alias_map[alias] = parsed_attribute
            self.attributes[attribute] = parsed_attribute
        elif isinstance(mosp_data, str):
            parsed_attribute = db.AttributeRef.parse(mosp_data, alias_map=self.table_alias_map)
            self.attributes[mosp_data] = parsed_attribute
        else:
            warnings.warn("Unknown attribute structure: {}".format(mosp_data))

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"Attributes: {self.attributes} || Aliases: {self.alias_map}"


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

    def parse_all_predicates(self):
        return MospPredicate.break_compound(self.join_predicate)

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


class MospWhereClause:
    def __init__(self, mosp_data, *, alias_map: dict = None):
        self.mosp_data = mosp_data
        self.alias_map = alias_map

    def break_conjunction(self) -> list:
        if not isinstance(self.mosp_data, dict):
            raise ValueError("Unknown predicate format: {}".format(self.mosp_data))
        operation = util.dict_key(self.mosp_data)
        if operation == "or" or operation == "not":
            parsed = MospCompoundPredicate.parse(self.mosp_data, alias_map=self.alias_map)
            if not parsed.is_filter():
                raise ValueError("Where clause cannot contain disjunctions or negations of "
                                 "join predicates: {}".format(self.mosp_data))
            return [parsed]
        elif operation != "and":
            return [MospBasePredicate(self.mosp_data, alias_map=self.alias_map)]

        # at this point we are sure that we indeed received a conjunction of some kind of predicates and we can
        # continue the parsing process
        mosp_predicates = self.mosp_data[operation]
        parsed_predicates = []

        for predicate in mosp_predicates:
            predicate_op = util.dict_key(predicate)
            if AbstractMospPredicate.is_compound_operation(predicate_op):
                parsed_predicates.extend(self._parse_compound_predicate(predicate))
            else:
                parsed_predicates.append(MospBasePredicate(predicate, alias_map=self.alias_map))

        return parsed_predicates

    def _parse_compound_predicate(self, predicate) -> list:
        operation = util.dict_key(predicate)

        if not AbstractMospPredicate.is_compound_operation(operation):
            return [MospBasePredicate(predicate, alias_map=self.alias_map)]

        if operation == "and":
            parsed_predicates = []
            for sub_predicate in predicate[operation]:
                parsed_predicates.extend(self._parse_compound_predicate(sub_predicate))
            return parsed_predicates

        return [MospCompoundPredicate.parse(predicate, alias_map=self.alias_map)]


# TODO: in the end, this should be renamed to MospPredicate. Currently we still need the Abstract prefix to ensure
# our wrecking-ball refactoring actually works and does not reuse this class accidentally
class AbstractMospPredicate(abc.ABC):
    @staticmethod
    def is_compound_operation(operation: str) -> bool:
        return operation in CompoundOperations

    def __init__(self, mosp_data: dict, alias_map: dict):
        self.mosp_data = mosp_data
        self.alias_map = alias_map

    @abc.abstractmethod
    def is_compound(self) -> bool:
        return NotImplemented

    def is_base(self) -> bool:
        return not self.is_compound()

    @abc.abstractmethod
    def is_join(self) -> bool:
        return NotImplemented

    def is_filter(self) -> bool:
        return not self.is_join()

    @abc.abstractmethod
    def collect_attributes(self) -> Set[db.AttributeRef]:
        """Provides all attributes that are part of any predicate."""
        return NotImplemented

    def collect_tables(self) -> Set[db.TableRef]:
        """Provides all tables that are part of any predicate."""
        return {attribute.table for attribute in self.collect_attributes()}

    def joins_table(self, table: db.TableRef) -> bool:
        """Checks, whether this predicate describes a join and one of the join partners is the given table."""
        return self.is_join() and table in {attribute.table for attribute in self.collect_attributes()}

    def attribute_of(self, table: db.TableRef) -> Union[db.AttributeRef, Set[db.AttributeRef]]:
        """Retrieves all attributes of the given table."""
        attributes = {attribute for attribute in self.collect_attributes() if attribute.table == table}
        if not attributes:
            raise ValueError("No attribute for table found: {}".format(table))

        if self.is_compound():
            return attributes
        else:
            return util.pull_any(attributes)

    @abc.abstractmethod
    def join_partner(self, table: db.TableRef) -> Union[db.AttributeRef, Set[db.AttributeRef]]:
        """Retrieves the attributes that are joined with the given table.

        This assumes that this predicate actually is a join. If it's a base predicate, a single attribute will be
        returned, otherwise all matching attributes will be wrapped in a set.
        """
        return NotImplemented

    def estimate_result_rows(self, *, sampling: bool = False, sampling_pct: int = 25,
                             dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
        tables = self.collect_tables()
        if util.contains_multiple(tables):
            raise ValueError("Can only estimate filters with a single table")
        base_table = util.simplify(tables)
        count_query = self._expand_predicate_to_mosp_query(base_table, count_query=True)

        formatted_query = mosp.format(count_query)

        # the trick to support sampling is incredibly dirty but sadly mo_sql_parsing does not support formatting with
        # tablesample, yet
        if sampling:
            table_sampled = f"FROM {base_table.full_name} AS {base_table.alias} TABLESAMPLE bernoulli ({sampling_pct})"
            original_from = f"FROM {base_table.full_name} AS {base_table.alias}"
            formatted_query = formatted_query.replace(original_from, table_sampled)

        return dbs.pg_estimate(formatted_query)

    @abc.abstractmethod
    def rename_table(self, from_table: db.TableRef, to_table: db.TableRef, *,
                     prefix_attribute: bool = False) -> "AbstractMospPredicate":
        return NotImplemented

    def _expand_predicate_to_mosp_query(self, base_table: db.TableRef, *, count_query: bool = False):
        proj = {"count": "*"} if count_query else "*"
        return {
            "select": proj,
            "from": {"value": base_table.full_name, "name": base_table.alias},
            "where": self.mosp_data
        }


class MospCompoundPredicate(AbstractMospPredicate):
    @staticmethod
    def parse(mosp_data, *, alias_map: dict = None) -> AbstractMospPredicate:
        operation = util.dict_key(mosp_data)
        if not AbstractMospPredicate.is_compound_operation(operation):
            return MospBasePredicate(mosp_data, alias_map=alias_map)

        if operation == "not":
            actual_predicate = MospCompoundPredicate.parse(mosp_data[operation], mosp_data=mosp_data,
                                                           alias_map=alias_map)
            return MospCompoundPredicate(operation, actual_predicate, alias_map=alias_map)
        elif operation == "or" or operation == "and":
            child_predicates = [MospCompoundPredicate.parse(child, alias_map=alias_map)
                                for child in mosp_data[operation]]
            return MospCompoundPredicate(operation, child_predicates, mosp_data=mosp_data, alias_map=alias_map)
        else:
            raise ValueError("Unknown compound predicate: {}".format(mosp_data))

    def __init__(self, operator, children: List[AbstractMospPredicate], *,
                 mosp_data: dict = None, alias_map: dict = None):
        super().__init__(mosp_data, alias_map)

        self.operation = operator
        self.children: List[AbstractMospPredicate] = util.enlist(children)
        self.negated = operator == "not"

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self.children)

    def collect_attributes(self) -> Set[db.AttributeRef]:
        return Set(util.flatten([child.collect_attributes() for child in self.children], flatten_set=True))

    def join_partner(self, table: db.TableRef) -> Union[db.AttributeRef, Set[db.AttributeRef]]:
        if not self.is_join():
            raise ValueError("Not a join predicate")
        partners = util.flatten([child.join_partner(table) for child in self.children], flatten_set=True)
        return set(partners)

    def rename_table(self, from_table: db.TableRef, to_table: db.TableRef, *,
                     prefix_attribute: bool = False) -> "AbstractMospPredicate":
        renamed_children = [child.rename_table(from_table, to_table, prefix_attribute=prefix_attribute) for child
                            in self.children]
        renamed_mosp_data = [child.mosp_data for child in self.children]
        if self.negated:
            renamed_mosp_data = util.simplify(renamed_mosp_data)
        return MospCompoundPredicate(self.operation, renamed_children, mosp_data=renamed_mosp_data,
                                     alias_map=self.alias_map)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.negated:
            return "NOT (" + str(util.simplify(self.children)) + ")"
        op_str = _OperationPrinting.get(self.operation, self.operation)
        return f" {op_str} ".join(str(child) for child in self.children)


class MospBasePredicate(AbstractMospPredicate):
    def __init__(self, mosp_data, *, alias_map: dict = None):
        super().__init__(mosp_data, alias_map)

        if not isinstance(mosp_data, dict):
            raise ValueError("Unknown predicate type: {}".format(mosp_data))
        self.operation = util.dict_key(mosp_data)

        if AbstractMospPredicate.is_compound_operation(self.operation):
            raise ValueError("Predicate may not be compound: {}".format(mosp_data))

        if self.operation == "exists" or self.operation == "missing":
            # exists and missing have no right-hand side, we need to improvise
            self.mosp_left = mosp_data[self.operation]
            self.mosp_right = None
        else:
            self.mosp_left, *self.mosp_right = mosp_data[self.operation]
            self.mosp_right = util.simplify(self.mosp_right)

        if alias_map:
            # if we received an alias map, we can actually parse the attributes
            left_table_alias, left_attribute = self._break_attribute(self.mosp_left)
            self.left_attribute = db.AttributeRef(alias_map[left_table_alias], left_attribute)
            if self._right_is_attribute():
                right_table_alias, right_attribute = self._break_attribute(self.mosp_right)
                self.right_attribute = db.AttributeRef(alias_map[right_table_alias], right_attribute)
            else:
                self.right_attribute = None
        else:
            warnings.warn("No alias map!")
            self.left_attribute, self.right_attribute = None

    def is_compound(self) -> bool:
        return False

    def is_join(self) -> bool:
        self._assert_alias_map()
        return (isinstance(self.right_attribute, db.AttributeRef)
                and self.left_attribute.table != self.right_attribute.table)

    def collect_attributes(self) -> Set[db.AttributeRef]:
        self._assert_alias_map()
        return Set([self.left_attribute, self.right_attribute])

    def join_partner(self, table: db.TableRef) -> Union[db.AttributeRef, Set[db.AttributeRef]]:
        self._assert_alias_map()
        if self.is_filter():
            raise ValueError("Not a join predicate")
        if self.left_attribute.table == table:
            return self.right_attribute
        elif self.right_attribute.table == table:
            return self.left_attribute
        else:
            raise ValueError("Table is not joined")

    def rename_table(self, from_table: db.TableRef, to_table: db.TableRef, *,
                     prefix_attribute: bool = False) -> "AbstractMospPredicate":
        updated_mosp_data = dict(self.mosp_data)
        renamed_predicate = MospBasePredicate(updated_mosp_data, alias_map=self.alias_map)
        renaming_performed = False

        if self.left_attribute.table == from_table:
            attribute_name = (f"{from_table.alias}_{self.left_attribute.attribute}" if prefix_attribute
                              else self.left_attribute.attribute)
            renamed_attribute = db.AttributeRef(to_table, attribute_name)
            renamed_predicate.left_attribute = renamed_attribute
            renaming_performed = True

        if self.is_join() and self.right_attribute.table == from_table:
            attribute_name = (f"{from_table.alias}_{self.right_attribute.attribute}" if prefix_attribute
                              else self.right_attribute.attribute)
            renamed_attribute = db.AttributeRef(to_table, attribute_name)
            renamed_predicate.right_attribute = renamed_attribute
            renaming_performed = True

        if renaming_performed:
            self.alias_map[to_table.alias] = to_table  # both alias maps reference the same dict so this is sufficient

        return renamed_predicate

    def _break_attribute(self, raw_attribute: str) -> Tuple[str, str]:
        return raw_attribute.split(".")

    def _right_is_attribute(self) -> bool:
        # Just to make sure we are making the right decision, this test is a little more verbose than necessary.
        # Theoretically, applying only the last check should suffice. However, some weird cases might fail the test:
        # the predicate a.value IN (b.value, 42) will fail the test. But in such a case, the correct decision is up to
        # debate anyway and such query structures are currently also unsupported by our framework.
        if not self.mosp_right:
            return False
        if self.operation == "exists" or self.operation == "missing":
            return False
        return isinstance(self.mosp_right, str)

    def _assert_alias_map(self):
        if not self.alias_map:
            raise ValueError("No alias map given")

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.operation == "exists":
            return self.mosp_left + " IS NOT NULL"
        elif self.operation == "missing":
            return self.mosp_left + " IS NULL"

        op_str = _OperationPrinting.get(self.operation, self.operation)

        right_is_str_value = not isinstance(self.mosp_right, list) and not util.represents_number(self.mosp_right)
        if self.is_filter() and right_is_str_value:
            right = f"'{right}'"
        else:
            right = self.mosp_right

        return f"{self.left_attribute} {op_str} {right}"


# class MospPredicate:
#     @staticmethod
#     def break_compound(mosp_data: dict, *, alias_map: dict = None) -> List["MospPredicate"]:
#         operation = util.dict_key(mosp_data)
#         if operation not in CompoundOperations:
#             return MospPredicate(mosp_data, alias_map=alias_map)
#         return util.flatten([MospPredicate.break_compound(sub_predicate, alias_map=alias_map)
#                              for sub_predicate in mosp_data[operation]])

#     def __init__(self, mosp_data, *, alias_map=None):
#         self.mosp_data = mosp_data
#         self.alias_map = alias_map
#         if not isinstance(mosp_data, dict):
#             raise TypeError("Predicate type not supported: " + str(mosp_data))
#         self.operation = util.dict_key(mosp_data)
#         if self.operation in CompoundOperations:
#             raise ValueError("Predicate may not be compound: " + str(mosp_data))
#         self.left, *self.right = util.dict_value(mosp_data)
#         if len(self.right) == 1:
#             self.right = self.right[0]
#         elif self.operation == "exists" or self.operation == "missing":
#             self.left = self.left + "".join(self.right)
#             self.right = ""

#     def has_literal_op(self) -> bool:
#         if self.right is None or not isinstance(self.right, str):
#             return True
#         if self.operation == "like" or self.operation == "exists" or self.operation == "missing":
#             return True
#         # FIXME: this heuristic is incomplete: a predicate like a.date (25, b.date) fails the tests
#         return False

#     def is_join_predicate(self) -> bool:
#         return not self.has_literal_op()

#     def is_compound(self) -> bool:
#         return False

#     def is_join_predicate_over(self, table: db.TableRef) -> bool:
#         return self.is_join_predicate() and table in self.parse_tables()

#     def left_op(self) -> str:
#         return self.left

#     def left_table(self) -> str:
#         return self._extract_table(self.left)

#     def left_attribute(self) -> str:
#         return self._extract_attribute(self.left)

#     def parse_left_attribute(self) -> db.AttributeRef:
#         if not self.alias_map:
#             raise ValueError("Cannot parse without alias map")
#         table = self.alias_map[self.left_table()]
#         return db.AttributeRef(table, self.left_attribute())

#     def right_op(self) -> str:
#         if self.has_literal_op():
#             return util.dict_value(self.right) if isinstance(self.right, dict) else self.right
#         return self.right

#     def right_table(self) -> str:
#         return None if self.has_literal_op() else self._extract_table(self.right)

#     def right_attribute(self) -> str:
#         return None if self.has_literal_op() else self._extract_attribute(self.right)

#     def parse_right_attribute(self) -> db.AttributeRef:
#         if self.has_literal_op():
#             raise ValueError("Can only parse attributes, not literal values")
#         if not self.alias_map:
#             raise ValueError("Cannot parse without alias map")
#         table = self.alias_map[self.right_table()]
#         return db.AttributeRef(table, self.right_attribute())

#     def parse_attributes(self) -> Tuple[db.AttributeRef]:
#         if not self.is_join_predicate():
#             raise ValueError("Filter predicates only have a left attribute")
#         return self.parse_left_attribute(), self.parse_right_attribute()

#     def join_partner(self, table: db.TableRef) -> db.AttributeRef:
#         if not self.is_join_predicate():
#             raise ValueError("Filter predicates have no join partner")
#         left, right = self.parse_attributes()
#         if left.table == table:
#             return right
#         elif right.table == table:
#             return left
#         else:
#             raise ValueError("Not in predicate: {}".format(table))

#     def attribute_of(self, table: db.TableRef) -> db.AttributeRef:
#         left, right = self.parse_attributes()
#         if left.table == table:
#             return left
#         elif right.table == table:
#             return right
#         else:
#             raise ValueError("Not in predicate: {}".format(table))

#     def operands(self) -> Tuple[str, Union[str, Any]]:
#         return (self.left, self.right)

#     def tables(self) -> Tuple[str, Union[str, Any]]:
#         return (self.left_table(), self.right_table())

#     def parse_tables(self) -> Union[db.TableRef, Tuple[db.TableRef, db.TableRef]]:
#         left_table = self.parse_left_attribute().table
#         if self.is_join_predicate():
#             right_table = self.parse_right_attribute().table
#             return left_table, right_table
#         return left_table

#     def attributes(self) -> Tuple[str, Union[str, Any]]:
#         return (self.left_attribute(), self.right_attribute())

#     def pretty_operation(self) -> str:
#         return _OperationPrinting.get(self.operation, self.operation)

#     def to_mosp(self):
#         if self.operation == "between":
#             return {"between": [self.left, *self.right]}
#         elif self.operation == "exists" or self.operation == "missing":
#             return {self.operation: self.left}
#         return {self.operation: [self.left, self.right]}

#     def estimate_result_rows(self, *, sampling: bool = False, sampling_pct: int = 25,
#                              dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
#         if self.is_join_predicate():
#             raise ValueError("Can only estimate filters, not joins")

#         base_table = self.parse_left_attribute().table
#         mosp_query = _expand_predicate_to_mosp_query(base_table, self.mosp_data, count_query=sampling)
#         formatted_query = mosp.format(mosp_query)

#         # the trick to support sampling is incredibly dirty but sadly mo_sql_parsing does not support formatting with
#         # tablesample, yet
#         if sampling:
#             table_sampled = f"FROM {base_table.full_name} AS {base_table.alias} TABLESAMPLE bernoulli ({sampling_pct})"
#             original_from = f"FROM {base_table.full_name} AS {base_table.alias}"
#             formatted_query = formatted_query.replace(original_from, table_sampled)
#             result = dbs.execute_query(formatted_query)
#             return result

#         return dbs.pg_estimate(formatted_query)

#     def rename_table(self, from_table: db.TableRef, to_table: db.TableRef, *,
#                      prefix_attribute: bool = False) -> "MospPredicate":
#         updated_mosp_data = dict(self.mosp_data)
#         renamed_predicate = MospPredicate(updated_mosp_data, alias_map=self.alias_map)
#         renaming_performed = False

#         left_attribute = self.parse_left_attribute()
#         if left_attribute.table == from_table:
#             attribute_name = (f"{from_table.alias}_{left_attribute.attribute}" if prefix_attribute
#                               else left_attribute.attribute)
#             renamed_attribute = db.AttributeRef(to_table, attribute_name)
#             renamed_predicate.left = str(renamed_attribute)
#             renaming_performed = True

#         if self.is_join_predicate():
#             right_attribute = self.parse_right_attribute()
#             if right_attribute.table == from_table:
#                 attribute_name = (f"{from_table.alias}_{right_attribute.attribute}" if prefix_attribute
#                                   else right_attribute.attribute)
#                 renamed_attribute = db.AttributeRef(to_table, attribute_name)
#                 renamed_predicate.right = str(renamed_attribute)
#                 renaming_performed = True

#         renamed_predicate.mosp_data = renamed_predicate.to_mosp()

#         if renaming_performed:
#             self.alias_map[to_table.alias] = to_table

#         return renamed_predicate

#     def _extract_table(self, op: str) -> str:
#         return op.split(".")[0]

#     def _extract_attribute(self, op: str) -> str:
#         return ".".join(op.split(".")[1:])

#     def __hash__(self) -> int:
#         return hash(str(self))

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, MospPredicate):
#             return False
#         return str(self) == str(other)

#     def __repr__(self) -> str:
#         return str(self)

#     def __str__(self) -> str:
#         if self.operation == "exists":
#             return self.left + " IS NOT NULL"
#         elif self.operation == "missing":
#             return self.left + " IS NULL"

#         op_str = _OperationPrinting.get(self.operation, self.operation)
#         right = self.right_op()

#         right_is_str_value = not isinstance(right, list) and not util.represents_number(right)
#         if self.has_literal_op() and right_is_str_value:
#             right = f"'{right}'"

#         return f"{self.left} {op_str} {right}"


# class CompoundMospFilterPredicate:
#     @staticmethod
#     def is_compound_predicate(mosp_data) -> bool:
#         operation = util.dict_key(mosp_data)
#         return operation in CompoundOperations

#     @staticmethod
#     def parse(mosp_data, *, skip_initial_level: bool = False,
#               alias_map: dict = None) -> Union["MospPredicate", "CompoundMospFilterPredicate"]:
#         operation = util.dict_key(mosp_data)
#         if not CompoundMospFilterPredicate.is_compound_predicate(mosp_data):
#             return MospPredicate(mosp_data, alias_map=alias_map)
#         parsed_children = [CompoundMospFilterPredicate.parse(child, alias_map=alias_map)
#                            for child in mosp_data[operation]]
#         if skip_initial_level:
#             return parsed_children
#         return CompoundMospFilterPredicate(parsed_children, operation)

#     @staticmethod
#     def build_and_predicate(children: List[Union["MospPredicate", "CompoundMospFilterPredicate"]]
#                             ) -> Union["MospPredicate", "CompoundMospFilterPredicate"]:
#         if len(children) == 1:
#             return children[0]

#         return CompoundMospFilterPredicate(children, "and")

#     def __init__(self, children: List[Union["MospPredicate", "CompoundMospFilterPredicate"]], operation: str):
#         if not children:
#             raise ValueError("Empty child list")
#         for child in children:
#             if isinstance(child, MospPredicate) and child.is_join_predicate():
#                 raise ValueError("CompoundFILTERPredicate can only be built over filters, not join '{}'".format(child))
#         self.children = children
#         self.operation = operation

#     def is_compound(self) -> bool:
#         return True

#     def is_join_predicate(self) -> bool:
#         return False

#     def is_and_compound(self) -> bool:
#         return self.operation == "and"

#     def parse_left_attribute(self) -> db.AttributeRef:
#         """Returns the attribute which is being filtered.

#         This does not work, if the compound predicate is specified over multiple attributes (e.g. as in
#         `R.a = 1 AND R.b = 2`). If such a predicate is given, a `ValueError` will be raised.

#         In that case, this most likely hints to some wrong assumptions by the client about the query structure.
#         A fix is necessary on the client side.

#         See `collect_left_attributes` to query for all attributes in the predicate.
#         """
#         left_attributes = set(child.parse_left_attribute() for child in self.children)
#         if len(left_attributes) != 1:
#             raise ValueError("Left is undefined for compound predicates over multiple attributes.")
#         return list(left_attributes)[0]

#     def collect_left_attributes(self) -> Set[db.AttributeRef]:
#         """In contrast to `parse_left_attribute` this method returns all attributes that are being filtered."""
#         attributes = set()
#         for child in self.children:
#             if child.is_compound():
#                 attributes |= child.collect_left_attributes()
#             else:
#                 attributes.add(child.parse_left_attribute())
#         return attributes

#     def parse_tables(self) -> List[db.TableRef]:
#         return util.flatten([child.parse_tables() for child in self.children], recursive=True)

#     def rename_table(self, from_table: db.TableRef, to_table: db.TableRef, *,
#                      prefix_attribute: bool = False) -> "CompoundMospFilterPredicate":
#         renamed_children = [child.rename_table(from_table, to_table, prefix_attribute=prefix_attribute) for child
#                             in self.children]
#         return CompoundMospFilterPredicate(renamed_children, self.operation)

#     def base_table(self) -> db.TableRef:
#         # we know this is not a join so there may only be one base table
#         return self.children[0].parse_left_attribute().table

#     def to_mosp(self):
#         return {self.operation: [child.to_mosp() for child in self.children]}

#     def estimate_result_rows(self, *, sampling: bool = False, sampling_pct: int = 25,
#                              dbs: db.DBSchema = db.DBSchema.get_instance()) -> int:
#         base_table = self.base_table()
#         mosp_query = _expand_predicate_to_mosp_query(base_table, self.to_mosp(), count_query=sampling)
#         formatted_query = mosp.format(mosp_query)

#         # the trick to support sampling is incredibly dirty but sadly mo_sql_parsing does not support formatting with
#         # tablesample, yet
#         if sampling:
#             table_sampled = f"FROM {base_table.full_name} AS {base_table.alias} TABLESAMPLE bernoulli ({sampling_pct})"
#             original_from = f"FROM {base_table.full_name} AS {base_table.alias}"
#             formatted_query = formatted_query.replace(original_from, table_sampled)
#             result = dbs.execute_query(formatted_query)
#             return result

#         return dbs.pg_estimate(formatted_query)

#     def __hash__(self) -> int:
#         return hash(str(self))

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, CompoundMospFilterPredicate):
#             return str(self) == str(other)

#     def __repr__(self):
#         return str(self)

#     def __str__(self):
#         op_str = _OperationPrinting.get(self.operation, self.operation)
#         return f" {op_str} ".join(str(child) for child in self.children)


# def flatten_and_predicate(predicates: List[Union[MospPredicate, CompoundMospFilterPredicate]]
#                           ) -> List[Union[MospPredicate, CompoundMospFilterPredicate]]:
#     """Simplifies a predicate tree, pulling all nested AND statements to the top level if possible.

#     A predicate like `R.a = 1 AND R.b = 2 AND (R.c = 3 AND R.d = 4)` will be flattened to
#     `R.a = 1 AND R.b = 2 AND R.c = 3 AND R.d = 4`. However, this transformation does not apply if the compound
#     predicate contains another conjunction higher up. E.g. `R.a = 1 AND R.b = 2 OR (R.c = 3 AND R.d = 4)` will not
#     be transformed.
#     """
#     flattened_predicates = []
#     for pred in predicates:
#         if pred.is_compound() and pred.is_and_compound():
#             flattened_predicates.extend(flatten_and_predicate(pred.children))
#         else:
#             flattened_predicates.append(pred)
#     return flattened_predicates


def parse(query):
    return mosp.parse(query)


def format(query):
    return mosp.format(query)
