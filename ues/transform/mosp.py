
import abc
import collections
import collections.abc
import copy
import re
import warnings
from typing import Iterator, List, Dict, Set, Any, Union, Tuple

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

    def predicates(self, *, include_where_clause: bool = True, include_join_on: bool = False,
                   recurse_subqueries: bool = False) -> "MospWhereClause":
        mosp_predicates = []
        if include_where_clause and self.where_clause():
            mosp_predicates.extend(util.enlist(self.where_clause()))
        if include_join_on:
            mosp_predicates.extend(join.join_predicate for join in self.joins())
        if recurse_subqueries:
            for join in [join for join in self.joins() if join.subquery]:
                mosp_predicates.extend(join.subquery.predicates(include_where_clause=include_where_clause,
                                                                include_join_on=include_join_on,
                                                                recurse_subqueries=recurse_subqueries))

        return MospWhereClause({"and": mosp_predicates}, alias_map=self._build_alias_map())

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

    def predicate(self) -> List["AbstractMospPredicate"]:
        return MospWhereClause.break_conjunction(self.join_predicate)

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
    "not_like": "NOT LIKE",
    "or": "OR",
    "and": "AND"
}

CompoundOperations = {
    "and", "or", "not"
}


class MospFilterMap(collections.abc.Mapping):
    def __init__(self, filter_predicates: List["AbstractMospPredicate"], *, alias_map: dict = None):
        self._predicates = filter_predicates
        self._filter_by_table = collections.defaultdict(list)
        for filter_predicate in filter_predicates:
            filtered_table = util.simplify(filter_predicate.collect_tables())
            self._filter_by_table[filtered_table].append(filter_predicate)

        self._merged_filters = []
        for filter_predicates in self._filter_by_table.values():
            self._merged_filters.append(MospCompoundPredicate.merge_and(filter_predicates, alias_map=alias_map))

    def __len__(self) -> int:
        return len(self._merged_filters)

    def __iter__(self) -> Iterator["AbstractMospPredicate"]:
        return self._merged_filters.__iter__()

    def __getitem__(self, key: db.TableRef) -> List["AbstractMospPredicate"]:
        return self._filter_by_table[key]


class MospJoinMap(collections.abc.Mapping):
    def __init__(self, join_predicates: List["AbstractMospPredicate"], *, alias_map: dict = None):
        self._predicates = join_predicates
        self._join_by_tables = collections.defaultdict(list)
        self._denormalized_join_by_tables = collections.defaultdict(lambda: collections.defaultdict(list))

        for join_predicate in join_predicates:
            join_tables = join_predicate.collect_tables()
            self._join_by_tables[frozenset(join_tables)].append(join_predicate)
            tab1, tab2 = join_tables
            self._denormalized_join_by_tables[tab1][tab2].append(join_predicate)
            self._denormalized_join_by_tables[tab2][tab1].append(join_predicate)

        self._merged_joins = []
        for join_predicates in self._join_by_tables.values():
            self._merged_joins.append(MospCompoundPredicate.merge_and(join_predicates, alias_map=alias_map))

    def contents(self) -> dict:
        return copy.deepcopy(self._denormalized_join_by_tables)

    def __len__(self) -> int:
        return len(self._merged_joins)

    def __iter__(self) -> Iterator["AbstractMospPredicate"]:
        return self._merged_joins.__iter__()

    def __getitem__(self, key: Union[db.TableRef, Tuple[db.TableRef, db.TableRef]]) -> List["AbstractMospPredicate"]:
        if isinstance(key, db.TableRef):
            return self._join_by_tables[key]
        tab1, tab2 = key
        return self._denormalized_join_by_tables[tab1][tab2]


class MospPredicateMap:
    def __init__(self, predicates: List["AbstractMospPredicate"], *, alias_map: dict = None):
        self.alias_map = alias_map

        filter_predicates = [pred for pred in predicates if pred.is_filter()]
        join_predicates = [pred for pred in predicates if pred.is_join()]

        self._join_map = MospJoinMap(join_predicates, alias_map=alias_map)
        self._filter_map = MospFilterMap(filter_predicates, alias_map=alias_map)

    def _get_filters(self) -> MospFilterMap:
        return self._filter_map

    def _get_joins(self) -> MospJoinMap:
        return self._join_map

    filters: MospFilterMap = property(_get_filters)
    joins: MospJoinMap = property(_get_joins)

    def __getitem__(self, key: Tuple[str, Union[db.TableRef, Tuple[db.TableRef]]]) -> List["AbstractMospPredicate"]:
        predicate_type, predicate_key = key
        if isinstance(predicate_type, db.TableRef):
            return self._join_map[(predicate_type, predicate_type)]
        elif predicate_type == "filter":
            return self._filter_map[predicate_key]
        elif predicate_type == "join":
            return self._join_map[predicate_key]
        else:
            raise ValueError("Unknown predicate type: {}".format(predicate_type))


class MospWhereClause:
    def __init__(self, mosp_data, *, alias_map: dict = None):
        self.mosp_data = mosp_data
        self.alias_map = alias_map

    def break_conjunction(self) -> List["AbstractMospPredicate"]:
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

    def predicate_map(self) -> MospPredicateMap:
        return MospPredicateMap(self.break_conjunction(), alias_map=self.alias_map)

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
        """Checks, whether this predicate is a conjunction/disjunction/negation of base predicates."""
        return NotImplemented

    def is_base(self) -> bool:
        """Checks, whether this predicate is a base predicate i.e. not a conjunction/disjunction/negation."""
        return not self.is_compound()

    @abc.abstractmethod
    def is_join(self) -> bool:
        """Checks, whether this predicate describes a join between two tables."""
        return NotImplemented

    def is_filter(self) -> bool:
        """Checks, whether this predicate is a filter on a base table rather than a join of base tables."""
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

    def base_predicates(self) -> List["MospBasePredicate"]:
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
            actual_predicate = MospCompoundPredicate.parse(mosp_data[operation], alias_map=alias_map)
            return MospCompoundPredicate(operation, actual_predicate, alias_map=alias_map)
        elif operation == "or" or operation == "and":
            child_predicates = [MospCompoundPredicate.parse(child, alias_map=alias_map)
                                for child in mosp_data[operation]]
            return MospCompoundPredicate(operation, child_predicates, alias_map=alias_map)
        else:
            raise ValueError("Unknown compound predicate: {}".format(mosp_data))

    @staticmethod
    def merge_and(predicates: List[AbstractMospPredicate], *, alias_map: dict = None) -> AbstractMospPredicate:
        flattened_predicates: List[AbstractMospPredicate] = []
        for predicate in predicates:
            if predicate.is_compound() and predicate.operation == "and":
                flattened_predicates.extend(predicate.children)
            else:
                flattened_predicates.append(predicate)
        return MospCompoundPredicate("and", flattened_predicates, alias_map=alias_map)

    def __init__(self, operator, children: List[AbstractMospPredicate], *, alias_map: dict = None):
        mosp_data = {operator: [child.mosp_data for child in children]}
        super().__init__(mosp_data, alias_map)

        self.operation = operator
        self.children: List[AbstractMospPredicate] = util.enlist(children)
        self.negated = operator == "not"

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self.children)

    def collect_attributes(self) -> Set[db.AttributeRef]:
        return set(util.flatten([child.collect_attributes() for child in self.children], flatten_set=True))

    def join_partner(self, table: db.TableRef) -> Union[db.AttributeRef, Set[db.AttributeRef]]:
        if not self.is_join():
            raise ValueError("Not a join predicate")
        partners = util.flatten([child.join_partner(table) for child in self.children], flatten_set=True)
        return set(partners)

    def base_predicates(self) -> List["MospBasePredicate"]:
        predicates = []
        for child in self.children:
            if child.is_base():
                predicates.append(child)
            else:
                predicates.extend(child.base_predicates())
        return predicates

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
        return f" {op_str} ".join(str(child) if child.is_base() else f"({child})" for child in self.children)


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
        if self.is_filter():
            return set([self.left_attribute])
        return set([self.left_attribute, self.right_attribute])

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

    def literal_value(self) -> Any:
        if self.is_join():
            raise ValueError("Join predicate has no literal value")
        return self._unwrap_literal()

    def base_predicates(self) -> List["MospBasePredicate"]:
        return [self]

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

    def _unwrap_literal(self):
        if not isinstance(self.mosp_right, dict) or "literal" not in self.mosp_right:
            return None
        return self.mosp_right["literal"]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.operation == "exists":
            return self.mosp_left + " IS NOT NULL"
        elif self.operation == "missing":
            return self.mosp_left + " IS NULL"

        op_str = _OperationPrinting.get(self.operation, self.operation)
        if self.is_join():
            return f"{self.left_attribute} {op_str} {self.right_attribute}"

        right_is_str_value = (not isinstance(self.mosp_right, list)
                              and not util.represents_number(self._unwrap_literal()))
        if self.is_filter() and right_is_str_value:
            right = f"'{self._unwrap_literal()}'"
        else:
            right = self._unwrap_literal()

        return f"{self.left_attribute} {op_str} {right}"


def parse(query):
    return mosp.parse(query)


def format(query):
    return mosp.format(query)
