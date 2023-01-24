
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


# TODO: a very long-term refactoring should target the MospQuery class: instead of treating all queries the same,
# our code should distinguish between queries with implicit joins as part of their FROM and WHERE clauses such as
# SELECT * FROM R, S WHERE R.a = S.b and queries with explicit joins such as SELECT * FROM R JOIN S ON R.a = S.b
# In the current design, some methods of MospQuery only work for the first query type, whereas other methods only work
# for the second one and it is often unclear, which these are.

class MospQuery:
    """Provides accessors to work more comfortably with MOSP parse trees."""
    @staticmethod
    def parse(query: str) -> "MospQuery":
        return MospQuery(mosp.parse(query))

    def __init__(self, mosp_data):
        if not isinstance(mosp_data, dict):
            raise TypeError("Unexpected mosp data:", mosp_data)
        self.query: dict = mosp_data
        self.alias_map = None

    def select_clause(self):
        return self.query["select"]

    def from_clause(self):
        return self.query.get("from", [])

    def where_clause(self):
        return self.query.get("where", {})

    def is_ordered(self):
        return "orderby" in self.query

    def base_table(self) -> "db.TableRef":
        tab = next(tab for tab in self.from_clause() if "value" in tab)
        return db.TableRef(tab["value"], tab["name"])

    def collect_tables(self, *, _include_subquery_targets: bool = False) -> List["db.TableRef"]:
        tables = [db.TableRef(tab["value"], tab["name"]) for tab in util.enlist(self.from_clause()) if "value" in tab]
        for join in self.joins(_skip_alias_map=True):
            tables.extend(join.collect_tables())
            if _include_subquery_targets and join.is_subquery() and join.join_target_table:
                tables.append(db.TableRef.virtual(join.join_target_table))
        return tables

    def projection(self) -> "MospProjection":
        return MospProjection(self.select_clause(), table_alias_map=self._build_alias_map())

    def joins(self, simplify=False, *, _skip_alias_map: bool = False) -> List["MospJoin"]:
        alias_map = self._build_alias_map() if not _skip_alias_map else None
        joins = [MospJoin(tab, alias_map=alias_map) for tab in self.from_clause() if "join" in tab]
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
            subquery_join_predicates = [str(join.predicate().joins.as_and_clause())
                                        for join in subquery.subquery.joins()]
            if pretty_predicate in subquery_join_predicates or swapped_pretty_predicate in subquery_join_predicates:
                return subquery
        return None

    def join_path(self, short: bool = False) -> str:
        path = []
        for join in self.joins():
            if join.is_subquery():
                path.append(f"({join.subquery.join_path(short=short)})")
            else:
                path.append(join.base_table().alias if short else str(join.base_table()))

        if short:
            path.insert(0, self.base_table().alias)
        else:
            path.insert(0, str(self.base_table()))

        join_separator = " " if short else " ⋈ "
        return join_separator.join(path)

    def count_result_tuples(self, dbs: db.DBSchema = None) -> int:
        dbs = db.DBSchema.get_instance() if dbs is None else dbs
        count_query = self.as_count_star().query
        query_str = mosp.format(count_query)
        n_tuples = dbs.execute_query(query_str)
        return n_tuples

    def as_count_star(self) -> "MospQuery":
        """Returns a new query based on this query, that counts all tuples instead of the normal projection."""
        count_query = dict(self.query)
        count_query["select"] = {"value": {"count": "*"}}
        return MospQuery(count_query)

    def extract_fragment(self, tables: List[db.TableRef]) -> "MospQuery":
        """
        Generates a new query that contains exactly those parts of the original query that touched the given tables.

        The fragment includes tables in the FROM clause, as well as filter and join predicates that are completely
        contained in the given tables.

        As of now, the fragment will always be of the form SELECT * FROM ... WHERE ... and "advanced" features such as
        groupings are not retained.
        """
        tables = util.enlist(tables)
        fragment_data = {"select": "*"}
        fragment_data["from"] = util.simplify([{"value": tab.full_name, "name": tab.alias}
                                               for tab in self.collect_tables() if tab in tables])

        predicate_map = self.predicates(include_where_clause=True, include_join_on=True, recurse_subqueries=True)
        where_clause = []
        for filter_pred in predicate_map.filters:
            if any(filter_pred.contains_table(tab) for tab in tables):
                where_clause.append(filter_pred.mosp_data)
        for join_pred in predicate_map.joins:
            first_table, second_table = join_pred.collect_tables()
            if first_table in tables and second_table in tables:
                where_clause.append(join_pred.mosp_data)

        if len(where_clause) > 1:
            fragment_data["where"] = {"and": where_clause}
        elif len(where_clause) == 1:
            fragment_data["where"] = where_clause[0]

        return MospQuery(fragment_data)

    def _build_alias_map(self) -> Dict[str, db.TableRef]:
        if self.alias_map:
            return self.alias_map
        self.alias_map = {}
        for tab in self.collect_tables(_include_subquery_targets=True):
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
    def build(base_table: "db.TableRef", predicate: dict) -> "MospJoin":
        mosp_data = {
            "join": {"value": base_table.full_name, "name": base_table.alias},
            "on": predicate
        }
        return MospJoin(mosp_data)

    def __init__(self, mosp_data, *, alias_map: dict = None):
        self.mosp_data = mosp_data
        self.alias_map = alias_map

        self.join_data: dict = self.mosp_data["join"]
        self.join_predicate: dict = self.mosp_data["on"]
        self.join_target_table: str = self.join_data.get("name", "")

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

    def is_subquery(self) -> bool:
        return isinstance(self.subquery, MospQuery)

    def predicate(self) -> "MospWhereClause":
        return MospWhereClause(self.join_predicate, alias_map=self.alias_map)

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
        return f"JOIN {join_body} ON {self.predicate()}"


_OperationPrinting = {
    "eq": "=",
    "lt": "<",
    "le": "<=",
    "lte": "<=",
    "gt": ">",
    "ge": ">=",
    "gte": ">=",
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
        self.alias_map = alias_map

        self._predicates = filter_predicates
        self._filter_by_table = collections.defaultdict(list)
        for filter_predicate in filter_predicates:
            filtered_table = util.simplify(filter_predicate.collect_tables())
            self._filter_by_table[filtered_table].append(filter_predicate)

        self._merged_filters = []
        for filter_predicates in self._filter_by_table.values():
            self._merged_filters.append(MospCompoundPredicate.merge_and(filter_predicates, alias_map=alias_map))

    def as_and_clause(self) -> "AbstractMospPredicate":
        return MospCompoundPredicate.merge_and(self._merged_filters, alias_map=self.alias_map)

    def contents(self) -> Dict[db.TableRef, "AbstractMospPredicate"]:
        return {table: MospCompoundPredicate.merge_and(filter_predicates, alias_map=self.alias_map)
                for table, filter_predicates in self._filter_by_table.items()}

    def __len__(self) -> int:
        return len(self._merged_filters)

    def __iter__(self) -> Iterator["AbstractMospPredicate"]:
        return self._merged_filters.__iter__()

    def __contains__(self, item: db.TableRef) -> bool:
        return item in self._filter_by_table

    def __getitem__(self, key: db.TableRef) -> List["AbstractMospPredicate"]:
        return self._filter_by_table[key]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return " AND ".join([str(pred) for pred in self._merged_filters])


class MospJoinMap(collections.abc.Mapping):
    def __init__(self, join_predicates: List["AbstractMospPredicate"], *, alias_map: dict = None):
        self._predicates = join_predicates
        self._join_by_tables: Dict[db.TableRef, List[AbstractMospPredicate]] = collections.defaultdict(list)
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

    def as_and_clause(self) -> "AbstractMospPredicate":
        return MospCompoundPredicate.merge_and(self._merged_joins)

    def joins_tables(self, *tables: db.TableRef) -> bool:
        for tab in tables:
            if not any(partner in self._denormalized_join_by_tables[tab] for partner in tables if partner != tab):
                return False
        return True

    def __len__(self) -> int:
        return len(self._merged_joins)

    def __iter__(self) -> Iterator["AbstractMospPredicate"]:
        return self._merged_joins.__iter__()

    def __contains__(self, item: Union[db.TableRef, Tuple[db.TableRef, db.TableRef]]) -> bool:
        if isinstance(item, tuple):
            tab1, tab2 = item
            return tab1 in self._join_by_tables and tab2 in self._join_by_tables[tab1]
        return item in self._join_by_tables

    def __getitem__(self, key: Union[db.TableRef, Tuple[db.TableRef, db.TableRef]]) -> List["AbstractMospPredicate"]:
        if isinstance(key, db.TableRef):
            return self._join_by_tables[key]
        tab1, tab2 = key
        return self._denormalized_join_by_tables[tab1][tab2]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return " AND ".join([str(pred) for pred in self._merged_joins])


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

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        filters_str = str(self._filter_map) if self._filter_map else ""
        joins_str = str(self._join_map) if self._join_map else ""
        if filters_str and joins_str:
            return f"{filters_str} AND {joins_str}"
        elif filters_str:
            return filters_str
        else:
            return joins_str


class MospWhereClause:
    def __init__(self, mosp_data: dict, *, alias_map: dict = None):
        self.mosp_data = mosp_data
        self.alias_map = alias_map
        self._predicate_map: MospPredicateMap = None

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
        self._inflate_predicate_map()
        return self._predicate_map

    def _get_joins(self) -> MospJoinMap:
        self._inflate_predicate_map()
        return self._predicate_map.joins

    def _get_filters(self) -> MospFilterMap:
        self._inflate_predicate_map()
        return self._predicate_map.filters

    filters: MospFilterMap = property(_get_filters)
    joins: MospJoinMap = property(_get_joins)

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

    def _inflate_predicate_map(self):
        if not self._predicate_map:
            self._predicate_map = MospPredicateMap(self.break_conjunction(), alias_map=self.alias_map)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        self._inflate_predicate_map()
        return str(self._predicate_map)


class NoJoinPredicateError(util.StateError):
    def __init__(self, msg: str = ""):
        super.__init__(msg)


class NoFilterPredicateError(util.StateError):
    def __init__(self, msg: str = ""):
        super.__init__(msg)


# TODO: in the end, this should be renamed to MospPredicate. Currently we still need the Abstract prefix to ensure
# our wrecking-ball refactoring actually works and does not reuse this class accidentally
class AbstractMospPredicate(abc.ABC):
    @staticmethod
    def is_compound_operation(operation: str) -> bool:
        return operation in CompoundOperations

    def __init__(self, mosp_data: dict, *, alias_map: dict):
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

    @abc.abstractmethod
    def sql_operation(self) -> str:
        """Parses the operation associated with this predicate into the equivalent SQL operation."""
        return NotImplemented

    def contains_table(self, table: db.TableRef) -> bool:
        """Checks, whether this predicate filters or joins an attribute of the given table."""
        return any(table == tab for tab in self.collect_tables())

    def joins_table(self, table: db.TableRef) -> bool:
        """Checks, whether this predicate describes a join and one of the join partners is the given table."""
        return self.is_join() and self.contains_table(table)

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
    def join_partner_of(self, table: db.TableRef) -> Union[db.AttributeRef, Set[db.AttributeRef]]:
        """Retrieves the attributes that are joined with the given table.

        This assumes that this predicate actually is a join. If it's a base predicate, a single attribute will be
        returned, otherwise all matching attributes will be wrapped in a set.
        """
        return NotImplemented

    @abc.abstractmethod
    def join_partners(self) -> List[Tuple[db.AttributeRef, db.AttributeRef]]:
        return NotImplemented

    def base_predicates(self) -> List["MospBasePredicate"]:
        return NotImplemented

    def estimate_result_rows(self, *, sampling: bool = False, sampling_pct: int = 25,
                             dbs: db.DBSchema = None) -> int:
        dbs = db.DBSchema.get_instance() if dbs is None else dbs
        tables = self.collect_tables()
        if util.contains_multiple(tables):
            raise ValueError("Can only estimate filters with a single table")
        base_table: db.TableRef = util.simplify(tables)
        count_query = self._as_mosp_query(base_table)

        formatted_query: str = mosp.format(count_query)

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

    @abc.abstractmethod
    def to_mosp(self) -> dict:
        return NotImplemented

    def as_full_query(self, *, count_query: bool = True) -> MospQuery:
        return MospQuery(self._as_mosp_query(self.collect_tables(), count_query=count_query))

    def _as_mosp_query(self, base_table: Union[db.TableRef, List[db.TableRef]], *, count_query: bool = False):
        proj = {"count": "*"} if count_query else "*"
        base_table = util.simplify(base_table)
        from_clause = ([{"value": tab.full_name, "name": tab.alias} for tab in base_table]
                       if util.contains_multiple(base_table)
                       else {"value": base_table.full_name, "name": base_table.alias})
        return {
            "select": proj,
            "from": from_clause,
            "where": self.to_mosp()
        }

    def _assert_alias_map(self):
        if not self.alias_map:
            raise ValueError("No alias map given")

    def _assert_join_predicate(self):
        if not self.is_join():
            raise NoJoinPredicateError()

    def _assert_filter_predicate(self):
        if not self.is_filter():
            raise NoFilterPredicateError()

    def __hash__(self) -> int:
        return util.dict_hash(self.mosp_data)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AbstractMospPredicate) and hash(self) == hash(other)


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
        if isinstance(predicates, AbstractMospPredicate):
            return predicates

        if len(predicates) == 1:
            return predicates[0]

        flattened_predicates: List[AbstractMospPredicate] = []
        for predicate in predicates:
            if predicate.is_compound() and predicate.operation == "and":
                flattened_predicates.extend(MospCompoundPredicate.pull_children_from_and_predicate(predicate))
            else:
                flattened_predicates.append(predicate)
        return MospCompoundPredicate("and", flattened_predicates, alias_map=alias_map)

    @staticmethod
    def extract_base_predicates(predicate: AbstractMospPredicate) -> List["MospBasePredicate"]:
        if not predicate.is_compound():
            return [predicate]

        base_predicates = []
        for child_predicate in predicate.children:
            base_predicates.extend(MospCompoundPredicate.extract_base_predicates(child_predicate))
        return base_predicates

    @staticmethod
    def pull_children_from_and_predicate(predicate: "MospCompoundPredicate") -> List[AbstractMospPredicate]:
        if not predicate.is_compound() or not predicate.operation == "and":
            raise ValueError("Not an AND predicate: {}".format(predicate))

        sub_predicates = []
        for child_predicate in predicate.children:
            if child_predicate.is_compound() and child_predicate.operation == "and":
                sub_predicates.extend(MospCompoundPredicate.pull_children_from_and_predicate(child_predicate))
            else:
                sub_predicates.append(child_predicate)
        return sub_predicates

    def __init__(self, operator, children: List[AbstractMospPredicate], *, alias_map: dict = None):
        mosp_data = {operator: [child.mosp_data for child in children]}
        super().__init__(mosp_data, alias_map=alias_map)

        self.operation = operator
        self.children: List[AbstractMospPredicate] = util.enlist(children)
        self.negated = operator == "not"

    def is_compound(self) -> bool:
        return True

    def is_join(self) -> bool:
        return any(child.is_join() for child in self.children)

    def sql_operation(self) -> str:
        return _OperationPrinting.get(self.operation, self.operation)

    def collect_attributes(self) -> Set[db.AttributeRef]:
        return set(util.flatten([child.collect_attributes() for child in self.children], flatten_set=True))

    def join_partner_of(self, table: db.TableRef) -> Union[db.AttributeRef, Set[db.AttributeRef]]:
        if not self.is_join():
            raise NoJoinPredicateError()
        partners = util.flatten([child.join_partner_of(table) for child in self.children], flatten_set=True)
        return set(partners)

    def base_predicates(self) -> List["MospBasePredicate"]:
        predicates = []
        for child in self.children:
            predicates.extend(child.base_predicates())
        return predicates

    def join_partners(self) -> List[Tuple[db.AttributeRef, db.AttributeRef]]:
        partners = []
        for child in self.children:
            partners.extend(child.join_partners())
        return partners

    def rename_table(self, from_table: db.TableRef, to_table: db.TableRef, *,
                     prefix_attribute: bool = False) -> "AbstractMospPredicate":
        renamed_children = [child.rename_table(from_table, to_table, prefix_attribute=prefix_attribute) for child
                            in self.children]
        renamed_mosp_data = [child.mosp_data for child in self.children]
        if self.negated:
            renamed_mosp_data = util.simplify(renamed_mosp_data)
        return MospCompoundPredicate(self.operation, renamed_children, mosp_data=renamed_mosp_data,
                                     alias_map=self.alias_map)

    def to_mosp(self) -> dict:
        if self.negated:
            return {"not": self.children[0].to_mosp()}
        return {self.operation: [child.to_mosp() for child in self.children]}

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.negated:
            return "NOT (" + str(util.simplify(self.children)) + ")"
        op_str = self.sql_operation()
        return f" {op_str} ".join(str(child) if child.is_base() else f"({child})" for child in self.children)


class MospBasePredicate(AbstractMospPredicate):
    def __init__(self, mosp_data, *, alias_map: dict = None):
        super().__init__(mosp_data, alias_map=alias_map)

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
            right_attr = self._extract_right_attribute()
            if right_attr:
                right_table_alias, right_attribute = self._break_attribute(right_attr)
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

    def sql_operation(self) -> str:
        return _OperationPrinting.get(self.operation, self.operation)

    def join_partner_of(self, table: db.TableRef) -> Union[db.AttributeRef, Set[db.AttributeRef]]:
        self._assert_alias_map()
        if self.is_filter():
            raise NoJoinPredicateError()
        if self.left_attribute.table == table:
            return self.right_attribute
        elif self.right_attribute.table == table:
            return self.left_attribute
        else:
            raise ValueError("Table is not joined")

    def join_partners(self) -> List[Tuple[db.AttributeRef, db.AttributeRef]]:
        self._assert_join_predicate()
        return [(self.left_attribute, self.right_attribute)]

    def literal_value(self) -> Any:
        if self.is_join():
            raise NoFilterPredicateError("Join predicates have no literal value!")
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
            renamed_predicate.mosp_left = str(renamed_attribute)
            renaming_performed = True

        if self.is_join() and self.right_attribute.table == from_table:
            attribute_name = (f"{from_table.alias}_{self.right_attribute.attribute}" if prefix_attribute
                              else self.right_attribute.attribute)
            renamed_attribute = db.AttributeRef(to_table, attribute_name)
            renamed_predicate.right_attribute = renamed_attribute
            renamed_predicate.mosp_right = str(renamed_attribute)
            renaming_performed = True

        if renaming_performed:
            renamed_predicate._refresh_mosp()
            self.alias_map[to_table.alias] = to_table  # both alias maps reference the same dict so this is sufficient

        return renamed_predicate

    def to_mosp(self) -> dict:
        return self.mosp_data

    def _break_attribute(self, raw_attribute: str) -> Tuple[str, str]:
        return raw_attribute.split(".")

    def _extract_right_attribute(self, _current_attr=None) -> Union[str, None]:
        _current_attr = self.mosp_right if not _current_attr else _current_attr
        if not _current_attr:
            return False

        if isinstance(_current_attr, dict):
            if "literal" in _current_attr:
                return None
            op = util.dict_key(_current_attr)
            if op in ["exists", "missing"]:
                return False
            return self._extract_right_attribute(_current_attr=_current_attr[op])
        elif isinstance(_current_attr, list):
            return self._extract_right_attribute(_current_attr=util.head(_current_attr))
        elif isinstance(_current_attr, str):
            return _current_attr

    def _unwrap_literal(self):
        if not isinstance(self.mosp_right, dict) or "literal" not in self.mosp_right:
            return self.mosp_right
        return self.mosp_right["literal"]

    def _refresh_mosp(self):
        """Updates mosp_data based on operation, mosp_left and mosp_right"""
        if self.operation == "exists" or self.operation == "missing":
            self.mosp_data = {self.operation: self.mosp_left}
        elif self.operation == "between":
            self.mosp_data = {"between": [self.mosp_left, *self.mosp_right]}
        else:
            self.mosp_data = {self.operation: [self.mosp_left, self.mosp_right]}

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.operation == "exists":
            return self.mosp_left + " IS NOT NULL"
        elif self.operation == "missing":
            return self.mosp_left + " IS NULL"

        op_str = self.sql_operation()
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
