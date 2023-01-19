
from typing import Any, List, Tuple, Union

from transform import db, mosp, util


class _TableReferences:
    def __init__(self):
        self.alias_map = dict()
        self.table_map = dict()
        self.virtual_tables = dict()

    def add(self, table_ref: db.TableRef, source_tables: List[db.TableRef] = None):
        if table_ref.is_virtual:
            self.virtual_tables[table_ref.alias] = source_tables if source_tables else []
        else:
            self.alias_map[table_ref.alias] = table_ref.full_name
            self.table_map[table_ref.full_name] = table_ref.alias

    def add_virtual(self, alias: str, source_tables: List[db.TableRef]):
        self.virtual_tables[alias] = source_tables

    def is_physical_table(self, alias: str) -> bool:
        if alias in self.virtual_tables:
            return False
        if alias in self.alias_map:
            return True
        else:
            raise KeyError("Unkown alias: " + alias)

    def is_virtual(self, alias: str) -> bool:
        return alias in self.virtual_tables

    def resolve(self, alias: str):
        if self.is_virtual(alias):
            raise KeyError("Alias describes a virtual table: " + alias)
        return self.alias_map[alias]

    def resolve_virtual(self, alias: str):
        if not self.is_virtual(alias):
            raise KeyError("Not a virtual table: " + alias)
        return self.virtual_tables[alias]

    def bind_attribute(self, table: Union[str, db.TableRef], attribute_name: str) -> str:
        if isinstance(table, db.TableRef):
            return table.bind_attribute(attribute_name)
        else:
            table_alias = self.table_map[table]
            return f"{table_alias}.{attribute_name}"

    def __str__(self):
        return f"Physical tables: {self.alias_map} | Virtual tables: {self.virtual_tables}"


class _FlattenedQueryBuilder:
    def __init__(self, query: mosp.MospQuery, base_table: db.TableRef, dbschema: db.DBSchema):
        self.query = query
        self.base_table = base_table
        self.dbschema = dbschema
        self.joins: List[Tuple[db.TableRef, Any]] = []
        self.table_references = _TableReferences()
        self.table_references.add(self.base_table)

    def include_join(self, join: mosp.MospJoin):
        # strategy for a simple join:
        if not join.is_subquery():
            self.table_references.add(join.base_table())
            join_predicate = self._rewrite_predicate_tree(join.predicate())
            self.joins.append((join.base_table(), join_predicate))
            return

        # strategy for a subquery join:
        # first up, extract the base table as "first-level" join
        base_join_table = join.base_table()
        self.table_references.add(base_join_table)
        self.table_references.add_virtual(join.name(), join.collect_tables())
        base_join_predicate = self._rewrite_predicate_tree(join.predicate())
        self.joins.append((base_join_table, base_join_predicate))
        # secondly, extract all the nested joins
        for subquery_join in join.subquery.joins():
            subquery_table = subquery_join.base_table()
            self.table_references.add(subquery_table)
            subquery_predicate = self._rewrite_predicate_tree(subquery_join.predicate())
            self.joins.append((subquery_table, subquery_predicate))

    def to_mosp(self):
        mosp_data = {"select": {"value": {"count": "*"}}}
        mosp_data["from"] = [self.base_table.to_mosp()]
        for join_table, join_predicate in self.joins:
            mosp_data["from"].append({"join": join_table.to_mosp(), "on": join_predicate})
        return mosp_data

    def _rewrite_predicate_tree(self, predicate_tree):
        if isinstance(predicate_tree, list):
            rewritten_subtree = [self._rewrite_predicate_tree(subtree) for subtree in predicate_tree]
            return rewritten_subtree
        elif isinstance(predicate_tree, dict):
            operation = util.dict_key(predicate_tree)
            if operation in mosp.CompoundOperations:
                subtree = util.dict_value(predicate_tree)
                rewritten_subtree = self._rewrite_predicate_tree(subtree)
                return {operation: rewritten_subtree}
            else:
                predicate = mosp.MospPredicate(predicate_tree)
                left_table, right_table = predicate.tables()
                left_attr, right_attr = predicate.attributes()
                if self.table_references.is_virtual(left_table):
                    predicate.left = self._resolve_virtual_attribute(left_table, left_attr)
                if not predicate.has_literal_op() and self.table_references.is_virtual(right_table):
                    predicate.right = self._resolve_virtual_attribute(right_table, right_attr)
                return predicate.to_mosp()
        else:
            raise TypeError("Unknown predicate tree structure: " + str(predicate_tree))

    def _resolve_virtual_attribute(self, virtual_table: str, attribute: str) -> str:
        candidate_tables = self.table_references.resolve_virtual(virtual_table)
        source_tables = self.dbschema.lookup_attribute(attribute, candidate_tables)
        return self.table_references.bind_attribute(source_tables, attribute)


def flatten_query(query, dbschema):
    query_tree = mosp.MospQuery.parse(query)
    flattened_query = _FlattenedQueryBuilder(query_tree, query_tree.base_table(), dbschema)
    for join in query_tree.joins():
        flattened_query.include_join(join)
    return flattened_query.to_mosp()
