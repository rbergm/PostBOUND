"""Tests for PostBOUND's query abstraction layer.

The tests here mainly act as regression tests to ensure that SqlQuery objects provide the correct information if
queried for joins, etc.
"""
from __future__ import annotations

import textwrap
import unittest

from postbound.qal import base, parser, expressions, predicates, qal

from tests import regression_suite


class SqlQueryTests(unittest.TestCase):
    def test_dependent_subqueries(self) -> None:
        dependent_query = "SELECT * FROM R WHERE R.a = S.b"
        parsed_query = parser.parse_query(dependent_query)
        self.assertTrue(parsed_query.is_dependent(), "Dependent query not recognized as dependent")

        independent_query = "SELECT * FROM R WHERE R.a = 42"
        parsed_query = parser.parse_query(independent_query)
        self.assertFalse(parsed_query.is_dependent(), "Independent query recognized as dependent")

    def test_join_detection(self) -> None:
        simple_query = "SELECT * FROM R, S WHERE R.a = 42 AND R.b = S.c"
        parsed_query = parser.parse_query(simple_query)
        self.assertTrue(len(parsed_query.predicates().joins()) == 1, "Should detect 1 join")

        subquery_join_query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.c = S.d)"
        parsed_query = parser.parse_query(subquery_join_query)
        self.assertTrue(len(parsed_query.predicates().joins()) == 0, "Should treat dependent subqueries as filters")

    def test_filter_detection(self) -> None:
        simple_query = "SELECT * FROM R, S WHERE R.a = 42 AND R.b = S.c"
        parsed_query = parser.parse_query(simple_query)
        self.assertTrue(len(parsed_query.predicates().filters()) == 1, "Should detect 1 filter")

        subquery_join_query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S WHERE R.c = S.d)"
        parsed_query = parser.parse_query(subquery_join_query)
        self.assertTrue(len(parsed_query.predicates().filters()) == 1,
                        "Should detect filters for dependent subquery")

        independent_subquery_query = "SELECT * FROM R WHERE R.a = (SELECT MIN(S.b) FROM S)"
        parsed_query = parser.parse_query(independent_subquery_query)
        self.assertTrue(len(parsed_query.predicates().filters()) == 1,
                        "Should detect 1 filter for independent subquery")


class PredicateTests(unittest.TestCase):
    def test_binary_predicate_joins(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a = S.b"
        with self.subTest("Direct equi join", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a < S.b"
        with self.subTest("Direct non-equi join", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = 42"
        with self.subTest("Direct filter", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE some_udf(R.a, S.b) = 42"
        with self.subTest("Join in UDF", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = some_udf(S.b)"
        with self.subTest("Join with UDF result", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE some_udf(R.a) = 42"
        with self.subTest("Filter with UDF", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a = S.b::integer"
        with self.subTest("Join with casted value", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a = (SELECT MIN(T.c) FROM T)"
        with self.subTest("Filter with subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a = (SELECT MIN(T.c) FROM T WHERE T.c = S.b)"
        with self.subTest("Filter with dependent subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_between_predicate(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND 42"
        with self.subTest("Direct BETWEEN filter", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND S.b"
        with self.subTest("Direct BETWEEN join, end", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN S.b AND S42"
        with self.subTest("Direct BETWEEN join, start", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN S.b AND S.b + 42"
        with self.subTest("Direct BETWEEN join, both ends", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a BETWEEN 24 AND (SELECT MIN(T.c) FROM T)"
        with self.subTest("BETWEEN filter with subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_in_predicate(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a IN (1, 2, 3)"
        with self.subTest("Direct IN filter", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE R.a IN (1, S.b, 3)"
        with self.subTest("Join in values list", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

        query = "SELECT * FROM R, S WHERE R.a IN (SELECT T.b FROM T WHERE T.c = 42)"
        with self.subTest("IN filter for independent subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

            query = "SELECT * FROM R, S WHERE R.a IN (SELECT T.b FROM T WHERE T.c = T.d)"
            with self.subTest("IN filter for dependent subquery", query=query):
                parsed = parser.parse_query(query)
                self.assertTrue(len(parsed.predicates().joins()) == 0)
                self.assertTrue(len(parsed.predicates().filters()) == 1)

    def test_unary_predicate(self) -> None:
        query = "SELECT * FROM R WHERE EXISTS (SELECT * FROM S WHERE R.a = S.b)"
        with self.subTest("EXISTS for dependent subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R WHERE NOT EXISTS (SELECT * FROM S WHERE R.a = S.b)"
        with self.subTest("NOT EXISTS for dependent subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE my_udf(R.a)"
        with self.subTest("Unary UDF filter", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 0)
            self.assertTrue(len(parsed.predicates().filters()) == 1)

        query = "SELECT * FROM R, S WHERE my_udf(R.a, S.b)"
        with self.subTest("Unary UDF join", query=query):
            parsed = parser.parse_query(query)
            self.assertTrue(len(parsed.predicates().joins()) == 1)
            self.assertTrue(len(parsed.predicates().filters()) == 0)

    def test_join_false_positives(self) -> None:
        query = textwrap.dedent("""
                                SELECT * FROM R
                                WHERE R.a = R.b
                                    AND my_udf_pred1(R.a)
                                    AND my_udf_pred2(R.a, R.c)
                                    AND R.a IN (24, R.b, 42)
                                """)
        parsed = parser.parse_query(query)
        self.assertFalse(parsed.predicates().joins())
        self.assertTrue(len(parsed.predicates().filters()) == 4)

    def test_nested_conjunction_disjunction_predicates(self) -> None:
        query = textwrap.dedent("""
                                SELECT *
                                FROM title t
                                    JOIN movie_info mi ON t.id = mi.movie_id
                                WHERE t.production_year < 2021
                                  OR t.season_nr > 10
                                  AND mi.info LIKE '%cartoon%'
                                  """)
        parsed = parser.parse_query(query)
        root_predicate = parsed.where_clause.predicate
        self.assertTrue(isinstance(root_predicate, predicates.CompoundPredicate))
        self.assertTrue(root_predicate.operation == expressions.LogicalSqlCompoundOperators.Or,
                        "OR predicate should come first")


class MockSchemaLookup:
    LookupData = {"a": base.TableReference("R"), "b": base.TableReference("R"), "c": base.TableReference("S")}

    def lookup_column(self, column: base.ColumnReference | str,
                      candidates: list[base.TableReference]) -> base.TableReference:
        column = column.name if isinstance(column, base.ColumnReference) else column
        return MockSchemaLookup.LookupData[column]


class TransformationTests(unittest.TestCase):

    def test_column_binding(self) -> None:
        """Column binding happens automatically during parsing.

        Therefore we parse normally and test the binding results for normal queries explicitly in this method.
        """
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        col_r_b = base.ColumnReference("b", tab_r)

        tab_s = base.TableReference("S")
        col_s_c = base.ColumnReference("c", tab_s)

        query = "SELECT R.a FROM R WHERE R.b = 42"
        with self.subTest("Simple binding through full name", query=query):
            parsed = parser.parse_query(query, bind_columns=False)  # bind_columns refers to live binding
            self.assertSetEqual(parsed.select_clause.columns(), {col_r_a})
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_b})
            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b})
            self.assertSetEqual(parsed.tables(), {tab_r})

        query = "SELECT S.c FROM R, S WHERE R.a = S.c AND R.b = 42"
        with self.subTest("Simple binding in join", query=query):
            parsed = parser.parse_query(query, bind_columns=False)  # bind_columns refers to live binding
            self.assertSetEqual(parsed.select_clause.columns(), {col_s_c})
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.tables(), {tab_r, tab_s})

    def test_alias_binding(self) -> None:
        """Column binding happens automatically during parsing.

        Therefore we parse normally and test the binding results for aliased queries explicitly in this method.
        """
        tab_r = base.TableReference("R", "r_tab")
        col_r_a = base.ColumnReference("a", tab_r)
        col_r_b = base.ColumnReference("b", tab_r)

        tab_s = base.TableReference("S")  # no alias!
        col_s_c = base.ColumnReference("c", tab_s)

        query = "SELECT r_tab.a FROM R r_tab WHERE r_tab.b = 42"
        with self.subTest("Simple binding through alias", query=query):
            parsed = parser.parse_query(query, bind_columns=False)  # bind_columns refers to live binding
            self.assertSetEqual(parsed.select_clause.columns(), {col_r_a})
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_b})
            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b})
            self.assertSetEqual(parsed.tables(), {tab_r})

        query = "SELECT S.c FROM R r_tab, S WHERE r_tab.a = S.c AND r_tab.b = 42"
        with self.subTest("Simple binding in join", query=query):
            parsed = parser.parse_query(query, bind_columns=False)  # bind_columns refers to live binding
            self.assertSetEqual(parsed.select_clause.columns(), {col_s_c})
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.tables(), {tab_r, tab_s})

    def test_schema_binding(self) -> None:
        """Column binding happens automatically during parsing.

        Therefore we parse normally and test the binding results explicitly in this method. In contrast to the other
        binding tests, these tests here focus on the binding that happens for unbound columns via the database schema.
        """
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        col_r_b = base.ColumnReference("b", tab_r)

        tab_s = base.TableReference("S")
        col_s_c = base.ColumnReference("c", tab_s)

        query = "SELECT a FROM R WHERE b = 42"
        with self.subTest("Simple binding through full name", query=query):
            parsed = parser.parse_query(query, db_schema=MockSchemaLookup(), bind_columns=True)
            self.assertSetEqual(parsed.select_clause.columns(), {col_r_a})
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_b})
            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b})
            self.assertSetEqual(parsed.tables(), {tab_r})

        query = "SELECT c FROM R, S WHERE a = S.c AND R.b = 42"
        with self.subTest("Simple binding in join", query=query):
            parsed = parser.parse_query(query, db_schema=MockSchemaLookup(), bind_columns=True)
            self.assertSetEqual(parsed.select_clause.columns(), {col_s_c})
            self.assertSetEqual(parsed.where_clause.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.columns(), {col_r_a, col_r_b, col_s_c})
            self.assertSetEqual(parsed.tables(), {tab_r, tab_s})

    def test_subquery_binding(self) -> None:
        tab_r = base.TableReference("R")
        col_r_a = base.ColumnReference("a", tab_r)
        col_r_b = base.ColumnReference("b", tab_r)

        tab_s = base.TableReference("S")
        col_s_b = base.ColumnReference("b", tab_s)
        col_s_c = base.ColumnReference("c", tab_s)

        query = "SELECT SUM(R.b) FROM R WHERE R.a < (SELECT MIN(S.c) FROM S)"
        all_cols = {col_r_a, col_r_b, col_s_c}
        with self.subTest("Column binding in anonymous subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertSetEqual(parsed.columns(), all_cols)

        query = "SELECT SUM(R.b) FROM R, (SELECT S.b, MIN(S.c) FROM S GROUP BY S.b) sub_s WHERE R.a = sub_s.b"
        sub_s = base.TableReference.create_virtual("sub_s")
        col_sub_s_b = base.ColumnReference("b", sub_s)
        all_cols = {col_r_a, col_r_b, col_s_b, col_s_c, col_sub_s_b, col_sub_s_b}
        with self.subTest("Column binding in virtual subquery", query=query):
            parsed = parser.parse_query(query)
            self.assertSetEqual(parsed.columns(), all_cols)

        query = "WITH s_cte AS (SELECT MIN(S.c) FROM S) SELECT SUM(R.b) FROM R WHERE R.a = s_cte.c"
        cte_s = base.TableReference.create_virtual("s_cte")
        col_cte_s_c = base.ColumnReference("c", cte_s)
        all_cols = {col_r_a, col_r_b, col_s_c, col_cte_s_c}
        with self.subTest("Column binding in CTE", query=query):
            parsed = parser.parse_query(query)
            self.assertSetEqual(parsed.columns(), all_cols)

    def test_end_to_end_binding(self) -> None:
        query = """
            WITH cte_a AS (SELECT R.r_a FROM R WHERE R.r_a < 42),
                cte_b AS (SELECT SUM(S.s_b) FROM S GROUP BY S.s_a)
            SELECT AVG(A.a_y * B.b_y)
            FROM A, B, cte_a, (SELECT M.m_a + N.n_a AS total FROM M, N WHERE M.m_b = N.n_b) AS sq_m_n
            WHERE A.a_x <> 100
                AND A.a_b = B.b_b
                AND B.b_a = cte_a.r_a
                AND A.a_t = sq_m_n.total
            GROUP BY A.a_z
            """
        parsed = parser.parse_query(query)
        expected_tables = {base.TableReference("R"), base.TableReference("S"),  # WITH clauses
                           base.TableReference("A"), base.TableReference("B"),  # FROM clauses
                           base.TableReference("M"), base.TableReference("N"),  # subquery tables
                           base.TableReference.create_virtual("cte_a"), base.TableReference.create_virtual("cte_b"),
                           base.TableReference.create_virtual("sq_m_n")}
        self.assertEqual(parsed.tables(), expected_tables)


class ParserTests(regression_suite.QueryTestCase):

    def test_parse_simple(self) -> None:
        query = "SELECT * FROM R, S WHERE R.a = S.b AND R.c = 42"
        parser.parse_query(query)

    def test_select_star(self) -> None:
        query = "SELECT * FROM R"
        parsed = parser.parse_query(query)
        self.assertSetEqual(parsed.columns(), set(), "* should not be considered a column")

    def test_parse_subquery_without_predicates(self) -> None:
        query = "SELECT * FROM R WHERE R.a IN (SELECT S.b FROM S)"
        parsed = parser.parse_query(query)
        self.assertTrue(len(parsed.subqueries()) == 1, "Should detect 1 subquery")

    def test_count_distinct(self) -> None:
        query = "SELECT COUNT(DISTINCT *) FROM R"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format COUNT(DISTINCT *) correctly.")

    def test_is_predicate(self) -> None:
        query = "SELECT * FROM R WHERE R.a IS NULL"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format IS NULL correctly.")
        self.assertTrue(len(parsed.predicates().filters()) == 1, "Should detect 1 filter for IS NULL")

        query = "SELECT * FROM R WHERE R.a IS NOT NULL"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format IS NOT NULL correctly.")
        self.assertTrue(len(parsed.predicates().filters()) == 1, "Should detect 1 filter for IS NOT NULL")

    def test_unary_udf_filter(self) -> None:
        query = "SELECT * FROM R WHERE my_udf(R.a)"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format unary UDF filter correctly.")
        self.assertTrue(len(parsed.predicates().filters()) == 1, "Should detect 1 filter for unary UDF filter")

    def test_implicit_from_clause(self) -> None:
        query = "SELECT * FROM R, S, T WHERE R.a = S.b AND S.b = T.c"
        parsed = parser.parse_query(query)
        self.assertIsInstance(parsed, qal.ImplicitSqlQuery, "Query should be parsed as implicit query")

        query = "SELECT * FROM R r, S s, T t WHERE r.a = s.b AND s.b = t.c"
        parsed = parser.parse_query(query)
        self.assertIsInstance(parsed, qal.ImplicitSqlQuery, "Query should be parsed as implicit query")

    def test_explicit_from_clause(self) -> None:
        query = "SELECT * FROM R JOIN S ON R.a = S.b WHERE R.c LIKE '%42%' AND R.c < S.b"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format explicit FROM clause correctly.")
        self.assertTrue(len(parsed.predicates().filters()) == 1, "Should detect 1 filter in WHERE clause")
        self.assertTrue(len(parsed.predicates().joins()) == 2, "Should detect 2 joins in WHERE clause")
        self.assertIsInstance(parsed, qal.ExplicitSqlQuery, "Query should be parsed as explicit query")

    def test_from_clause_subquery(self) -> None:
        query = "SELECT * FROM R, (SELECT * FROM S WHERE S.d < 42) AS s WHERE R.a = s.b"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format FROM clause with subquery correctly.")
        self.assertTrue(len(parsed.predicates().filters()) == 1, "Should detect 1 filter in WHERE clause")
        self.assertTrue(len(parsed.predicates().joins()) == 1, "Should detect 1 join in WHERE clause")
        self.assertIsInstance(parsed, qal.MixedSqlQuery, "Query should be parsed as mixed query")

    def test_single_cte(self) -> None:
        query = "WITH cte_r AS (SELECT * FROM R) SELECT * FROM cte_r JOIN R ON cte_r.id = R.id"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format CTE query correctly")
        self.assertTrue(len(parsed.cte_clause.queries) == 1, "Did not recognize CTE correctly")

    def test_multiple_cte(self) -> None:
        query = "WITH cte_r AS (SELECT * FROM R), cte_s AS (SELECT MIN(S.c) FROM S WHERE S.c < 42) SELECT * FROM cte_r, cte_s"
        parsed = parser.parse_query(query)
        self.assertQueriesEqual(query, parsed, "Did not parse/format CTE query correctly")
        self.assertTrue(len(parsed.cte_clause.queries) == 2, "Did not recognize CTEs correctly")


class RegressionTests(unittest.TestCase):
    # No regressions so far!
    pass
