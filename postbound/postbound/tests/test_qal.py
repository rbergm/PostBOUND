"""Tests for PostBOUND's query abstraction layer.

The tests here mainly act as regression tests to ensure that SqlQuery objects provide the correct information if
queried for joins, etc.
"""

import sys
import textwrap
import unittest

sys.path.append("../../")

from postbound.qal import base, parser  # noqa: E402

from postbound.tests import regression_suite  # noqa: E402


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


class RegressionTests(unittest.TestCase):
    # No regressions so far!
    pass
