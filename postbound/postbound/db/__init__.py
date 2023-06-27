"""Contains utilities to interact with various database instances.

This includes accessing *schema information* such as primary key/foreign key indices, retrieving *metadata* such as
tuple counts or histograms, and the ability to execute arbitrary *SQL queries* on the database.

In addition, this package contains functionality to generate system-specific query information for (optimized) SQL
queries. Such information is used to enforce the otimization decisions made by PostBOUND when actually executing a query.

Furthermore, a basic model of query plans (as commonly obtained by running an ``EXPLAIN`` query) is also provided. This model
should not be confused with the query plans that are generated in the `optimizer` package. The model provided by the `db`
package is much more relaxed. It should be thought of as a  concept that just happens to share a lot of structure and
attributes with PostBOUND's optimizer plans.

The `db` package follows an interface/implementation approach: for each aspect of database functionality (e.g.
retrieving statistics or generating queries) a basic abstract interface exists. In order to make this functionality
accessible for a specific database system (e.g. PostgreSQL, MySQL, Oracle...), the interfaces must be implemented
according to the concrete rules of the database system.
"""
