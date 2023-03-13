"""Contains utilities to interact with various database instances.

This includes accessing *schema information* such as primary key/foreign key indices, retrieving *metadata* such as
tuple counts or histograms, and the ability to execute arbitrary *SQL queries* on the database.

In addition, this package contains functionality to generate system-specific query information for (optimized) SQL
queries.

The `db` package follows an interface/implementation approach: for each aspect of database functionality (e.g.
retrieving statistics or generating queries) a basic abstract interface exists. In order to make this functionality
accessible for a specific database system (e.g. PostgreSQL, MySQL, Oracle...), the interfaces must be implemented
according to the concrete rules of the database system.
"""
