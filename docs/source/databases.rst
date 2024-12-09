Interaction with databases
==========================

PostBOUND interacts with databases using an abstract interface. This interface enables PostBOUND to pull information such as
statistics or schema information and to interact with the query optimizer of the database system. Furthermore, the interface
provides means to generate plan information that enforces optimization decisions made by PostBOUND during the actual runtime
of the query. This enables PostBOUND to operate independently from a specific database or database system for most of the time.
Since all database functionality is encapsuled by a common interface, different database systems can easily be added to
PostBOUND. This only requires the implementation of the database interface. Take a look at the `postbound.db` package for more
details. Furthermore, the `postbound` package provides a more general introduction into the interaction of PostBOUND with
physical databases.


Connecting to databases
-----------------------

Although most parts of the database interaction are precisely specified, one that was intentionally left out from the
standardization is the connection to the physical databases. This is because there is no universal standard and different
systems could require vastly different information. Compare for example remote database servers with connection strings
(Postgres, MySQL, ...) to file-based databases such as SQLite. Therefore, it is up to the specific database systems to provide
the most appropriate means to establish a connection. Nevertheless, a common pattern is to provide a ``connect`` method in the
database's module, which accepts the required parameters and provides an instance of the database interface. Typically, this
method also handles registration of the database in the `DatabasePool` for easier access.

Currently, PostBOUND supports two database systems: PostgreSQL and MySQL. Both of them make use of the *connect* pattern.
Connections to these systems can be obtained as follows:

Postgres
    The connect method operates in two different modes: the connect string can be supplied directly as a parameter. However,
    the preferred way of obtaining the connection is to store the connect string in a configuration file. This prevents
    sensitive connection information from being leaked or from becoming visible in the Python history, etc. By default, the
    connect file is a hidden file called ``.psycopg_connection``. It has to be located in the current working directory when
    the connection is established. The name of the file can also be customized by the connect method, which allows
    establishment of connections to different Postgres databases simultaneously. The allowed contents of the connect string are
    specified by Psycopg 3/Postgres [1]_. See `postbound.db.postgres.connect` for more details on the connection method.

MySQL
    The connect method also operates in two modes. However, it requires an explicit connection object instead of a plain text
    string. Instances of this object can either be created explicitly, or once again inferred from a configuration file. By
    default, this file is called ``.mysql_connection.config`` and is required to be a valid INI file. For specific details on
    the supported syntax see [2]_. The allowed parameters are documented in [3]_. See `postbound.db.mysql.connect` for details
    on the connection method.

Notice that for both PostgreSQL as well as MySQL PostBOUND already provides utilities to set up instances of these database
systems, as well as to automatically generate the connection configuration files. See the top-level README for more details.

Links
-----

.. [1] PostgreSQL connection string: https://www.postgresql.org/docs/current/libpq-connect.html#LIBPQ-CONNSTRING
.. [2] INI file structure: https://docs.python.org/3/library/configparser.html#supported-ini-file-structure
.. [3] MySQL connection parameters: https://dev.mysql.com/doc/connector-python/en/connector-python-connectargs.html
