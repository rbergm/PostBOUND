Setup
=====

PostBOUND can be installed either directly on the system using a manual setup, or as a Docker container.
Both installation methods are described in the following sections.

Manual Installation
-------------------

A manual installation requires a UNIX-based system, with Linux being the most well supported.
Windows users should use the Windows Subsystem for Linux (WSL) to run PostBOUND (which we do as well).
By default, PostBOUND is installed within a Python virtual environment using the ``tools/setup-py-venv.sh`` script.

.. note::

    We test PostBOUND mostly on a WSL-based Ubuntu system and sporadically on MacOS.
    Therefore, these are the most well-supported platforms.
    Sadly, at the current time we cannot provide support for different setups.

Basically, the setup script takes care of all the necessary installation steps of the framework you should be good to go
after running it. Use ``--help`` to view the available options.
Once the framework is installed, you just need to configure the database connection.

A key requirement for PostBOUND is a running database server to execute queries against.
See the :ref:`hinting` description for more details on why this is necessary and what  functionality the database has to
provide.
Currently, the most well-supported DBS is PostgreSQL.
Limited support for MySQL is also available.
In the following, we assume that PostBOUND should interact with a Postgres server.

To ensure the smoothest interaction between PostBOUND and Postgres and to have the least configuration effort, the Postgres
server and PostBOUND should run on the same address space (i.e. on the same machine or within the same virtualized
environment. Notably, a manual installation of PostBOUND and a Docker-based installation of Postgres does not work).
Basically, there are three different options to setup a Postgres server:

1. Installing Postgres on your own using the package manager or a binary distribution. This requires you to also install
   the `pg_hint_plan <https://github.com/ossc-db/pg_hint_plan>`__ extension to enable query hinting.
2. Installing Postgres using the `pg_lab <https://github.com/rbergm/pg_lab>`__ build tools. This provides the most complete
   hinting functionality, e.g. with support for base table cardinalities and parallel query plans.
3. Using the build tools that are shipped with PostBOUND. These create a local Postgres server by compiling from source and
   take care of setting up all required extensions (such as *pg_hint_plan*). The remainder of this section documents this
   installation method in detail.

.. admonition:: pg_lab vs. pg_hint_plan

    On the surface, *pg_hint_plan* and *pg_lab* provide very similar functionality.
    Both provide a hinting mechanism to embed optimizer decisions into the raw query text using special SQL comments.
    However, *pg_lab* goes far beyond query hinting.
    In fact, it is an extension of Postgres that adds additional extension points to the query optimizer.
    These allow extensions to modify the optimizer's behavior in a fine-grained manner.
    In a way, *pg_lab* can be thought of as a low-level alternative to PostBOUND for Postgres-specific research.
    That is, if you want to learn all of the internal details of the Postgres optimizer and how to modify them.

The Postgres build tools compile a fresh server instance from source.
Therefore, you need a number of tools and libraries available on you system.
For Ubuntu-based distributions, these can be installed like so:

.. code-block:: bash

    sudo apt install \
        build-essential flex bison curl pkg-config llvm clang \
        libreadline-dev libssl-dev libicu-dev liblz4-dev libossp-uuid-dev python3-dev \
        git unzip zstd

Adjust the packages depending on your distribution.
Once the required packages are installed, you can use the build tools from the ``db-support/postgres`` directory to set up
a Postgres server.
All scripts support ``--help`` to view the available options.

.. code-block:: bash

    cd db-support/postgres
    ./postgres-setup.sh --pg-ver 17 --stop
    . ./postgres-start.sh

This will pull the latest stable release of Postgres 17, compile it from source and initialize a new database cluster along
with all required extensions (e.g., pg_hint_plan).
The server will have a default user that correponds to your system username.
All binaries, data directory, etc. are placed under *postgres-server* in the working directory.
A central challenge of this manual setup is that the Postgres binaries, etc. are not globally available, e.g., on your
PATH.
This means that you cannot simply psql into your server.
To mitigate this situation, PostBOUND ships a number of management scripts:

- ``postgres-load-env.sh`` takes care of setting up all environment variables (PATH, C_INCLUDE_PATH, etc.) as necessary
- ``postgres-start.sh`` starts Postgres and also initializes your environment
- ``postgres-stop.sh`` shuts down a running server and undoes all changes to your environment

Since all of these scripts make changes to your global environment, they have to be sourced instead of executed.

You can also install multiple different Postgres versions side-by-side.
To do so, all scripts accept an optional installation path as parameter:

.. code-block:: bash

    ./postgres-setup.sh --pg-ver 16 --stop --dir /my/path
    . ./postgres-start.sh /my/path

.. tip:: 

    We recommend to always setup the Postgres server with the ``--stop`` option and not source this script directly.
    This ensures that your shell does not terminate in case the setup runs into any issues.
    Once the setup is completed, just start the server using the ``postgres-start.sh``.

After your server is setup and running, you can populate it with some well-known benchmarks, such as JOB, Stats or Stack.
PostBOUND provides simple setup scripts for these out-of-the-box:

.. code-block:: bash
    ./workload-job-setup.sh

These scripts assume that your Postgres server is running and you can simply use *psql* to connect to it.
Once again, you can use the ``--help`` option to view the available options (including ways to adapt the connection
parameters).

One last question is how to connect to the database server from within PostBOUND.
Internally, PostBOUND uses the `psycopg <https://www.psycopg.org/>`__ library to connect to Postgres.
You can use the ``postgres-psycopg-setup.sh`` script to create a connection file with the necessary parameters to connect
to the Postgres.

Now, you should be able to connect to the Postgres server like so:

.. ipython:: python

    import postbound as pb
    pg_instance = pb.postgres.connect(config_file=".psycopg_connection")
    pg_instance

Putting things together, you can create an entirely new Postgres server like so:

.. code-block:: bash

    cd db-support/postgres
    ./postgres-setup.sh --pg-ver 17 --stop
    . ./postgres-start.sh
    ./workload-job-setup.sh
    ./postgres-psycopg-setup.sh job imdb
    cp .psycopg_connect_job ../..

Docker Installation
-------------------