Setup
=====

PostBOUND can be installed either directly on the system using a manual setup, or as a Docker container.
Both installation methods are described in the following sections.

.. tip::

    If you want to update an existing (virtual environment-based) installation of PostBOUND, you can just
    use the ``tools/setup-py-venv.sh`` script. It takes care of loading the latest PostBOUND release and
    updating all required packages. This also works within the Docker-based installation.


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
See the documentation of :func:`postgres.connect() <postbound.db.postgres.connect>` for more details on the config file
and alternative ways to establish a connection.

Now, you should be able to connect to the Postgres server using the following code:

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

The Docker-based installation essentially automates the manual installation process described above.
The resulting Docker container contains a virtual environment-based installation of PostBOUND and a Postgres (or pg_lab)
server completely configured and ready to use.
Optionally, you can also obtain an optimized Postgres server configuration and setup different benchmarks.

.. note::

    Right now, the Docker setup creates a tailored image for PostBOUND and Postgres.
    Sadly, this prevents us from using volumes to persist the Postgres data or to make the complete PostBOUND framework
    available on the host.
    This situation is not ideal and we will probably use a different approach in the (near) future.
    But for now, this is the best that we have.

To create the Docker image, simply run ``docker build`` in the main PostBOUND directory.
The build process will take a while, since it has to download and compile Postgres from source.
You can customize the build process with the following ``--build-arg`` options:

+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| Argument               | Allowed values                | Description                                                                                                                                                                                                                                                            | Default       |
+========================+===============================+========================================================================================================================================================================================================================================================================+===============+
| ``TIMEZONE``           | Any valid timezone identifier | Timezone of the Docker container (and hence the Postgres server). It is probably best to just use the value of ``cat /etc/timezone``.                                                                                                                                  | ``UTC``       |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``USERNAME``           | Any valid UNIX username       | The username within the Docker container. This will also be the Postgres user and password.                                                                                                                                                                            | ``postbound`` |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``SETUP_IMDB``         | *true* or *false*             | Whether an `IMDB <https://doi.org/10.14778/2850583.2850594>`__ instance should be created as part of the Postgres setup. PostBOUND can connect to the database using the ``.psycopg_connection_job`` config file.                                                      | *false*       |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``SETUP_STATS``        | *true* or *false*             | Whether a `Stats <https://doi.org/10.14778/3503585.3503586>`__ instance should be created as part of the Postgres setup. PostBOUND can connect to the database using the ``.psycopg_connection_stats`` config file.                                                    | *false*       |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``SETUP_STACK``        | *true* or *false*             | Whether a `Stack <https://doi.org/10.1145/3448016.3452838>`__ instance should be created as part of the Postgres setup. PostBOUND can connect to the database using the ``.psycopg_connection_stack`` config file.                                                     | *false*       |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``OPTIMIZE_PG_CONFIG`` | *true* or *false*             | Whether the Postgres configuration parameters should be automatically set based on your hardware platform. Rules are based on `PGTune <https://pgtune.leopard.in.ua/>`__ by `le0pard <https://github.com/le0pard>`__. See :ref:`pg-server-config` for more details.    | *false*       |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``PG_DISK_TYPE``.      | *SSD* or *HDD*                | In case the Postgres server is automatically configured (see ``OPTIMIZE_PG_CONFIG``) this indicates the kind of storage for the actual database. In turn, this influences the relative cost of sequential access and index-based access for the query optimizer.       | *SSD*         |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``PG_VER``             | 16, 17, ...                   | The Postgres version to use. Notice that pg_lab supports fewer versions. This value is passed to the ``postgres-setup.sh`` script of the Postgres tooling (either under ``db-support`` or from pg_lab), which provides the most up to date list of supported versions. | *17*          |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+
| ``USE_PGLAB``          | *true* or *false*             | Whether to initialize a `pg_lab <https://github.com/rbergm/pg_lab>`__ server instead of a normal Postgres server. pg_lab provides advanced hinting capabilities and offers additional extension points for the query optimizer.                                        | *false*       |
+------------------------+-------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------+

The Docker container makes port 5432 available to bind on the system.
This enables you to connect to the Postgres server from outside.
Likewise, a volume is created at ``/postbound/public`` to easily copy experiment scripts into the container and to export
results back out.
The PostBOUND installation itself is located at ``/postbound``.
If a vanilla Postgres server is used, it is installed at ``/postbound/db-support/postgres/postgres-server``.
pg_lab servers are installed at ``/pg_lab``.

Once you log in to the container, the PostBOUND virtual environment will be activated automatically.
Likewise, all Postgres binaries are available on the *PATH*.

Putting things together, you can create a Docker container with PostBOUND and Postgres like so:

.. code-block:: bash

    docker build -t postbound \
        --build-arg TIMEZONE=$(cat /etc/timezone) \
        --build-arg SETUP_IMDB=true \
        --build-arg SETUP_STATS=true \
        --build-arg OPTIMIZE_PG_CONFIG=true \
        --build-arg PG_DISK_TYPE=SSD \
        --build-arg PG_VER=17 \
        --build-arg USE_PGLAB=true \
        .

    docker run -dt \
        --shm-size 4G \
        --name postbound \
        --volume $PWD/postbound-docker:/postbound/public \
        --publish 5432:5432 \
        postbound

    docker exec -it postbound /bin/bash

.. tip::

    Building the Docker image will take a while.
    This is expected and nothing to worry about.
    The build process involves downloading and compiling Postgres from source, as well as optionally setting up the
    databases for JOB, Stats and the like (which also includes downloading and importing them).
