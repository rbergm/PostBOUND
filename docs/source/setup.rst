Setup
=====

Manual installation
-------------------


A complete setup of PostBOUND includes the following steps:

1. Setting up a database system and loading an actual database
2. Setting up a supported Python version
3. Generating a new virtual environment for PostBOUND
4. Building and installing PostBOUND as a local pip package

These steps will be discussed in more detail in the next sections. Notice that PostBOUND currently only targets Linux
distributions (including WSL). While other systems might still work, there is no guarantee and bugs can occur.


Database setup
--------------

PostBOUND provides utilities to automatically setup a database system and database. Take a look at the top-level README as well
as the specialized READMEs for the different database systems for more details. For example, to setup a PostgreSQL instance
with the IMDB database, the following commands can be used (starting from the root PostBOUND directory):

.. code-block:: bash

    $ cd db-support/postgres

    # Install a local Postgres server
    $ . ./postgres-setup.sh --stop

    # Load the IMDB database
    $ . ./postgres-start.sh
    $ ./workload-job-setup.sh

    # Setup the configuration file for the IMDB.
    # This file has to be moved to the working directory from which the Python interpreter is started
    # later on. In this case, this directory is ../postbound. The set-workload.sh utility creates a
    # symlink to make sure this is the default configuration file
    $ ./postgres-psycopg-setup.sh job imdb
    $ mv .psycopg_connection_job ../postbound/
    $ cd ../postbound
    $ ./set-workload.sh job


Python setup
------------

PostBOUND requires at least Python version 3.10 to run properly. If this is not provided by your package manager, it can be
installed via the utilities in the ``python-3.10`` directory. In order to setup a custom Python installation, use the following
commands:

.. code-block:: bash

    $ ./python-setup.sh
    $ . ./python-load-path.sh

Take a look at the dedicated README for more details.


Virtual environment setup
-------------------------

Since PostBOUND has a number of Python dependencies, it is recommended to install it in a dedicated virtual environment to
prevent dependency hell as good as possible. This can also include installing PostBOUND itself as a local pip package,
although this step is optional. For example, you can use the following commands:

.. code-block:: bash

    $ python3 -m venv pb-venv
    $ . pb-venv/bin/activate

    # If the PostBOUND source code should be used directly, the requirements have to be installed manually
    (pb-venv) $ pip install -r requirements.txt

    # Alternatively, PostBOUND can be installed as a local pip/wheel package
    (pb-venv) $ pip install build wheel
    (pb-venv) $ python3 -m build
    (pb-venv) $ pip install dist/PostBOUND-<file suffix>.whl

This entire process (and more) is automated under ``tools/setup-py-venv.sh``.
