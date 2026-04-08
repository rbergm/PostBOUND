# Postgres management

This folder provides utilities to

- build a Postgres server instance from source `postgres-setup.sh`. Check with `--help` for details
- install a number of recommended and/or required extensions, most importantly
  [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan) (this is also handled as part of the general Postgres setup)
- manage an installed Postgres server (start via `postgres-start.sh` and stop via `postgres-stop.sh`)
- import popular benchmark databases via the <code>workload-_\<benchmark name\>_-setup.sh</code> scripts. Use `--help` to see
  supported options

Notice that the Postgres setup performs a _local_ installation, no global paths or public directories are modified.
Therefore, use the `postgres-load-env.sh` utility to setup your *PATH* environment variable correctly for the current
session. The `*-start`, `*-stop` and `*-setup` scripts can also be sourced to keep the *PATH* up to date (i.e. adding the
Postgres binaries to the *PATH* upon server start and removing them again afterwards). If the Postgres server was installed
to a custom location, this location has to be supplied when calling the management scripts.

Finally, the `postgres-psycopg-setup` utility generates a config file that can be used by PostBOUND's database interaction
code to automatically connect to the database. See the documentation in `postgres.py` for details.

The `util` directory contains a collection of stored procedures for different introspection methods.


## Dependencies

To create a custom Postgres build, a number of packages have to be available on your system. Since packages names, etc. are
not standardized across distributions, we sadly cannot simply list them here. Consult the
[official PG documentation](https://www.postgresql.org/docs/current/install-requirements.html) for more details.
On Ubuntu-based systems, the requirements can be installed like so:

```sh
sudo apt install -y \
    build-essential flex bison curl pkg-config llvm clang meson ninja-build \
    libreadline-dev libssl-dev libicu-dev liblz4-dev libossp-uuid-dev python3-dev \
    liburing-dev \
    git unzip zstd
```


## Enabling Remote Access

To allow external applications to connect to the Postgres server, you need to specify a password via the `--remote-password`
option in the `postgres-setup.sh` script. This functions as the trigger for enabling remote access, `pg_hba.conf`, etc.
will be updated accordingly. After the server is setup, you can connect to it remotely using the specified password and the
username.

If you did not specify a password during setup, you can still enable remote access by manually updating the `pg_hba.conf` file
and `postgresql.conf` manually.


## Optimizing the Server Config

We provide a simple configuration utility based on [PGTune](https://pgtune.leopard.in.ua/) to help you optimize the server
configuration for your hardware. The `postgres-config-generator.py` automatically detects your current hardware environment
and generates appropriate configuration options using the PGTune rules. It produces an SQL file that can be executed on the
server to apply the generated configuration.

Note that the internal hardware detection logic is tuned for Linux-based systems and bare-metal setups. Especially the
detection of SSD vs. HDD might not work on virtualized environments. If you run into any issues, help the utility by providing
the correct hardware information via the command line options. Check with `--help` for details. We also welcome issues and
contributions to harden the script in the future.

## Creating Benchmark Databases

You can setup commonly-used benchmark databases such as IMDB/JOB, Stats, etc. using the `workload-<benchmark name>-setup.sh`
scripts. These scripts will download the necessary data, create the database and import the data into the Postgres server.
Check with `--help` for details on the supported options.
