# Postgres management

This folder provides utilities to

- build a Postgres server instance from source `postgres-setup.sh`. Check with `--help` for details
- install a number of recommended and/or required extensions (this is also handled as part of the general Postgres setup)
- manage an installed Postgres server (start via `postgres-start.sh` and stop via `postgres-stop.sh`)
- import popular benchmark databases via the <code>workload-_\<benchmark name\>_-setup.sh</code> scripts. Use `--help` to see supported options

Notice that the Postgres setup performs a _local_ installation, no global paths or public
directories are modified. Therefore, use the `postgres-load-env.sh` utility to setup you `PATH`
environment variable correctly for the current session. The `-start`, `-stop` and `-setup`
scripts can also be sourced to keep the `PATH` up to date (i.e. adding the Postgres binaries
to the `PATH` upon server start and removing them again afterwards). If the Postgres server
was installed to a custom location, this location has to be supplied when calling the
management scripts.

Finally, the `postgres-psycopg-setup` utility generates a config file that can be used by
PostBOUND's database interaction code to automatically connect to the database. See the
documentation in `postgres.py` for details.

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
    git unzip zstd
```
