# DuckDB management

To use DuckDB with PostBOUND, we need to perform some modifications to the vanilla DuckDB source code.
These modifications add the necessary hooks for PostBOUND to influence the internal optimizer behavior.
All modifications are captured in the [quacklab](https://github.com/rbergm/quacklab) project.
Here, we use the [Python-packaged version](https://github.com/rbergm/quacklab-python) of quacklab.

## Installation

The simplest way to install the DuckDB/quacklab Python package is via pip:

```bash
pip install quacklab
```

quacklab provides pre-built binary wheels for a variety of Python version for Linux and MacOS.
An installation from source will take significantly longer, as DuckDB needs to be compiled from scratch.

## Manual Setup

The Python package can be built using the `duckdb-setup.sh` script. This script will:

1. Pull the latest (or a specified) version of quacklab
2. Compile DuckDB/quacklab
3. Compile the Python bindings for DuckDB/quacklab
4. Create the Python package

Since we compile DuckDB from source, this process may take some time.
You also need to satisfy the build dependencies for DuckDB (currently a C++ toolchain and uv, but see the
[official documentation](https://duckdb.org/docs/stable/dev/building/python) for the authoritative list).
If not already installed, uv will be installed wihtin the target environmnent.

The setup script requires a Python virtual environment to function properly. If not specified, an environment will be
automatically created at `.venv` in the current directory. Use `--help` to see all available options.

## Databases

You can create instances of popular databases (currently JOB/IMDB and Stats) using the `workload-setup.py` script.
Use `--help` to see all available options.
