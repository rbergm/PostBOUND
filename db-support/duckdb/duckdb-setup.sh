#!/bin/bash

WD="$(pwd)"
VENV="$WD/.venv"
EXPLICIT_VENV="false"
DUCKDB_VER="v1.4-andium"
TARGET_DIR="$WD/quacklab"
SKIP_PULL="false"
SKIP_INSTALL="false"

function show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"

    echo -e "\n--venv <venv>"
    echo -e "\tPath to the Python virtual environment to setup the build environment."
    echo -e "\tIf an environment is already active, it will be used unless this option is provided."
    echo -e "\tIf the specified environment does not exist, it will be created."

    echo -e "\n--duckdb-ver"
    echo -eecho ".. Activating virtual environment at $VENV" "\tThe DuckDB version to install. Defaults to $DUCKDB_VER."

    echo -e "\n-d | --dir <directory>"
    echo -e "\tThe directory to install the DuckDB binary distribution in."
    echo -e "\tDefaults to '$TARGET_DIR'."

    echo -e "\n--no-update"
    echo -e "\tDon't try to pull the latest DuckDB version"

    echo -e "\n--no-install"
    echo -e "\tDon't install quacklab package into venv. Only create the wheel."
    exit $RET
}

while [ $# -gt 0 ] ; do
    case $1 in
        --venv)
            if [[ "$2" = /* ]] ; then
                VENV="$2"
            else
                VENV="$WD/$2"
            fi
            EXPLICIT_VENV="true"
            shift
            shift
            ;;
        --duckdb-ver)
            DUCKDB_VER=$2
            shift
            shift
            ;;
        -d|--dir)
            if [[ "$2" = /* ]] ; then
                TARGET_DIR="$2"
            else
                TARGET_DIR="$WD/$2"
            fi
            shift
            shift
            ;;
        --no-update)
            SKIP_PULL="true"
            shift
            ;;
        --no-install)
            SKIP_INSTALL="true"
            shift
            ;;
        --help)
            show_help 0
            ;;
        *)
            show_help 1
            ;;
    esac
done

if [ -z "$VIRTUAL_ENV" ] || [ "$EXPLICIT_VENV" = "true" ] ; then

    if [ -d "$VENV" ] ; then
        echo ".. Using existing virtual environment at $VENV"
    else
        echo ".. Creating new Python virtual environment at $VENV"
        python3 -m venv "$VENV"
    fi

fi

if [ -z "$VIRTUAL_ENV" ] && [  "$EXPLICIT_VENV" = "false" ] ; then
    echo ".. Using default virtual environment at $VENV"

    if [ ! -d "$VENV" ] ; then
        python3 -m venv "$VENV"
    fi
fi

if [[ "$DUCKDB_VER" != quacklab-* ]] ; then
    DUCKDB_VER="quacklab-$DUCKDB_VER"
fi

source "$VENV/bin/activate"
python3 -m pip install --upgrade pip

if [ ! $(command -v uv) ] ; then
    echo ".. Setting up uv"
    python3 -m pip install uv
fi

echo ".. Setting up hinting-aware DuckDB"
if [ ! -d "$TARGET_DIR" ] ; then
    git clone --recurse-submodules https://github.com/rbergm/quacklab-python.git "$TARGET_DIR"
elif [ "$SKIP_PULL" = "false" ] ; then
    cd "$TARGET_DIR"
    git pull --recurse-submodules
    git fetch --tags
    cd "$TARGET_DIR/external/duckdb"
    git fetch --tags
fi

cd "$TARGET_DIR"
git switch "$DUCKDB_VER"

echo ".. Building quacklab hinting grammar"
cd "$TARGET_DIR/external/duckdb/third_party/antlr4"
java -jar antlr-4.13.2-complete.jar -Dlanguage=Cpp "$TARGET_DIR/external/duckdb/src/hinting/grammar/HintBlock.g4"

echo ".. Compiling DuckDB and building Python package"
cd "$TARGET_DIR"
uv build

LATEST_WHEEL=$(ls dist/*.whl | sort -V | tail -n 1)
cp "$LATEST_WHEEL" "$WD"

if [ "$SKIP_INSTALL" = "false" ] ; then
    echo ".. Installing quacklab package into $VENV"
    python3 -m pip install "$LATEST_WHEEL"
fi

echo ".. Done."
cd "$WD"
