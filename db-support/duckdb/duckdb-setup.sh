#!/bin/bash

WD=$(pwd)
VENV="$WD/../../pb-venv"
EXPLICIT_VENV="false"
TARGET_DIR="$WD/duckdb-lab"

function show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "--venv <venv>\tpath to the Python virtual environment to install the DuckDB package into. Defaults to '$VENV'."
    echo -e "-d | --dir <directory>\tthe directory to install the DuckDB binary distribution in. Defaults to '$TARGET_DIR'."
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
        -d|--dir)
            if [[ "$2" = /* ]] ; then
                TARGET_DIR="$2"
            else
                TARGET_DIR="$WD/$2"
            fi
            shift
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

echo ".. Setting up hinting-aware DuckDB"

if [ -d "$TARGET_DIR" ] ; then
    echo ".. Re-using existing DuckDB build directory at $TARGET_DIR"
    cd "$TARGET_DIR"
    git pull
    git fetch --tags
else
    git clone https://github.com/rbergm/ducklab.git "$TARGET_DIR"
    cd "$TARGET_DIR"
    git fetch --tags
fi

source "$VENV/bin/activate"

python3 -m pip install --upgrade pip

BUILD_PYTHON=1 make -j 12 release

if [ -n "$BASH_VERSION" -a "$BASH_SOURCE" != "$0" ] || [ -n "$ZSH_VERSION" -a "$ZSH_EVAL_CONTEXT" != "toplevel" ] ; then
    echo ".. Adding DuckDB executable to the system PATH"
    export PATH="$TARGET_DIR/build/release:$PATH"
fi

echo ".. Done."
cd $WD
