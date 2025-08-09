#!/bin/bash

ACTION="load"
WD=$(pwd)
TARGET_DIR="$WD/quacklab"

show_help() {
    RET=$1
    echo "Usage: $0 <options> <action>"
    echo "Allowed options:"
    echo -e "<action>\tAction to perform: 'load' to load the environment variables, 'unload' to unload them."
    echo -e "-d | --dir\tDirectory containing the DuckDB build."
    echo -e "-h | --help\tShow this help message."
    exit $RET
}

while [ $# -gt 0 ] ; do
    case $1 in
        -d|--dir)
            if [[ "$2" = /* ]] ; then
                TARGET_DIR="$2"
            else
                TARGET_DIR="$WD/$2"
            fi
            shift
            shift
            ;;
        -h|--help)
            show_help 0
            ;;
        *)
            if [[ "$1" == "load" || "$1" == "unload" ]] ; then
                ACTION="$1"
                shift
            else
                show_help 1
            fi
            ;;
    esac
done

if [ -n "$BASH_VERSION" -a "$BASH_SOURCE" = "$0" ] || [ -n "$ZSH_VERSION" -a "$ZSH_EVAL_CONTEXT" = "toplevel" ] ; then
    echo "$0 must be sourced!" 1>&2
    exit 2
fi

DUCKDB_EXEC_PATH="$TARGET_DIR/build/release"

if [ "$ACTION" = "load" ] ; then
    export PATH="$DUCKDB_EXEC_PATH:$PATH"
else
    export PATH="${PATH//$DUCKDB_EXEC_PATH:}"
fi
