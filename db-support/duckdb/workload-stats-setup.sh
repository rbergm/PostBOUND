#!/bin/bash

set -e  # exit on error

WD="$(pwd)"
DB_NAME="$WD/stats.duckdb"
FORCE_CREATION="false"
TARGET_DIR="../stats_data"
CLEANUP="false"

show_help() {
    RET="$1"
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir <directory>\tspecifies the directory to store/load the Stats data files, defaults to '$TARGET_DIR'"
    echo -e "-f | --force\t\tdelete existing instance of the database if necessary"
    echo -e "-t | --target <db name>\tname of the DuckDB database to create. Defaults to 'stats.duckdb' in the current directory."
    echo -e "--no-fkeys\t\tdoes not load foreign key indexes to the database (includes both foreign key constraints as well as the actual indexes)"
    echo -e "--cleanup\t\tremove the data files after import"
    exit "$RET"
}

while [ $# -gt 0 ] ; do
    case $1 in
        -d|--dir)
            TARGET_DIR="$2"
            shift
            shift
            ;;
        -f|--force)
            FORCE_CREATION="true"
            shift
            ;;
        -t|--target)
            if [[ "$2" = /* ]] ; then
                DB_NAME="$2"
            else
                DB_NAME="$WD/$2"
            fi
            shift
            shift
            ;;
        --cleanup)
            CLEANUP="true"
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

echo ".. Working directory is $WD"

if [ -f "$DB_NAME" ] && [ "$FORCE_CREATION" = "false" ] ; then
    echo ".. Stats exists, doing nothing"
    exit 2
fi

if [ -f "$DB_NAME" ] ; then
    echo ".. Removing existing Stats database"
    rm "$DB_NAME"
fi

echo ".. Stats source directory is $TARGET_DIR"

# Even if the current implementation of downloading to an outside directory first and then copying
# the database around might seem unnecessarily complex, we keep this logic for a simple reason:
# this allows us to to have a "working copy" of the database under DB_NAME. In case users mess around
# with the data and need to restore the original dump, we can do so without any additional downloads.

SRC_FILE="$TARGET_DIR/stats.duckdb"
if [ -f "$SRC_FILE" ] ; then
    echo ".. Re-using existing Stats input data"
else
    echo ".. Stats source directory does not exist, re-creating"
    echo ".. Fetching Stats data"
    mkdir -p "$TARGET_DIR"
    curl -o "$SRC_FILE" "https://zenodo.org/records/19131189/files/stats.duckdb?download=1"
fi

echo ".. Creating Stats database"
cp "$SRC_FILE" "$DB_NAME"

if [ "$CLEANUP" = "true" ] ; then
    echo ".. Removing Stats input data"
    rm "$SRC_FILE"
fi

echo ".. Done"
