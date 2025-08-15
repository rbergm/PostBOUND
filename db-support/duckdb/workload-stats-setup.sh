#!/bin/bash

set -e  # exit on error

WD=$(pwd)
DB_NAME="$WD/stats.duckdb"
FORCE_CREATION="false"
TARGET_DIR="../stats_data"
CLEANUP="false"

show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir <directory>\tspecifies the directory to store/load the Stats data files, defaults to '../stats_data'"
    echo -e "-f | --force\t\tdelete existing instance of the database if necessary"
    echo -e "-t | --target <db name>\tname of the DuckDB database to create. Defaults to 'stats.duckdb' in the current directory."
    echo -e "--no-fkeys\t\tdoes not load foreign key indexes to the database (includes both foreign key constraints as well as the actual indexes)"
    echo -e "--cleanup\t\tremove the data files after import"
    exit $RET
}

while [ $# -gt 0 ] ; do
    case $1 in
        -d|--dir)
            TARGET_DIR=$2
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

if [ -f "$DB_NAME" ] && [ $FORCE_CREATION = "false" ] ; then
    echo ".. Stats exists, doing nothing"
    exit 2
fi

if [ -f "$DB_NAME" ] ; then
    echo ".. Removing existing Stats database"
    rm $DB_NAME
fi

echo ".. Stats source directory is $TARGET_DIR"

if [ -d $TARGET_DIR ] ; then
    echo ".. Re-using existing Stats input data"
else
    echo ".. Stats source directory does not exist, re-creating"
    echo ".. Fetching Stats data"
    mkdir $TARGET_DIR
    curl -o $TARGET_DIR/csv.zip "https://db4701.inf.tu-dresden.de:8443/public.php/dav/files/p8eRRMEERQE9nXC"
    curl -o $TARGET_DIR/schema.sql "https://db4701.inf.tu-dresden.de:8443/public.php/dav/files/q7jcDYDSPqsDJ8j"
    curl -o $TARGET_DIR/import.sql "https://db4701.inf.tu-dresden.de:8443/public.php/dav/files/9LPBERBcwtcqg3M"

    echo ".. Extracting Stats data"
    unzip $TARGET_DIR/csv.zip -d $TARGET_DIR
fi

echo ".. Creating Stats database"
cd $TARGET_DIR
cp $WD/sql/stats-schema.sql schema_duckdb.sql
cp $WD/sql/stats-import.sql import_duckdb.sql
duckdb $DB_NAME -f schema_duckdb.sql
duckdb $DB_NAME -f import_duckdb.sql