#!/bin/bash

set -e  # exit on error

WD=$(pwd)
DB_NAME="$WD/imdb.duckdb"
FORCE_CREATION="false"
TARGET_DIR="../imdb_data"
CLEANUP="false"

show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir <directory>\tspecifies the directory to store/load the IMDB data files, defaults to '../imdb_data'"
    echo -e "-f | --force\t\tdelete existing instance of the database if necessary"
    echo -e "-t | --target <db name>\tname of the DuckDB database to create. Defaults to 'imdb.duckdb' in the current directory."
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
    echo ".. IMDB exists, doing nothing"
    exit 2
fi

if [ -f "$DB_NAME" ] ; then
    echo ".. Removing existing IMDB database"
    rm $DB_NAME
fi

echo ".. IMDB source directory is $TARGET_DIR"

if [ -d $TARGET_DIR ] ; then
    echo ".. Re-using existing IMDB input data"
else
    echo ".. IMDB source directory does not exist, re-creating"
    echo ".. Fetching IMDB data"
    mkdir $TARGET_DIR
    curl -L -o $TARGET_DIR/csv.zip "https://db4701.inf.tu-dresden.de:8443/index.php/s/H7TKaEBr5JmdaNA/download/csv.zip"
    curl -L -o $TARGET_DIR/create.sql "https://db4701.inf.tu-dresden.de:8443/index.php/s/e35mDHTCZx88y6p/download/create.sql"
    curl -L -o $TARGET_DIR/import.sql "https://db4701.inf.tu-dresden.de:8443/index.php/s/bNzMwSpmQESRz6P/download/import.sql"

    echo ".. Extracting IMDB data"
    unzip $TARGET_DIR/csv.zip -d $TARGET_DIR
fi

echo ".. Creating IMDB database"
cd $TARGET_DIR
cp $WD/sql/imdb-schema.sql schema_duckdb.sql
cp $WD/sql/imdb-import.sql import_duckdb.sql
duckdb $DB_NAME -f schema_duckdb.sql
duckdb $DB_NAME -f import_duckdb.sql