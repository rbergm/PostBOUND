#!/bin/bash

PWD=$(pwd)
DB_NAME="imdb"
FORCE_CREATION="false"
IMDB_DIR="../imdb_data"

show_help() {
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo "-d | --dir <directory> specifies the directory to store/load the IMDB data files, defaults to '../imdb_data'"
    echo "-f | --force delete existing instance of the database if necessary"
    echo "-t | --target <db name> name of the IMDB database, defaults to 'imdb'"
    exit 1
}

while [ $# -gt 0 ] ; do
    case $1 in
        -d|--dir)
            IMDB_DIR=$2
            shift
            shift
            ;;
        -f|--force)
            FORCE_CREATION="true"
            shift
            ;;
        -t|--target)
            DB_NAME=$2
            shift
            shift
            ;;
        *)
            show_help
            ;;
    esac
done

EXISTING_DBS=$(psql -l | grep "$DB_NAME")

if [ ! -z "$EXISTING_DBS" ] && [ $FORCE_CREATION = "false" ] ; then
    echo ".. IMDB exists, doing nothing"
    exit 0
fi

if [ ! -z "$EXISTING_DBS" ] ; then
    dropdb $DB_NAME
fi

echo ".. IMDB source directory is $IMDB_DIR"

if [ -d $IMDB_DIR ] ; then
    echo ".. Re-using existing IMDB input data"
else
    echo ".. IMDB source directory does not exist, re-creating"
    echo ".. Fetching IMDB data"
    mkdir $IMDB_DIR
    curl -o $IMDB_DIR/csv.zip "https://cloudstore.zih.tu-dresden.de/index.php/s/eqWWK53CgkxMxfA/download?path=%2F&files=csv.zip"
    curl -o $IMDB_DIR/create.sql "https://cloudstore.zih.tu-dresden.de/index.php/s/eqWWK53CgkxMxfA/download?path=%2F&files=create.sql"
    curl -o $IMDB_DIR/import.sql "https://cloudstore.zih.tu-dresden.de/index.php/s/eqWWK53CgkxMxfA/download?path=%2F&files=import.sql"
    curl -o $IMDB_DIR/fkindexes.sql "https://cloudstore.zih.tu-dresden.de/index.php/s/eqWWK53CgkxMxfA/download?path=%2F&files=fkindexes.sql"

    echo ".. Extracting IMDB data"
    unzip $IMDB_DIR/csv.zip -d $IMDB_DIR
fi

echo ".. Creating IMDB database"
createdb $DB_NAME
psql $DB_NAME -c "CREATE EXTENSION pg_buffercache;"
psql $DB_NAME -c "CREATE EXTENSION pg_prewarm;"
psql $DB_NAME -c "CREATE EXTENSION pg_hint_plan;"

echo ".. Loading IMDB database schema"
psql $DB_NAME -f $IMDB_DIR/create.sql

echo ".. Inserting IMDB data into database"
cd $IMDB_DIR
psql $DB_NAME -f import.sql

echo ".. Creating IMDB foreign key indices"
psql $DB_NAME -f fkindexes.sql
psql $DB_NAME -c "CREATE INDEX IF NOT EXISTS subject_id_complete_cast ON complete_cast(subject_id)"
psql $DB_NAME -c "CREATE INDEX IF NOT EXISTS status_id_complete_cast ON complete_cast(status_id)"

echo ".. Done"
cd $PWD
