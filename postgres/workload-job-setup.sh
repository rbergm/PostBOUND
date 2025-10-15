#!/bin/bash

set -e  # exit on error

WD=$(pwd)
DB_NAME="imdb"
FORCE_CREATION="false"
TARGET_DIR="../imdb_data"
PG_CONN="-U $USER"
SKIP_FKEYS="false"
SKIP_EXTENSIONS="false"
SKIP_VACUUM="false"

show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir\t<directory> specifies the directory to store/load the IMDB data files, defaults to '../imdb_data'"
    echo -e "-f | --force\tdelete existing instance of the database if necessary"
    echo -e "-t | --target\t<db name> name of the IMDB database, defaults to 'imdb'"
    echo -e "--pg-conn\t<connection> connection string to the PostgreSQL server (server, port, user) if required, e.g. '-h localhost -U admin'"
    echo -e "--no-fkeys\tdoes not load foreign key indexes to the database (includes both foreign key constraints as well as the actual indexes)"
    echo -e "--no-ext\tdoes not install any extensions"
    echo -e "--no-vacuum\tdoes not run VACUUM ANALYZE after the import"
    exit $RET
}

attempt_pg_ext_install() {
    EXTENSION=$1
    AVAILABLE_EXTS=$(psql $PG_CONN $DB_NAME -t -c "SELECT name FROM pg_available_extensions" | grep "$EXTENSION" || true)
    if [ -z "$AVAILABLE_EXTS" ] ; then
        echo ".. Extension $EXTENSION not available, skipping"
        return
    fi
    psql $PG_CONN $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $EXTENSION;"
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
            DB_NAME=$2
            shift
            shift
            ;;
        --no-fkeys)
            SKIP_FKEYS="true"
            shift
            ;;
        --pg-conn)
            PG_CONN=$2
            shift
            shift
            ;;

        --no-ext)
            SKIP_EXTENSIONS="true"
            shift
            ;;
        --no-vacuum)
            SKIP_VACUUM="true"
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

EXISTING_DBS=$(psql $PG_CONN -l | grep "$DB_NAME" || true)

echo ".. Working directory is $WD"

if [ ! -z "$EXISTING_DBS" ] && [ $FORCE_CREATION = "false" ] ; then
    echo ".. IMDB exists, doing nothing"
    exit 0
fi

if [ ! -z "$EXISTING_DBS" ] ; then
    dropdb $PG_CONN $DB_NAME
fi

echo ".. IMDB source directory is $TARGET_DIR"

if [ -d $TARGET_DIR ] ; then
    echo ".. Re-using existing IMDB input data"
else
    echo ".. IMDB source directory does not exist, re-creating"
    echo ".. Fetching IMDB data"
    mkdir $TARGET_DIR
    curl -k -L -o $TARGET_DIR/csv.zip "https://db4701.inf.tu-dresden.de:8443/index.php/s/H7TKaEBr5JmdaNA/download/csv.zip"
    curl -k -L -o $TARGET_DIR/create.sql "https://db4701.inf.tu-dresden.de:8443/index.php/s/e35mDHTCZx88y6p/download/create.sql"
    curl -k -L -o $TARGET_DIR/import.sql "https://db4701.inf.tu-dresden.de:8443/index.php/s/bNzMwSpmQESRz6P/download/import.sql"

    echo ".. Extracting IMDB data"
    unzip $TARGET_DIR/csv.zip -d $TARGET_DIR
fi

echo ".. Creating IMDB database"
createdb $PG_CONN $DB_NAME

if [ $SKIP_EXTENSIONS == "false" ] ; then
    attempt_pg_ext_install "pg_buffercache"
    attempt_pg_ext_install "pg_prewarm"
    attempt_pg_ext_install "pg_cooldown"
    attempt_pg_ext_install "pg_hint_plan"
else
    echo ".. Skipping extension generation"
fi

echo ".. Loading IMDB database schema"
psql $PG_CONN $DB_NAME -f $TARGET_DIR/create.sql

echo ".. Inserting IMDB data into database"
cd $TARGET_DIR
psql $PG_CONN $DB_NAME -f import.sql

if [ $SKIP_FKEYS == "false" ] ; then
	echo ".. Creating IMDB foreign key indices"
	psql $PG_CONN $DB_NAME -f $WD/workload-job-fk-indexes.sql
else
	echo ".. Skipping IMDB foreign key creation"
fi

if [ $SKIP_VACUUM == "false" ] ; then
    echo ".. Vacuuming database"
    psql $PG_CONN $DB_NAME -c "VACUUM ANALYZE;"
else
    echo ".. Skipping vacuuming"
fi

echo ".. Done"
cd $WD
