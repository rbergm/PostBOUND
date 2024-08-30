#!/bin/bash

set -e  # exit on error

DB_NAME=stack
WD=$(pwd)
SSB_DIR=$WD/../stack_data
CORES=$(($(nproc --all) / 2))
FORCE_CREATION="false"
SKIP_EXTENSIONS="false"
SKIP_VACUUM="false"

show_help() {
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo "-d | --dir <directory> specifies the directory to store/load the Stack data files, defaults to '../stack_data'"
    echo "-t | --target <db name> name of the Stack database, defaults to 'stack'"
    echo "-j | --jobs <count> configure the number of worker processes to load the database. Defaults to 1/2 of CPU cores"
    echo "-f | --force delete existing instance of the database if necessary"
    exit 1
}

attempt_pg_ext_install() {
    EXTENSION=$1
    AVAILABLE_EXTS=$(psql $DB_NAME -t -c "SELECT name FROM pg_available_extensions" | grep "$EXTENSION" || true)
    if [ -z "$AVAILABLE_EXTS" ] ; then
        echo ".. Extension $EXTENSION not available, skipping"
        return
    fi
    psql $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $EXTENSION;"
}

while [ $# -gt 0 ] ; do
    case $1 in
        -d|--dir)
            SSB_DIR=$2
            shift
            shift
            ;;
        -j|--jobs)
            CORES=$2
            shift
            ;;
        -t|--target)
            DB_NAME=$2
            shift
            shift
            ;;
        -f|--force)
            FORCE_CREATION="true"
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
        *)
            show_help
            ;;
    esac
done

EXISTING_DBS=$(psql -l | grep "$DB_NAME" || true)

if [ ! -z "$EXISTING_DBS" ] && [ $FORCE_CREATION = "false" ] ; then
    echo ".. Stack database exists, doing nothing"
    exit 0
fi

if [ ! -z "$EXISTING_DBS" ] ; then
    dropdb $DB_NAME
fi

echo ".. Initializing Stack data dir at $SSB_DIR"
mkdir -p $SSB_DIR

if [ -f $SSB_DIR/stack_dump ] ;  then
    echo ".. Re-using existing Stack dump"
else
    echo ".. Downloading Stack database dump"
    curl --location "https://www.dropbox.com/s/55bxfhilcu19i33/so_pg13?dl=1" --output $SSB_DIR/stack_dump
fi

echo ".. Creating Stack database"
createdb $DB_NAME

if [ $SKIP_EXTENSIONS == "false" ] ; then
    attempt_pg_ext_install "pg_buffercache"
    attempt_pg_ext_install "pg_prewarm"
    attempt_pg_ext_install "pg_cooldown"
    attempt_pg_ext_install "pg_hint_plan"
else
    echo ".. Skipping extension generation"
fi

echo ".. Loading Stack dump"
pg_restore -O -x --exit-on-error --jobs=$CORES --dbname=$DB_NAME $SSB_DIR/stack_dump

if [ $SKIP_VACUUM == "false" ] ; then
    echo ".. Vacuuming database"
    psql $DB_NAME -c "VACUUM ANALYZE;"
else
    echo ".. Skipping vacuuming"
fi

echo ".. Done"
cd $WD
