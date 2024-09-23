#!/bin/bash

set -e  # exit on error

WD=$(pwd)
DB_NAME="stats"
FORCE_CREATION="false"
STATS_DIR="../stats_data"
PG_CONN="-U $USER"
SKIP_EXTENSIONS="false"
SKIP_VACUUM="false"

show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir\t<directory> specifies the directory to store/load the stats data files, defaults to '../stats_data'"
    echo -e "-f | --force\tdelete existing instance of the database if necessary"
    echo -e "-t | --target\t<db name> name of the stats database, defaults to 'stats'"
    echo -e "--pg-conn\t<connection> connection string to the PostgreSQL server (server, port, user) if required, e.g. '-h localhost -U admin'"
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
            STATS_DIR=$2
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
    echo ".. Stats exists, doing nothing"
    exit 0
fi

if [ ! -z "$EXISTING_DBS" ] ; then
    dropdb $PG_CONN $DB_NAME
fi

echo ".. Stats source directory is $STATS_DIR"

if [ -d $STATS_DIR ] ; then
    echo ".. Re-using existing Stats input data"
else
    echo ".. Stats source directory does not exist, re-creating"
    echo ".. Fetching Stats data"
    mkdir $STATS_DIR
    curl -o $STATS_DIR/csv.zip "https://db4701.inf.tu-dresden.de:8443/index.php/s/JSeYtLidKkaERGR/download?path=%2F&files=csv.zip"
    curl -o $STATS_DIR/schema.sql "https://db4701.inf.tu-dresden.de:8443/index.php/s/JSeYtLidKkaERGR/download?path=%2F&files=schema.sql"
    curl -o $STATS_DIR/import.sql "https://db4701.inf.tu-dresden.de:8443/index.php/s/JSeYtLidKkaERGR/download?path=%2F&files=import.sql"

    echo ".. Extracting Stats data"
    unzip $STATS_DIR/csv.zip -d $STATS_DIR
fi

echo ".. Creating Stats database"
createdb $PG_CONN $DB_NAME

if [ $SKIP_EXTENSIONS == "false" ] ; then
    attempt_pg_ext_install "pg_buffercache"
    attempt_pg_ext_install "pg_prewarm"
    attempt_pg_ext_install "pg_cooldown"
    attempt_pg_ext_install "pg_hint_plan"
else
    echo ".. Skipping extension generation"
fi

echo ".. Loading Stats database schema"
psql $PG_CONN $DB_NAME -f $STATS_DIR/schema.sql

echo ".. Inserting Stats data into database"
cd $STATS_DIR
psql $PG_CONN $DB_NAME -f import.sql

if [ $SKIP_VACUUM == "false" ] ; then
    echo ".. Vacuuming database"
    psql $PG_CONN $DB_NAME -c "VACUUM ANALYZE;"
else
    echo ".. Skipping vacuuming"
fi

echo ".. Done"
cd $WD
