#!/bin/bash

set -e  # exit on error

WD=$(pwd)
DB_NAME="ssb"
FORCE_CREATION="false"
TARGET_DIR="../ssb_data"
PG_CONN="-U $USER"
SKIP_EXTENSIONS="false"
SKIP_VACUUM="false"
SF=10

attempt_pg_ext_install() {
    EXTENSION=$1
    AVAILABLE_EXTS=$(psql $DB_NAME -t -c "SELECT name FROM pg_available_extensions" | grep "$EXTENSION" || true)
    if [ -z "$AVAILABLE_EXTS" ] ; then
        echo ".. Extension $EXTENSION not available, skipping"
        return
    fi
    psql $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $EXTENSION;"
}

show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir\t\t<directory> specifies the directory to store/load the stats data files, defaults to '../ssb_data'"
    echo -e "-s | --scale-factor\t<scale factor> specifies the scale factor to use (defaults to 10)"
    echo -e "-f | --force\t\tdelete existing instance of the database if necessary"
    echo -e "-t | --target\t\t<db name> name of the ssb database, defaults to 'ssb'"
    echo -e "--pg-conn\t\t<connection> connection string to the PostgreSQL server (server, port, user) if required, e.g. '-h localhost -U admin'"
    echo -e "--no-ext\t\tdoes not install any extensions"
    echo -e "--no-vacuum\t\tdoes not run VACUUM ANALYZE after the import"
    exit $RET
}

while [ $# -gt 0 ] ; do
    case $1 in
        -d|--dir)
            TARGET_DIR=$2
            shift
            shift
            ;;
        -s|--scale-factor)
            SF=$2
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
    echo ".. SSB exists, doing nothing"
    exit 0
fi

if [ ! -z "$EXISTING_DBS" ] ; then
    dropdb $PG_CONN $DB_NAME
fi

echo ".. SSB source directory is $TARGET_DIR"

if [ -d "$TARGET_DIR/ssb_data_$SF" ] ; then
    echo ".. Re-using existing SSB input data"
else
    echo ".. SSB source directory does not exist, re-creating"
    cd $TARGET_DIR
    TARGET_DIR=$(pwd)  # get absolute path -- resolves relative target dirs
    mkdir -p "$TARGET_DIR/ssb_data_$SF"

    if [ ! -d "$TARGET_DIR/ssb-kit" ] ; then
        echo ".. Building SSB utility"
        git clone https://github.com/gregrahn/ssb-kit.git
        patch -fs ssb-kit/dbgen/bm_utils.c ../util/ssb_dbgen.patch || true
        cd ssb-kit/dbgen
        make MACHINE=LINUX DATABASE=POSTGRESQL
    fi

    echo ".. Setting up environment"
    cd $TARGET_DIR
    export DSS_CONFIG="$TARGET_DIR/ssb-kit/dbgen"
    export DSS_QUERY="$DSS_CONFIG/queries"
    export DSS_PATH="$TARGET_DIR/ssb_data_$SF"
    sed -i "4s#.*#\\\set path '$TARGET_DIR/ssb_data_$SF/'#" ssb-kit/scripts/pg_load.sql

    echo ".. Generating SSB data (SF = $SF)"
    ssb-kit/dbgen/dbgen -vf -s $SF -T a
fi

echo ".. Creating SSB database schema"
createdb $PG_CONN $DB_NAME
attempt_pg_ext_install "pg_buffercache"
attempt_pg_ext_install "pg_prewarm"
attempt_pg_ext_install "pg_cooldown"
attempt_pg_ext_install "pg_hint_plan"

echo ".. Loading SSB database schema"
cd $TARGET_DIR
psql $PG_CONN $DB_NAME -f $TARGET_DIR/ssb-kit/scripts/pg_schema.sql

echo ".. Inserting SSB data into database"
psql $PG_CONN $DB_NAME -f $TARGET_DIR/ssb-kit/scripts/pg_load.sql

echo ".. Creating SSB Foreign Key indices"
psql $PG_CONN $DB_NAME -f $WD/workload-ssb-fk-indexes.sql

if [ $SKIP_VACUUM == "false" ] ; then
    echo ".. Vacuuming database"
    psql $PG_CONN $DB_NAME -c "VACUUM ANALYZE;"
else
    echo ".. Skipping vacuuming"
fi

echo ".. Done"
cd $WD
