#!/bin/bash

set -e  # exit on error

WD=$(pwd)
DB_NAME="ssb"
FORCE_CREATION="false"
TARGET_DIR="../ssb_data"
PG_CONN="-U $(whoami)"
SKIP_EXTENSIONS="false"
SKIP_VACUUM="false"
CLEANUP="false"
SF=10

attempt_pg_ext_install() {
    EXTENSION=$1
    AVAILABLE_EXTS=$(psql $PG_CONN $DB_NAME -t -c "SELECT name FROM pg_available_extensions" | grep "$EXTENSION" || true)
    if [ -z "$AVAILABLE_EXTS" ] ; then
        echo ".. Extension $EXTENSION not available, skipping"
        return
    fi
    psql $PG_CONN $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $EXTENSION;"
}

show_help() {
    RET=$1
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo -e "-d | --dir <directory>\tspecifies the directory to store/load the SSB data files, defaults to '../ssb_data'"
    echo -e "-s | --scale-factor\t<scale factor> specifies the scale factor to use (defaults to 10)"
    echo -e "-f | --force\t\tdelete existing instance of the database if necessary"
    echo -e "-t | --target <db name>\tname of the SSB database, defaults to 'ssb'"
    echo -e "--pg-conn <connection>\tconnection string to the PostgreSQL server (server, port, user) if required, e.g. '-h localhost -U admin'"
    echo -e "--no-fkeys\t\tdoes not load foreign key indexes to the database (includes both foreign key constraints as well as the actual indexes)"
    echo -e "--no-ext\t\tdoes not install any extensions"
    echo -e "--no-vacuum\t\tdoes not run VACUUM ANALYZE after the import"
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
    mkdir -p "$TARGET_DIR/ssb_data_$SF"
    cd $TARGET_DIR
    TARGET_DIR=$(pwd)  # get absolute path -- resolves relative target dirs

    if [ ! -d "$TARGET_DIR/ssb-kit" ] ; then
        echo ".. Building SSB utility"
        git clone https://github.com/gregrahn/ssb-kit.git
        patch -fs ssb-kit/dbgen/bm_utils.c $WD/workload-ssb-dbgen.patch || true
        cd ssb-kit/dbgen
        make MACHINE=LINUX DATABASE=POSTGRESQL
    fi

    echo ".. Setting up environment"
    cd $TARGET_DIR
    export DSS_CONFIG="$TARGET_DIR/ssb-kit/dbgen"
    export DSS_QUERY="$DSS_CONFIG/queries"
    export DSS_PATH="$TARGET_DIR/ssb_data_$SF"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "4s#.*#\\\set path '$TARGET_DIR/ssb_data_$SF/'#" ssb-kit/scripts/pg_load.sql
    else
        sed -i "4s#.*#\\\set path '$TARGET_DIR/ssb_data_$SF/'#" ssb-kit/scripts/pg_load.sql
    fi

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

if [ $CLEANUP == "true" ] ; then
    echo ".. Removing data files"
    rm -rf "$TARGET_DIR/ssb_data_$SF"
fi

echo ".. Done"
cd $WD
