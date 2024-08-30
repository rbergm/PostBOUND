#!/bin/bash

set -e  # exit on error

attempt_pg_ext_install() {
    EXTENSION=$1
    AVAILABLE_EXTS=$(psql $DB_NAME -t -c "SELECT name FROM pg_available_extensions" | grep "$EXTENSION" || true)
    if [ -z "$AVAILABLE_EXTS" ] ; then
        echo ".. Extension $EXTENSION not available, skipping"
        return
    fi
    psql $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS $EXTENSION;"
}

# use custom TPCH scale factor if supplied
if [ -z "$1" ] ; then
    SF=10
else
    SF=$1
fi

root=$(pwd)

echo ".. Building SSB utility"
patch -fs ssb-kit/dbgen/bm_utils.c ../util/ssb_dbgen.patch || true
cd ssb-kit/dbgen
make MACHINE=LINUX DATABASE=POSTGRESQL

cd $root
echo ".. Setting up environment"
export DSS_CONFIG=$(pwd)/ssb-kit/dbgen
export DSS_QUERY=$DSS_CONFIG/queries
export DSS_PATH=../ssb_data

echo ".. Generating SSB data (SF = $SF)"
mkdir -p ../ssb_data
ssb-kit/dbgen/dbgen -vf -s $SF -T a

echo ".. Creating SSB database schema"
createdb ssb
attempt_pg_ext_install "pg_buffercache"
attempt_pg_ext_install "pg_prewarm"
attempt_pg_ext_install "pg_cooldown"
attempt_pg_ext_install "pg_hint_plan"

wget -nv --output-document ../ssb_data/pg_schema.sql https://raw.githubusercontent.com/gregrahn/ssb-kit/master/scripts/pg_schema.sql
psql ssb -f ../ssb_data/pg_schema.sql

echo ".. Inserting SSB data into database"
wget -nv --output-document ../ssb_data/pg_load.sql https://raw.githubusercontent.com/gregrahn/ssb-kit/master/scripts/pg_load.sql

# this patch may never fail since we just downloaded the live version (unless this version is changed)
patch ../ssb_data/pg_load.sql ../util/ssb_pg_load.patch
psql ssb -f ../ssb_data/pg_load.sql

echo ".. Creating SSB Foreign Key indices"
psql ssb -f workload-ssb-fk-indexes.sql

psql $DB_NAME -c "VACUUM ANALYZE;"
