#!/bin/bash

# use custom TPCH scale factor if supplied
if [ -z "$1" ] ; then
    SF=10
else
    SF=$1
fi

root=$(pwd)

echo ".. Building SSB utility"
patch -fs ssb-kit/dbgen/bm_utils.c ssb_dbgen.patch
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
wget -nv --output-document ../ssb_data/pg_schema.sql https://raw.githubusercontent.com/gregrahn/ssb-kit/master/scripts/pg_schema.sql
psql ssb -f ../ssb_data/pg_schema.sql

echo ".. Inserting SSB data into database"
wget -nv --output-document ../ssb_data/pg_load.sql https://raw.githubusercontent.com/gregrahn/ssb-kit/master/scripts/pg_load.sql

# this patch may never fail since we just downloaded the live version (unless this version is changed)
patch ../ssb_data/pg_load.sql ssb_pg_load.patch
psql ssb -f ../ssb_data/pg_load.sql


echo ".. Creating SSB Foreign Key indices"
psql ssb -f ssb_fk_indexes.sql
