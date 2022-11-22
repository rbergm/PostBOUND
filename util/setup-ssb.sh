#!/bin/bash

# use custom TPCH scale factor if supplied
if [ -z "$1" ] ; then
    SF=10
else
    SF=$1
fi

root=$(pwd)

echo ".. Building TPC-H utility"
patch -fs ssb-kit/dbgen/bm_utils.c tpch_dbgen.patch
cd ssb-kit/dbgen
make MACHINE=LINUX DATABASE=POSTGRESQL

cd $root
echo ".. Setting up environment"
export DSS_CONFIG=$(pwd)/ssb-kit/dbgen
export DSS_QUERY=$DSS_CONFIG/queries
export DSS_PATH=../tpch_data

echo ".. Generating TPCH data (SF = $SF)"
mkdir -p ../tpch_data
ssb-kit/dbgen/dbgen -vf -s $SF -T a

echo ".. Creating TPC-H database schema"
createdb tpch
wget -nv --output-document ../tpch_data/pg_schema.sql https://raw.githubusercontent.com/gregrahn/ssb-kit/master/scripts/pg_schema.sql
psql tpch -f ../tpch_data/pg_schema.sql

echo ".. Inserting TPC-H data into database"
wget -nv --output-document ../tpch_data/pg_load.sql https://raw.githubusercontent.com/gregrahn/ssb-kit/master/scripts/pg_load.sql

# this patch may never fail since we just downloaded the live version (unless this version is changed)
patch ../tpch_data/pg_load.sql tpch_pg_load.patch
psql tpch -f ../tpch_data/pg_load.sql


echo ".. Creating SSB Foreign Key indices"
psql tpch -f tpch_fk_indexes.sql
