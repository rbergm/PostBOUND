#!/bin/bash

root=$(pwd)

echo ".. Building TPC-H utility"
patch ssb-kit/dbgen/bm_utils.c tpch_dbgen.patch
cd ssb-kit/dbgen
make MACHINE=LINUX DATABASE=POSTGRESQL

cd $root
echo ".. Setting up environment"
export DSS_CONFIG=$(pwd)/ssb-kit/dbgen
export DSS_QUERY=$DSS_CONFIG/queries
export DSS_PATH=../tpch_data

echo ".. Generating TPCH data (SF = 0.5)"
mkdir ../tpch_data
ssb-kit/dbgen/dbgen -vf -s 0.5 -T a

echo ".. Inserting TPCH data into database"
wget -nv --output-document ../tpch_data/pg_schema.sql https://raw.githubusercontent.com/gregrahn/ssb-kit/master/scripts/pg_schema.sql
psql tpch -f ../tpch_data/pg_schema.sql

wget -nv --output-document ../tpch_data/pg_load.sql https://raw.githubusercontent.com/gregrahn/ssb-kit/master/scripts/pg_load.sql
patch ../tpch_data/pg_load.sql tpch_pg_load.patch
psql tpch -f ../tpch_data/pg_load.sql
