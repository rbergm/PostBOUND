#!/bin/bash

WD=$(pwd)
USER=$(whoami)

echo ".. Cloning Postgres 14"
git clone --depth 1 --branch REL_14_STABLE https://github.com/postgres/postgres.git postgres-server
cd postgres-server

echo ".. Downloading pg_hint_plan extension"
curl -L https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/REL14_1_4_0.tar.gz -o contrib/pg_hint_plan.tar.gz

echo ".. Building Postgres v14"
./configure --prefix=$(pwd)/build
make clean && make && make install
export PATH="$(pwd)/build/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/build/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$(pwd)/build/include/server:$C_INCLUDE_PATH"

echo ".. Installing pg_hint_plan extension"
cd contrib
tar xzvf pg_hint_plan.tar.gz
mv pg_hint_plan-REL14_1_4_0 pg_hint_plan
cd pg_hint_plan
make && make install

echo ".. Initializing Postgres Server environment"
cd $WD/postgres-server
echo "... Creating cluster"
initdb -D $(pwd)/data
echo "... Adding pg_hint_plan to preload libraries"
sed -i "s/shared_preload_libraries.*/shared_preload_libraries = 'pg_hint_plan'/" $(pwd)/data/postgresql.conf
echo "... Starting Postgres (log file is pg.log)"
pg_ctl -D $(pwd)/data -l pg.log start
echo "... Creating user database for $USER"
createdb $USER

if [ "$1" = "--stop" ] ; then
    pg_ctl -D $(pwd)/postgres-server/data stop
    echo ".. Setup done"
else
    echo ".. Setup done, ready to connect"
fi
