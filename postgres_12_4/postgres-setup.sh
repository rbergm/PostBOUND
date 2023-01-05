#!/bin/bash

WD=$(pwd)

echo ".. Cloning Postgres 12.4"
git clone --depth 1 --branch REL_12_4 https://github.com/postgres/postgres.git postgres-server
cd postgres-server


echo ".. Building Postgres v12.4"
./configure --prefix=$(pwd)/build
make clean && make && make install
export PATH="$(pwd)/build/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/build/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$(pwd)/build/include/server:$C_INCLUDE_PATH"

echo ".. Initializing Postgres Server environment"
cd $WD/postgres-server
echo "... Creating cluster"
initdb -D $(pwd)/data
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
