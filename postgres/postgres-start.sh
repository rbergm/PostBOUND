#!/bin/sh

WD=$(pwd)
. ./postgres-load-env.sh "$1"

cd postgres-server
pg_ctl -D $(pwd)/data -l pg.log start
cd ..
