#!/bin/sh

WD=$(pwd)

. ./postgres-load-env.sh

cd postgres-server

pg_ctl -D $(pwd)/data -l pg.log start

