#!/bin/sh

WD=$(pwd)

if [ "$THESIS_PG_ENV_LOADED" != "true" ] ; then
	. ./postgres-load-env.sh
fi

cd postgres-server

pg_ctl -D $(pwd)/data -l pg.log start

cd ..

