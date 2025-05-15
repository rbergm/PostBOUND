#!/bin/bash


if [ "$USE_PGLAB" = "true" ] ; then
    PG_PATH=/pg_lab
else
    PG_PATH=/postbound/db-support/postgres
fi

cd $PG_PATH
. ./postgres-load-env.sh
postgres -D $PGDATA
