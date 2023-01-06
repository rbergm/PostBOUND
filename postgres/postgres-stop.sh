#!/bin/bash

echo ".. Stopping Postgres Server"
. ./postgres-load-env.sh
pg_ctl -D $(pwd)/postgres-server/data stop
export PATH="${PATH//$PG_CTL_PATH:}"
