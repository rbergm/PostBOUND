#!/bin/bash

WORKLOAD_DIR=../workloads
BIN_DIR=../util/bin
PG_CTL_DIR=../postgres
CWD=$(pwd)

for query in $(ls "$WORKLOAD_DIR/JOB-Queries");
do
    label=${query%.sql}
    echo "At label $label"
    $BIN_DIR/drop-caches
    cd $PG_CTL_DIR
    . ./postgres-start.sh
    cd $CWD
    psql -c "SELECT * FROM pg_buffercache_table_summary();" imdb
    python3 experiment-job-prewarming-handler.py $label
    cd $PG_CTL_DIR
    . ./postgres-stop.sh
    cd $CWD
done
