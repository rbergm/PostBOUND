#!/bin/bash

# Setup script specific to the ReadTheDocs environment

cd db-support/postgres
./postgres-setup.sh --stop
. ./postgres-start.sh
./workload-stats-setup.sh
./postgres-psycopg-setup.sh stats stats
cp .psycopg_connection_stats ../../.psycopg_connection
