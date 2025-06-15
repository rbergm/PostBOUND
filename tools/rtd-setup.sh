#!/bin/bash

# Setup script specific to the ReadTheDocs environment

cd db-support/postgres
./postgres-setup.sh --stop
. ./postgres-start.sh
./workload-job-setup.sh
./postgres-psycopg-setup.sh job imdb
cp .psycopg_connection_job ../../.psycopg_connection
