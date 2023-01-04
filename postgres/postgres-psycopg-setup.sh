#!/bin/bash

if [ $# -lt 2 ] || [ "$1" = "--help" ] ; then
    echo "Usage: $0 <file suffix> <db name>"
    exit 0
fi

touch .psycopg_connection_$1
echo "dbname=$2 user=$USER host=localhost" >> .psycopg_connection_$1
