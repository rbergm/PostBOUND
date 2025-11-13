#!/bin/bash

# Setup script specific to the ReadTheDocs environment

echo "dbname=$PGDATABASE host=$PGHOST user=$PGOWNER password=$PGPASSWORD sslmode=require channel_binding=require" > docs/source/.psycopg_connection
