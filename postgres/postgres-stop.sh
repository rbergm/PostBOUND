#!/bin/sh

echo ".. Stopping Postgres Server"
pg_ctl -D $(pwd)/postgres-server/data stop

