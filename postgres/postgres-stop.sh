#!/bin/bash

WD=$(pwd)

if [ -z "$1" ] ; then
	PG_INSTALL_DIR=$WD/postgres-server
elif [[ "$1" = /* ]] ; then
	PG_INSTALL_DIR="$1"
else
	PG_INSTALL_DIR="$WD/$1"
fi

echo ".. Stopping Postgres Server"
. ./postgres-load-env.sh "$1"

cd $PG_INSTALL_DIR
pg_ctl -D $PG_INSTALL_DIR/data stop
export PATH="${PATH//$PG_CTL_PATH:}"

cd $WD
