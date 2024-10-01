#!/bin/bash

WD=$(pwd)

if [ -z "$1" ] ; then
	PG_INSTALL_DIR=$WD/postgres-server
elif [[ "$1" = /* ]] ; then
	PG_INSTALL_DIR="$1"
else
	PG_INSTALL_DIR="$WD/$1"
fi

cd $PG_INSTALL_DIR

PG_BIN_PATH="$PG_INSTALL_DIR/build/bin"
INIT=$(echo "$PATH" | grep "$PG_BIN_PATH")
PGPORT=$(grep "port =" data/postgresql.conf | awk '{print $3}')

if [ -z "$INIT" ] ; then
	export PG_BIN_PATH
	export PGPORT
	export PG_CTL_PATH="$WD"
	export PGDATA="$PG_INSTALL_DIR/data"
	export PATH="$PG_BIN_PATH:$PATH"
	export LD_LIBRARY_PATH="$PG_INSTALL_DIR/build/lib:$LD_LIBRARY_PATH"
	export C_INCLUDE_PATH="$PG_INSTALL_DIR/build/include/server:$C_INCLUDE_PATH"
fi

cd $WD
