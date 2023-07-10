#!/bin/bash

WD=$(pwd)
cd "$WD/mysql-server/mysql-server"
MYSQL_CTL_PATH="$(pwd)/bin"
INIT=$(echo "$PATH" | grep "$MYSQL_CTL_PATH")

if [ -z "$INIT" ] ; then
	export MYSQL_CTL_PATH
	export PATH="$MYSQL_CTL_PATH:$PATH"
fi

cd $WD
