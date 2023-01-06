#!/bin/bash

WD=$(pwd)
cd postgres-server
PG_CTL_PATH="$(pwd)/build/bin"
INIT=$(echo "$PATH" | grep "$PG_CTL_PATH")

if [ -z $INIT ] ; then
	export PG_CTL_PATH
	export PATH="$PG_CTL_PATH:$PATH"
	export LD_LIBRARY_PATH="$(pwd)/build/lib:$LD_LIBRARY_PATH"
fi

cd $WD
