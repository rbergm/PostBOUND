#!/bin/bash

WD=$(pwd)
PY_PATH="$WD/build/bin"
INIT=$(echo "$PATH" | grep "$PG_CTL_PATH")

if [ -z "$INIT" ] ; then
    export PATH_ORIG=$PATH
    export PY_PATH
    export PATH="$PY_PATH:$PATH"
fi

cd $WD
