#!/bin/bash

WD=$(pwd)
PY_PATH="$WD/build/bin/"
PY_LIB_PATH="$WD/build/lib/"
INIT=$(echo "$PATH" | grep "$PY_PATH")

if [ -z "$INIT" ] ; then
    export PATH_ORIG=$PATH
    export LD_LIBRARY_PATH_ORIG=$LD_LIBRARY_PATH

    export PY_PATH
    export PATH="$PY_PATH:$PATH"

    export LIB_PATH
    export LD_LIBRARY_PATH="$PY_LIB_PATH:$LD_LIBRARY_PATH"
fi

cd $WD
