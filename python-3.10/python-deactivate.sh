#!/bin/bash

if [ -z "$PY_PATH" ] ; then
    exit 1
fi

export PATH="${PATH//$PY_PATH:}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH//$PY_LIB_PATH:}"
