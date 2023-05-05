#!/bin/bash

if [ -z "$PY_PATH" ] ; then
    exit 1
fi

export PATH="${PATH//$PY_PATH:}"
