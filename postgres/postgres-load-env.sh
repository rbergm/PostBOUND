#!/bin/sh

WD=$(pwd)
cd postgres-server

export PATH="$(pwd)/build/bin:$PATH"
export LD_LIBRARY_PATH="$(pwd)/build/lib:$LD_LIBRARY_PATH"

cd $WD

