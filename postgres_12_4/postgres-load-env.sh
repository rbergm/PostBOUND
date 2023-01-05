#!/bin/sh

if [ "$THESIS_PG_ENV_LOADED" != "true" ] || [ "$1" = "--force" ] ; then
	export THESIS_PG_ENV_LOADED="true"

	WD=$(pwd)
	cd postgres-server

	export PATH="$(pwd)/build/bin:$PATH"
	export LD_LIBRARY_PATH="$(pwd)/build/lib:$LD_LIBRARY_PATH"

	cd $WD
fi
