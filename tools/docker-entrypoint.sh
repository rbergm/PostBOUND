#!/bin/bash

if [ "$USE_PGLAB" = "true" ] ; then
    cd /pg_lab
else
    cd /postbound/db-support/postgres
fi
. ./postgres-start.sh

/bin/bash
