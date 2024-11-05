#!/bin/bash

TARGET_NAME=.mysql_connection.config
USER=$(whoami)

if [ $# -lt 2 ] || [ "$1" = "--help" ] ; then
    echo "Usage: $0 <db name> [docker | native]"
    exit 0
fi

if [ "$2" == "docker" ] ; then
    USER="root"
fi

rm -f $TARGET_NAME
touch $TARGET_NAME
echo "[MYSQL]" >> $TARGET_NAME
echo "User = $USER" >> $TARGET_NAME
echo "Database = $1" >> $TARGET_NAME
