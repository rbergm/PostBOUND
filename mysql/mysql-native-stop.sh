#!/bin/bash

WD=$(pwd)
MYSQL_INSTALL_DIR=mysql-server/mysql-server

echo ".. Stopping MySQL Server"
. ./mysql-load-env.sh

cd $MYSQL_INSTALL_DIR
kill $(cat mysqld.pid)

export PATH="${PATH//$MYSQL_CTL_PATH:}"

cd $WD
