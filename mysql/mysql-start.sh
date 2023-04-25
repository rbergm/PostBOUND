#!/bin/bash

WD=$(pwd)
$MYSQL_INSTALL_DIR=mysql-server/mysql-server
. ./mysql-load-env.sh

cd $MYSQL_INSTALL_DIR
bin/mysqld --basedir=$MYSQL_INSTALL_DIR \
    --datadir=$MYSQL_INSTALL_DIR/mysql-data \
    --pid-file=$MYSQL_INSTALL_DIR/mysqld.pid \
    &
cd $WD
