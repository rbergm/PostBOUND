#!/bin/bash

WD=$(pwd)
USER=$(whoami)
MAKE_CORES=$(($(nproc --all) / 2))


mkdir mysql-server
cd $WD/mysql-server
MYSQL_INSTALL_ROOT=$(pwd)

echo ".. Cloning MySQL 8.0"
git clone --depth 1 --branch 8.0 https://github.com/mysql/mysql-server.git mysql-src

echo ".. Building MySQL 8.0"
mkdir $MYSQL_INSTALL_ROOT/mysql-build
cd $MYSQL_INSTALL_ROOT/mysql-build
cmake $MYSQL_INSTALL_ROOT/mysql-src \
    -DCMAKE_INSTALL_PREFIX=/ \
    -DDOWNLOAD_BOOST=1 -DWITH_BOOST=$MYSQL_INSTALL_ROOT/boost-src
make -j $MAKE_CORES
make install DESTDIR=$MYSQL_INSTALL_ROOT/mysql-server

echo ".. Initializing MySQL server environment"
cd $MYSQL_INSTALL_ROOT/mysql-server
MYSQL_INSTALL_DIR=$(pwd)
mkdir mysql-data
bin/mysqld --initialize-insecure \
    --basedir=$MYSQL_INSTALL_DIR \
    --datadir=$MYSQL_INSTALL_DIR/mysql-data

echo ".. Starting MySQL server"
bin/mysqld --basedir=$MYSQL_INSTALL_DIR \
    --datadir=$MYSQL_INSTALL_DIR/mysql-data \
    --pid-file=$MYSQL_INSTALL_DIR/mysqld.pid \
    &
echo ".. Setup done, ready to connect"
