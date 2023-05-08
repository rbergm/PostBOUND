#!/bin/bash

WD=$(pwd)
PYTHON_ARCHIVE_NAME="Python-3.10.tar.gz"
PYTHON_SRC_DIR="$WD/Python-3.10.11"
OPENSSL_ARCHIVE_NAME="openssl-1.1.1t.tar.gz"
OPENSSL_SRC_DIR="$WD/openssl-1.1.1t"
BIN_DIR="$WD/build"
LIB_DIR="$WD/build"
NCORES=$(($(nproc --all) / 2))

echo ".. Pulling Python 3.10"
if [ ! -f "$PYTHON_ARCHIVE_NAME" ] ; then
    curl "https://www.python.org/ftp/python/3.10.11/Python-3.10.11.tgz" --output $PYTHON_ARCHIVE_NAME
fi

echo ".. Pulling OpenSSL 1.1"
if [ ! -f "$OPENSSL_ARCHIVE_NAME" ] ; then
    curl "https://www.openssl.org/source/openssl-1.1.1t.tar.gz" --output $OPENSSL_ARCHIVE_NAME
fi

echo ".. Unpacking archives"
tar xzf $PYTHON_ARCHIVE_NAME
tar xzf $OPENSSL_ARCHIVE_NAME

echo ".. Building OpenSSL 1.1"
cd $OPENSSL_SRC_DIR
./config shared --prefix="$LIB_DIR" | tee openssl-1.1.1-configure.log
make -j $NCORES | tee openssl-1.1.1-make.log | tee openssl-1.1.1-make.log
make install
export LD_LIBRARY_PATH="$LIB_DIR/lib:$LD_LIBRARY_PATH"

echo ".. Building Python 3.10"
cd $PYTHON_SRC_DIR
./configure --enable-optimizations --prefix="$BIN_DIR" --with-openssl="$LIB_DIR" | tee python3.10-configure.log
make -j $NCORES | tee python3.10-make.log
make altinstall

cd $BIN_DIR/bin
ln -s python3.10 python3

echo ".. Done. Use '. ./python-load-path.sh' to use the new Python executable"
cd $WD
