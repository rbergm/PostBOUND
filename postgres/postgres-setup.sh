#!/bin/bash

WD=$(pwd)
USER=$(whoami)
PG_VER_PRETTY=14
PG_VERSION=REL_14_STABLE
PG_PLAN_HINT_VERSION=REL_14_1_4_0
PG_TARGET_DIR=postgres-server
MAKE_CORES=$(($(nproc --all) / 2))

show_help() {
    echo "Usage: $0 <options>"
    echo "Allowed options:"
    echo "--pg-ver <version> Setup Postgres with the given version (currently allowed values: 12.4, 14 (default), 15)"
    echo "-d | --dir <director> Install Postgres server to the designated directory (postgres-server by default)."
    exit 1
}

while [ $# -gt 0 ] ; do
    case $1 in
        --pg-ver)
            case $2 in
                12.4)
                    PG_VER_PRETTY="12.4"
                    PG_VERSION=REL_12_4
                    PG_HINT_PLAN_VERSION=REL12_1_3_8
                    ;;
                14)
                    PG_VER_PRETTY="14"
                    PG_VERSION=REL_14_STABLE
                    PG_HINT_PLAN_VERSION=REL14_1_4_0
                    ;;
                15)
                    PG_VER_PRETTY="15"
                    PG_VERSION=REL_15_STABLE
                    PG_HINT_PLAN_VERSION=REL15_1_5_0
                    ;;
                *)
                    show_help
                    ;;
            esac
            shift
            shift
            ;;
        -d|--dir)
            if [[ "$2" = /* ]] ; then
                PG_TARGET_DIR=$2
            else
                PG_TARGET_DIR=$WD/$2
                echo "... Normalizing relative target directory to $PG_TARGET_DIR"
            fi
            shift
            shift
            ;;
        *)
            show_help
            ;;
    esac
done

echo ".. Cloning Postgres $PG_VER_PRETTY"
git clone --depth 1 --branch $PG_VERSION https://github.com/postgres/postgres.git $PG_TARGET_DIR
cd $PG_TARGET_DIR

echo ".. Downloading pg_hint_plan extension"
curl -L https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/$PG_HINT_PLAN_VERSION.tar.gz -o contrib/pg_hint_plan.tar.gz

echo ".. Building Postgres $PG_VER_PRETTY"
./configure --prefix=$PG_TARGET_DIR/build
make clean && make -j $MAKE_CORES && make install
export PATH="$PG_TARGET_DIR/build/bin:$PATH"
export LD_LIBRARY_PATH="$PG_TARGET_DIR/build/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$PG_TARGET_DIR/build/include/server:$C_INCLUDE_PATH"

echo ".. Installing pg_hint_plan extension $(pwd)"
cd $PG_TARGET_DIR/contrib
echo "... [DEBUG] $PG_TARGET_DIR/contrib"
tar xzvf pg_hint_plan.tar.gz
mv pg_hint_plan-$PG_HINT_PLAN_VERSION pg_hint_plan
cd pg_hint_plan
make -j $MAKE_CORES && make install
cd $PG_TARGET_DIR

echo ".. Installing pg_prewarm extension"
cd $PG_TARGET_DIR/contrib/pg_prewarm
make -j $MAKE_CORES && make install
cd $PG_TARGET_DIR

echo ".. Initializing Postgres Server environment"
cd $PG_TARGET_DIR

echo "... Creating cluster"
initdb -D $PG_TARGET_DIR/data

echo "... Adding pg_hint_plan and pg_prewarm to preload libraries"
sed -i "s/#\{0,1\}shared_preload_libraries.*/shared_preload_libraries = 'pg_hint_plan,pg_prewarm'/" $PG_TARGET_DIR/data/postgresql.conf
echo "pg_prewarm.autoprewarm = false" >>  $PG_TARGET_DIR/data/postgresql.conf

echo "... Starting Postgres (log file is pg.log)"
pg_ctl -D $PG_TARGET_DIR/data -l pg.log start

echo "... Creating user database for $USER"
createdb $USER

if [ "$1" = "--stop" ] ; then
    pg_ctl -D $PG_TARGET_DIR/postgres-server/data stop
    echo ".. Setup done"
else
    echo ".. Setup done, ready to connect"
fi

cd $WD
