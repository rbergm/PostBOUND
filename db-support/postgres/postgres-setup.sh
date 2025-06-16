#!/bin/bash

set -e  # exit on error

WD=$(pwd)
USER=$(whoami)
PG_VER_PRETTY=17
PG_VERSION=REL_17_STABLE
PG_HINT_PLAN_VERSION=REL17_1_7_0
PG_TARGET_DIR="$WD/postgres-server"
PG_DEFAULT_PORT=5432
PG_PORT=5432
MAKE_CORES=$(($(nproc --all) / 2))
ENABLE_REMOTE_ACCESS="false"
USER_PASSWORD=""
STOP_AFTER="false"

show_help() {
    NEWLINE="\n\t\t\t\t"
    echo -e "Usage: $0 <options>"
    echo -e "Setup a local Postgres server with the given options. The default user is the current UNIX username.\n"
    echo -e "Allowed options:"
    echo -e "--pg-ver <version>\t\tSetup Postgres with the given version.${NEWLINE}Currently allowed values: 12.4, 14, 15, 16, 17 (default))"
    echo -e "-d | --dir <directory>\t\tInstall Postgres server to the designated directory (postgres-server by default)."
    echo -e "-p | --port <port number>\tConfigure the Postgres server to listen on the given port (5432 by default)."
    echo -e "--remote-password <password>\tEnable remote access for the current user, based on the given password.${NEWLINE}Remote access is disabled if no password is provided."
    echo -e "--stop\t\t\t\tStop the Postgres server process after installation and setup finished"
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
                    PG_HINT_PLAN_VERSION=REL14_1_4_3
                    ;;
                15)
                    PG_VER_PRETTY="15"
                    PG_VERSION=REL_15_STABLE
                    PG_HINT_PLAN_VERSION=REL15_1_5_2
                    ;;
                16)
                    PG_VER_PRETTY="16"
                    PG_VERSION=REL_16_STABLE
                    PG_HINT_PLAN_VERSION=REL16_1_6_1
                    ;;
                17)
                    PG_VER_PRETTY="17"
                    PG_VERSION=REL_17_STABLE
                    PG_HINT_PLAN_VERSION=REL17_1_7_0
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
        -p|--port)
            PG_PORT=$2
            shift
            shift
            ;;
        --remote-password)
            ENABLE_REMOTE_ACCESS="true"
            USER_PASSWORD=$2
            shift
            shift
            ;;
        --stop)
            STOP_AFTER="true"
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
./configure --prefix=$PG_TARGET_DIR/build \
        --with-ssl=openssl \
        --with-python \
        --with-llvm \
        --with-lz4 \
        --with-zstd
make clean && make -j $MAKE_CORES && make install
export PATH="$PG_TARGET_DIR/build/bin:$PATH"
export LD_LIBRARY_PATH="$PG_TARGET_DIR/build/lib:$LD_LIBRARY_PATH"
export C_INCLUDE_PATH="$PG_TARGET_DIR/build/include/server:$C_INCLUDE_PATH"

echo ".. Installing pg_hint_plan extension"
cd $PG_TARGET_DIR/contrib
tar xzvf pg_hint_plan.tar.gz
mv pg_hint_plan-$PG_HINT_PLAN_VERSION pg_hint_plan
cd pg_hint_plan
make -j $MAKE_CORES && make install
cd $PG_TARGET_DIR

echo ".. Installing pg_prewarm extension"
cd $PG_TARGET_DIR/contrib/pg_prewarm
make -j $MAKE_CORES && make install
cd $PG_TARGET_DIR

echo ".. Installing pg_cooldown extension"
cd $PG_TARGET_DIR/contrib
git clone https://github.com/rbergm/pg_cooldown.git pg_cooldown
cd pg_cooldown
make -j $MAKE_CORES && make install
cd $PG_TARGET_DIR

echo ".. Installing pg_buffercache extension"
cd $PG_TARGET_DIR/contrib/pg_buffercache
make -j $MAKE_CORES && make install
cd $PG_TARGET_DIR

echo ".. Initializing Postgres Server environment"
cd $PG_TARGET_DIR

echo "... Creating cluster"
initdb -D $PG_TARGET_DIR/data

if [ "$PG_PORT" != "$PG_DEFAULT_PORT" ] ; then
    echo "... Updating Postgres port to $PG_PORT"
    sed -i "s/#\{0,1\}port = 5432/port = $PG_PORT/" $PG_TARGET_DIR/data/postgresql.conf
fi

echo "... Adding pg_buffercache, pg_hint_plan and pg_prewarm to preload libraries"
sed -i "s/#\{0,1\}shared_preload_libraries.*/shared_preload_libraries = 'pg_buffercache,pg_hint_plan,pg_prewarm'/" $PG_TARGET_DIR/data/postgresql.conf
echo "pg_prewarm.autoprewarm = false" >>  $PG_TARGET_DIR/data/postgresql.conf

echo "... Starting Postgres (log file is pg.log)"
pg_ctl -D $PG_TARGET_DIR/data -l pg.log start

echo "... Creating user database for $USER"
createdb -p $PG_PORT $USER

if [ "$ENABLE_REMOTE_ACCESS" == "true" ] ; then
    echo "... Enabling remote access for $USER"
    echo -e "#customization\nhost all $USER 0.0.0.0/0 md5" >> $PG_TARGET_DIR/data/pg_hba.conf
    sed -i "s/#\{0,1\}listen_addresses = 'localhost'/listen_addresses = '*'/" $PG_TARGET_DIR/data/postgresql.conf
    psql -c "ALTER USER $USER WITH PASSWORD '$USER_PASSWORD';"
fi

if [ "$STOP_AFTER" == "true" ] ; then
    pg_ctl -D $PG_TARGET_DIR/data stop
    echo ".. Setup done"
else
    echo ".. Setup done, ready to connect"
fi

cd $WD
