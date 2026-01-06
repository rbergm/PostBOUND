#!/bin/bash

sed_wrapper() {
    REPLACEMENT="$1"
    FILE="$2"

    if [ "$OS_TYPE" = "Darwin" ]; then
        sed -i '' -e "$REPLACEMENT" "$FILE"
    else
        sed -i -e "$REPLACEMENT" "$FILE"
    fi
}

version_earlier() {
    printf '%s\n' "$1" "$2" | sort -C -V
}

# Assert that we are not sourcing
if [ -n "$BASH_VERSION" -a "$BASH_SOURCE" != "$0" ] || [ -n "$ZSH_VERSION" -a "$ZSH_EVAL_CONTEXT" != "toplevel" ] ; then
    echo "This script should not be sourced. Please run it as ./postgres-setup.sh" 1>&2
    return 1
fi

set -e  # exit on error

WD=$(pwd)
USER=$(whoami)
PG_VER_PRETTY=18
PG_VERSION=REL_18_STABLE
PG_HINT_PLAN_VERSION=REL18_1_8_0
PG_TARGET_DIR="$WD/postgres-server"
PG_DEFAULT_PORT=5432
PG_PORT=5432
ENABLE_REMOTE_ACCESS="false"
USER_PASSWORD=""
STOP_AFTER="false"
UUID_LIBRARY="none"
LLVM_OPTS="auto"

show_help() {
    NEWLINE="\n\t\t\t\t"
    echo -e "Usage: $0 <options>"
    echo -e "Setup a local Postgres server with the given options. The default user is the current UNIX username.\n"
    echo -e "Allowed options:"
    echo -e "--pg-ver <version>\t\tSetup Postgres with the given version.${NEWLINE}Currently allowed values: 12, 14, 15, 16, 17, 18 (default).\n"
    echo -e "-d | --dir <directory>\t\tInstall Postgres server to the designated directory (postgres-server by default).\n"
    echo -e "-p | --port <port number>\tConfigure the Postgres server to listen on the given port (5432 by default).\n"
    echo -e "--remote-password <password>\tEnable remote access for the current user, based on the given password.${NEWLINE}Remote access is disabled if no password is provided.\n"
    echo -e "--stop\t\t\t\tStop the Postgres server process after installation and setup finished.\n"
    echo -e "--uuid-lib <library>\t\tSpecify the UUID library to use or 'none' to disable UUID support (default: none).${NEWLINE}Currently supported libraries are 'ossp' and 'e2fs'.\n"
    echo -e "--llvm-jit <on|off|auto>\tEnable or disable LLVM-based JIT compilation (default: on only if LLVM is available).\n"
    exit 1
}

while [ $# -gt 0 ] ; do
    case $1 in
        --pg-ver)
            case $2 in
                12)
                    PG_VER_PRETTY="12"
                    PG_VERSION=REL_12_STABLE
                    PG_HINT_PLAN_VERSION=REL12_1_3_10
                    ;;
                12.4)
                    PG_VER_PRETTY="12"
                    PG_VERSION=REL_12_4
                    PG_HINT_PLAN_VERSION=REL12_1_3_10
                    ;;
                14)
                    PG_VER_PRETTY="14"
                    PG_VERSION=REL_14_STABLE
                    PG_HINT_PLAN_VERSION=REL14_1_4_4
                    ;;
                15)
                    PG_VER_PRETTY="15"
                    PG_VERSION=REL_15_STABLE
                    PG_HINT_PLAN_VERSION=REL15_1_5_3
                    ;;
                16)
                    PG_VER_PRETTY="16"
                    PG_VERSION=REL_16_STABLE
                    PG_HINT_PLAN_VERSION=REL16_1_6_2
                    ;;
                17)
                    PG_VER_PRETTY="17"
                    PG_VERSION=REL_17_STABLE
                    PG_HINT_PLAN_VERSION=REL17_1_7_1
                    ;;
                18)
                    PG_VER_PRETTY="18"
                    PG_VERSION=REL_18_STABLE
                    PG_HINT_PLAN_VERSION=REL18_1_8_0
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
            USER_PASSWORD="$2"
            shift
            shift
            ;;
        --stop)
            STOP_AFTER="true"
            shift
            ;;
        --uuid-lib)
            case $2 in
                ossp)
                    UUID_LIBRARY="ossp"
                    ;;
                e2fs)
                    UUID_LIBRARY="e2fs"
                    ;;
                none)
                    UUID_LIBRARY="none"
                    ;;
                *)
                    show_help
                    ;;
            esac
            shift
            shift
            ;;
        --llvm-jit)
            LLVM_OPTS="$2"
            shift
            shift
            ;;
        *)
            show_help
            ;;
    esac
done

OS_TYPE=$(uname)
if [[ "$OS_TYPE" = "Darwin" ]]; then
    MAKE_CORES=$(sysctl -n hw.logicalcpu)
elif [[ "$OS_TYPE" = "Linux" ]]; then
    MAKE_CORES=$(nproc)
else
    echo "Unsupported operating system: $OS_TYPE"
    exit 1
fi


echo ".. Cloning Postgres $PG_VER_PRETTY"
if [ ! -d "$PG_TARGET_DIR" ] ; then
    git clone --depth 1 --branch $PG_VERSION https://github.com/postgres/postgres.git $PG_TARGET_DIR
fi
cd $PG_TARGET_DIR

echo ".. Downloading pg_hint_plan extension"
curl -L https://github.com/ossc-db/pg_hint_plan/archive/refs/tags/$PG_HINT_PLAN_VERSION.tar.gz -o contrib/pg_hint_plan.tar.gz

echo ".. Building Postgres $PG_VER_PRETTY"
if [ "$PG_VER_PRETTY" -lt "16" ] ; then
    # Postgres releases before v16 only support the classic configure/make build system.
    # In this mode, we need to do most of our customizations manually.
    #
    # This check includes some arcane logic to determine whether we can use the LLVM-based JIT or not. The reason is that
    # Postgres 12 required LLVM 3.x or newer, which is very old by now. Some changes in later LLVM version broke compatibility
    # with such old PG releases. Therefore, we have to disable LLVM there (and pray that we caught all such cases).
    if [ "$LLVM_OPTS" == "auto" ] ; then
        if [ -x "$(command -v llvm-config)" ] && version_earlier "$(llvm-config --version)" "16.0.0" ; then
            LLVM_OPTS="on"
        else
            LLVM_OPTS="off"
        fi
    fi
    if [ "$LLVM_OPTS" == "on" ] ; then
    echo ".. Enabling LLVM-based JIT compilation"
        LLVM_OPTS="--with-llvm"
    else
        echo ".. Disabling LLVM-based JIT compilation"
        LLVM_OPTS=""
    fi

    ./configure --prefix=$PG_TARGET_DIR/build \
                --with-openssl \
                --with-icu \
                $LLVM_OPTS
    make -j $MAKE_CORES && make install
else
    # Postgres v16 introduced the new Meson/Ninja build system. This allows to detect many of the features manually
    # (depending on whether the required dependencies are installed on the system).

    if [ -f GNUmakefile ] ; then
        echo ".. Cleaning previous build"
        make distclean
    fi

    meson setup build --prefix=$PG_TARGET_DIR/build \
        --buildtype=release \
        -Dicu=enabled \
        -Dssl=openssl \
        -Dplpython=auto \
        -Dlz4=auto \
        -Dzstd=auto \
        -Dllvm=$LLVM_OPTS \
        -Duuid=$UUID_LIBRARY
    cd $PG_TARGET_DIR/build
    ninja all && ninja install
fi
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

if [ "$PG_VER_PRETTY" -lt "16" ] ; then
    # Meson/Ninja automatically builds all contrib modules but it is only available since PG 16
    # So we need to manually build the required extensions here for older versions
    echo ".. Installing pg_prewarm extension"
    cd $PG_TARGET_DIR/contrib/pg_prewarm
    make -j $MAKE_CORES && make install
    cd $PG_TARGET_DIR

    echo ".. Installing pg_buffercache extension"
    cd $PG_TARGET_DIR/contrib/pg_buffercache
    make -j $MAKE_CORES && make install
    cd $PG_TARGET_DIR
fi

echo ".. Initializing Postgres Server environment"
cd $PG_TARGET_DIR

echo "... Creating cluster"
initdb -D $PG_TARGET_DIR/data

if [ "$PG_PORT" != "$PG_DEFAULT_PORT" ] ; then
    echo "... Updating Postgres port to $PG_PORT"
    sed_wrapper "s/#\{0,1\}port = 5432/port = $PG_PORT/" $PG_TARGET_DIR/data/postgresql.conf
fi

echo "... Adding pg_buffercache, pg_hint_plan and pg_prewarm to preload libraries"
sed_wrapper "s/#\{0,1\}shared_preload_libraries.*/shared_preload_libraries = 'pg_buffercache,pg_hint_plan,pg_prewarm'/" $PG_TARGET_DIR/data/postgresql.conf
echo "pg_prewarm.autoprewarm = false" >>  $PG_TARGET_DIR/data/postgresql.conf

echo "... Starting Postgres (log file is pg.log)"
pg_ctl -D $PG_TARGET_DIR/data -l pg.log start

echo "... Creating user database for $USER"
createdb -p $PG_PORT $USER

if [ "$ENABLE_REMOTE_ACCESS" == "true" ] ; then
    echo "... Enabling remote access for $USER"
    echo -e "#customization\nhost all $USER 0.0.0.0/0 md5" >> $PG_TARGET_DIR/data/pg_hba.conf
    sed_wrapper "s/#\{0,1\}listen_addresses = 'localhost'/listen_addresses = '*'/" $PG_TARGET_DIR/data/postgresql.conf
    psql -c "ALTER USER $USER WITH PASSWORD '$USER_PASSWORD';"
fi

if [ "$STOP_AFTER" == "true" ] ; then
    pg_ctl -D $PG_TARGET_DIR/data stop
    echo ".. Setup done"
else
    echo ".. Setup done, ready to connect"
fi

cd $WD
