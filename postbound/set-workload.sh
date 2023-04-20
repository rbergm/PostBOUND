#!/bin/bash

show_help() {
    echo "set-workload [--help | --check] [job | ssb | stack | tpch]"
    echo
    echo "set-workload is a utility to conveniently set active .psycopg_connection files for different workloads."
    echo "These files are used by PostBOUND to connect to the active database more easily."
    echo
    echo "--help                     .. shows this help dialog"
    echo "--check                    .. checks, which .psycopg_connection files are available"
    echo "[job | tpch | ssb | stack] .. sets the active .psycopg_connection file to the given workload"
}

check_profiles() {
    if [ ! -f ".psycopg_connection_job" ] ; then
        echo "No JOB connection file (expected name: .psycopg_connection_job)"
    else
        echo "JOB connection .. OK"
    fi

    if [ ! -f ".psycopg_connection_tpch" ] ; then
        echo "No TPC-H connection file (expected name: .psycopg_connection_tpch)"
    else
        echo "TPC-H connection .. OK"
    fi

    if [ ! -f ".psycopg_connection_ssb" ] ; then
        echo "No SSB connection file (expected name: .psycopg_connection_ssb)"
    else
        echo "SSB connection .. OK"
    fi

    if [ ! -f ".psycopg_connection_stack" ] ; then
        echo "No Stack connection file (expected name: .psycopg_connection_stack)"
    else
        echo "Stack connection .. OK"
    fi
}

set_profile() {
    case "$1" in
        job | tpch | stack | ssb)
            PROFILE="$1"
            ;;
        *)
            echo "Unknown profile: '$1'"
            exit 1
    esac
    ln -sf ".psycopg_connection_$PROFILE" .psycopg_connection
    ln -sf ".psycopg_connection_$PROFILE" tests/.psycopg_connection
}

if [ $# -eq 0 ] ; then
    show_help
    exit
fi

case "$1" in
    "--help")
        show_help
        ;;
    "--check")
        check_profiles
        ;;
    *)
        set_profile "$1"
esac
