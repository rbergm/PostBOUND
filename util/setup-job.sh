#!/bin/bash

PWD=$(pwd)
DB_NAME="imdb"

if [ -z "$1" ] ; then
    IMDB_DIR="../imdb_data"
else
    IMDB_DIR="$1"
fi

if psql -l | grep "$DB_NAME" | wc -l ; then
    echo ".. IMDB exists, doing nothing"
    exit 0
fi

echo ".. IMDB source directory is $IMDB_DIR"

if [ -d $IMDB_DIR ] ; then
    echo ".. Re-using existing IMDB input data"
else
    echo ".. IMDB source directory does not exist, re-creating"
    echo ".. Fetching IMDB data"
    mkdir $IMDB_DIR
    curl -o $IMDB_DIR/csv.zip "https://cloudstore.zih.tu-dresden.de/index.php/s/eqWWK53CgkxMxfA/download?path=%2F&files=csv.zip"
    curl -o $IMDB_DIR/create.sql "https://cloudstore.zih.tu-dresden.de/index.php/s/eqWWK53CgkxMxfA/download?path=%2F&files=create.sql"
    curl -o $IMDB_DIR/import.sql "https://cloudstore.zih.tu-dresden.de/index.php/s/eqWWK53CgkxMxfA/download?path=%2F&files=import.sql"
    curl -o $IMDB_DIR/fkindexes.sql "https://cloudstore.zih.tu-dresden.de/index.php/s/eqWWK53CgkxMxfA/download?path=%2F&files=fkindexes.sql"

    echo ".. Extracting IMDB data"
    unzip $IMDB_DIR/csv.zip -d $IMDB_DIR
fi

echo ".. Creating IMDB database schema"
createdb $DB_NAME
psql imdb -f $IMDB_DIR/create.sql

echo ".. Inserting IMDB data into database"
cd $IMDB_DIR
psql imdb -f import.sql

echo ".. Creating IMDB foreign key indices"
psql imdb -f fkindexes.sql
psql imdb -c "create index if not exists subject_id_complete_cast on complete_cast(subject_id)"
psql imdb -c "create index if not exists status_id_complete_cast on complete_cast(status_id)"

echo ".. Done"
cd $PWD
