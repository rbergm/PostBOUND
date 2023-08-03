#!/bin/bash

WD=$(pwd)
DB_NAME="imdb"
IMDB_DIR="../imdb_data"
DOCKER_CONTAINER_NAME=mysql

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

cd $IMDB_DIR
docker cp create.sql $DOCKER_CONTAINER_NAME:/
docker cp fkindexes.sql $DOCKER_CONTAINER_NAME:/

cd $IMDB_DIR/csv
for file in $(ls | grep csv); do
    docker exec -it $DOCKER_CONTAINER_NAME test -f "/var/lib/mysql-files/$file"
    exists=$?
    if [ $exists -eq 0 ]; then
        echo "Skipping transfer of existing file $file"
    else
        echo "Transfering $file to docker image"
        docker cp $file $DOCKER_CONTAINER_NAME:/var/lib/mysql-files
    fi
done

cd $WD
docker cp imdb-docker-import.py $DOCKER_CONTAINER_NAME:/
docker exec $DOCKER_CONTAINER_NAME python3 imdb-docker-import.py

cd $WD
