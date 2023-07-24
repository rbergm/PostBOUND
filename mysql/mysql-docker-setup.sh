#!/bin/bash

MYSQL_VERSION=8

docker pull mysql:$MYSQL_VERSION
docker run --name pb-mysql -e MYSQL_ALLOW_EMPTY_PASSWORD=yes -d mysql:$MYSQL_VERSION
