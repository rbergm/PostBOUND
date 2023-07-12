#!/bin/bash

rm -rf q*/
curl --remote-name https://rmarcus.info/so_queries.tar.zst
zstd -d so_queries.tar.zst
tar xvf so_queries.tar
mv so_queries/* .
rm -r so_queries*
