#!/bin/bash

TARGET_DIR=../pb-venv/

if [ ! -z "$1" ] ; then
  TARGET_DIR="$1"
fi

echo ".. Setting up Python virtual environment at $TARGET_DIR"
python3 -m venv $TARGET_DIR
. $TARGET_DIR/bin/activate

echo ".. Installing dependencies"
pip install -r ../postbound/requirements.txt

