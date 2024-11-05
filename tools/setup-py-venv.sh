#!/bin/bash

TARGET_DIR=pb-venv/

if [ "$1" == "--help" ] ; then
  echo "Usage: $0 [target_dir]"
  echo "  target_dir: The directory where the virtual environment will be created."
  echo "              Default: ./pb-venv/"
  exit 0
fi

if [ ! -z "$1" ] ; then
  TARGET_DIR="$1"
fi

echo ".. Setting up Python virtual environment at $TARGET_DIR"
python3 -m venv $TARGET_DIR
. $TARGET_DIR/bin/activate

echo ".. Installing dependencies"
# TODO: install PostBOUND as a Python package
pip install -r requirements.txt
