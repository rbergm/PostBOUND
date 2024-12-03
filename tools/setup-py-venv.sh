#!/bin/bash

set -e

WD=$PWD
TARGET_DIR=pb-venv/
EXTRAS=""
MINIMAL_EXTRAS=""
ALL_EXTRAS="[mysql,vis]"

show_help() {
  RET=$1
  echo -e "Usage: $0 <options>"
  echo -e ""
  echo -e "Installs PostBOUND into a (possibly existing) Python virtual environment. This script is assumed to be run from the "
  echo -e "root of the PostBOUND repository, i.e. as tools/setup-py-venv.sh."
  echo -e ""
  echo -e "Allowed options:"
  echo -e "--venv <dir>"
  echo -e "\tPath to the virtual environment where PostBOUND will be installed. If this venv exists, it will be "
  echo -e "\tused. Otherwise, an empty venv will be created. Defaults to ./pb-venv/"
  echo -e "--features <features>"
  echo -e "\tOptional extras to install with PostBOUND. These are specified as a comma-separated list."
  echo -e "\tSupported extras are: 'mysql' for installing the MySQL backend and 'vis' for using the visualization utilities."
  echo -e "\tFurthermore, 'all' can be used to install all available extras and 'minimal' only installs the core package."
  exit $RET
}

while [ $# -gt 0 ] ; do
  case $1 in
    --venv)
      TARGET_DIR="$2"
      shift
      shift
      ;;
    --features)
      case $2 in
        all)
          EXTRAS=$ALL_EXTRAS
          ;;
        minimal)
          EXTRAS=$MINIMAL_EXTRAS
          ;;
        mysql)
          EXTRAS="[mysql]"
          ;;
        vis)
          EXTRAS="[vis]"
          ;;
        *)
          EXTRAS="[$2]"
          ;;
      esac
      shift
      shift
      ;;
    --help)
      show_help 0
      ;;
    *)
      show_help 1
      ;;
  esac
done

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
  echo ".. Default Python appears to be older than 3.10. Trying to set up local Python 3.10."
  PYTHON="$WD/tools/python-3.10"
  cd $PYTHON
  ./python-setup.sh
  . ./python-load-path.sh
  echo ".. Setup complete, continuing with Python 3.10 (installed locally at $PYTHON)"
  cd $WD
fi

if [ -d "$TARGET_DIR" ] ; then
  echo ".. Installing into existing virtual environment $TARGET_DIR"
else
  echo ".. Creating new virtual environment $TARGET_DIR"
  python3 -m venv "$TARGET_DIR"
fi

. $TARGET_DIR/bin/activate

echo ".. Building PostBOUND package"
pip install build wheel
python3 -m build

echo ".. Installing PostBOUND package"
LATEST_WHEEL=$(ls dist/*.whl | sort -V | tail -n 1)
pip install "$LATEST_WHEEL$EXTRAS"

echo ".. Done. Activate venv as '. $TARGET_DIR/bin/activate'"
cd "$WD"
