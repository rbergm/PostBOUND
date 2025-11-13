#!/bin/bash

set -e

WD=$PWD
TARGET_DIR=pb-venv/
EXPLICIT_TARGET="false"
EXTRAS=""
MINIMAL_EXTRAS=""
ALL_EXTRAS="[mysql,vis]"
BUILD_DOC="false"
GIT_PULL="true"

show_help() {
  RET=$1
  echo -e "Usage: $0 <options>"
  echo -e ""
  echo -e "Installs PostBOUND into a (possibly existing) Python virtual environment. This script is assumed to be run from the "
  echo -e "root of the PostBOUND repository, i.e. as tools/setup-py-venv.sh."
  echo -e "If PostBOUND is already installed, it will be upgraded."
  echo -e ""
  echo -e "Allowed options:"
  echo -e "\n--venv <dir>"
  echo -e "\tPath to the virtual environment where PostBOUND will be installed. If this venv exists, it will be "
  echo -e "\tused. Otherwise, an empty venv will be created at the location. Defaults to ./pb-venv/. This parameter is ignored"
  echo -e "\tif a venv is already active."
  echo -e "\n--features <features>"
  echo -e "\tOptional extras to install with PostBOUND. These are specified as a comma-separated list."
  echo -e "\tSupported extras are: 'mysql' for installing the MySQL backend and 'vis' for using the visualization utilities."
  echo -e "\tFurthermore, 'all' can be used to install all available extras and 'minimal' only installs the core package."
  echo -e "\n--include-doc"
  echo -e "\tAlso build the documentation."
  echo -e "\n--skip-pull"
  echo -e "\tDon't pull the latest version of the repository before building. Notice that a pull will not update this script"
  echo -e "\twhile it is running. If there should be any issues with the setup script, please pull the latest version "
  echo -e "\tmanually and try again."
  echo -e "\n--help"
  echo -e "\tShow this help message."
  exit $RET
}

while [ $# -gt 0 ] ; do
  case $1 in
    --venv)
      TARGET_DIR="$2"
      EXPLICIT_TARGET="true"
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
    --include-doc)
      BUILD_DOC="true"
      shift
      ;;
    --skip-pull)
      GIT_PULL="false"
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

if [ "$GIT_PULL" == "true" ] ; then
  echo ".. Checking for latest version of PostBOUND"
  git pull
fi

if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
  echo ".. Default Python appears to be older than 3.10. Trying to set up local Python 3.10."
  PYTHON="$WD/tools/python-3.10"
  cd $PYTHON
  ./python-setup.sh
  . ./python-load-path.sh
  echo ".. Setup complete, continuing with Python 3.10 (installed locally at $PYTHON)"
  cd $WD
fi

if [ -z "$VIRTUAL_ENV" ] || [ "$EXPLICIT_TARGET" = "true" ] ; then

  # We are not in a virtual environment, so we need to create or activate one.

  if [ -d "$TARGET_DIR" ] ; then
    echo ".. Installing into existing virtual environment $TARGET_DIR"
  else
    echo ".. Creating new virtual environment $TARGET_DIR"
    python3 -m venv "$TARGET_DIR"
  fi

  . $TARGET_DIR/bin/activate

else

  echo ".. Using active virtual environment $VIRTUAL_ENV"

fi

echo ".. Building PostBOUND package"
pip install build wheel ipython
python3 -m build

echo ".. Installing PostBOUND package"
LATEST_WHEEL=$(ls dist/*.whl | sort -V | tail -n 1)
pip install -r requirements.txt  # this skips unnecessary updates
pip install --force-reinstall --no-deps "$LATEST_WHEEL$EXTRAS"  # this always forces the installation of the latest binary

if [ "$BUILD_DOC" == "true" ] ; then
  echo ".. Building documentation"
  cd $WD/docs
  sphinx-apidoc --force \
                --ext-autodoc \
                --maxdepth 4 \
                --module-first \
                -o source/generated \
                ../postbound
  make html
fi

echo ".. Done. Activate venv as '. $VIRTUAL_ENV/bin/activate' or 'source $VIRTUAL_ENV/bin/activate'."
cd "$WD"
