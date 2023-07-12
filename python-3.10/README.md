# Python management

The utilities in this folder can be used to build an installation of Python 3.10 from scratch
and to integrate it into the current environment. Similarly to the setup scripts in the
`postgres` directory, the installation is completely local, i.e. no global paths, etc. are
modified. Therefore, the `python-load-path.sh` and `python-deactive.sh` scripts have to be
used to enforce the usage of the correct binary.
