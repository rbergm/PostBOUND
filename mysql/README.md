# MySQL management

The scripts in this directory can be used to setup, start and stop different MySQL database servers and to import some
commonly used workloads. More precisely, the following utilities are provided:

- scripts starting with `mysql-native-` can be used to pull a source distribution of the MySQL server, to compile and to
  configure it (this is all handled by the `setup` script). Furthermore, the `start`, `stop` and `load-env` scripts can be
  used to start the installed server, to stop it and to load the required binaries to the `PATH`
- scripts starting with `mysql-docker-` have a similar purpose as their native counterparts, but dont' perform a compilation/
  binary installation. Instead, the MySQL instance is installed as a docker image. Therefore, no `PATH` manipulation is
  performed (and hence no `load-env` script exists). To interact with the server directly, either connect to the docker
  instance (using the `connect` script), or install a MySQL CLI using your distribution-specific package manager
- scripts starting with `imdb-docker` are utilities to automatically setup an instance of the IMDB (Internet Movie Database)
  used in the popular Join Order Benchmark (JOB). This currently only works for the Docker installation of MySQL. Use the
  `setup` script to start this process. The Python `import` program is only used internally.
- the `mysql-connect-setup.sh` utility can be used to generated PostBOUND-compatible connect files to establish a connection to
  the database via the PostBOUND's database abstraction
