#!/bin/bash

# Setup local Postgres or pg_lab
if [ "$USE_PGLAB" = "true" ] ; then
    git clone --depth 1 --branch=main https://github.com/rbergm/pg_lab /pg_lab
    cd /pg_lab && ./postgres-setup.sh --remote-password "" --stop
    . ./postgres-start.sh
else
    cd /postbound/db-support/postgres && ./postgres-setup.sh --remote-password "" --stop
    . ./postgres-start.sh
fi

cd /postbound/db-support/postgres
if [ "$OPTIMIZE_PG_CONFIG" = "true" ] ; then
    python3 postgres-config-generator.py --out pg-conf.sql --disk-type "$PG_DISK_TYPE" "$PGDATA"
    psql -f pg-conf.sql
fi

# Install PostBOUND package
cd /postbound && tools/setup-py-venv.sh --venv /postbound/pb-venv

# Setup databases
cd /postbound/db-support/postgres
if [ "$SETUP_IMDB" = "true" ] ; then
        ./workload-job-setup.sh
fi
if [ "$SETUP_STATS" = "true" ] ; then
    ./workload-stats-setup.sh
fi
if [ "$SETUP_STACK" = "true" ] ; then
    ./workload-stack-setup.sh
fi

# User config
if [ "$USE_PGLAB" = "true" ] ; then
    echo "cd /pg_lab/" >> /home/$USERNAME/.bashrc
else
    echo "cd /postbound/db-support/postgres/" >> /home/$USERNAME/.bashrc
fi
echo "source ./postgres-load-env.sh" >> /home/$USERNAME/.bashrc
echo "cd /postbound" >> /home/$USERNAME/.bashrc
echo "source /postbound/pb-venv/bin/activate" >> /home/$USERNAME/.bashrc


# Teardown
if [ "$USE_PGLAB" = "true" ] ; the
    cd /pg_lab
else
    cd /postbound/db-support/postgres
fi
. ./postgres-stop.sh
