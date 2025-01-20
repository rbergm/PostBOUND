FROM ubuntu:noble

EXPOSE 5432

# Install dependencies
RUN apt update && apt install -y \
    build-essential sudo tzdata procps \
    bison flex curl pkg-config libicu-dev libreadline-dev cmake \
    git vim unzip default-jre \
    python3 python3-venv python3-pip

# Configure some general settings
ARG USERNAME=postbound
ARG TIMEZONE=UTC
ARG SETUP_IMDB=false
ARG SETUP_STATS=false
ARG SETUP_STACK=false
ARG OPTIMIZE_PG_CONFIG=false
ARG PG_DISK_TYPE=SSD
ARG USE_PGLAB=false
ENV TZ=$TIMEZONE
ENV USER=$USERNAME
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create a new user
WORKDIR /postbound
WORKDIR /pg_lab
RUN useradd -ms /bin/bash $USERNAME
RUN chown -R $USERNAME:$USERNAME /postbound
RUN chown -R $USERNAME:$USERNAME /pg_lab
RUN chmod 755 /postbound
RUN chmod 755 /pg_lab
RUN echo "$USERNAME:$USERNAME" | chpasswd
RUN usermod -aG sudo $USERNAME
USER $USERNAME

RUN git clone --depth 1 --branch=main https://github.com/rbergm/PostBOUND /postbound

# Setup local Postgres or pg_lab
RUN if [ "$USE_PGLAB" = "true" ] ; then \
        git clone --depth 1 --branch=main https://github.com/rbergm/pg_lab /pg_lab ; \
        cd /pg_lab && ./postgres-setup.sh --remote-password "" --stop && . ./postgres-start.sh ; \
    else \
        cd /postbound/db-support/postgres && ./postgres-setup.sh --remote-password "" --stop && . ./postgres-start.sh ; \
    fi ; \
    cd /postbound/db-support/postgres ; \
    if [ "$OPTIMIZE_PG_CONFIG" = "true" ] ; then \
        python3 postgres-config-generator.py --out pg-conf.sql --disk-type "$PG_DISK_TYPE" "$PGDATA" ; \
        psql -f pg-conf.sql ; \
    fi ; \
    if [ "$USE_PGLAB" = "true" ] ; then \
        cd /pg_lab && . ./postgres-stop.sh ; \
    else \
        . ./postgres-stop.sh ; \
    fi

# Install PostBOUND package
RUN cd /postbound && tools/setup-py-venv.sh --venv /postbound/pb-venv

# Setup databases
RUN if [ "$USE_PGLAB" = "true" ] ; then \
        cd /pg_lab && . ./postgres-start.sh ; \
    else \
        cd /postbound/db-support/postgres && . ./postgres-start.sh ; \
    fi ; \
    cd /postbound/db-support/postgres ; \
    if [ "$SETUP_IMDB" = "true" ] ; then \
         ./workload-job-setup.sh ; \
    fi ; \
    if [ "$SETUP_STATS" = "true" ] ; then \
        ./workload-stats-setup.sh ; \
    fi ; \
    if [ "$SETUP_STACK" = "true" ] ; then \
        ./workload-stack-setup.sh ; \
    fi ; \
    if [ "$USE_PGLAB" = "true" ] ; then \
        cd /pg_lab && . ./postgres-stop.sh ; \
    else \
        cd /postbound/db-support/postgres && . ./postgres-stop.sh ; \
    fi


# User config
RUN if [ "$USE_PGLAB" = "true" ] ; then \
        echo "cd /pg_lab/" >> /home/$USERNAME/.bashrc ; \
    else \
        echo "cd /postbound/db-support/postgres/" >> /home/$USERNAME/.bashrc ; \
    fi ; \
    echo "source ./postgres-load-env.sh" >> /home/$USERNAME/.bashrc ; \
    echo "cd /postbound" >> /home/$USERNAME/.bashrc ; \
    echo "source /postbound/pb-venv/bin/activate" >> /home/$USERNAME/.bashrc

# Final container config
WORKDIR /postbound
VOLUME /postbound/public
CMD /postbound/db-support/postgres/postgres-start.sh /postbound/db-support/postgres/postgres-server && /bin/bash
