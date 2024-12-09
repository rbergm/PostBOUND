FROM ubuntu:noble

EXPOSE 5432

# Install dependencies
RUN apt update && apt install -y \
    build-essential sudo tzdata procps \
    bison flex curl pkg-config libicu-dev libreadline-dev \
    git vim unzip \
    python3 python3-venv python3-pip

# Configure some general settings
ARG USERNAME=postbound
ARG TIMEZONE=UTC
ARG SETUP_IMDB=false
ARG SETUP_STATS=false
ARG SETUP_STACK=false
ARG OPTIMIZE_PG_CONFIG=false
ARG PG_DISK_TYPE=SSD
ENV TZ=$TIMEZONE
ENV USER=$USERNAME
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create a new user
WORKDIR /postbound
RUN useradd -ms /bin/bash $USERNAME
RUN chown -R $USERNAME:$USERNAME /postbound
RUN chmod 755 /postbound
RUN echo "$USERNAME:$USERNAME" | chpasswd
RUN usermod -aG sudo $USERNAME
USER $USERNAME

RUN git clone --depth 1 --branch=quality/structure-simplification https://github.com/rbergm/PostBOUND /postbound

# Setup local Postgres
WORKDIR /postbound/db-support/postgres
RUN ./postgres-setup.sh --stop
RUN if [  "$OPTIMIZE_PG_CONFIG" = "true" ] ; then \
        python3 postgres-config-generator.py --out pg-conf.sql --disk-type "$PG_DISK_TYPE" /postbound/db-support/postgres/postgres-server/data ; \
        . ./postgres-start.sh && psql -f pg-conf.sql ; \
    fi

# TODO: optionally use pg_lab

# Install PostBOUND package
WORKDIR /postbound
RUN tools/setup-py-venv.sh --venv /postbound/pb-venv

# Setup databases
WORKDIR /postbound/db-support/postgres
RUN if [ "$SETUP_IMDB" = "true" ] ; then \
        . ./postgres-start.sh && ./workload-job-setup.sh ; \
    fi
RUN if [ "$SETUP_STATS" = "true" ] ; then \
        . ./postgres-start.sh && ./workload-stats-setup.sh ; \
    fi
RUN if [ "$SETUP_STACK" = "true" ] ; then \
        . ./postgres-start.sh && ./workload-stack-setup.sh ; \
    fi

# User config
RUN echo "cd /postbound/db-support/postgres/" >> /home/$USERNAME/.bashrc
RUN echo "source ./postgres-load-env.sh" >> /home/$USERNAME/.bashrc
RUN echo "cd /postbound" >> /home/$USERNAME/.bashrc
RUN echo "source /postbound/pb-venv/bin/activate" >> /home/$USERNAME/.bashrc

# Final container config
WORKDIR /postbound
VOLUME /postbound/public
CMD /postbound/db-support/postgres/postgres-start.sh /postbound/db-support/postgres/postgres-server && /bin/bash
