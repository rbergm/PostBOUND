FROM ubuntu:noble

EXPOSE 5432

# Install dependencies
RUN apt update && apt install -y \
    build-essential sudo tzdata procps \
    bison flex curl pkg-config libicu-dev libreadline-dev cmake \
    git vim unzip zstd default-jre \
    python3 python3-venv python3-pip

# Configure some general settings
ARG USERNAME=postbound
ENV USERNAME=$USERNAME
ENV USER=$USERNAME

ARG TIMEZONE=UTC
ENV TZ=$TIMEZONE
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ARG SETUP_IMDB=false
ENV SETUP_IMDB=$SETUP_IMDB
ARG SETUP_STATS=false
ENV SETUP_STATS=$SETUP_STATS
ARG SETUP_STACK=false
ENV SETUP_STACK=$SETUP_STACK

ARG OPTIMIZE_PG_CONFIG=false
ENV OPTIMIZE_PG_CONFIG=$OPTIMIZE_PG_CONFIG
ARG PG_DISK_TYPE=SSD
ENV PG_DISK_TYPE=$PG_DISK_TYPE

ARG USE_PGLAB=false
ENV USE_PGLAB=$USE_PGLAB

# Create a new user
WORKDIR /postbound
WORKDIR /pg_lab
RUN useradd -ms /bin/bash $USERNAME ; \
    chown -R $USERNAME:$USERNAME /postbound ; \
    chown -R $USERNAME:$USERNAME /pg_lab ; \
    chmod 755 /postbound ; \
    chmod 755 /pg_lab ; \
    echo "$USERNAME:$USERNAME" | chpasswd ; \
    usermod -aG sudo $USERNAME
USER $USERNAME

# PostBOUND and database setup
RUN git clone --depth 1 --branch=main https://github.com/rbergm/PostBOUND /postbound
RUN /bin/bash /postbound/tools/docker-setup.sh

# Final container config
WORKDIR /postbound
VOLUME /postbound/public
CMD ["/postbound/tools/docker-entrypoint.sh"]
