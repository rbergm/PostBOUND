FROM oraclelinux:8

ARG ORACLE_PASSWORD=mypass
ARG XE_RPM_NAME=oracle-database-xe-21c-1.0-1.ol8.x86_64.rpm
ARG XE_DOWNLOAD_PATH=https://download.oracle.com/otn-pub/otn_software/db-express/${XE_RPM_NAME}

RUN dnf -y install oracle-database-preinstall-21c wget which vim
RUN wget ${XE_DOWNLOAD_PATH}
RUN mkdir /xe_logs
ENV ORACLE_DOCKER_INSTALL=true
RUN yum -y localinstall ./${XE_RPM_NAME} > /xe_logs/XEsilentinstall.log 2>&1
RUN (echo "${ORACLE_PASSWORD}"; echo "${ORACLE_PASSWORD}") | /etc/init.d/oracle-xe-21c configure >> /xe_logs/XEsilentinstall.log 2>&1

COPY bash_profile.patch /home/oracle/bash_profile.patch
RUN cat /home/oracle/bash_profile.patch >> /home/oracle/.bash_profile

CMD /etc/init.d/oracle-xe-21c start && bash
