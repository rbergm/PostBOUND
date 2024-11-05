# Oracle management

This folder provides a Dockerfile that sets up a Docker image for the Oracle Database Express
edition (Oracle XE). Just use the normal Docker tools to get it up and running:

```bash
# Install the Docker image
$ docker build -t pb-oracle .
# Create a container instance
$ docker run --it --name pb-oracle -p 1521:1521 -p 5500:5500 pb-oracle
# Connect to the container
$ docker exec -it pb-oracle bash
```
