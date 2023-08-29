#! /bin/bash

# make distroless-buildkit
# makeName=miniconda
makeName=distroless-buildkit
imageName="vegi_esc_server_$makeName"
mapFromDockerImagePort=5002
mapToHostPort=2001
open http://127.0.0.1:$mapToHostPort/vegi-users
# make sure the port below matches the bound port in the docker file run CMD
# ~ https://docs.docker.com/engine/reference/commandline/run/#options
# ~ https://docs.docker.com/engine/reference/commandline/run/#env
# docker run -e MYVAR1 --env MYVAR2=foo --env-file ./env.list ubuntu bash
docker stats --no-stream
docker run \
    --memory=15g \
    --cpus=2 \
    --platform linux/amd64 \
    --env DATABASE_HOST=host.docker.internal \
    --env-file ./.env \
    -it \
    -p 2001:5002 \
    $imageName
