#! /bin/bash

# make distroless-buildkit
# makeName=miniconda
makeName=distroless-buildkit
if make $makeName
then
    echo "Docker built fine"
else
    echo "Docker build failed: run `docker system prune --all`"
    docker system prune --all
    exit 1
fi
imageName="vegi-esc-server-$makeName"
echo "building docker image tagged: \"$imageName\""
# make sure the port below matches the bound port in the docker file run CMD
# ~ https://stackoverflow.com/a/43267603
set -a # automatically export all variables
source ./.env
set +a
mapFromDockerImagePort=$PORT
mapToHostPort=2001
echo "Will listen to port [$mapFromDockerImagePort] within docker container and expose on local at [$mapToHostPort]"
# ~ https://docs.docker.com/engine/reference/commandline/run/#options
# ~ https://docs.docker.com/engine/reference/commandline/run/#env
# docker run -e MYVAR1 --env MYVAR2=foo --env-file ./env.list ubuntu bash

docker stats --no-stream

# use -i to attach STDIN to send commands to image from terminal
# use -t to Docker states that the -t option will "allocate a pseudo-TTY" to the process inside the container. TTY stands for Teletype and can be interpreted as a device that offers basic input-output. The reason it's a pseudo-TTY is that there's no physical teletype needed, and it's emulated using a combination of display driver and keyboard driver.

# The zsh equivalent of bash's read -p prompt is
# read "?Here be dragons. Continue?"
# ~ https://superuser.com/a/556006
# if [[ "$BASH_VERSINFO" =~ ^[Yy]$ ]] then
# ~ https://www.notion.so/gember/MacOS_Commands-84c0274f5cde48f38b642316782a6be9?pvs=4#9f742f10db3440b3863445aadf17753b
if [ -n "${BASHVERSIONINFO+x}" ]; then
    read -p "Do you want to run the new docker container locally (privately) tagged: "$imageName"? (y/n): " response
else
    read "response?Do you want to run the new docker container locally (privately) tagged: "$imageName"? (y/n): "
fi

if [[ "$response" =~ ^[Yy]$ ]] 
then
    open http://127.0.0.1:$PORT/success/fenton
    docker run \
        --env DATABASE_HOST=host.docker.internal \
        --env-file ./.env_docker \
        --platform linux/amd64 \
        -p $mapToHostPort:$mapFromDockerImagePort \
        -it \
        --memory=15g \
        --cpus=2 \
        $imageName
fi
