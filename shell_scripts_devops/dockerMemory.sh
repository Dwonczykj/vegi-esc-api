#! /bin/zsh

# ~ https://www.notion.so/gember/Docker-d0138e9a2dd84e4f9f3102ba726ef174?pvs=4#8b3f41377bc7494abd21998ae3ea4753
# * print the max docker image size
docker info | grep Memory

# print size of disk that docker is running in:
# docker info | grep "Data Space Total"

# update the size of docker script
echo "See https://stackoverflow.com/a/73121097 and paste \"storage-opts\" into file"
echo "See config at https://docs.docker.com/config/daemon/ on mac with docker desktop: https://docs.docker.com/desktop/settings/mac/#docker-engine && https://www.notion.so/gember/Docker-d0138e9a2dd84e4f9f3102ba726ef174?pvs=4#8b3f41377bc7494abd21998ae3ea4753"
nano ~/.docker/daemon.json

# Specifically for Docker for Mac, because it's a "GUI" app, there's a workaround (http://osxdaily.com/2014/09/05/gracefully-quit-application-command-line/):
osascript -e 'quit app "Docker"'

# Since you'd want to restart, here's the way to open it from the command line (https://superuser.com/a/1107622/243446):
open -a Docker
