#! /bin/zsh

dockerFileExt=${1:-"distroless-buildkit"}

# Check if the file "./docker/Dockerfile.$dockerFileExt" exists
if [ ! -f "./docker/Dockerfile.$dockerFileExt" ]; then
    echo "Error: The requested Docker file does not exist."
    exit 1
fi


imageName=vegi-esc-server-$dockerFileExt
condaVenvName=esc-llm
dockerFileName=docker/Dockerfile.$dockerFileExt
herokuWebAppName=vegi-esc-server

dockerBuildOutput=$( DOCKER_BUILDKIT=1 \
    docker buildx build \
    --platform=linux/amd64 \
    --tag $imageName \
    --load -f $dockerFileName \
    . 2>&1 | tee /dev/fd/2 )
if grep -q "no space left on device" $dockerBuildOutput; then
    docker ps -a --format 'table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.CreatedAt}}\t{{.Status}}'
    docker container prune --filter "label=$imageName" # --filter "until=24h30m"
    docker image prune --filter "label=$imageName" # --filter "until=24h30m"
    # docker system prune
else
    docker image list #  --filter "label=$imageName"  # --filter "until=24h30m"
fi

docker run \
    --env DATABASE_HOST=host.docker.internal \
    --env-file ./.env \
    --platform linux/amd64 \
    -it \
    -p 2001:5001 \
    $imageName