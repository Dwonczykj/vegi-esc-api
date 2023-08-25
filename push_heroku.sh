#! /bin/zsh

# push_heroku.sh
# this script:
#    - optionally rebuilds your conda environment from what is currently in the esc-llm conda venv
#    - optionally builds the docker image using `buildx` for a linux/amd64 docker image
#    - it then tags the new image locally with a name conforming to the heroku docker registry and then pushes the container to the registry
#    - it finally opens the newly deployed heroku app
#    - ! if the container push fails as unauthorised with heroku, then need to first `heroku login` and then `heroku container:login`

# Set dockerFileExt to first positional argument if provided, else default to 'distroless-buildkit'
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

echo "Running push_heroku.sh in ($condaVenvName) to build and tag \"$imageName\" from dockerfile: \"$dockerFileName\" and pushing to heroku webapp \"$herokuWebAppName\""


chmod +x entrypoint.sh

conda activate $condaVenvName

# ~ https://unix.stackexchange.com/a/346683
# ! for zsh the syntax is `read username"?What's you name? "` for rest its `read -p "?What's you name? " username`
read "response?Do you want to recreate conda-lock environment from the $condaVenvName virtual conda environment? (y/n): "
if [ "$response" = "y" ]; then
    unset response
    read -p "Do you want to export a new $condaVenvName env --from-history? (y/n): " response
    if [ "$response" = "y" ]; then
        conda env export --from-history > environment.yml
        cp environment.yml predict-environment.yml
    fi
    conda-lock \
        -f predict-environment.yml \
        -p linux-64 \
        -k explicit \
        --filename-template "predict-{platform}.lock"
    # conda-lock \
    #     -f predict-environment.yml \
    #     -f pot-environment.yml \
    #     -p linux-64 \
    #     -k explicit \
    #     --filename-template "predict-{platform}.lock"
else
    echo "Skipping recreation of conda-lock environment."
fi

unset response
read "response?Do you want to rebuild the docker image? (y/n): "

if [ "$response" = "y" ]; then
    # NOTE: Our setup.py file that is called to install a module into /pkg in the dockerfile has the package named to my_distinct_package_name in the setup.py file
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
else
    echo "Skipping rebuild of docker image."
fi

# Tag the image for Heroku's container registry
docker tag $imageName registry.heroku.com/$herokuWebAppName/web

# Use mktemp to create a temporary file
tmpfile=$(mktemp)

# Push the image to Heroku
# Run the command, capture the output in the temporary file and also write to stdout
# docker push registry.heroku.com/$herokuWebAppName/web | tee $tmpfile 

# ~ https://devcenter.heroku.com/articles/container-registry-and-runtime#pushing-an-existing-image
# ~ https://unix.stackexchange.com/a/526905
# 2 refers to STDERR. 2>&1 will send STDERR to the same location as 1 (STDOUT).
output=$( \
    docker push registry.heroku.com/$herokuWebAppName/web \
    2>&1 | \
    tee /dev/fd/2 \
    )
# if get an error here about being unauthorised, then need `heroku login -i & heroku containers:login`
# So if the output contains "unauthorised"
if grep -q "unauthorised" $output; then
    echo "get auth token for login from the Authorizations subheading of https://dashboard.heroku.com/account/applications"
    heroku login -i
    heroku container:login
fi

# ~ https://devcenter.heroku.com/articles/container-registry-and-runtime#releasing-an-image
output=$( \
    heroku container:release web --app $herokuWebAppName -v \
    2>&1 | \
    tee /dev/fd/2 \
    )

# Remove the temporary file
rm $tmpfile

unset response
read "response?Do you want to locally run the new reomte image stored at \"registry.heroku.com/$herokuWebAppName/web\"? (y/n): "

if [ "$response" = "y" ]; then
    heroku local web
fi

heroku open --app $herokuWebAppName
