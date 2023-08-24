@ ~ https://docs.docker.com/engine/reference/commandline/buildx_build/
% : docker/Dockerfile.%
	DOCKER_BUILDKIT=1 docker buildx build --platform=linux/amd64 -t vegi-esc-server-$@ --load -f $< .

lock:
	conda lock -p osx-64 -p linux-64 -f environment.yml
	conda lock -p osx-64 -p linux-amd-64 -f predict-environment.yml --filename-template 'predict-{platform}.lock'