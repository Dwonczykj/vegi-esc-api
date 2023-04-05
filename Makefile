% : docker/Dockerfile.%
	docker buildx build -t vegi-esc-server-$@ --load -f $< .

lock:
	conda lock --mamba -p osx-arm64 -p linux-64 -f environment.yml
	conda lock --mamba -p osx-arm64 -p linux-64 -f predict-environment.yml --filename-template 'predict-{platform}.lock'