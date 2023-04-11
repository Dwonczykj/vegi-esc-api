#! /bin/zsh

docker buildx build -t vegi_esc_server-distroless-buildkit-expert --load -f docker/Dockerfile.distroless-buildkit-expert .
docker run -ti -p 5002:5002 vegi_esc_server-distroless-buildkit-expert