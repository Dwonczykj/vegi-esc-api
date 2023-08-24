# syntax=docker/dockerfile:experimental
# Container for building the environment
# ~ https://stackoverflow.com/a/71611002 + first comment for docker run adding a --platform linux/x86_64 as first arg to command
FROM --platform=linux/amd64 condaforge/mambaforge:4.9.2-5 as conda

#Set the working directory
WORKDIR /


RUN apt-get update && \
    apt-get install -y libpq-dev gcc

COPY predict-linux-64.lock .

# RUN mamba create --copy -p /env --file predict-linux-64.lock && conda clean -afy
RUN --mount=type=cache,target=/opt/conda/pkgs mamba create --copy -p /env --file predict-linux-64.lock && echo 4
RUN [ "sh", "-c", "echo $HOME" ]
COPY . /pkg
RUN conda run -p /env python -m pip install --no-deps /pkg

# Distroless for execution
# FROM gcr.io/distroless/base-debian10
# FROM gcr.io/distroless/base-debian10:debug-nonroot
# ~ https://github.com/sigstore/cosign#dockerfile
FROM gcr.io/projectsigstore/cosign:v1.13.0 as cosign-bin

# Source: https://github.com/chainguard-images/static
FROM cgr.dev/chainguard/static:latest
COPY --from=cosign-bin /ko-app/cosign /usr/local/bin/cosign
ENTRYPOINT [ "cosign" ]
# ~ https://github.com/GoogleContainerTools/distroless#debug-images
# Debug Images
# Distroless images are minimal and lack shell access. The :debug image set for each language provides a busybox shell to enter.
# RUN [ "sh", "-c", "cosign verify "gcr.io/distroless/base-debian11:debug-nonroot" --certificate-oidc-issuer https://accounts.google.com  --certificate-identity keyless@distroless.iam.gserviceaccount.com" ]
FROM gcr.io/distroless/base-debian11:debug-nonroot

# Set the working directory
# WORKDIR /app

# Add a shell to distroless image for debugging
# COPY --from=busybox:1.35.0-uclibc /bin/sh /bin/sh
# Add bash for debugging
# To use a different shell, other than '/bin/sh', use the exec form passing in the desired shell. For example:
# ~ https://docs.docker.com/engine/reference/builder/#run:~:text=To%20use%20a%20different%20shell%2C%20other%20than%20%E2%80%98/bin/sh%E2%80%99%2C%20use%20the%20exec%20form%20passing%20in%20the%20desired%20shell.%20For%20example%3A
RUN [ "sh", "-c", "echo $HOME" ]
# RUN [ "sh", "-c", "apk add --update bash" ]
# RUN [ "sh", "-c", "apt-get update && apt-get install -y bash" ]
# RUN ["/bin/busybox", "-c", "echo hello"]
# RUN ["/bin/busybox", "-c", "apt-get update && apt-get install -y bash"]
# ENV PATH=/bin/bash:$PATH


COPY --from=conda /env /env
# COPY --from=conda /env /app/env
RUN ["sh", "-c", "ls -la /bin"]
RUN ["sh", "-c", "ls -la /env"]

# Copy models from  localmachine to container to save downloading them
# RUN mkdir -p /models
# COPY models/word2vec-google-news-300 /models
# COPY models/word2vec-google-news-300.vectors.npy /models
# COPY /models /models

# COPY /.chromadb /app/.chromadb
COPY /.chromadb /.chromadb

# Note that distroless images by default do not contain a shell. That means the Dockerfile ENTRYPOINT command, when defined, must be specified in vector form, to avoid the container runtime prefixing with a shell.
CMD [ \
    "sh", "-c", \
    "/env/bin/gunicorn --bind 0.0.0.0:$PORT vegi_esc_api.app:gunicorn_app --timeout 600" \
    ]

# CMD [ \
#     # "/env/bin/gunicorn", "-b", "0.0.0.0:5002", "-k", "uvicorn.workers.UvicornWorker", "nyc_taxi_fare.serve:app" \
#     "/env/bin/gunicorn", "--bind", "0.0.0.0:5002", "vegi_esc_api.app:gunicorn_app\(\)", "--timeout", "600" \
#     # "gunicorn", "--bind", "0.0.0.0:5002", "vegi_esc_api.app:gunicorn_app(verbose=True)", "--timeout", "600" \
#     # "/app/env/bin/gunicorn", "--bind", "0.0.0.0:5002", "vegi_esc_api.app:gunicorn_app", "--timeout", "600" \
#     ]

# ENV PG_MAJOR=9.3
# ENV PG_VERSION=9.3.4
# ENV PATH=/usr/local/postgres-$PG_MAJOR/bin:$PATH
# ENV PATH=/app/env/bin/gunicorn:$PATH
# ENTRYPOINT [ "/app/env/bin/gunicorn" ]
# ENV PATH=/env/bin/gunicorn:$PATH
# ENTRYPOINT [ "/env/bin/gunicorn" ]
# CMD [ \
#     # "--bind", "0.0.0.0:5002", "vegi_esc_api.app:gunicorn_app(verbose=True)", "--timeout", "600" \
#     "--bind", "0.0.0.0:5002", "vegi_esc_api.app:gunicorn_app\(\)", "--timeout", "600" \
#     ]
