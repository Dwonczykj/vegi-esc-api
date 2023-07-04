# syntax=docker/dockerfile:experimental
# Container for building the environment
# ~ https://stackoverflow.com/a/71611002 + first comment for docker run adding a --platform linux/x86_64 as first arg to command
FROM --platform=linux/amd64 condaforge/mambaforge:4.9.2-5 as conda

RUN apt-get update && \
    apt-get install -y libpq-dev gcc

COPY predict-linux-64.lock .

# RUN mamba create --copy -p /env --file predict-linux-64.lock && conda clean -afy
RUN --mount=type=cache,target=/opt/conda/pkgs mamba create --copy -p /env --file predict-linux-64.lock && echo 4
COPY . /pkg
RUN conda run -p /env python -m pip install --no-deps /pkg

# Distroless for execution
FROM gcr.io/distroless/base-debian10

COPY --from=conda /env /env

# Copy models from  localmachine to container to save downloading them
# RUN mkdir -p /models
# COPY models/word2vec-google-news-300 /models
# COPY models/word2vec-google-news-300.vectors.npy /models
COPY /models /models
COPY /.chromadb /.chromadb

CMD [ \
    # "/env/bin/gunicorn", "-b", "0.0.0.0:5002", "-k", "uvicorn.workers.UvicornWorker", "nyc_taxi_fare.serve:app" \
    "/env/bin/gunicorn", "--bind", "0.0.0.0:5002", "vegi_esc_api.app:gunicorn_app(verbose=True)", "--timeout", "600" \
    ]
