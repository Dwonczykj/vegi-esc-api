#! /bin/bash

python word2vec-api.py \
    --model ./models/GoogleNews-vectors-negative300.bin  \
    --binary BINARY \
    --path /word2vec \
    --host 0.0.0.0 \
    --port 5002