import vegi_esc_api.logger as logger
from typing import Any
from gensim import models
import argparse
import os
# import gensim.models.keyedvectors as word2vec


def getModel(args: argparse.Namespace | None = None):
    model: Any
    models_path = "./models"
    model_name = "word2vec-google-news-300"
    # model_name = "glove-twitter-25"
    # model_path = f"{models_path}/GoogleNews-vectors-negative300"
    model_path = f"{models_path}/{model_name}"
    if args and args.model:
        logger.info("Loading model from args path...")
        mp = args.model
        if mp.endswith(".bin"):
            binary = True if args.binary else mp.endswith(".bin") if mp else False
            binary_mode = "BINARY_MODE" if binary else "NON_BINARY_MODE"
            logger.verbose(f'Running "{mp}" in {binary_mode}')
            model = models.KeyedVectors.load_word2vec_format(mp, binary=binary)
            model.save(model_path)
        else:
            model = models.KeyedVectors.load(mp)
    elif os.path.exists(f"{model_path}"):
        model = models.KeyedVectors.load(model_path)
    elif os.path.exists(f"{model_path}.bin"):
        mp = f"{model_path}.bin"
        binary = mp.endswith(".bin") if model_path else False
        binary_mode = "BINARY_MODE" if binary else "NON_BINARY_MODE"
        logger.verbose(f'Running "{mp}" in {binary_mode}')
        model = models.KeyedVectors.load_word2vec_format(mp, binary=binary)
        model.save(model_path)
    else:
        if os.path.exists(models_path) is False:
            os.mkdir(path=models_path)
        logger.info("Loading model from gensim downloader...")
        import gensim.downloader as api

        model = api.load(model_name)
        # ~ https://stackoverflow.com/a/59912447
        # model.wv.save_word2vec_format(f"{model_path}.bin", binary=True)
        model.save(model_path)

    # ~ https://stackoverflow.com/a/43067907
    # model_save_path = "./models/GoogleNews-vectors-negative300.model"
    # if not os.path.exists(model_save_path):
    #     model.save()
    return model
