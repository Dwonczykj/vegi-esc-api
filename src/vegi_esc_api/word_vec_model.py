from __future__ import annotations
from typing import Self, Type, Any
from gensim import models
import argparse
import os
from flask import Flask
import vegi_esc_api.logger as logger
# import gensim.models.keyedvectors as word2vec


def _getModel(args: argparse.Namespace | None = None):
    model: Any
    models_path = "./models"
    # model_abs_dir = os.path.abspath(models_path)
    if os.path.exists(models_path) is False:
        os.mkdir(path=models_path)
    model_name = "word2vec-google-news-300"
    # model_name = "glove-twitter-25"
    # model_path = f"{models_path}/GoogleNews-vectors-negative300"
    model_path = f"{models_path}/{model_name}"
    model_abs_path = os.path.abspath(model_path)
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
        logger.info(f'Loading model from saved location: "{model_abs_path}"')
        model = models.KeyedVectors.load(model_path)
    elif os.path.exists(f"{model_path}.bin"):
        mp = f"{model_path}.bin"
        logger.info(f'Loading model from saved location: "{mp}"')
        binary = mp.endswith(".bin") if model_path else False
        binary_mode = "BINARY_MODE" if binary else "NON_BINARY_MODE"
        logger.verbose(f'Running "{mp}" in {binary_mode}')
        model = models.KeyedVectors.load_word2vec_format(mp, binary=binary)
        model.save(model_path)
    else:
        if os.path.exists(models_path) is False:
            os.mkdir(path=models_path)
        logger.info(f"Loading model from gensim downloader as doesn't exist in {models_path}...")
        import gensim.downloader as api

        model = api.load(model_name)
        # ~ https://stackoverflow.com/a/59912447
        # model.wv.save_word2vec_format(f"{model_path}.bin", binary=True)
        logger.info(f"Saving model downloaded from gensim downloader to {model_path}...")
        model.save(model_path)

    # ~ https://stackoverflow.com/a/43067907
    # model_save_path = "./models/GoogleNews-vectors-negative300.model"
    # if not os.path.exists(model_save_path):
    #     model.save()
    return model


class Word_Vec_Model:
    def __init__(
        self,
        model: Any,
    ) -> None:
        self._model = model

    @property
    def model(self):
        return self._model

    @classmethod
    def getModel(cls, app: Flask, args: argparse.Namespace | None = None) -> Self:
        _model = _getModel(args=args)
        self = Word_Vec_Model(model=_model)
        return self
