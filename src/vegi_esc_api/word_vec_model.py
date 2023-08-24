from __future__ import annotations
from typing import Self, Type, Any
from gensim import models
import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors
import argparse
import os
from flask import Flask
import vegi_esc_api.logger as Logger
from vegi_esc_api.init_flask_config import init_flask_config

# import gensim.models.keyedvectors as word2vec


def _getModel(args: argparse.Namespace | None = None):
    model: KeyedVectors
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
        Logger.info("Loading model from args path...")
        mp = args.model
        if mp.endswith(".bin"):
            binary = True if args.binary else mp.endswith(".bin") if mp else False
            binary_mode = "BINARY_MODE" if binary else "NON_BINARY_MODE"
            Logger.verbose(f'Running "{mp}" in {binary_mode}')
            model = models.KeyedVectors.load_word2vec_format(mp, binary=binary)
            model.save(model_path)
        else:
            model = models.KeyedVectors.load(mp)
    elif os.path.exists(f"{model_path}"):
        Logger.info(f'Loading model from saved location: "{model_abs_path}"')
        model = models.KeyedVectors.load(model_path)
    elif os.path.exists(f"{model_path}.bin"):
        mp = f"{model_path}.bin"
        Logger.info(f'Loading model from saved location: "{mp}"')
        binary = mp.endswith(".bin") if model_path else False
        binary_mode = "BINARY_MODE" if binary else "NON_BINARY_MODE"
        Logger.verbose(f'Running "{mp}" in {binary_mode}')
        model = models.KeyedVectors.load_word2vec_format(mp, binary=binary)
        model.save(model_path)
    else:
        if os.path.exists(models_path) is False:
            os.mkdir(path=models_path)
        Logger.info(f"Loading model from gensim downloader as doesn't exist in {models_path}...")

        _model = api.load(model_name)
        if isinstance(_model, str):
            Logger.warn(f'Unable to load word_vec_model["{model_name}"] with result: {_model}')
            return None
        model = _model
        # ~ https://stackoverflow.com/a/59912447
        # model.wv.save_word2vec_format(f"{model_path}.bin", binary=True)
        Logger.info(f"Saving model downloaded from gensim downloader to {model_path}...")
        model.save(model_path)

    # ~ https://stackoverflow.com/a/43067907
    # model_save_path = "./models/GoogleNews-vectors-negative300.model"
    # if not os.path.exists(model_save_path):
    #     model.save()
    return model


class Word_Vec_Model:
    def __init__(
        self,
        model: KeyedVectors,
    ) -> None:
        self._model = model

    @property
    def model(self):
        return self._model

    @classmethod
    def getModel(cls, app: Flask, args: argparse.Namespace | None = None) -> Self:
        _model = _getModel(args=args)
        assert _model is not None, f"Couldnt load word2vec model in {cls.__name__}.getModel()"
        self = Word_Vec_Model(model=_model)
        return self
    
    @classmethod
    def parseCLArgsToInitApp(cls, app: Flask):
        # ----------- Parsing Arguments ---------------
        p = argparse.ArgumentParser()
        default_host = "0.0.0.0"
        default_port = os.environ.get("PORT", 5002)
        p.add_argument("--model", help="Path to the trained model")
        p.add_argument(
            "--binary", help="Specifies the loaded model is binary", default=False
        )
        p.add_argument(
            "--host", help=f"Host name (default: {default_host})", default=default_host
        )
        p.add_argument(
            "--port", help=f"Port (default: {default_port})", default=default_port
        )
        p.add_argument("--path", help="Path (default: /word2vec)", default="/word2vec")
        p.add_argument(
            "--Xfrozen_modules", help="Python helper", required=False, default="on"
        )
        p.add_argument(
            "--norm",
            help="How to normalize vectors. clobber: Replace loaded vectors with normalized versions. Saves a lot of memory if exact vectors aren't needed. both: Preserve the original vectors (double memory requirement). already: Treat model as already normalized. disable: Disable 'most_similar' queries and do not normalize vectors. (default: both)",
        )
        args = p.parse_args()

        runApp = Word_Vec_Model.initFlaskAppModel(app=app, args=args)
        return runApp

    @classmethod
    def initFlaskAppModel(cls, app: Flask, args: argparse.Namespace | None = None):
        global model
        global norm
        if args:
            model = Word_Vec_Model.getModel(app=app, args=args).model

            Logger.verbose("App model loaded, now running initApp")
            app, host, port = init_flask_config(app=app, args=args)
            norm = args.norm if args.norm else "both"
            norm = norm.lower()
            if norm in ["clobber", "replace"]:
                norm = "clobber"
                Logger.verbose("Normalizing (clobber)...")
                # model.init_sims(replace=True)
                model.fill_norms(force=True)
            elif norm == "already":
                model.vectors_norm = (
                    model.vectors
                )  # prevent recalc of normed vectors (model.syn0norm = model.syn0)
            elif norm in ["disable", "disabled"]:
                norm = "disable"
            else:
                norm = "both"
                Logger.verbose("Normalizing...")
                model.fill_norms()
            if norm == "both":
                Logger.verbose("Model loaded.")
            else:
                Logger.verbose(("Model loaded. (norm=", norm, ")"))
        else:
            # model_path = "./models/GoogleNews-vectors-negative300.bin" #"./model.bin.gz"
            # binary = model_path.endswith('.bin')
            # binary_mode = 'BINARY_MODE' if binary else 'NON_BINARY_MODE'
            app, host, port = init_flask_config(app=app, args=args)
            # logger.verbose("Loading model...")
            # model = models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
            # model = models.Word2Vec.load_word2vec_format(model_path, binary=binary)
            model = Word_Vec_Model.getModel(app=app).model
            norm = "both"
            Logger.verbose("Normalizing...")
            model.fill_norms()
            model.index_to_key
            model.n_similarity
        return lambda: app.run(host=host, port=port)
