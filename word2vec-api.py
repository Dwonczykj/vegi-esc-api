'''
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/word2vec/n_similarity?ws1=sushi&ws1=shop&ws2=japanese&ws2=restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''
from __future__ import print_function
from logger import LOG_LEVEL, LogLevel
# import gensim.models.keyedvectors as word2vec
from gensim import models

from future import standard_library
standard_library.install_aliases()
from builtins import str
from flask import Flask, request, jsonify
from flask import Flask, redirect, url_for, request
import logger
import os
import json

LOG_LEVEL = LogLevel.info

# from flask_restful import Resource, Api, reqparse
from gensim import utils, matutils
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
     uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
import pickle
import argparse
import base64


app = Flask(__name__)


def filter_words(words):
    if words is None:
        return
    # if not model.key_to_index[words]:
    #     return
    # return words
    # Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.
    # See https: // github.com/RaRe-Technologies/gensim/wiki/Migrating-from -Gensim-3.x-to-4
    return [word for word in words if word in model.key_to_index.keys()]


@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name


@app.route('/n_similarity')
def n_similarity():
    args = request.args
    print(args)
    result = model.n_similarity(filter_words([args['ws1']]),filter_words([args['ws2']])).item()
    print(result)
    return f"Success: {result}"


@app.route('/similarity')
def similarity():
    if (norm == "disable"):
        return ("most_similar disabled", 400)
    args = request.args
    result = model.similarity(args['w1'], args['w2']).item()
    print(result)
    return f"Success: {result}"


@app.route('/most_similar')
def most_similar():
    args = request.args
    pos = filter_words(args.get('positive', []))
    neg = filter_words(args.get('negative', []))
    t = args.get('topn', 10)
    pos = [] if pos == None else pos
    neg = [] if neg == None else neg
    t = 10 if t == None else t
    logger.verbose("positive: " + str(pos) + " negative: " +
          str(neg) + " topn: " + str(t))
    try:
        res = model.most_similar_cosmul(positive=pos, negative=neg, topn=t)
        print(res)
        return f"Success: {res}"
    except Exception as e:
        logger.error(e)
        logger.error(res)
        return "An error occured!"


@app.route('/model')
def modelCall():
    args = request.args
    try:
        res = model[args['word']]
        res = base64.b64encode(res).decode()
        print(res)
        return f"Success: {res}"
    except Exception as e:
        logger.error(e)
        return "An error occured!"


@app.route('/model_word_set')
def model_word_set():
    args = request.args
    try:
        res = base64.b64encode(pickle.dumps(
            set(model.index2word))).decode()
        print(res)
        return f"Success: {res}"
    except Exception as e:
        logger.error(e)
        return "An error occured!"



@app.errorhandler(404)
def pageNotFound(error):
    return "page not found"

@app.errorhandler(500)
def raiseError(error):
    return error

if __name__ == '__main__':
    global model
    global norm

    #----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Path to the trained model")
    p.add_argument("--binary", help="Specifies the loaded model is binary")
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 5000)")
    p.add_argument("--path", help="Path (default: /word2vec)")
    p.add_argument("--norm", help="How to normalize vectors. clobber: Replace loaded vectors with normalized versions. Saves a lot of memory if exact vectors aren't needed. both: Preserve the original vectors (double memory requirement). already: Treat model as already normalized. disable: Disable 'most_similar' queries and do not normalize vectors. (default: both)")
    args = p.parse_args()

    model_path = args.model if args.model else "./model.bin.gz"
    binary = True if args.binary else False
    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/word2vec"
    port = int(args.port) if args.port else 5000
    if not args.model:
        logger.verbose("Usage: word2vec-apy.py --model path/to/the/model [--host host --port 1234]")

    logger.verbose("Loading model...")
    # model = models.Word2Vec.load_word2vec_format(model_path, binary=binary)
    model = models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
    print(model["queen"])

    norm = args.norm if args.norm else "both"
    norm = norm.lower()
    if (norm in ["clobber", "replace"]):
        norm = "clobber"
        logger.verbose("Normalizing (clobber)...")
        model.init_sims(replace=True)
    elif (norm == "already"):
        model.wv.vectors_norm = model.wv.vectors  # prevent recalc of normed vectors (model.syn0norm = model.syn0)
    elif (norm in ["disable", "disabled"]):
        norm = "disable"
    else:
        norm = "both"
        logger.verbose("Normalizing...")
        model.init_sims()
    if (norm == "both"):
        logger.verbose("Model loaded.")
    else:
        logger.verbose(("Model loaded. (norm=",norm,")"))

    # api.add_resource(N_Similarity, path+'/n_similarity')
    # api.add_resource(Similarity, path+'/similarity')
    # api.add_resource(MostSimilar, path+'/most_similar')
    # api.add_resource(Model, path+'/model')
    # api.add_resource(ModelWordSet, '/word2vec/model_word_set')
    app.run(host=host, port=port, debug=True)



