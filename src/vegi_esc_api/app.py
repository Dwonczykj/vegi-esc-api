"""
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/word2vec/n_similarity?ws1=sushi&ws1=shop&ws2=japanese&ws2=restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
"""
from __future__ import print_function
from vegi_esc_api.models_wrapper import ESCRatingExplainedResult
from vegi_esc_api.vegi_repo import SSHRepo
from vegi_esc_api.sustained import SustainedAPI
from vegi_esc_api.sustained_models import SustainedProductBase
from vegi_esc_api.create_app import create_app
from vegi_esc_api.logger import LogLevel
import vegi_esc_api.logger as logger
from typing import Any
from dotenv import load_dotenv
from flask import redirect, url_for, request
import pickle
import base64
import argparse
import os
import json

# import gensim.models.keyedvectors as word2vec
from gensim import models
from future import standard_library

standard_library.install_aliases()


logger.set_log_level(LogLevel.verbose)


VEGI_SERVER_P_KEY_FILE: str = ""
VEGI_SERVER_PUBLIC_HOSTNAME = ""
VEGI_SERVER_PUBLIC_IP_ADDRESS = ""
VEGI_SERVER_PRIVATE_IP_ADDRESS = ""
VEGI_SERVER_USERNAME = ""
SSH_ARGS = ""
SQL_USER = ""
SQL_PASSWORD = ""
SQL_DB_NAME = ""
# with open('hosts.yml') as f:
# hostKVPs = yaml.load(f, Loader=yaml.FullLoader)
# config = hostKVPs['all']['hosts']['vegi-backend-qa']

load_dotenv()
config = os.environ
if config:
    VEGI_SERVER_IP_ADDRESS = config.get("ansible_ssh_host", "")
    VEGI_SERVER_PRIVATE_IP_ADDRESS = config.get("ansible_ssh_private_ip", "")
    VEGI_SERVER_PUBLIC_HOSTNAME = (
        "ec2-"
        + config.get("ansible_ssh_host", "").replace(".", "-")
        + ".compute-1.amazonaws.com"
    )  # ec2-54-221-0-234.compute-1.amazonaws.com
    VEGI_SERVER_USERNAME = config.get("ansible_user", "")
    # VEGI_SERVER_P_KEY_FILE = config.get('ansible_ssh_private_key_file','').replace('~', '/Users/joey')
    VEGI_SERVER_P_KEY_FILE = os.path.expanduser(
        config.get("ansible_ssh_private_key_file", "")
    )
    SSH_ARGS = config.get("ansible_ssh_extra_args", "")
    SQL_USER = config.get("mysql_production_user", "")
    SQL_PASSWORD = config.get("mysql_production_password", "")
    SQL_DB_NAME = config.get("mysql_production_database", "")

# app = Flask(__name__)
# db = SQLAlchemy(app)


app = create_app(None)


@app.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        return redirect(url_for("success", name=user))
    else:
        user = request.args.get("nm")
        return redirect(url_for("success", name=user))


@app.route("/product/<id>")
def product(id: int):
    product = None
    with SSHRepo(
        # server_hostname=VEGI_SERVER_PRIVATE_IP_ADDRESS,
        server_hostname=VEGI_SERVER_PUBLIC_HOSTNAME,
        ssh_user=VEGI_SERVER_USERNAME,
        ssh_pkey=VEGI_SERVER_P_KEY_FILE,
        db_hostname="localhost",
        db_host_inbound_port="3306",
        db_username=SQL_USER,
        db_password=SQL_PASSWORD,
        db_name=SQL_DB_NAME,
    ) as repoConn:
        repoConn._connect_ssh()
        product = repoConn.get_product_to_rate(id)

    return json.dumps(product)


@app.route("/rate-product/<id>")
def rate_product(id: int):
    with SSHRepo(
        # server_hostname=VEGI_SERVER_PRIVATE_IP_ADDRESS,
        server_hostname=VEGI_SERVER_PUBLIC_HOSTNAME,
        ssh_user=VEGI_SERVER_USERNAME,
        ssh_pkey=VEGI_SERVER_P_KEY_FILE,
        db_hostname="localhost",
        db_host_inbound_port="3306",
        db_username=SQL_USER,
        db_password=SQL_PASSWORD,
        db_name=SQL_DB_NAME,
    ) as repoConn:
        repoConn._connect_ssh()
        product = repoConn.get_product_to_rate(id)
        rating_explanations = repoConn.get_product_ratings(id)
        print(rating_explanations)
        # todo: choose the product rating if exists that closest matches a category from sustained
        rating = {
            "rating": 5 if str(product["name"]).upper().startswith("A") else 0,
            "explanation": "product starts with an A",
        }

    return json.dumps(rating)


@app.route("/users")
def users():
    users = []
    with SSHRepo(
        # server_hostname=VEGI_SERVER_PRIVATE_IP_ADDRESS,
        server_hostname=VEGI_SERVER_PUBLIC_HOSTNAME,
        ssh_user=VEGI_SERVER_USERNAME,
        ssh_pkey=VEGI_SERVER_P_KEY_FILE,
        db_hostname="localhost",
        db_host_inbound_port="3306",
        db_username=SQL_USER,
        db_password=SQL_PASSWORD,
        db_name=SQL_DB_NAME,
    ) as repoConn:
        repoConn._connect_ssh()
        users = repoConn.read_all_records_from_users()

    return json.dumps(users)


def filter_words(words):
    if words is None:
        return
    # if not model.key_to_index[words]:
    #     return
    # return words
    # Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.
    # See https: // github.com/RaRe-Technologies/gensim/wiki/Migrating-from -Gensim-3.x-to-4
    return [word for word in words if word in model.key_to_index.keys()]


@app.route("/success/<name>")
def success(name: str):
    return "welcome %s" % name


@app.route("/n_similarity")
def n_similarity():
    args = request.args
    print(args)
    result = model.n_similarity(
        filter_words(args["ws1"]), filter_words(args["ws2"])
    ).item()
    print(result)
    return f"Success: {result}"


@app.route("/similarity")
def similarity():
    if norm == "disable":
        return ("most_similar disabled", 400)
    args = request.args
    result = model.similarity(args["w1"], args["w2"]).item()
    print(result)
    return f"Success: {result}"


@app.route("/sentence_similarity")
def sentence_similarity():
    if norm == "disable":
        return ("most_similar disabled", 400)
    args = request.args
    result = model.wmdistance(
        args["s1"], args["s2"]
    )  # see wmd.py for use (compare sentence1 and sentence2)
    print(result)
    return f"Success: {result}"


@app.route("/sustained/refresh")
def sustained_refresh():
    ss = SustainedAPI()
    ss.refresh_products_lists()
    return "Success"


def _sustained_most_similar_category_id_spaced(sentence1: str):
    ss = SustainedAPI()
    sustained_categories = ss.get_category_ids(replace=("-", " "))
    similarities = dict()
    for category_id_spaced in sustained_categories:
        similarities[category_id_spaced] = model.wmdistance(
            category_id_spaced, sentence1
        )
        print(f"'{category_id_spaced}': {similarities[category_id_spaced]}")
    print(similarities)
    min_v = min([v for v in similarities.values()])
    most_similar = next((k for k, v in similarities.items() if v == min_v))
    return f"{most_similar}"


@app.route("/sustained/most-similar-category")
def sustained_most_similar_category():
    args = request.args
    print(args)
    most_sim_cat_id = _sustained_most_similar_category_id_spaced(args["s1"])
    ss = SustainedAPI()
    cat = ss.get_cat_for_space_delimited_id(most_sim_cat_id, replace=("-", " "))
    if cat:
        return cat["name"]
    raise Exception("Category not found")


def _sustained_product_to_vegi_esc_rating(sProd: SustainedProductBase):
    ss = SustainedAPI()
    return ss.get_product_with_impact(sustainedProductId=sProd.id)


def _sustained_most_similar_product(sentence1: str):
    most_sim_cat_id = _sustained_most_similar_category_id_spaced(sentence1)
    ss = SustainedAPI()
    cat = ss.get_cat_for_space_delimited_id(most_sim_cat_id, replace=("-", " "))
    if not cat:
        raise Exception("Category not found")
    sustained_products = ss.get_products(category_name=cat["name"])
    similarities = dict()
    for product in sustained_products:
        product_name = product.name
        similarities[product_name] = model.wmdistance(product_name, sentence1)
        print(f"'{product_name}': {similarities[product_name]}")
    min_v = min([v for v in similarities.values()])
    most_similar = next((k for k, v in similarities.items() if v == min_v))
    products_with_matching_name = [
        p for p in sustained_products if p.name.lower() == most_similar.lower()
    ]
    assert (
        products_with_matching_name
    ), "Products_with_matching_name somehow lost product name - Should never happen"
    most_similar_product = products_with_matching_name[0]
    vegiRating = _sustained_product_to_vegi_esc_rating(sProd=most_similar_product)
    # adjust rating for wmdistance away if < 0.5 keep same as v similar product if > 1.2, v different products and reduce rating by 50%
    if min_v <= 0.6:
        pass  # no adjustment needed as fairly close match
    elif min_v >= 1.2:
        vegiRating.rating.rating *= 0.5
    else:
        # min_v = 0.6 -> 0.5 -> 1
        # min_v = 1.2 -> 0 -> 0.5
        vegiRating.rating.rating *= (((1.2 - min_v) / (1.2 - 0.6)) * 0.5) + 0.5

    return ESCRatingExplainedResult(
        rating=vegiRating.rating,
        explanations=vegiRating.explanations,
        original_search_term=sentence1,
        wmdistance=min_v,
        _sustainedProduct=most_similar_product,
    )


@app.route("/sustained/most-similar-product")  # type: ignore
def sustained_most_similar():
    args = request.args
    print(args)
    most_similar_product_name = _sustained_most_similar_product(args["s1"])
    return most_similar_product_name.toJson()


@app.route("/most_similar")
def most_similar():
    args = request.args
    pos = filter_words(args.get("positive", []))
    neg = filter_words(args.get("negative", []))
    t = args.get("topn", 10)
    pos = [] if pos is None else pos
    neg = [] if neg is None else neg
    t = 10 if t is None else t
    logger.verbose(
        "positive: " + str(pos) + " negative: " + str(neg) + " topn: " + str(t)
    )
    try:
        res = model.most_similar_cosmul(positive=pos, negative=neg, topn=t)
        print(res)
        return f"Success: {res}"
    except Exception as e:
        logger.error(e)
        return "An error occured!"


@app.route("/model")
def modelCall():
    args = request.args
    try:
        res = model[args["word"]]
        res = base64.b64encode(res).decode()
        print(res)
        return f"Success: {res}"
    except Exception as e:
        logger.error(e)
        return "An error occured!"


@app.route("/model_word_set")
def model_word_set():
    # args = request.args
    try:
        res = base64.b64encode(pickle.dumps(set(model.index2word))).decode()
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


def getModel(args: argparse.Namespace | None = None):
    model: Any
    model_path = "./models/GoogleNews-vectors-negative300"
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
        logger.info("Loading model from gensim downloader...")
        import gensim.downloader as api

        model = api.load("word2vec-google-news-300")
        # ~ https://stackoverflow.com/a/59912447
        # model.wv.save_word2vec_format(f"{model_path}.bin", binary=True)
        model.save(model_path)

    # ~ https://stackoverflow.com/a/43067907
    # model_save_path = "./models/GoogleNews-vectors-negative300.model"
    # if not os.path.exists(model_save_path):
    #     model.save()
    return model


def initApp(app, args: argparse.Namespace | None = None):
    global model
    global norm
    if args:
        model = getModel(args=args)

        host = args.host if args.host else "localhost"
        # path = args.path if args.path else "/word2vec"
        port = int(os.environ.get("PORT", int(args.port) if args.port else 5002))

        # if not args.model:
        #     logger.verbose("Usage: word2vec-apy.py --model path/to/the/model [--host host --port 1234]")

        # # model = models.Word2Vec.load_word2vec_format(model_path, binary=binary)
        # model = models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
        # import gensim.downloader as api
        # model = api.load('word2vec-google-news-300')
        # print(model["queen"])

        norm = args.norm if args.norm else "both"
        norm = norm.lower()
        if norm in ["clobber", "replace"]:
            norm = "clobber"
            logger.verbose("Normalizing (clobber)...")
            # model.init_sims(replace=True)
            model.fill_norms(replace=True)
        elif norm == "already":
            model.wv.vectors_norm = (
                model.wv.vectors
            )  # prevent recalc of normed vectors (model.syn0norm = model.syn0)
        elif norm in ["disable", "disabled"]:
            norm = "disable"
        else:
            norm = "both"
            logger.verbose("Normalizing...")
            model.fill_norms()
        if norm == "both":
            logger.verbose("Model loaded.")
        else:
            logger.verbose(("Model loaded. (norm=", norm, ")"))
    else:
        # model_path = "./models/GoogleNews-vectors-negative300.bin" #"./model.bin.gz"
        # binary = model_path.endswith('.bin')
        # binary_mode = 'BINARY_MODE' if binary else 'NON_BINARY_MODE'
        host = "localhost"
        # path = "/word2vec"
        port = int(
            os.environ.get("PORT", 5002)
        )  # allow heroku to set the path, no need for python-dotenv package to get this env variable either...
        # logger.verbose("Loading model...")
        # model = models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
        # model = models.Word2Vec.load_word2vec_format(model_path, binary=binary)
        model = getModel()
        norm = "both"
        logger.verbose("Normalizing...")
        model.fill_norms()

    logger.info(f"Running app from {host}:{port}")
    logger.info(f"Check the app is up by running: `{host}:{port}/success/fenton`")
    return lambda: app.run(host=host, port=port)


if __name__ == "__main__":
    # ----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Path to the trained model")
    p.add_argument("--binary", help="Specifies the loaded model is binary")
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 5002)")
    p.add_argument("--path", help="Path (default: /word2vec)")
    p.add_argument(
        "--norm",
        help="How to normalize vectors. clobber: Replace loaded vectors with normalized versions. Saves a lot of memory if exact vectors aren't needed. both: Preserve the original vectors (double memory requirement). already: Treat model as already normalized. disable: Disable 'most_similar' queries and do not normalize vectors. (default: both)",
    )
    args = p.parse_args()

    runApp = initApp(app=app, args=args)
    # api.add_resource(N_Similarity, path+'/n_similarity')
    # api.add_resource(Similarity, path+'/similarity')
    # api.add_resource(MostSimilar, path+'/most_similar')
    # api.add_resource(Model, path+'/model')
    # api.add_resource(ModelWordSet, '/word2vec/model_word_set')

    # create_app = app
    # create_app.run()
    runApp()
else:
    logger.info(
        f'Thread name running app is "{__name__}"'
    )  # Thread name running app is "app" if command run is `gunicorn --bind 127.0.0.1:5002 app:gunicorn_app --timeout 90`
    runApp = initApp(app=app)
    gunicorn_app = app  # gunicorn --bind 127.0.0.1:5002 app:gunicorn_app --timeout 90 --log-level=debug
