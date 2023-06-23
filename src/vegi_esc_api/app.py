"""
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/word2vec/n_similarity?ws1=sushi&ws1=shop&ws2=japanese&ws2=restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
"""
from __future__ import print_function
from collections import defaultdict

from vegi_esc_api.models_wrapper import ESCRatingExplainedResult
from vegi_esc_api.vegi_repo import (
    VegiRepo,
)
from vegi_esc_api.vegi_esc_repo import (
    Vegi_ESC_Repo,
    ESCRatingSql,
    NewRating,
)
from vegi_esc_api.sustained import SustainedAPI
from vegi_esc_api.create_app import create_app
from vegi_esc_api.models import ESCProductInstance, ESCExplanationCreate, ServerError
from vegi_esc_api.logger import LogLevel, slow_call_timer
import vegi_esc_api.logger as logger
from vegi_esc_api.word_vec_model import getModel
from dotenv import load_dotenv
from datetime import datetime, timedelta
from flask import redirect, url_for, request, Flask
import pickle
import base64
import argparse
import os
import json
import sys
from difflib import SequenceMatcher
from typing import Any


# import gensim.models.keyedvectors as word2vec
from future import standard_library

standard_library.install_aliases()


if "--verbose" in sys.argv:
    logger.set_log_level(LogLevel.verbose)
    logger.info("Verbose mode enabled.")
else:
    logger.info("Verbose mode not specified.")


load_dotenv()
config = os.environ

# app = Flask(__name__)
# todo: refactor to be on the flask db config instead of bespoke ssh repo connection... then we can use the that in methods directly on dataclasses as in vegi_esc_repo.py
# db = SQLAlchemy(app)

server, vegi_db_session = create_app(None)


def simple_word_similarity(a: str, b: str):
    return SequenceMatcher(None, a, b).ratio()


@slow_call_timer
@server.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        return redirect(url_for("success", name=user))
    else:
        user = request.args.get("nm")
        return redirect(url_for("success", name=user))


@slow_call_timer
@server.route("/rating/<id>")
def rating(id: str):
    rating = None

    repoConn = Vegi_ESC_Repo(app=server)
    r_id = int(id)
    rating = repoConn.get_rating(rating_id=r_id, since_time_delta=timedelta(days=5))
    if not rating:
        return json.dumps(
            ServerError(message="Rating not found for id", code="NOT_FOUND")
        )
    return json.dumps(rating.serialize())


@slow_call_timer
@server.route("/rate-product")
def rate_product(
    name: str,
    product_external_id_on_source: str,
    # vendorInternalId: str,
    source: int,
    description: str,
    category: str,
    keyWords: list[str],
    imageUrl: str,
    ingredients: str,
    packagingType: str,
    stockUnitsPerProduct: int,
    sizeInnerUnitValue: float,
    sizeInnerUnitType: str,
    productBarCode: str,
    supplier: str,
    brandName: str,
    origin: str,
    finalLocation: str,
    taxGroup: str,
    dateOfBirth: datetime,
    finalDate: datetime,
):
    escRepoConn = Vegi_ESC_Repo(app=server)
    # TODO: if product already exists in esc_db then use it
    ss = SustainedAPI(app=server)
    esc_product = escRepoConn.get_esc_product(
        name=name, source=ss.get_sustained_escsource().id
    )
    if esc_product:
        # TODO: Rate the existing product and return
        new_rating, most_similar_esc_product = _rate_product(
            product_name=esc_product.name,
            product_category_name=esc_product.category,
            product_names_in_same_category=[],
        )
        if not new_rating:
            return json.dumps({})
        return json.dumps(
            {
                "product": esc_product.serialize(),
                "most_similar_esc_product": most_similar_esc_product.serialize(),
                **new_rating.serialize(),
            }
        )
    new_rating, most_similar_esc_product = _rate_product(
        product_name=name,
        product_category_name=category,
        product_names_in_same_category=[],
    )
    if not new_rating:
        return json.dumps({})
    return json.dumps(
        {
            "product": name,
            "most_similar_esc_product": most_similar_esc_product.serialize(),
            **new_rating.serialize(),
        }
    )
    # product = escRepoConn.add_product_if_not_exists(
    #     name=name,
    #     product_external_id_on_source=product_external_id_on_source,
    #     source=source,
    #     description=description,
    #     category=category,
    #     keyWords=keyWords,
    #     imageUrl=imageUrl,
    #     ingredients=ingredients,
    #     packagingType=packagingType,
    #     stockUnitsPerProduct=stockUnitsPerProduct,
    #     sizeInnerUnitValue=sizeInnerUnitValue,
    #     sizeInnerUnitType=sizeInnerUnitType,
    #     productBarCode=productBarCode,
    #     supplier=supplier,
    #     brandName=brandName,
    #     origin=origin,
    #     finalLocation=finalLocation,
    #     taxGroup=taxGroup,
    #     dateOfBirth=dateOfBirth,
    #     finalDate=finalDate,
    # )


@slow_call_timer
@server.route("/rate-vegi-product/<id>")
def rate_vegi_product(id: str):
    product, new_rating, most_similar_esc_product = _rate_vegi_product(id=int(id))
    if new_rating is None or product is None:
        return json.dumps({})

    return json.dumps(
        {
            "product": product.serialize(),
            "most_similar_esc_product": most_similar_esc_product.serialize(),
            "new_rating": new_rating.serialize(),
        }
    )


def _rate_vegi_product(id: int):
    p_id = int(id)
    repoConn = VegiRepo(app=server)
    # * get vegi product details from vegi repository
    product, products_in_category, category = repoConn.get_product_category_details(
        p_id
    )
    # NOTE Get the product and all other products in same category from vegi repo
    # NOTE So similarity checks of all product names in same cat and category name with higher weight in weighted average to get most similar sustained category
    if not products_in_category or not category or not product:
        return (
            None,
            None,
            ServerError(message="no products matched for id", code="PRODUCT_NOT_FOUND"),
        )
    new_rating, most_similar_esc_product = _rate_product(
        product_name=product.name,
        product_category_name=category.name,
        product_names_in_same_category=[p.name for p in products_in_category],
    )
    return product, new_rating, most_similar_esc_product


def _find_similar_rated_product(
    product_name: str,
    product_names_in_same_category: list[str],
    product_category_name: str,
):
    if product_name not in product_names_in_same_category:
        product_names_in_same_category.append(product_name)

    n = float(len(product_names_in_same_category))

    # * find most similar product in sustained
    # store the distance from the category matched in teh below function too
    most_sim_cats_to_category_map = _sustained_most_similar_category_id_spaced_map(
        product_category_name
    )
    # cat = ss.get_cat_for_space_delimited_id(most_sim_cat_id, replace=("-", " "))
    category_weight = 0.4
    # product_weight = 1 - category_weight
    # _average_wmds_for_sustained_cats = {
    #     sustained_cat: category_weight * cat_cat_wmd
    #     for sustained_cat, cat_cat_wmd in most_sim_cats_to_category_map.items()
    # }
    _averages: dict[str, dict[str, float]] = {}
    for i, _product_name in enumerate(product_names_in_same_category):
        # progress_percent = i / n
        _averages[_product_name] = _sustained_most_similar_category_id_spaced_map(
            _product_name
        )

    _sscat_id_name_to_prod_avg_wmd: defaultdict[str, float] = defaultdict(float)
    for _product_name in _averages.keys():
        for ss_cat_id_sp in _averages[_product_name]:
            if ss_cat_id_sp not in _sscat_id_name_to_prod_avg_wmd.keys():
                _sscat_id_name_to_prod_avg_wmd[ss_cat_id_sp] = 0.0
            _sscat_id_name_to_prod_avg_wmd[ss_cat_id_sp] += (
                _averages[_product_name][ss_cat_id_sp] / n
            )
    _sscat_id_name_to_avg_wmd: dict[str, float] = {}
    for sscat_name in _sscat_id_name_to_prod_avg_wmd.keys():
        _sscat_id_name_to_avg_wmd[sscat_name] = (
            _sscat_id_name_to_prod_avg_wmd[sscat_name] * (1 - category_weight)
            + most_sim_cats_to_category_map[sscat_name] * category_weight
        )
    most_sim_v = min([v for v in _sscat_id_name_to_avg_wmd.values()])
    most_similar_ss_cat_id_spaced = next(
        (k for k, v in _sscat_id_name_to_avg_wmd.items() if v == most_sim_v)
    )
    # * Then match most similar product in that category same as already done and use its rating for now
    most_similar_ss_product_result = (
        _sustained_most_similar_product_for_ss_cat_space_delimited_id(
            search_product_name=product_name,
            most_sim_cat_id=most_similar_ss_cat_id_spaced,
        )
    )
    return most_similar_ss_product_result


def _rate_product(
    product_name: str,
    product_names_in_same_category: list[str],
    product_category_name: str,
):
    ss = SustainedAPI(app=server)
    sustained_escsource = ss.get_sustained_escsource()
    escRepoConn = Vegi_ESC_Repo(app=server)
    most_similar_ss_product_result = _find_similar_rated_product(
        product_name=product_name,
        product_names_in_same_category=product_names_in_same_category,
        product_category_name=product_category_name,
    )
    esc_product = most_similar_ss_product_result._sustainedProduct
    new_rating = escRepoConn.add_rating_for_product(
        # product_id=p_id,
        new_rating=ESCRatingSql(
            product=esc_product.id,
            product_id=most_similar_ss_product_result.rating.product_id,
            product_name=most_similar_ss_product_result.rating.product_name,
            rating=most_similar_ss_product_result.rating.rating,
            calculated_on=most_similar_ss_product_result.rating.calculated_on,
        ),
        explanationsCreate=[
            ESCExplanationCreate(
                title=f"Proxy from similar product [{most_similar_ss_product_result._sustainedProduct.name}]: {e.title}",
                measure=e.measure,
                reasons=e.reasons,
                evidence=e.evidence,
            )
            for e in most_similar_ss_product_result.explanations
        ],
        source=sustained_escsource.id,
    )
    return new_rating, esc_product


@slow_call_timer
@server.route("/rate-latest")
def rate_latest():
    """load latest `n`[=10] with none or expired ratings, rate them and add lines to esc db for them with (`this is for similar product` flag)"""
    args = request.args
    n = 10
    try:
        n = int(args["n"])
    except Exception:
        n = 10
    n = max(1, n)
    repoConn = VegiRepo(app=server)
    product_ids = repoConn.get_products_to_rate(limit=10, days_ago=5)
    products, ratings = repoConn.get_products_and_ratings_from_ids(ids=product_ids)
    new_ratings: list[NewRating] = []
    for p in products:
        p_id = p.id
        new_rating = _rate_vegi_product(id=p_id)
        if isinstance(new_rating, NewRating):
            new_ratings.append(new_rating)
        # # NOTE: 1. Find most simliar product by name existing already in ESC DB
        # product_name = p.name
        # most_similar_ss_product_result = _sustained_most_similar_product(product_name)
        # # NOTE: 2. Copy the explanations for this product to results here prepending to each explanation that it is taken from the ESCExplanation of a simialar product.
        # new_rating = repoConn.add_rating_for_product(
        #     product_id=p_id,
        #     new_rating=VegiESCRatingSql(
        #         productPublicId=most_similar_ss_product_result.rating.product_id,
        #         rating=most_similar_ss_product_result.rating.rating,
        #         calculatedOn=most_similar_ss_product_result.rating.calculated_on,
        #         product=p_id,
        #     ),
        #     explanations=[
        #         VegiESCExplanationSql(
        #             title=f"Proxy from similar product [{most_similar_ss_product_result._sustainedProduct.name}]: {e.title}",
        #             measure=e.measure,
        #             reasons=e.reasons,
        #             evidence=e.evidence,
        #             escrating=e.rating,
        #             escsource=e.source,
        #         )
        #         for e in most_similar_ss_product_result.explanations
        #     ],
        # )
        # new_ratings += [new_rating]

    return json.dumps([nr.serialize() for nr in new_ratings])


@server.route("/vegi-users")
def vegi_users():
    users = []
    repoConn = VegiRepo(app=server)
    users = repoConn.get_users()

    return json.dumps([u.serialize() for u in users])


def filter_words(words: list[str] | str):
    if words is None:
        return
    # if not model.key_to_index[words]:
    #     return
    # return words
    # Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.
    # See https: // github.com/RaRe-Technologies/gensim/wiki/Migrating-from -Gensim-3.x-to-4
    return [word for word in words if word in model.key_to_index.keys()]


@slow_call_timer
@server.route("/success/<name>")
def success(name: str):
    return "welcome %s" % name


@slow_call_timer
@server.route("/n_similarity")
def n_similarity():
    args = request.args
    a = str(args["ws1"]).lower()
    b = str(args["ws2"]).lower()
    result = model.n_similarity(filter_words(a), filter_words(b)).item()
    logger.info(result)
    return f"Success: {result}"


@slow_call_timer
@server.route("/similarity")
def similarity():
    if norm == "disable":
        return ("most_similar disabled", 400)
    args = request.args
    a = str(args["w1"]).lower()
    b = str(args["w2"]).lower()
    if a not in model.key_to_index.keys():
        a_sims = _lookup_most_similar_words_in_language_model(a)
        if a_sims:
            a = a_sims[0]
        # else we leave a as a in the knowledge that it will throw a KeyError below...
    if b not in model.key_to_index.keys():
        b_sims = _lookup_most_similar_words_in_language_model(b)
        if b_sims:
            a = b_sims[0]
        # else we leave a as a in the knowledge that it will throw a KeyError below...
    result = model.similarity(a, b).item()
    logger.info(result)
    return f"Success: {result}"


def _lookup_most_similar_words_in_language_model(
    a: str, tolerance: float = 0.75
) -> list[str]:
    if a in model.key_to_index.keys():
        return [a]
    similar_vocab = [
        word
        for word in model.key_to_index.keys()
        if simple_word_similarity(a, word) >= tolerance
    ]
    return similar_vocab


@slow_call_timer
@server.route("/nearest-word-in-model")
def lookup_most_similar_word_in_language_model():
    if norm == "disable":
        return ("most_similar disabled", 400)
    args = request.args
    a = str(args["w1"]).lower()
    result = _lookup_most_similar_words_in_language_model(a)
    logger.info(result)
    return f"Success: {result}"


@server.route("/connection")
def connection():
    args = request.args
    a = str(args["w1"]).lower()
    b = str(args["w2"]).lower()
    result = model.similarity(a, b).item()
    logger.info(result)
    return f"Success: {result}"


@slow_call_timer
@server.route("/sentence_similarity")
def sentence_similarity():
    if norm == "disable":
        return ("most_similar disabled", 400)
    args = request.args
    result = model.wmdistance(
        args["s1"], args["s2"]
    )  # see wmd.py for use (compare sentence1 and sentence2)
    logger.info(result)
    return f"Success: {result}"


@server.route("/sustained/refresh")
def sustained_refresh():
    ss = SustainedAPI(app=server)
    ss.refresh_products_lists()
    return "Success"


def _sustained_most_similar_category_id_spaced_map(sentence1: str):
    """returns a map from `category_id` (using spaces in key name not underscores) to the wmdistance between the `sentence1` to that `category_id`"""
    ss = SustainedAPI(app=server)
    sustained_categories = ss.get_category_ids(replace=("-", " "))
    similarities: dict[str, float] = dict()
    for category_id_spaced in sustained_categories:
        similarities[category_id_spaced] = model.wmdistance(
            category_id_spaced, sentence1
        )
        logger.info(f"'{category_id_spaced}': {similarities[category_id_spaced]}")
    logger.info(similarities)
    # most_sim_v = min([v for v in similarities.values()])
    # most_similar = next((k for k, v in similarities.items() if v == most_sim_v))
    return similarities


def _sustained_most_similar_category_id_spaced(sentence1: str):
    similarities = _sustained_most_similar_category_id_spaced_map(sentence1=sentence1)
    most_sim_v = min([v for v in similarities.values()])
    most_similar = next((k for k, v in similarities.items() if v == most_sim_v))
    return f"{most_similar}"


@server.route("/sustained/most-similar-category")
def sustained_most_similar_category():
    args = request.args
    logger.info(args)
    most_sim_cat_id = _sustained_most_similar_category_id_spaced(args["s1"])
    ss = SustainedAPI(app=server)
    cat = ss.get_cat_for_space_delimited_id(most_sim_cat_id, replace=("-", " "))
    if cat:
        return cat.name
    raise Exception("Category not found")


def _sustained_product_to_vegi_esc_rating(sProd: ESCProductInstance):
    ss = SustainedAPI(app=server)
    return ss.get_product_with_impact(
        sustainedProductId=sProd.product_external_id_on_source
    )


def _sustained_most_similar_product(sentence1: str):
    most_sim_cat_id = _sustained_most_similar_category_id_spaced(sentence1)
    return _sustained_most_similar_product_for_ss_cat_space_delimited_id(
        search_product_name=sentence1, most_sim_cat_id=most_sim_cat_id
    )


def _sustained_most_similar_product_for_ss_cat_space_delimited_id(
    search_product_name: str, most_sim_cat_id: str
):
    ss = SustainedAPI(app=server)
    cat = ss.get_cat_for_space_delimited_id(most_sim_cat_id, replace=("-", " "))
    if not cat:
        raise Exception("Category not found")
    sustained_products = ss.get_products(category_name=cat.name)
    similarities = dict()
    for product in sustained_products:
        product_name = product.name
        similarities[product_name] = model.wmdistance(product_name, search_product_name)
        logger.verbose(f"'{product_name}': {similarities[product_name]}")
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
        original_search_term=search_product_name,
        wmdistance=min_v,
        _sustainedProduct=most_similar_product,
    )


@slow_call_timer
@server.route("/sustained/most-similar-product")  # type: ignore
def sustained_most_similar():
    args = request.args
    logger.info(args)
    most_similar_product_name = _sustained_most_similar_product(args["s1"])
    return most_similar_product_name.toJson()


@slow_call_timer
@server.route("/most_similar")
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
        logger.info(res)
        return f"Success: {res}"
    except Exception as e:
        logger.error(e)
        return "An error occured!"


@server.route("/model")
def modelCall():
    args = request.args
    try:
        res = model[args["word"]]
        res = base64.b64encode(res).decode()
        logger.info(res)
        return f"Success: {res}"
    except Exception as e:
        logger.error(e)
        return "An error occured!"


@server.route("/model_word_set")
def model_word_set():
    # args = request.args
    try:
        res = base64.b64encode(pickle.dumps(set(model.index2word))).decode()
        logger.info(res)
        return f"Success: {res}"
    except Exception as e:
        logger.error(e)
        return "An error occured!"


@server.errorhandler(404)
def pageNotFound(error):
    return "page not found"


@server.errorhandler(500)
def raiseError(error):
    return error


def initApp(app: Flask, args: argparse.Namespace | None = None):
    global model
    global norm
    if args:
        model = getModel(args=args)

        logger.verbose("App model loaded, now running initApp")
        host = args.host
        port = int(os.environ.get("PORT", int(args.port)))

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
    logger.info(
        f"Check the app is up by openning: `http://{host}:{port}/success/fenton` unless in docker where need to map the port from {port} to your forward port..."
    )
    return lambda: app.run(host=host, port=port)


if __name__ == "__main__":
    # ----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    default_host = "0.0.0.0"
    default_port = 5002
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

    runApp = initApp(app=server, args=args)
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
    runApp = initApp(app=server)
    
    # Gunicorn entry point generator
    def _app(*args: Any, **kwargs: Any):
        # Gunicorn CLI args are useless.
        # https://stackoverflow.com/questions/8495367/
        #
        # Start the application in modified environment.
        # https://stackoverflow.com/questions/18668947/
        #
        import sys
        sys.argv = ['--gunicorn']
        for k in kwargs:
            sys.argv.append("--" + k)
            sys.argv.append(kwargs[k])
        return server
    gunicorn_app = _app  # gunicorn --bind 127.0.0.1:5002 app:gunicorn_app(**kwargs) --timeout 90 --log-level=debug
