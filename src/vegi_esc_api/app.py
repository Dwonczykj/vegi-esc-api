"""
Simple web service wrapping a Word2Vec as implemented in Gensim
"""
from __future__ import print_function
import asyncio
from collections import defaultdict

from vegi_esc_api.models_wrapper import ESCRatingExplainedResult
# from vegi_esc_api.vegi_repo import (
#     VegiRepo,
# )
from vegi_esc_api.vegi_api import (
    VegiApi,
)
from vegi_esc_api.vegi_repo_models import GColumn
from vegi_esc_api.vegi_esc_repo import (
    Vegi_ESC_Repo,
    ESCRatingSql,
    NewRating,
)
from vegi_esc_api.sustained import SustainedAPI
from vegi_esc_api.create_app import create_app
from vegi_esc_api.models import ESCProductInstance, ESCExplanationCreate, ServerError
from vegi_esc_api.logger import LogLevel, slow_call_timer
import vegi_esc_api.logger as Logger
from vegi_esc_api.llm_model import LLM
from vegi_esc_api.i_llm import I_Am_LLM
from vegi_esc_api.init_flask_config import init_flask_config
# from vegi_esc_api.word_vec_model import Word_Vec_Model, initFlaskAppModel
from chromadb.api.types import Document
from dotenv import load_dotenv
from datetime import datetime, timedelta
from flask import redirect, url_for, request
# import pickle
# import base64
import os
import json
import sys
from difflib import SequenceMatcher
from typing import Any
import cachetools.func


# import gensim.models.keyedvectors as word2vec
from future import standard_library

standard_library.install_aliases()


if "--verbose" in sys.argv:
    Logger.set_log_level(LogLevel.verbose)
    Logger.info("Verbose mode enabled.")
else:
    Logger.info("Verbose mode not specified.")


load_dotenv()
config = os.environ

# app = Flask(__name__)
# todo: refactor to be on the flask db config instead of bespoke ssh repo connection... then we can use the that in methods directly on dataclasses as in vegi_esc_repo.py
# db = SQLAlchemy(app)

Logger.log('Creating FLASK APP')
server, vegi_db_session = create_app(None)


# def _llm_getter() -> I_Am_LLM:
def _llm_getter() -> LLM:
    model = LLM.getModel(app=server).model
    return model


def simple_word_similarity(a: str, b: str):
    return SequenceMatcher(None, a, b).ratio()


# * Connection Status Endpoints
@slow_call_timer
@server.route("/success/<string:name>")
def success(name: str):
    return "welcome %s" % name


@server.route("/connection")
def connection():
    return f"Success: {Vegi_ESC_Repo.db_session.is_active}"


# * Dash Auth Endpoints
@slow_call_timer
@server.route("/login", methods=["POST", "GET"])
def login():
    if request.method == "POST":
        user = request.form["nm"]
        return redirect(url_for("success", name=user))
    else:
        user = request.args.get("nm")
        return redirect(url_for("success", name=user))


# * LLM Admin
@slow_call_timer
@server.route("/reset-llm-db")
def reset_LLM_db():
    llm = _llm_getter()
    reinit: bool = False
    if request.args and request.args.get("reinit"):
        reinit = request.args.get("reinit") == "True"
    return llm.chromadb_reset_db(reinit=reinit)
    
    # collection = llm.chroma_get_esc_collection(llm.chroma_esc_product_collection_name)
    # if collection:
    #     return json.dumps(collection.peek())
    # return "db reset complete"


# * LLM Queries
@slow_call_timer
@server.route("/llm/view-vector-store")
@cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
def view_vector_store():
    llm = _llm_getter()
    collection = llm.chroma_get_esc_collection()
    return collection.json() if collection else "no vector store found"
    
    
@slow_call_timer
@server.route("/llm/view-vector-store-documents")
@cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
def view_vector_store_documents():
    llm = _llm_getter()
    query_texts = []
    _top_n = request.args.get("n")
    top_n = None
    if _top_n:
        top_n = int(_top_n)
    _query_texts = request.args.get("query")
    if _query_texts is not None:
        query_texts = _query_texts.split(",")
        if not top_n:
            top_n = 10
        result = llm.chroma_query_vector_store(["documents"], top_n, None, *query_texts)
    else:
        result = llm.chroma_get_vector_store_contents(["documents"], top_n)
    assert result is not None, "chromadb QueryResult in LLM().chroma_most_similar_sustained_products didnt obtain a query result"
    documents = result["documents"]
    assert documents is not None, "chromadb QueryResult in LLM().chroma_most_similar_sustained_products didnt contain documents"
    if not documents:
        return []
    if isinstance(documents[0], list):
        return [d for dl in documents for d in dl]
    elif isinstance(documents[0], Document):
        return [d for d in documents]
    return [d for dl in documents for d in dl]


@slow_call_timer
@server.route("/llm/query-vector-store")
def query_vector_store():
    llm = _llm_getter()
    top_n = int(request.args.get("n", default=10))
    query_texts = []
    _query_texts = request.args.get("query")
    if _query_texts is not None:
        query_texts = _query_texts.split(",")
    result = llm.chroma_query_vector_store(["documents", "metadatas"], top_n, None, *query_texts)
    assert result is not None, "chromadb QueryResult in LLM().chroma_most_similar_sustained_products didnt obtain a query result"
    return result


# * ESC Endpoints
# TODO: Organise route functions into sections and condense into just the ones we need for backend server.
# ~ https://stackoverflow.com/a/57647307
@slow_call_timer
@server.route("/rating/<int:id>", methods=['GET'])
@cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
def rating(id: int):
    rating = None
    vegiEscConn = Vegi_ESC_Repo(app=server)
    r_id = int(id)
    rating = vegiEscConn.get_rating(rating_id=r_id, since_time_delta=timedelta(days=5))
    if not rating:
        return json.dumps(
            ServerError(message="Rating not found for id", code="NOT_FOUND")
        )
    res = rating.serialize()
    assert isinstance(res, dict)
    return res


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
            # product_category_name=esc_product.category,
            # product_names_in_same_category=[],
        )
        if not new_rating:
            return {}
        return ({
            "product": esc_product.serialize(),
            "most_similar_esc_product": most_similar_esc_product.serialize(),
            **new_rating.serialize(),
        })
        
    new_rating, most_similar_esc_product = _rate_product(
        product_name=name,
        # product_category_name=category,
        # product_names_in_same_category=[],
    )
    if not new_rating:
        return {}
    return ({
        "product": name,
        "most_similar_esc_product": most_similar_esc_product.serialize(),
        **new_rating.serialize(),
    })
    
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
@cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
def rate_vegi_product(id: str):
    # so that the module contains all the methods that we might want to call on the vec model.
    product_response, new_rating, most_similar_esc_product = _rate_vegi_product(id=int(id))
    if new_rating is None or product_response is None:
        return {}

    return ({
        "product": product_response.product.serialize(),
        "category": product_response.category.serialize(),
        "most_similar_esc_product": most_similar_esc_product.serialize(),
        "new_rating": new_rating.serialize(),
    })


def _rate_vegi_product(id: int):
    p_id = int(id)
    # repoConn = VegiRepo(app=server)
    vegiApiConn = VegiApi()
    # * get vegi product details from vegi repository
    product_response = asyncio.run(vegiApiConn.get_product(
        p_id
    ))
    # NOTE Get the product and all other products in same category from vegi repo
    # NOTE So similarity checks of all product names in same cat and category name with higher weight in weighted average to get most similar sustained category
    if not product_response:
        return (
            None,
            None,
            ServerError(message="no products matched for id", code="PRODUCT_NOT_FOUND"),
        )
    new_rating, most_similar_esc_product = _rate_product(
        product_name=product_response.product.name,
    )
    return product_response, new_rating, most_similar_esc_product


def _rate_product(
    product_name: str,
    # product_names_in_same_category: list[str],
    # product_category_name: str | GColumn[str],
):
    llm = _llm_getter()
    escRepoConn = Vegi_ESC_Repo(app=server)
    ss = SustainedAPI(app=server)
    sustained_escsource = ss.get_sustained_escsource()
    
    most_similar_esc_products = llm.most_similar_esc_products(product_name)  # todo: get the cosine distance of similarity between similar product and search term name foe the min_v below
    assert most_similar_esc_products is not None, f"llm.most_similar_esc_products(\"{product_name}\") should not return None"
    esc_product, store_query_result = list(most_similar_esc_products.values())[0]  # first index as we only passed one product to *args
    assert esc_product is not None, f"llm.most_similar_esc_products(\"{product_name}\") should not return a None dictionary value"
    vegi_rated_most_sim_product = _convert_sustained_product_to_vegi_rating(
        esc_product=esc_product,
        min_v=store_query_result["distance"] if store_query_result["distance"] else 0.5,
        original_search_term=product_name,
    )
    new_rating = escRepoConn.add_rating_for_product(
        # product_id=p_id,
        new_rating=ESCRatingSql(
            product=esc_product.id,
            product_id=vegi_rated_most_sim_product.rating.product_id,
            product_name=vegi_rated_most_sim_product.rating.product_name,
            rating=vegi_rated_most_sim_product.rating.rating,
            calculated_on=vegi_rated_most_sim_product.rating.calculated_on,
        ),
        explanationsCreate=[
            ESCExplanationCreate(
                title=f"Proxy from similar product [{vegi_rated_most_sim_product._sustainedProduct.name}]: {e.title}",
                measure=e.measure,
                reasons=e.reasons,
                evidence=e.evidence,
            )
            for e in vegi_rated_most_sim_product.explanations
        ],
        source=sustained_escsource.id,
    )
    # if most_similar_ss_product_result is None:
    #     ss = SustainedAPI(app=server)
    #     sustained_escsource = ss.get_sustained_escsource()
    #     escRepoConn = Vegi_ESC_Repo(app=server)
    #     most_similar_ss_product_result = _find_similar_rated_product(
    #         product_name=product_name,
    #         product_names_in_same_category=product_names_in_same_category,
    #         product_category_name=product_category_name,
    #     )
    
    #     esc_product = most_similar_ss_product_result._sustainedProduct
    #     new_rating = escRepoConn.add_rating_for_product(
    #         # product_id=p_id,
    #         new_rating=ESCRatingSql(
    #             product=esc_product.id,
    #             product_id=most_similar_ss_product_result.rating.product_id,
    #             product_name=most_similar_ss_product_result.rating.product_name,
    #             rating=most_similar_ss_product_result.rating.rating,
    #             calculated_on=most_similar_ss_product_result.rating.calculated_on,
    #         ),
    #         explanationsCreate=[
    #             ESCExplanationCreate(
    #                 title=f"Proxy from similar product [{most_similar_ss_product_result._sustainedProduct.name}]: {e.title}",
    #                 measure=e.measure,
    #                 reasons=e.reasons,
    #                 evidence=e.evidence,
    #             )
    #             for e in most_similar_ss_product_result.explanations
    #         ],
    #         source=sustained_escsource.id,
    #     )
    return new_rating, esc_product


# remove this function as needs to be started by the vegi-backednd server when its looking for new ratings.
# @slow_call_timer
# @server.route("/rate-latest")
# def rate_latest():
#     """load latest `n`[=10] with none or expired ratings, rate them and add lines to esc db for them with (`this is for similar product` flag)"""
#     args = request.args
#     n = 10
#     try:
#         n = int(args["n"])
#     except Exception:
#         n = 10
#     n = max(1, n)
#     vegiApiConn = VegiRepo(app=server)
#     product_ids = vegiApiConn.get_products_to_rate(limit=10, days_ago=5)
#     products, ratings = vegiApiConn.get_products_and_ratings_from_ids(ids=product_ids)
#     new_ratings: list[NewRating] = []
#     for p in products:
#         p_id = p.id
#         new_rating = _rate_vegi_product(id=p_id)
#         if isinstance(new_rating, NewRating):
#             new_ratings.append(new_rating)
#         # # NOTE: 1. Find most simliar product by name existing already in ESC DB
#         # product_name = p.name
#         # most_similar_ss_product_result = _sustained_most_similar_product(product_name)
#         # # NOTE: 2. Copy the explanations for this product to results here prepending to each explanation that it is taken from the ESCExplanation of a simialar product.
#         # new_rating = vegiApiConn.add_rating_for_product(
#         #     product_id=p_id,
#         #     new_rating=VegiESCRatingSql(
#         #         productPublicId=most_similar_ss_product_result.rating.product_id,
#         #         rating=most_similar_ss_product_result.rating.rating,
#         #         calculatedOn=most_similar_ss_product_result.rating.calculated_on,
#         #         product=p_id,
#         #     ),
#         #     explanations=[
#         #         VegiESCExplanationSql(
#         #             title=f"Proxy from similar product [{most_similar_ss_product_result._sustainedProduct.name}]: {e.title}",
#         #             measure=e.measure,
#         #             reasons=e.reasons,
#         #             evidence=e.evidence,
#         #             escrating=e.rating,
#         #             escsource=e.source,
#         #         )
#         #         for e in most_similar_ss_product_result.explanations
#         #     ],
#         # )
#         # new_ratings += [new_rating]

#     return [nr.serialize() for nr in new_ratings]


@server.route("/vegi-users")
@cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
def vegi_users():
    users = []
    # vegiRepo = VegiRepo(app=server)
    vegiApiConn = VegiApi()
    users = asyncio.run(vegiApiConn.get_users())
    return [u.serialize() for u in users]


@server.route("/vegi-account")
@cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
def vegi_account():
    vegiApiConn = VegiApi()
    account = asyncio.run(vegiApiConn.view_account())
    return account


def filter_words(words: list[str] | str):
    model = _llm_getter()
    if words is None:
        return []
    # if not model.key_to_index[words]:
    #     return
    # return words
    # Use KeyedVector's .key_to_index dict, .index_to_key list, and methods .get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.
    # See https: // github.com/RaRe-Technologies/gensim/wiki/Migrating-from -Gensim-3.x-to-4
    return [word for word in words if word in model.key_to_index.keys()]


# * ESC DB Admin Endpoints
@server.route("/products/add")  # type: ignore
def add_product():
    name: str = request.args.get('name', '')
    assert name, "name argument must be required and non-empty"
    product_external_id_on_source: str = request.args.get('product_external_id_on_source', '')
    assert product_external_id_on_source, "product_external_id_on_source must be required and non-empty"
    source: int = int(request.args.get('source', '0'))
    assert source, "source argument must be required and non-empty"
    description: str = request.args.get('description', default='')
    category: str = request.args.get('category', default='')
    assert category, "category argument must be required and non-empty"
    keyWords: list[str] = json.loads(s=request.args.get('keyWords', default='[]'))
    imageUrl: str = request.args.get('imageUrl', default='')
    ingredients: str = request.args.get('ingredients', default='')
    packagingType: str = request.args.get('packagingType', default='')
    stockUnitsPerProduct: int = int(request.args.get('stockUnitsPerProduct', default='0'))
    sizeInnerUnitValue: float = float(request.args.get('sizeInnerUnitValue', default='1.0'))
    sizeInnerUnitType: str = request.args.get('sizeInnerUnitType', default='g')
    productBarCode: str = request.args.get('productBarCode', default='')
    supplier: str = request.args.get('supplier', default='')
    assert supplier, "supplier argument must be required and non-empty"
    brandName: str = request.args.get('brandName', default='')
    assert brandName, "brandName argument must be required and non-empty"
    origin: str = request.args.get('origin', default='')
    taxGroup: str = request.args.get('taxGroup', default='')
    dateOfBirth: datetime = datetime.strptime(request.args.get('dateOfBirth', default=datetime.now().strftime('%Y-%M-%d')), '%Y-%M-%d')
    finalLocation: str = request.args.get('finalLocation', '')
    finalDate: datetime = datetime.strptime(request.args.get('finalDate', default=datetime.now().strftime('%Y-%M-%d')), '%Y-%M-%d')
    vegiEscConn = Vegi_ESC_Repo(app=server)
    newProduct = vegiEscConn.add_product_if_not_exists(
        name=name,
        product_external_id_on_source=product_external_id_on_source,    
        source=source,    
        description=description,    
        category=category,    
        keyWords=keyWords,    
        imageUrl=imageUrl,    
        ingredients=ingredients,    
        packagingType=packagingType,    
        stockUnitsPerProduct=stockUnitsPerProduct,    
        sizeInnerUnitValue=sizeInnerUnitValue,    
        sizeInnerUnitType=sizeInnerUnitType,    
        productBarCode=productBarCode,    
        supplier=supplier,    
        brandName=brandName,    
        origin=origin,    
        taxGroup=taxGroup,    
        dateOfBirth=dateOfBirth,
        finalDate=finalDate,
        finalLocation=finalLocation,
    )
    if newProduct:
        return newProduct.serialize()
    else:
        return ({
            'message': 'Error creating new product',
            'code': 'Product_not_created',
        })
        

@server.route("/explanations/add")
def add_explanation():
    title: str = request.args.get('title', '')
    assert title, "title argument must be required and non-empty"
    product: int = int(request.args.get('product', '0'))
    assert product, "product argument must be required and non-empty"
    source: int = int(request.args.get('source', '0'))
    assert source, "source argument must be required and non-empty"
    measure: float = float(request.args.get('measure', '0.0'))
    assert measure, "measure argument must be required and non-empty"
    evidence: str = request.args.get('evidence', '')
    assert evidence, "evidence must be required and non-empty"
    reasons: list[str] = json.loads(s=request.args.get('reasons', default='[]'))
    vegiEscConn = Vegi_ESC_Repo(app=server)
    newProduct = vegiEscConn.add_explanation_for_product(
        product=product,
        source=source,
        title=title,
        measure=measure,
        evidence=evidence,
        reasons=reasons,
    )
    if newProduct:
        return newProduct.serialize()
    else:
        return ({
            'message': 'Error creating new explanation for product',
            'code': 'Product_explanation_not_created',
        })
        

# * ESC Source Endpoints
@server.route("/sustained/refresh")
def sustained_refresh():
    ss = SustainedAPI(app=server)
    ss.refresh_products_lists()
    return "Success"
    

# * Deprecated endpoints
# @slow_call_timer
# @server.route("/n_similarity")
# def n_similarity():
#     args = request.args
#     a = str(args["ws1"]).lower()
#     b = str(args["ws2"]).lower()
#     result = model.n_similarity(
#         filter_words(a),
#         filter_words(b),
#     ).item()
#     logger.info(result)
#     return f"Success: {result}"


# @slow_call_timer
# @server.route("/similarity")
# def similarity():
#     # if norm == "disable":
#     #     return ("most_similar disabled", 400)
#     args = request.args
#     a = str(args["w1"]).lower()
#     b = str(args["w2"]).lower()
#     if a not in model.key_to_index.keys():
#         a_sims = _lookup_most_similar_words_in_language_model(a)
#         if a_sims:
#             a = a_sims[0]
#         # else we leave a as a in the knowledge that it will throw a KeyError below...
#     if b not in model.key_to_index.keys():
#         b_sims = _lookup_most_similar_words_in_language_model(b)
#         if b_sims:
#             a = b_sims[0]
#         # else we leave a as a in the knowledge that it will throw a KeyError below...
#     result = model.similarity(a, b).item()
#     logger.info(result)
#     return f"Success: {result}"


# def _lookup_most_similar_words_in_language_model(
#     a: str, tolerance: float = 0.75
# ) -> list[str]:
#     if a in model.key_to_index.keys():
#         return [a]
#     similar_vocab = [
#         word
#         for word in model.key_to_index.keys()
#         if simple_word_similarity(a, word) >= tolerance
#     ]
#     return similar_vocab


# @slow_call_timer
# @server.route("/nearest-word-in-model")
# def lookup_most_similar_word_in_language_model():
#     args = request.args
#     a = str(args["w1"]).lower()
#     result = _lookup_most_similar_words_in_language_model(a)
#     return f"Success: {result}"


# @slow_call_timer
# @server.route("/sentence_similarity")
# def sentence_similarity():
#     # if norm == "disable":
#     #     return ("most_similar disabled", 400)
#     args = request.args
#     result = model.wmdistance(
#         args["s1"], args["s2"]
#     )  # see wmd.py for use (compare sentence1 and sentence2)
#     logger.info(result)
#     return f"Success: {result}"


# def _sustained_most_similar_category_id_spaced_map(sentence1: str):
#     """returns a map from `category_id` (using spaces in key name not underscores) to the wmdistance between the `sentence1` to that `category_id`"""
#     ss = SustainedAPI(app=server)
#     sustained_categories = ss.get_category_ids(replace=("-", " "))
#     similarities: dict[str, float] = dict()
#     for category_id_spaced in sustained_categories:
#         similarities[category_id_spaced] = model.wmdistance(
#             category_id_spaced, sentence1
#         )
#         logger.info(f"'{category_id_spaced}': {similarities[category_id_spaced]}")
#     logger.info(similarities)
#     # most_sim_v = min([v for v in similarities.values()])
#     # most_similar = next((k for k, v in similarities.items() if v == most_sim_v))
#     return similarities


# def _sustained_most_similar_category_id_spaced(sentence1: str):
#     similarities = _sustained_most_similar_category_id_spaced_map(sentence1=sentence1)
#     most_sim_v = min([v for v in similarities.values()])
#     most_similar = next((k for k, v in similarities.items() if v == most_sim_v))
#     return f"{most_similar}"


# @server.route("/sustained/most-similar-category")
# def sustained_most_similar_category():
#     args = request.args
#     logger.info(args)
#     most_sim_cat_id = _sustained_most_similar_category_id_spaced(args["s1"])
#     ss = SustainedAPI(app=server)
#     cat = ss.get_cat_for_space_delimited_id(most_sim_cat_id, replace=("-", " "))
#     if cat:
#         return cat.name
#     raise Exception("Category not found")


def _sustained_product_to_vegi_esc_rating(sProd: ESCProductInstance):
    ss = SustainedAPI(app=server)
    return ss.get_product_with_impact(
        sustainedProductId=sProd.product_external_id_on_source
    )


# def _sustained_most_similar_product(sentence1: str):
#     most_sim_cat_id = _sustained_most_similar_category_id_spaced(sentence1)
#     most_similar_ss_product_result, min_v = (
#         _sustained_most_similar_product_for_ss_cat_space_delimited_id(
#             search_product_name=sentence1,
#             most_sim_cat_id=most_sim_cat_id,
#         )
#     )
#     vegi_rated_most_sim_product = _convert_sustained_product_to_vegi_rating(
#         esc_product=most_similar_ss_product_result,
#         min_v=min_v,
#         original_search_term=sentence1,
#     )
#     return vegi_rated_most_sim_product


# def _sustained_most_similar_product_for_ss_cat_space_delimited_id(
#     search_product_name: str, most_sim_cat_id: str
# ):
#     ss = SustainedAPI(app=server)
#     cat = ss.get_cat_for_space_delimited_id(most_sim_cat_id, replace=("-", " "))
#     if not cat:
#         raise Exception("Category not found")
#     sustained_products = ss.get_products(category_name=cat.name)
#     similarities: dict[str, float] = dict()
#     for product in sustained_products:
#         product_name = product.name
#         similarities[product_name] = model.wmdistance(product_name, search_product_name)
#         logger.verbose(f"'{product_name}': {similarities[product_name]}")
#     min_v: float = min([v for v in similarities.values()])
#     most_similar = next((k for k, v in similarities.items() if v == min_v))
#     products_with_matching_name = [
#         p for p in sustained_products if p.name.lower() == most_similar.lower()
#     ]
#     assert (
#         products_with_matching_name
#     ), "Products_with_matching_name somehow lost product name - Should never happen"
#     most_similar_product = products_with_matching_name[0]
#     return most_similar_product, min_v

# def _find_similar_rated_product(
#     product_name: str,
#     product_names_in_same_category: list[str],
#     product_category_name: str,
# ):
#     if product_name not in product_names_in_same_category:
#         product_names_in_same_category.append(product_name)

#     n = float(len(product_names_in_same_category))

#     # * find most similar product in sustained
#     # store the distance from the category matched in teh below function too
#     most_sim_cats_to_category_map = _sustained_most_similar_category_id_spaced_map(
#         product_category_name
#     )
#     # cat = ss.get_cat_for_space_delimited_id(most_sim_cat_id, replace=("-", " "))
#     category_weight = 0.4
#     # product_weight = 1 - category_weight
#     # _average_wmds_for_sustained_cats = {
#     #     sustained_cat: category_weight * cat_cat_wmd
#     #     for sustained_cat, cat_cat_wmd in most_sim_cats_to_category_map.items()
#     # }
#     _averages: dict[str, dict[str, float]] = {}
#     for i, _product_name in enumerate(product_names_in_same_category):
#         # progress_percent = i / n
#         _averages[_product_name] = _sustained_most_similar_category_id_spaced_map(
#             _product_name
#         )

#     _sscat_id_name_to_prod_avg_wmd: defaultdict[str, float] = defaultdict(float)
#     for _product_name in _averages.keys():
#         for ss_cat_id_sp in _averages[_product_name]:
#             if ss_cat_id_sp not in _sscat_id_name_to_prod_avg_wmd.keys():
#                 _sscat_id_name_to_prod_avg_wmd[ss_cat_id_sp] = 0.0
#             _sscat_id_name_to_prod_avg_wmd[ss_cat_id_sp] += (
#                 _averages[_product_name][ss_cat_id_sp] / n
#             )
#     _sscat_id_name_to_avg_wmd: dict[str, float] = {}
#     for sscat_name in _sscat_id_name_to_prod_avg_wmd.keys():
#         _sscat_id_name_to_avg_wmd[sscat_name] = (
#             _sscat_id_name_to_prod_avg_wmd[sscat_name] * (1 - category_weight)
#             + most_sim_cats_to_category_map[sscat_name] * category_weight
#         )
#     most_sim_v = min([v for v in _sscat_id_name_to_avg_wmd.values()])
#     most_similar_ss_cat_id_spaced = next(
#         (k for k, v in _sscat_id_name_to_avg_wmd.items() if v == most_sim_v)
#     )
#     # * Then match most similar product in that category same as already done and use its rating for now
#     most_similar_ss_product_result, min_v = (
#         _sustained_most_similar_product_for_ss_cat_space_delimited_id(
#             search_product_name=product_name,
#             most_sim_cat_id=most_similar_ss_cat_id_spaced,
#         )
#     )
#     vegi_rated_most_sim_product = _convert_sustained_product_to_vegi_rating(
#         esc_product=most_similar_ss_product_result,
#         min_v=min_v,
#         original_search_term=product_name,
#     )
#     return vegi_rated_most_sim_product


def _convert_sustained_product_to_vegi_rating(esc_product: ESCProductInstance, min_v: float, original_search_term: str):
    ''' use the ESCProductInstance to create a new vegi ESC rating'''
    vegiRating = _sustained_product_to_vegi_esc_rating(sProd=esc_product)
    # adjust rating for wmdistance away if < 0.5 keep same as v similar product if > 1.2, v different products and reduce rating by 50%
    vegiRating.rating.rating = adjust_rating_from_similar_product(
        min_v=min_v,
        similar_product_rating=vegiRating.rating.rating,
    )
    
    return ESCRatingExplainedResult(
        rating=vegiRating.rating,
        explanations=vegiRating.explanations,
        original_search_term=original_search_term,
        wmdistance=min_v,
        _sustainedProduct=esc_product,
    )
    

def adjust_rating_from_similar_product(min_v: float, similar_product_rating: float):
    '''
    adjust rating for wmdistance away if < 0.5 keep same as v similar product if > 1.2, v different products and reduce rating by 50%
    '''
    if min_v <= 0.6:
        return similar_product_rating  # no adjustment needed as fairly close match
    elif min_v >= 1.2:
        return similar_product_rating * 0.5
    else:
        # min_v = 0.6 -> 0.5 -> 1
        # min_v = 1.2 -> 0 -> 0.5
        return similar_product_rating * (((1.2 - min_v) / (1.2 - 0.6)) * 0.5) + 0.5


# @slow_call_timer
# @server.route("/sustained/most-similar-product")  # type: ignore
# def sustained_most_similar():
#     args = request.args
#     logger.info(args)
#     most_similar_product_name = _sustained_most_similar_product(args["s1"])
#     return most_similar_product_name.toJson()


# @slow_call_timer
# @server.route("/most_similar")
# def most_similar():
#     args = request.args
#     pos = filter_words(args.get("positive", []))
#     neg = filter_words(args.get("negative", []))
#     t = int(args.get("topn", 10))
#     pos = [] if pos is None else pos
#     neg = [] if neg is None else neg
#     t = 10 if t is None else t
#     logger.verbose(
#         "positive: " + str(pos) + ", negative: " + str(neg) + ", topn: " + str(t)
#     )
#     try:
        
#         res = model.most_similar_cosmul(positive=pos, negative=neg, topn=t)
#         logger.info(res)
#         return f"Success: {res}"
#     except Exception as e:
#         logger.error(e)
#         return "An error occured!"


# @server.route("/model")
# def modelCall():
#     args = request.args
#     try:
#         res = model[args["word"]]
#         res = base64.b64encode(res).decode()
#         logger.info(res)
#         return f"Success: {res}"
#     except Exception as e:
#         logger.error(e)
#         return "An error occured!"


# @server.route("/model_word_set")
# def model_word_set():
#     # args = request.args
#     try:
#         res = base64.b64encode(pickle.dumps(set(model.index2word))).decode()
#         logger.info(res)
#         return f"Success: {res}"
#     except Exception as e:
#         logger.error(e)
#         return "An error occured!"


@server.errorhandler(404)
def pageNotFound(error: Any):
    return "page not found"


@server.errorhandler(500)
def raiseError(error: Any):
    return error


if __name__ == "__main__":
    app, host, port = init_flask_config(app=server)
    # model: I_Am_LLM = LLM.getModel(app=server).model
    app.run(host=host, port=port)
    # runApp = Word_Vec_Model.initLocalFlaskAppWithArgs(app=server)
    # runApp()
else:
    Logger.info(
        f'Thread name running app is "{__name__}"'
    )  # Thread name running app is "app" if command run is `gunicorn --bind 127.0.0.1:5002 app:gunicorn_app --timeout 90`
    # runApp = initFlaskAppModel(app=server)
    
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
        Logger.info(f'Running gunicorn app with args: [{sys.argv}]')
        _server, host, port = init_flask_config(app=server)
        # model: I_Am_LLM = LLM.getModel(app=server).model
        return server
    gunicorn_app = _app()  # gunicorn --bind 127.0.0.1:5002 app:gunicorn_app(**kwargs) --timeout 90 --log-level=debug
