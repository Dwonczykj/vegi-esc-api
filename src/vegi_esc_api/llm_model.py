from __future__ import annotations
import asyncio
from typing import Callable, Optional, Self, Type, Any, TypeVar
from typing import Tuple, Final, Generic, TypedDict
import threading
from dataclasses import dataclass

# import json
import os
import re
import argparse
from nltk.corpus import stopwords
import nltk
from flask import Flask
from unidecode import unidecode
from ftfy import fix_text
from pprint import pformat
import scipy.sparse

# from scipy.stats import entropy
# from scipy.linalg import get_blas_funcs, triu

# from scipy.linalg.lapack import get_lapack_funcs
# from scipy.special import psi  # gamma function utils
import math
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.api.fastapi import FastAPI
from chromadb.api.local import LocalAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    # EmbeddingFunction,
    Metadata,
    Embedding,
    ID,
    Where,
    WhereDocument,
    OneOrMany,
    Include,
    GetResult,
    QueryResult,
    Document,
)

# from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, CollectionMetadata
# from chromadb.errors import ChromaError, error_types


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

from vegi_esc_api.constants import AppConstants
from vegi_esc_api.vegi_esc_repo import Vegi_ESC_Repo
# from vegi_esc_api.vegi_repo import VegiRepo
from vegi_esc_api.vegi_api import VegiApi
import vegi_esc_api.logger as Logger


V = TypeVar("V", str, int)


class QuerySingleResult(TypedDict):
    id: ID
    embedding: Optional[Embedding]
    document: Optional[Document]
    metadata: Optional[Metadata]
    distance: Optional[float]


@dataclass
class CleanProductNameCategoryTuple(Generic[V]):
    name: str
    category: str
    product_id: V
    original_name: str
    original_category_name: str
    source_id: int
    source_name: str


@dataclass
class VegiProductsFromJson:
    product_names: list[CleanProductNameCategoryTuple]
    product_categories: list[CleanProductNameCategoryTuple]
    # product_categories: list[str]
    source: Final[str] = "vegi"

    def limit(self, n: int):
        return type(self)(
            source=self.source,
            product_categories=self.product_categories[:n],
            product_names=self.product_names[:n],
        )


@dataclass
class ExternalProductsFromJson:
    source: str
    source_id: int
    product_names: list[CleanProductNameCategoryTuple]
    product_categories: list[CleanProductNameCategoryTuple]

    def limit(self, n: int):
        return type(self)(
            source=self.source,
            source_id=self.source_id,
            product_categories=self.product_categories[:n],
            product_names=self.product_names[:n],
        )


T = TypeVar("T")


def id_from_name(name: str):
    return re.sub(pattern=r"[^0-9A-Za-z]", repl="", string=name.replace(" ", "_"))


def _listify(vals: list[T] | T) -> list[T]:
    _vals = vals if isinstance(vals, list) else [vals]
    return _vals


def split_array(array: list[T], batch_size: int) -> list[list[T]]:
    split_list = [array[i : i + batch_size] for i in range(0, len(array), batch_size)]
    return split_list


cachedStopWords: list[str] = []


try:
    cachedStopWords = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")
    cachedStopWords = stopwords.words("english")


def clean_words(words: str):
    words = unidecode(words)
    words = fix_text(words)
    words = " ".join([word for word in words.split() if word not in cachedStopWords])
    words = re.sub(pattern=r"[^0-9A-Za-z\s]", string=words, repl="")
    return words


def scatter_text(
    x: str,
    y: str,
    hue: str,
    palette,
    data: pd.DataFrame,
    legend: str,
    alpha: float,
    labels_column: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    origin_index: int = -1,
):
    if not xlabel:
        xlabel = x
    if not ylabel:
        ylabel = y
    if not title:
        title = f"{xlabel} vs {ylabel}"

    """Scatter plot with country codes on the x y coordinates
        Based on this answer: https://stackoverflow.com/a/54789170/2641825"""
    # Create the scatter plot
    p1 = sns.scatterplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette=palette,
        alpha=alpha,
        legend=legend,
        size=12,
    )

    # if origin_index > -1:
    #     origin = {'x':data[x][origin_index], 'y': data[y][origin_index]}
    #     # Plot the origin point with a larger marker and different color
    #     p1 = sns.scatterplot(x=origin['x'], y=origin['y'], s=100, color='red')

    #     # Label the origin point
    #     p1.text(origin['x'], origin['y'], 'Origin', ha='center', va='bottom', fontsize=12, color='red')

    # Add text besides each point
    for line in range(0, data.shape[0]):
        if line == origin_index:
            p1.text(
                data[x][line] + 0.01,
                data[y][line],
                data[labels_column][line],
                horizontalalignment="left",
                size="large",
                color="green",
                weight="bold",
            )
        else:
            p1.text(
                data[x][line] + 0.01,
                data[y][line],
                data[labels_column][line],
                horizontalalignment="left",
                size="medium",
                color="black",
                weight="semibold",
            )
    # def label_point(x, y, val, ax):
    #     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    #     for i, point in a.iterrows():
    #         ax.text(point['x']+.02, point['y'], str(point['val']))

    # label_point(data[x], data[y], data[labels_column], plt.gca())
    # Set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1


_Chroma_ESC_product_collection_name = "esc_product_vectors"


class LLM:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if not cls._instance:
                # ! cls._instance = super(LLM, cls).__new__(cls, *args, **kwargs)
                cls._instance = super(LLM, cls).__new__(cls)
        return cls._instance

    __chromadb_client: FastAPI | LocalAPI

    __instance: LLM

    def __init__(
        self,
        app: Flask,
    ) -> None:
        '''
        Only init from within flask route context
        '''
        Logger.log('INIT LLM Model - Should only be called once.')
        self._app = app
        self.chromadb_reinit()
        # if hasattr(type(self), "_LLM__instance") is False or not type(self).__instance:
        # else:
        #     raise Exception(f"Cant load duplicate instance of {type(self).__name__}")

    @property
    def chroma_client(self):
        if (
            hasattr(type(self), "_LLM__chromadb_client") is False
            or not LLM.__chromadb_client
        ):
            # ~ https://docs.trychroma.com/usage-guide#initiating-a-persistent-chroma-client
            # ! Having many in-memory clients that are loading and saving to the same path can cause strange behavior including data deletion.
            # ! As a general practice, create an in-memory Chroma client once in your application,
            # ! and pass it around instead of creating many clients.
            type(self).__chromadb_client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=os.path.join(
                        os.getcwd(), ".chromadb"
                    ),  # Optional, defaults to .chromadb/ in the current directory
                )
            )
            return type(self).__chromadb_client
        return type(self).__chromadb_client

    @property
    def chroma_esc_product_collection_name(self):
        return self._chroma_esc_product_collection_name

    @property
    def server(self):
        return self._app

    @property
    def vegi_products(self):
        return self._vegi_products

    @property
    def model(self):
        return self

    @property
    def vocab(self) -> list[str]:
        # raise AttributeError(
        #     "The vocab attribute was removed from KeyedVector in Gensim 4.0.0.\n"
        #     "Use KeyedVector's .key_to_index dict, .index_to_key list, and methods "
        #     ".get_vecattr(key, attr) and .set_vecattr(key, attr, new_val) instead.\n"
        #     "See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4"
        # )
        result = self.chroma_query_vector_store(["documents"], 10, None)
        return self._result_to_docs(result)

    def most_similar(self, words: list[str] | str, top_n: int = 1):
        _words = _listify(words)
        result = self.chroma_query_vector_store(["documents"], top_n, None, *_words)
        return self._result_to_docs(result)

    @property
    def esc_collection(self):
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is None:
            if not self._chroma_collection:
                self._run_downloads()
            collection = self._chroma_collection
        assert (
            collection is not None
        ), f"Collection with name {self.chroma_esc_product_collection_name} should not be empty in vector store"
        return collection

    @property
    def key_to_index(self) -> dict[str, str]:
        result = self.esc_collection.get(include=["documents"])
        assert (
            result["documents"] is not None
        ), "GetResult documents cannot be none in key_to_index property of LLM"
        return {k: v for k, v in zip(result["ids"], result["documents"])}

    @property
    def index_to_key(self) -> list[str]:
        result = self.esc_collection.get(include=["documents"])
        return result["ids"]

    def __getitem__(self, document_id: str) -> GetResult | None:
        """
        gets the document from the vector store with matching id.
        """
        return self.chroma_get_result_for_id(id=document_id)

    def get_mean_vector(self, words: list[str], pre_normalize: bool = False):
        pass

    def make_unit_vector(self, scalar: float | int, dims: int):
        pass

    @property
    def vectors(self):
        pass

    @property
    def norms(self):
        pass

    def get_vector(self, words: list[str], norm: bool = True):
        pass

    # def n_similarity(self, ws1: list[str], ws2: list[str]):
    #     """Compute cosine similarity between two sets of keys.

    #     Parameters
    #     ----------
    #     ws1 : list of str
    #         Sequence of keys.
    #     ws2: list of str
    #         Sequence of keys.

    #     Returns
    #     -------
    #     numpy.ndarray
    #         Similarities between `ws1` and `ws2`.

    #     """
    #     if not (len(ws1) and len(ws2)):
    #         raise ZeroDivisionError('At least one of the passed list is empty.')
    #     mean1 = self.get_mean_vector(ws1, pre_normalize=False)
    #     mean2 = self.get_mean_vector(ws2, pre_normalize=False)
    #     return np.dot(unitvec(mean1), unitvec(mean2))

    # def most_similar_cosmul(self, positive: list[str], negative: list[str], topn: int, restrict_vocab: int | None = None):
    #     '''
    #     See:
    #     - https://stackoverflow.com/a/31720951
    #     - https://stackoverflow.com/questions/47027711/finding-closest-related-words-using-word2vec

    #     Find the top-N most similar words, using the multiplicative combination objective,
    #     proposed by `Omer Levy and Yoav Goldberg "Linguistic Regularities in Sparse and Explicit Word Representations"
    #     <http://www.aclweb.org/anthology/W14-1618>`_. Positive words still contribute positively towards the similarity,
    #     negative words negatively, but with less susceptibility to one large distance dominating the calculation.
    #     In the common analogy-solving case, of two positive and one negative examples,
    #     this method is equivalent to the "3CosMul" objective (equation (4)) of Levy and Goldberg.

    #     Additional positive or negative examples contribute to the numerator or denominator,
    #     respectively - a potentially sensible but untested extension of the method.
    #     With a single positive example, rankings will be the same as in the default
    #     :meth:`~gensim.models.keyedvectors.KeyedVectors.most_similar`.

    #     Allows calls like most_similar_cosmul('dog', 'cat'), as a shorthand for
    #     most_similar_cosmul(['dog'], ['cat']) where 'dog' is positive and 'cat' negative

    #     Parameters
    #     ----------
    #     positive : list of str, optional
    #         List of words that contribute positively.
    #     negative : list of str, optional
    #         List of words that contribute negatively.
    #     topn : int or None, optional
    #         Number of top-N similar words to return, when `topn` is int. When `topn` is None,
    #         then similarities for all words are returned.
    #     restrict_vocab : int or None, optional
    #         Optional integer which limits the range of vectors which are searched for most-similar values.
    #         For example, restrict_vocab=10000 would only check the first 10000 node vectors in the vocabulary order.
    #         This may be meaningful if vocabulary is sorted by descending frequency.

    #     Returns
    #     -------
    #     list of (str, float) or numpy.array
    #         When `topn` is int, a sequence of (word, similarity) is returned.
    #         When `topn` is None, then similarities for all words are returned as a
    #         one-dimensional numpy array with the size of the vocabulary.

    #     '''
    #     if isinstance(topn, Integral) and topn < 1:
    #         return []

    #     # allow passing a single string-key or vector for the positive/negative arguments
    #     positive = _ensure_list(positive)
    #     negative = _ensure_list(negative)

    #     self.init_sims()

    #     if isinstance(positive, str):
    #         # allow calls like most_similar_cosmul('dog'), as a shorthand for most_similar_cosmul(['dog'])
    #         positive = [positive]

    #     if isinstance(negative, str):
    #         negative = [negative]

    #     all_words = {
    #         self.get_index(word) for word in positive + negative
    #         if not isinstance(word, np.ndarray) and word in self.key_to_index
    #     }

    #     positive = [
    #         self.get_vector(word, norm=True) if isinstance(word, str) else word
    #         for word in positive
    #     ]
    #     negative = [
    #         self.get_vector(word, norm=True) if isinstance(word, str) else word
    #         for word in negative
    #     ]

    #     if not positive:
    #         raise ValueError("cannot compute similarity with no input")

    #     # equation (4) of Levy & Goldberg "Linguistic Regularities...",
    #     # with distances shifted to [0,1] per footnote (7)
    #     pos_dists = [((1 + np.dot(self.vectors, term) / self.norms) / 2) for term in positive]
    #     neg_dists = [((1 + np.dot(self.vectors, term) / self.norms) / 2) for term in negative]
    #     dists = np.prod(pos_dists, axis=0) / (np.prod(neg_dists, axis=0) + 0.000001)

    #     if not topn:
    #         return dists
    #     best = argsort(dists, topn=topn + len(all_words), reverse=True)
    #     # ignore (don't return) words from the input
    #     result = [(self.index_to_key[sim], float(dists[sim])) for sim in best if sim not in all_words]
    #     return result[:topn]

    # def wmdistance(self, document1: list[str], document2: list[str], norm: bool = True):
    #     """Compute the Word Mover's Distance between two documents.

    #     When using this code, please consider citing the following papers:

    #     * `Rémi Flamary et al. "POT: Python Optimal Transport"
    #       <https://jmlr.org/papers/v22/20-451.html>`_
    #     * `Matt Kusner et al. "From Word Embeddings To Document Distances"
    #       <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.

    #     Parameters
    #     ----------
    #     document1 : list of str
    #         Input document.
    #     document2 : list of str
    #         Input document.
    #     norm : boolean
    #         Normalize all word vectors to unit length before computing the distance?
    #         Defaults to True.

    #     Returns
    #     -------
    #     float
    #         Word Mover's distance between `document1` and `document2`.

    #     Warnings
    #     --------
    #     This method only works if `POT <https://pypi.org/project/POT/>`_ is installed.

    #     If one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity)
    #     will be returned.

    #     Raises
    #     ------
    #     ImportError
    #         If `POT <https://pypi.org/project/POT/>`_  isn't installed.

    #     """
    #     # If POT is attempted to be used, but isn't installed, ImportError will be raised in wmdistance
    #     from ot import emd2

    #     # Remove out-of-vocabulary words.
    #     len_pre_oov1 = len(document1)
    #     len_pre_oov2 = len(document2)
    #     document1 = [token for token in document1 if token in self.vocab]
    #     document2 = [token for token in document2 if token in self.vocab]
    #     diff1 = len_pre_oov1 - len(document1)
    #     diff2 = len_pre_oov2 - len(document2)
    #     if diff1 > 0 or diff2 > 0:
    #         Logger.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', diff1, diff2)

    #     if not document1 or not document2:
    #         Logger.warn("At least one of the documents had no words that were in the vocabulary.")
    #         return float('inf')

    #     dictionary = Dictionary(documents=[document1, document2])
    #     vocab_len = len(dictionary)

    #     if vocab_len == 1:
    #         # Both documents are composed of a single unique token => zero distance.
    #         return 0.0

    #     doclist1 = list(set(document1))
    #     doclist2 = list(set(document2))
    #     v1 = np.array([self.get_vector(token, norm=norm) for token in doclist1])
    #     v2 = np.array([self.get_vector(token, norm=norm) for token in doclist2])
    #     doc1_indices = dictionary.doc2idx(doclist1)
    #     doc2_indices = dictionary.doc2idx(doclist2)

    #     # Compute distance matrix.
    #     distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.double)
    #     distance_matrix[np.ix_(doc1_indices, doc2_indices)] = cdist(v1, v2)

    #     if abs(np_sum(distance_matrix)) < 1e-8:
    #         # `emd` gets stuck if the distance matrix contains only zeros.
    #         Logger.info('The distance matrix is all zeros. Aborting (returning inf).')
    #         return float('inf')

    #     def nbow(document):
    #         d = np.zeros(vocab_len, dtype=np.double)
    #         nbow = dictionary.doc2bow(document)  # Word frequencies.
    #         doc_len = len(document)
    #         for idx, freq in nbow:
    #             d[idx] = freq / float(doc_len)  # Normalized word frequencies.
    #         return d

    #     # Compute nBOW representation of documents. This is what POT expects on input.
    #     d1 = nbow(document1)
    #     d2 = nbow(document2)

    #     # Compute WMD.
    #     return emd2(d1, d2, distance_matrix)

    def _result_to_docs(self, result: GetResult | QueryResult | None):
        assert (
            result is not None
        ), "chromadb QueryResult in LLM().chroma_query_vector_store didnt obtain a query result"
        documents = result["documents"]
        assert (
            documents is not None
        ), "chromadb QueryResult in LLM().chroma_query_vector_store didnt contain documents"
        if not documents:
            return []
        if isinstance(documents[0], list) and isinstance(documents[0][0], Document):
            return [str(d) for dl in documents for d in dl]
        elif isinstance(documents[0], Document):
            return [str(d) for d in documents]
        return [str(d) for dl in documents for d in dl]

    def similarity(
        self, words: list[str] | str, measure_similarity_to_word: str
    ) -> dict[str, float]:
        """
        inverse of distance (defaults to cosine)
        @returns `dict[str, float]` where the keys are the words that we are comparing similarity of `measure_similarity_to_word` with and the values are the similarity measure
        """
        words = _listify(words)
        result = self.chroma_query_vector_store(
            ["documents", "distances"], 1, None, measure_similarity_to_word, *words
        )
        assert (
            result is not None
        ), "chromadb QueryResult in LLM().chroma_query_vector_store didnt obtain a query result"
        documents = result["documents"]
        assert (
            documents is not None
        ), "chromadb QueryResult in LLM().chroma_query_vector_store didnt contain documents"
        if not documents:
            return {}
        assert (
            "distances" in result
        ), "distances key should be returned from LLM().chroma_query_vector_store but wasnt"
        distances = result["distances"]
        assert (
            distances is not None
        ), "chromadb QueryResult in LLM().chroma_query_vector_store didnt contain distances"
        if not distances:
            return {}
        if isinstance(documents[0], list) and isinstance(documents[0][0], Document):
            return {
                d: 1.0 / w
                for (dl, wl) in list(zip(documents, distances))
                for (d, w) in list(zip(dl, wl))
            }
        elif isinstance(documents[0], Document):
            return {
                str(d): float(np.mean([1.0 / w for w in wl]))
                for (d, wl) in list(zip(documents, distances))
            }
        return {
            d: 1.0 / w
            for (dl, wl) in list(zip(documents, distances))
            for (d, w) in list(zip(dl, wl))
        }

    def distance(
        self, words: list[str] | str, measure_similarity_to_word: str
    ) -> dict[str, float]:
        """
        inverse of similarity (defaults to cosine)
        @returns `dict[str, float]` where the keys are the words that we are comparing similarity of `measure_similarity_to_word` with and the values are the distance measure
        """
        words = _listify(words)
        result = self.chroma_query_vector_store(
            ["documents", "distances"], 1, None, measure_similarity_to_word, *words
        )
        assert (
            result is not None
        ), "chromadb QueryResult in LLM().chroma_query_vector_store didnt obtain a query result"
        documents = result["documents"]
        assert (
            documents is not None
        ), "chromadb QueryResult in LLM().chroma_query_vector_store didnt contain documents"
        if not documents:
            return {}
        assert (
            "distances" in result
        ), "distances key should be returned from LLM().chroma_query_vector_store but wasnt"
        distances = result["distances"]
        assert (
            distances is not None
        ), "chromadb QueryResult in LLM().chroma_query_vector_store didnt contain distances"
        if not distances:
            return {}
        if isinstance(documents[0], list) and isinstance(documents[0][0], Document):
            return {
                d: w
                for (dl, wl) in list(zip(documents, distances))
                for (d, w) in list(zip(dl, wl))
            }
        elif isinstance(documents[0], Document):
            return {
                str(d): float(np.mean([w for w in wl]))
                for (d, wl) in list(zip(documents, distances))
            }
        return {
            d: w
            for (dl, wl) in list(zip(documents, distances))
            for (d, w) in list(zip(dl, wl))
        }

    @property
    def sustained_products(self):
        return self._sustained_products

    def _init_products_cache(self):
        nltk.download("wordnet")  # approx 30 seconds
        sustained_source = self._repoConn.get_source(
            source_name=AppConstants.SUSTAINED_DOMAIN_NAME
        )
        sustained_products = ExternalProductsFromJson(
            source="Unknown",
            source_id=-1,
            product_names=[],
            product_categories=[],
        )
        vegi_products = VegiProductsFromJson(
            source="Unknown",
            product_names=[],
            product_categories=[],
        )
        if sustained_source:
            sustained_categories = self._repoConn.get_categories(
                item_source=AppConstants.SUSTAINED_DOMAIN_NAME
            )
            sustained_categories_names = [
                CleanProductNameCategoryTuple(
                    source_id=sustained_source.id,
                    source_name=sustained_source.name,
                    product_id=c.id,
                    name=clean_words(c.name),
                    category=f"{sustained_source.name}_category",
                    original_name=c.name,
                    original_category_name=f"{sustained_source.name}_category",
                )
                for c in sustained_categories
            ]
            sustained_product_names: list[CleanProductNameCategoryTuple] = []
            _sustained_products = self._repoConn.get_products(
                item_source=sustained_source.id
            )
            sustained_product_names = [
                CleanProductNameCategoryTuple(
                    source_id=sustained_source.id,
                    source_name=sustained_source.name,
                    product_id=p.id,
                    name=clean_words(p.name),
                    category=clean_words(p.category),
                    original_name=p.name,
                    original_category_name=p.category,
                )
                for p in _sustained_products
            ]
            sustained_products = ExternalProductsFromJson(
                source=sustained_source.name,
                source_id=sustained_source.id,
                product_names=sustained_product_names,
                product_categories=sustained_categories_names,
            )
            # purple_carrot_json = requests.get(f'http://qa-vegi.vegiapp.co.uk/api/v1/vendors/{vendorId}')
            vendor_outcode = 'L1'
            vendorId = 1
            _vegi_vendors = asyncio.run(self._vegiApiConn.get_vendors(outcode=vendor_outcode))
            if _vegi_vendors:
                vendorId = _vegi_vendors[0].id
            _vegi_products = asyncio.run(self._vegiApiConn.get_products(vendorId=vendorId))  # todo: need to not harcode this id.
            _vegi_categories = asyncio.run(self._vegiApiConn.get_product_categories(vendor=vendorId))
            vegi_products = VegiProductsFromJson(
                product_names=[
                    CleanProductNameCategoryTuple(
                        source_id=-1,
                        source_name="vegi",
                        product_id=p.id,
                        name=clean_words(p.name),
                        category=clean_words(p.category.name),
                        original_name=p.name,
                        original_category_name=p.category.name,
                    )
                    for p in _vegi_products
                ],
                product_categories=[
                    CleanProductNameCategoryTuple(
                        source_id=-1,
                        source_name="vegi",
                        product_id=k.id,
                        name=clean_words(k.name),
                        category="vegi_category",
                        original_category_name="vegi_category",
                        original_name=k.name,
                    )
                    for k in _vegi_categories
                    # clean_words(k.name)
                    # for k in _vegi_categories
                ],
            )
        self._sustained_products = sustained_products
        self._vegi_products = vegi_products
        return self

    def _run_downloads(self):
        self._chroma_collection = self._chroma_init_esc_products_collection(
            name=self.chroma_esc_product_collection_name,
            # embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            #     api_key=os.environ["OPEN_AI_SECRET"],
            #     model_name=self._text_embedding_openai_model,
            # ),
            vegi_products=self._vegi_products,
            external_products=self._sustained_products,
            # vegi_products=vegi_products.limit(n=10),
            # external_products=sustained_products.limit(n=10),
            get_or_create=True,  # gets collection if already exists.
            batch_sizes=250,
        )
        return self._chroma_collection.get()

    @classmethod
    def getModel(
        cls: Type[Self], app: Flask, args: argparse.Namespace | None = None
    ) -> Self:
        """
        Takes care of downloading, initialisation, singleton fetching and is a factory method that returns an object of this class
        """
        if hasattr(cls, "__instance") is False or not cls.__instance:
            return LLM(
                app=app,
            )
        return cls.__instance

    def most_similar_esc_products(self, *vegi_product_names: str):
        vegi_to_esc_product_map = self.chroma_most_similar_sustained_products(
            *vegi_product_names
        )
        if vegi_to_esc_product_map is None:
            return None
        return {
            vegi_product_name: (
                self._repoConn.get_esc_product_by_id(
                    id=int(query_result["metadata"]["product_esc_id"] if query_result["metadata"] else -1),
                    source=int(query_result["metadata"]["source_id"] if query_result["metadata"] else -999),
                ),
                query_result,
            )
            for (vegi_product_name, query_result) in vegi_to_esc_product_map.items()
            if int(query_result["metadata"]["source_id"] if query_result["metadata"] else 0) > 0
        }

    def _chroma_add_to_collection(
        self,
        products: list[CleanProductNameCategoryTuple],
        is_products: bool,
        source: str,
        collection: Collection,
        batch_sizes: int = -1,
    ):
        _unique_document_ids = list(set([id_from_name(p.name) for p in products]))
        result = collection.get(ids=_unique_document_ids)
        Logger.log(f'Read {len(result)} existing ids from chromadb.')
        _still_to_create_products = list(
            {
                id_from_name(p.name): (p, id_from_name(p.name))
                for p in products
                if id_from_name(p.name) not in result["ids"]
            }.values()
        )
        _to_modify_products = list(
            {
                id_from_name(p.name): (p, id_from_name(p.name))
                for p in products
                if id_from_name(p.name) in result["ids"]
            }.values()
        )
        
        if _to_modify_products:
            def metadata_from_product(p: CleanProductNameCategoryTuple):
                return {
                    "isProduct": True,
                    "category": p.original_category_name,
                    "product_name": p.original_name,
                    "product_esc_id": p.product_id,
                    "source_id": p.source_id,
                    # "source": source,
                    # "source_name": p.source_name,
                }
            _dummy_keys = list(metadata_from_product(products[0]).keys())
            metadata_out_of_date = any((k not in _dummy_keys for k in result["metadatas"][0].keys())) if result else False
            if metadata_out_of_date:
                print(
                    f"Can modify {len(_to_modify_products)}/{len(products)} of the {source} products"
                )
                try:
                    batches = [_to_modify_products]
                    if batch_sizes > -1 and batch_sizes < len(_to_modify_products):
                        batches = split_array(_to_modify_products, batch_sizes)
                    rows_calculated = 0
                    for i, b in enumerate(batches):
                        Logger.info(
                            f"Computing batch for modify {i} ({i*batch_sizes}/{len(_to_modify_products)})"
                        )
                        collection.update(
                            ids=[id for (p, id) in batches[i]],  # unique for each doc
                            # documents=[
                            #     p.name for (p, id) in batches[i]
                            # ],  # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
                            # ! NOTE: Max keys formetadata is 5
                            # ! NOTE: Metadata must have same strucutre as existing metadata, if there is a difference, updates NEED to be done first...
                            metadatas=[
                                {
                                    "isProduct": True,
                                    "category": p.original_category_name,
                                    "product_name": p.original_name,
                                    "product_esc_id": p.product_id,
                                    "source_id": p.source_id,
                                    # "source": source,
                                    # "source_name": p.source_name,
                                }
                                for (p, id) in batches[i]
                            ],  # filter on these!
                        )
                        rows_calculated += len(batches[i])
                        if rows_calculated >= 250:
                            Logger.log(f'Persisted the last {rows_calculated} rows to modify to chromadb')
                            self.chroma_client.persist()
                except Exception as e:
                    print(e)
                    raise e
        if _still_to_create_products:
            print(
                f"Adding {len(_still_to_create_products)}/{len(products)} of the {source} {'products' if is_products else 'categories'}"
            )
            try:
                batches = [_still_to_create_products]
                if batch_sizes > -1 and batch_sizes < len(_still_to_create_products):
                    batches = split_array(_still_to_create_products, batch_sizes)
                rows_calculated = 0
                for i, b in enumerate(batches):
                    Logger.info(
                        f"Computing batch to add {i} ({i*batch_sizes}/{len(_still_to_create_products)})"
                    )
                    collection.add(
                        ids=[id for (p, id) in batches[i]],  # unique for each doc
                        documents=[
                            p.name for (p, id) in batches[i]
                        ],  # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
                        # ! NOTE: Max keys formetadata is 5
                        # ! NOTE: Metadata must have same strucutre as existing metadata, if there is a difference, updates NEED to be done first...
                        metadatas=[
                            {
                                "isProduct": is_products,
                                "category": p.original_category_name,
                                "product_name": p.original_name,
                                "product_esc_id": p.product_id,
                                "source_id": p.source_id,
                                # "source": source,
                                # "source_name": p.source_name,
                            }
                            for (p, id) in batches[i]
                        ],  # filter on these!
                    )
                    rows_calculated += len(batches[i])
                    if rows_calculated >= 250 or rows_calculated == len(
                        _still_to_create_products
                    ):
                        Logger.log(f'Persisted the last {rows_calculated} rows to add to chromadb')
                        self.chroma_client.persist()
            except Exception as e:
                print(e)
                raise e

    def _chroma_init_esc_products_collection(
        self,
        name: str = _Chroma_ESC_product_collection_name,
        # embedding_function: Optional[Callable] = None,
        # NOTE l2 is the default, https://docs.trychroma.com/usage-guide#changing-the-distance-function
        # NOTE and https://github.com/nmslib/hnswlib/tree/master#supported-distances
        metadata: dict[Any, Any] | None = {"hnsw:space": "cosine"},
        get_or_create: bool = False,
        vegi_products: VegiProductsFromJson | None = None,
        external_products: ExternalProductsFromJson | None = None,
        batch_sizes: int = -1,
    ):
        """
        - `embedding_function` by default is the sentence_transformer, but can be set to the openAi transformer 'text-embedding-ada-002' for example
        - `metadata`: defines what distance function to use. see https://docs.trychroma.com/usage-guide#changing-the-distance-function
        - `get_or_create`:  If True, will return the collection if it already exists,
        """
        # Create collection. get_collection, get_or_create_collection, delete_collection also available!
        collection = self.chroma_client.create_collection(
            name=self.chroma_esc_product_collection_name,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.environ["OPEN_AI_SECRET"],
                model_name=self._text_embedding_openai_model,
            ),
            metadata=metadata,
            get_or_create=get_or_create,
        )
        self.chroma_client.persist()
        self.chroma_esc_collection = collection

        if vegi_products:
            self._chroma_add_to_collection(
                products=vegi_products.product_names,
                is_products=True,
                source=vegi_products.source,
                collection=collection,
                batch_sizes=batch_sizes,
            )

            self._chroma_add_to_collection(
                products=vegi_products.product_categories,
                is_products=False,
                source=vegi_products.source,
                collection=collection,
                batch_sizes=batch_sizes,
            )

        if external_products:
            self._chroma_add_to_collection(
                products=external_products.product_names,
                is_products=True,
                source=external_products.source,
                collection=collection,
                batch_sizes=batch_sizes,
            )

            self._chroma_add_to_collection(
                products=external_products.product_categories,
                is_products=False,
                source=external_products.source,
                collection=collection,
                batch_sizes=batch_sizes,
            )

        # ! In a normal python program, .persist() will happening automatically if you set it. But in a Jupyter Notebook you will need to manually call client.persist().

        self.chroma_client.persist()
        return collection

    def chroma_get_esc_collection(
        self,
        name: str = "",
        # embedding_used_on_create_collection: EmbeddingFunction | None = embedding_functions.DefaultEmbeddingFunction(),
    ):
        """
        WARNING:

        If you later wish to get_collection, you MUST do so with the embedding function you supplied while creating the collection
        """
        if not name:
            name = self.chroma_esc_product_collection_name
        try:
            return self.chroma_client.get_collection(
                name=name,
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ["OPEN_AI_SECRET"],
                    model_name=self._text_embedding_openai_model,
                ),
            )
        except Exception as e:
            Logger.error(
                f'Failed to get collection named "{name}", perhaps it had incorrect name or this collection was creating using a different embedding function? Error: {e}'
            )
            Logger.error(e)
            return None

    def chroma_remove_collection(self, name: str):
        return self.chroma_client.delete_collection(name=name)

    def chroma_get_esc_product_vectors(
        self,
        vegi_only: bool = False,
        collection: Collection | None = None,
        get_distances: bool = False,
        neighbourhood_of_product: Tuple[str, str] = ("", ""),
        n_results: int = 10,
    ):
        """
        Returns a Tuple[ GetResult.Embeddings, GetResult.Documents, GetResult ]

        A GetResult which is a dict with keys: ['ids', 'embeddings', 'documents', 'metadatas']
        """
        if collection is None:
            collection = self.chroma_get_esc_collection(
                name=self.chroma_esc_product_collection_name,
            )
        # ~ https://docs.trychroma.com/usage-guide#using-where-filters
        where_products: dict[str, Any] = {
            "isProduct": True,  # ~ https://docs.trychroma.com/usage-guide#querying-a-collection:~:text=Using%20the%20%24eq%20operator%20is%20equivalent%20to%20using%20the%20where%20filter.
        }
        where_categories: dict[str, Any] = {
            "isProduct": False,
        }
        if vegi_only:
            where_products["source"] = {"$eq": self._vegi_products.source}
            where_categories["source"] = {"$eq": self._vegi_products.source}

        query_texts = None
        # where_products_documents: dict[str, Any] = {}
        if neighbourhood_of_product[0]:
            if get_distances:
                query_texts = [neighbourhood_of_product[0]]
            else:
                # what we want is all products in same category, + other categories
                products_in_same_category = [
                    p.name
                    for p in self._vegi_products.product_names
                    if p.category == neighbourhood_of_product[1]
                ]
                other_category_names = [
                    *self._vegi_products.product_categories,
                    *self._sustained_products.product_categories,  # how to always init these?.
                ]
                # where_products_documents['$contains'] = neighbourhood_of_product
                query_texts = [*products_in_same_category, *other_category_names]
        else:
            get_distances = False

        if collection is not None:
            result: QueryResult | GetResult
            ind: int
            if get_distances:
                if query_texts:
                    print(
                        f"chroma db `collection.query` will return len(query_texts)={len(query_texts)} results with each result containing the nearest {n_results} results. Use get to return exact matches on an id"
                    )
                _include: Include = [
                    "embeddings",
                    "documents",
                    "metadatas",
                    "distances",
                ]
                result = collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=None if query_texts else where_products,
                    include=_include,
                )
                if query_texts is not None and len(query_texts) == 1:

                    def _flatten(a: list):
                        arr = np.array(a)
                        return arr.reshape(-1, arr.shape[-1]).reshape(-1, arr.shape[-1])
                        # return arr.ravel()

                    for k in ["ids", *_include]:
                        result[k] = np.array(result[k])

                    # for k in ["embeddings", "metadatas"]:
                    #     result[k] = _flatten(result[k])

                    # # result["distances"] = np.squeeze(result["distances"])#.reshape(-1, np.array(result["distances"]).shape[-1])
                    # result["documents"] = np.squeeze(result["documents"])
                    # result["ids"] = np.squeeze(result["ids"])
                    # result["metadatas"] = np.squeeze(result["metadatas"])
                    # np.squeeze(x, axis=(2,)).shape
                    Logger.info({k: result[k].shape for k in ["ids", *_include]})
                ind = -1
            else:
                ids = (
                    list(set([id_from_name(q) for q in query_texts]))
                    if query_texts
                    else None
                )
                _include: Include = ["embeddings", "documents", "metadatas"]
                result = collection.get(
                    ids=ids,
                    where=None if query_texts else where_products,
                    include=_include,
                )
                ind = (
                    result["ids"].index(id_from_name(neighbourhood_of_product[0]))
                    if ids
                    else -1
                )
            return result, ind
        else:
            return None, -1

    def chroma_nearest_category_vectors_to_products(
        self,
        neighbourhood_of_product: Tuple[str, str] = ("", ""),
        n_results: int = 10,
    ):
        result, id = self.chroma_get_esc_product_vectors(
            vegi_only=False,
            get_distances=True,
            neighbourhood_of_product=neighbourhood_of_product,
            n_results=n_results,
        )
        if not result:
            return None, None, None
        # embeddings = np.array(result["embeddings"])
        metadatas = result["metadatas"]
        assert (
            metadatas is not None
        ), "chromadb QueryResult in LLM().chroma_nearest_category_vectors_to_products didnt contain metadatas"
        source = np.array(
            [
                [mdi["source"] for mdi in md] if isinstance(md, list) else md["source"]
                for md in metadatas
            ]
        ).ravel()
        category = np.array(
            [
                [mdi["category"] for mdi in md]
                if isinstance(md, list)
                else md["category"]
                for md in metadatas
            ]
        ).ravel()
        df = pd.DataFrame(
            data={
                "documents": result["documents"],
                "source": source,
                "y": category if neighbourhood_of_product[0] else source,
                "category": category,
            }
        )
        if "distances" in result and type(result) == QueryResult:
            df["cos_diff"] = result["distances"]
        df.sort_values(by=["cos_diff"], ascending=True, inplace=True)
        return df

    def chroma_visualise_esc_product_vectors(
        self,
        vegi_only: bool = False,
        neighbourhood_of_product: Tuple[str, str] = ("", ""),
        n_results: int = 5,
    ):
        result, id = self.chroma_get_esc_product_vectors(
            vegi_only=vegi_only,
            get_distances=False,
            neighbourhood_of_product=neighbourhood_of_product,
            n_results=n_results,
        )
        if not result:
            return None, None, None
        embeddings = np.array(result["embeddings"])
        metadatas = result["metadatas"]
        assert (
            metadatas is not None
        ), "chromadb QueryResult in LLM().chroma_nearest_category_vectors_to_products didnt contain metadatas"
        source = np.array(
            [
                [mdi["source"] for mdi in md] if isinstance(md, list) else md["source"]
                for md in metadatas
            ]
        ).ravel()
        category = np.array(
            [
                [mdi["category"] for mdi in md]
                if isinstance(md, list)
                else md["category"]
                for md in metadatas
            ]
        ).ravel()
        # Logger.info(result['embeddings'])
        # Logger.info(np.array(result['embeddings']).shape)
        # Logger.info(result['documents'])
        # Logger.info(np.array(result['documents']).shape)
        # Logger.info(np.array(result['metadatas']).shape)
        # Logger.info(metadatas)

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(embeddings)
        # if neighbourhood_of_product[0]:
        #     # what we want is all products in same category, + other categories
        #     products_in_same_category = [p for p,c in vegi_products.product_names if c == neighbourhood_of_product[1]]
        #     other_category_names = [
        #         *vegi_products.product_categories,
        #         *sustained_products.product_categories,
        #     ]
        #     # where_products_documents['$contains'] = neighbourhood_of_product
        #     query_texts = [
        #         *products_in_same_category,
        #         *other_category_names
        #     ]
        df = pd.DataFrame(
            data={
                "documents": result["documents"],
                "source": source,
                "y": category if neighbourhood_of_product[0] else source,
                "category": category,
            }
        )
        df["pca-one"] = pca_result[:, 0]
        df["pca-two"] = pca_result[:, 1]
        df["pca-three"] = pca_result[:, 2]

        print(
            "Explained variation per principal component: {}".format(
                pca.explained_variance_ratio_
            )
        )

        np.random.seed(42)
        rndperm = np.random.permutation(df.shape[0])

        plt.figure(figsize=(16, 10))
        scatter_text(
            x="pca-one",
            y="pca-two",
            hue="y",
            # palette=sns.color_palette("hls", 10),
            palette=sns.color_palette("bright", df["y"].shape[0]),
            data=df.loc[rndperm, :],
            legend="full",
            alpha=0.3,
            labels_column="documents",
            title="PCA",
            origin_index=id,
        )
        time_start = time.time()
        tsne = TSNE(
            n_components=2,
            verbose=1,
            perplexity=max(1, min(40, embeddings.shape[0] / 5)),
            n_iter=300,
        )
        tsne_results = tsne.fit_transform(embeddings)
        df["tsne-2d-one"] = tsne_results[:, 0]
        df["tsne-2d-two"] = tsne_results[:, 1]

        plt.figure(figsize=(16, 10))
        scatter_text(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="y",
            # palette=sns.color_palette("hls", 10),
            palette=sns.color_palette("bright", df["y"].shape[0]),
            data=df.loc[rndperm, :],
            legend="full",
            alpha=0.3,
            labels_column="documents",
            title="TSNE",
            origin_index=id,
        )

        print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))
        return df, pca_result, tsne_results

    def chroma_get_vectors(
        self,
        query_embeddings: OneOrMany[Embedding] | None = None,
        query_texts: OneOrMany[Document] | None = None,
        n_results: int = 10,
        where: Where | None = None,
        where_document: WhereDocument | None = None,
    ):
        """
        For filtering by metadata, see https://docs.trychroma.com/usage-guide#filtering-by-metadata
        """
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is None:
            return None
        if query_embeddings or query_texts:
            include: Include = ["embeddings", "metadatas", "documents", "distances"]
            return collection.query(
                query_embeddings=query_embeddings,
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include,  # https://docs.trychroma.com/troubleshooting#using-get-or-query-embeddings-say-none
            )
        else:
            include: Include = ["embeddings", "metadatas", "documents"]
            return collection.get(
                where=where,
                where_document=where_document,
                include=include,
                limit=n_results,
            )

    def chroma_update_vectors_embeddings(
        self,
        ids: OneOrMany[ID],
        embeddings: OneOrMany[Embedding] | None = None,
    ):
        """
        See https://docs.trychroma.com/usage-guide#updating-data-in-a-collection
        """
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is None:
            return None
        return collection.update(
            ids=ids,
            embeddings=embeddings,  # [[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
        )

    def chroma_update_vectors(
        self,
        ids: OneOrMany[ID],
        metadatas: OneOrMany[Metadata] | None = None,
        documents: OneOrMany[Document] | None = None,
    ):
        """
        See https://docs.trychroma.com/usage-guide#updating-data-in-a-collection
        """
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is None:
            return None
        return collection.update(
            ids=ids,
            metadatas=metadatas,  # [{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
            documents=documents,  # ["doc1", "doc2", "doc3", ...],
        )

    def chroma_get_result_for_id(self, id: str):
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is not None:
            return collection.get(ids=id)
        else:
            return None

    def chroma_get_vectors_for_texts(self, *texts: str):
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is not None:
            return collection.get(ids=[id_from_name(name=name) for name in texts])
        else:
            return None

    def chromadb_check_connection(self):
        """returns a nanosecond heartbeat. Useful for making sure the client remains connected."""
        return self.chroma_client.heartbeat()

    def chromadb_reset_db(self, reinit: bool = False):
        """Empties and completely resets the database. ⚠️ This is destructive and not reversible."""
        self.chroma_client.reset()
        if reinit:
            return self.chromadb_reinit()
        return []

    def chromadb_reinit(self):
        '''
        Only all from 
        '''
        self._repoConn = Vegi_ESC_Repo(app=self._app)
        # self._vegiDbRepoConn = VegiRepo(app=self._app)
        self._vegiApiConn = VegiApi()
        self._chroma_esc_product_collection_name = _Chroma_ESC_product_collection_name
        self._text_embedding_openai_model = "text-embedding-ada-002"
        self._init_products_cache()
        return self._run_downloads()

    def chromadb_check_db_for_metadata_updates(self):
        """Empties and completely resets the database. ⚠️ This is destructive and not reversible."""
        self._run_downloads()

    def chroma_get_vector_store_contents(
        self,
        include: Include = ["documents"],
        limit: int | None = None,
    ):
        
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is None:
            _collections = self.chroma_client.list_collections()
            Logger.warn(
                f"Unable to get chroma esc collection: '{self.chroma_esc_product_collection_name}', possible collection names are: {pformat(_collections)}"
            )
            return None
        result = collection.get(
            include=include,
            limit=limit,
        )
        return result
    
    def chroma_query_vector_store(
        self,
        include: Include,
        top_n: int,
        where: Where | None,
        *query_texts: str
    ):
        """
        You can also query by a set of `query_texts`.
        Chroma will first embed each `query_text` with the collection's `embedding` function,
        and then perform the query with the `generated embedding`.

        The query will return the `n_results=10` closest matches to each `query_embedding`, in order.
        this will look like mapping a list of `query_texts` with len 3 -> a list of `embeddings` of len 3 -> a list: `[(top_n similar matches to emb in order) for emb in embeddings]`
        ========================================================================
        collection.query notes:
        top_n = 10 # by default
        return [
            {
                'documents':
                    [ nth_most_similar_document_in_collection_to(text, n) for n in range(topn) ]
            }
            for text in query_texts
        ]
        ========================================================================

        Returns
        -------
        dict[Include, list[list[...]]]
            where this looks like:
        {
            'ids': [N,t] as list[list[ID]],
            'documents': [N,t] as list[list[Document]],
            'distances': [N,t] as list[list[float]],
        }
        where N := len(query_texts)
        where t := top_n nearest matches for each text to search for similarities for
        ========================================================================
        """
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is None:
            _collections = self.chroma_client.list_collections()
            Logger.warn(
                f"Unable to get chroma esc collection: '{self.chroma_esc_product_collection_name}', possible collection names are: {pformat(_collections)}"
            )
            return None
        result = (
            collection.query(
                query_texts=[*query_texts],
                where=where,
                include=include,
                n_results=top_n,
            )
        )
        return result

    def chroma_most_similar_sustained_products(self, *vegi_product_names: str):
        """
        requests the single most similar prodduct from vector store for each vegi_product_name using `n_results=1`
        ========================================================================
        collection.query notes:
        top_n = 10 # by default
        return [
            {
                'documents':
                    [ nth_most_similar_document_in_collection_to(text, n) for n in range(topn) ]
            }
            for text in query_texts
        ]
        ========================================================================
        """
        _include: Include = ["embeddings", "documents", "distances", "metadatas"]
        _n_results: Final[int] = 1
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is None:
            Logger.warn(
                f"Unable to get chroma esc collection: '{self.chroma_esc_product_collection_name}'"
            )
            return None
        query_texts = [*vegi_product_names]
        result = collection.query(
            query_texts=query_texts,
            where={"source_id": {"$eq": self._sustained_products.source_id}},
            # where={"source": {"$eq": self._sustained_products.source}},
            # where={"source_id": {"$eq": "sustained"}},
            n_results=3,
            include=_include,
        )
        result = self.chroma_query_vector_store(
            _include, _n_results, {"source_id": {"$eq": self._sustained_products.source_id}}, *query_texts
        )
        # documents = result["documents"]
        # assert documents is not None, "chromadb QueryResult in LLM().chroma_most_similar_sustained_products didnt contain documents"
        # metadatas = result["metadatas"]
        # assert metadatas is not None, "chromadb QueryResult in LLM().chroma_most_similar_sustained_products didnt contain metadatas"
        # distances = result["distances"]
        # assert distances is not None, "chromadb QueryResult in LLM().chroma_most_similar_sustained_products didnt contain distances"
        
        # def _f(i: int) -> QuerySingleResult:
        #     assert (
        #         result is not None
        #     ), "empty result returned from LLM.chroma_query_vector_store in LLM().chroma_most_similar_sustained_products"
        #     assert (
        #         result["distances"] is not None
        #     ), "empty result distances returned from LLM.chroma_query_vector_store in LLM().chroma_most_similar_sustained_products"
        #     assert (
        #         result["metadatas"] is not None
        #     ), "empty result metadatas returned from LLM.chroma_query_vector_store in LLM().chroma_most_similar_sustained_products"
        #     assert (
        #         result["documents"] is not None
        #     ), "empty result documents returned from LLM.chroma_query_vector_store in LLM().chroma_most_similar_sustained_products"
        #     return ({
        #         "embedding": result["embeddings"][i][_n_results - 1] if result["embeddings"] else None,
        #         "distance": result["distances"][i][_n_results - 1]
        #         if result["distances"]
        #         else None,
        #         "metadata": result["metadatas"][i][_n_results - 1]
        #         if result["metadatas"]
        #         else None,
        #         "document": result["documents"][i][_n_results - 1]
        #         if result["documents"]
        #         else None,
        #         "id": result["ids"][i][_n_results - 1],
        #     })
        assert (
            result is not None
        ), "empty result returned from LLM.chroma_query_vector_store in LLM().chroma_most_similar_sustained_products"
        output_result_lookup: dict[str, QuerySingleResult] = {
            text: {
                "embedding": result["embeddings"][i][_n_results - 1]
                if result["embeddings"]
                else None,
                "distance": result["distances"][i][_n_results - 1]
                if result["distances"]
                else None,
                "metadata": result["metadatas"][i][_n_results - 1]
                if result["metadatas"]
                else None,
                "document": result["documents"][i][_n_results - 1]
                if result["documents"]
                else None,
                "id": result["ids"][i][_n_results - 1],
            }
            for i, text in enumerate(query_texts)
        }
        return output_result_lookup

    def chroma_most_similar_sustained_product(self, vegi_product_name: str):
        return self.chroma_most_similar_sustained_products(vegi_product_name)


# def blas(name: str, ndarray: np.ndarray):
#     """Helper for getting the appropriate BLAS function, using :func:`scipy.linalg.get_blas_funcs`.

#     Parameters
#     ----------
#     name : str
#         Name(s) of BLAS functions, without the type prefix.
#     ndarray : numpy.ndarray
#         Arrays can be given to determine optimal prefix of BLAS routines.

#     Returns
#     -------
#     object
#         BLAS function for the needed operation on the given data type.

#     """
#     return get_blas_funcs((name,), (ndarray,))[0]


# blas_nrm2 = blas("nrm2", np.array([], dtype=float))
# blas_scal = blas("scal", np.array([], dtype=float))


# def unitvec(
#     vec: np.ndarray | scipy.sparse.sparray | list[int | float],
#     norm="l2",
#     return_norm=False,
# ):
#     """Scale a vector to unit length.

#     Parameters
#     ----------
#     vec : {numpy.ndarray, scipy.sparse, list of (int, float)}
#         Input vector in any format
#     norm : {'l1', 'l2', 'unique'}, optional
#         Metric to normalize in.
#     return_norm : bool, optional
#         Return the length of vector `vec`, in addition to the normalized vector itself?

#     Returns
#     -------
#     numpy.ndarray, scipy.sparse, list of (int, float)}
#         Normalized vector in same format as `vec`.
#     float
#         Length of `vec` before normalization, if `return_norm` is set.

#     Notes
#     -----
#     Zero-vector will be unchanged.

#     """
#     supported_norms = ("l1", "l2", "unique")
#     if norm not in supported_norms:
#         raise ValueError(
#             "'%s' is not a supported norm. Currently supported norms are %s."
#             % (norm, supported_norms)
#         )

#     if scipy.sparse.issparse(vec):  # type: ignore
#         vec = vec.tocsr()
#         if norm == "l1":
#             veclen = np.sum(np.abs(vec.data))
#         if norm == "l2":
#             veclen = np.sqrt(np.sum(vec.data**2))
#         if norm == "unique":
#             veclen = vec.nnz
#         if veclen > 0.0:
#             if np.issubdtype(vec.dtype, np.integer):
#                 vec = vec.astype(float)
#             vec /= veclen
#             if return_norm:
#                 return vec, veclen
#             else:
#                 return vec
#         else:
#             if return_norm:
#                 return vec, 1.0
#             else:
#                 return vec

#     if isinstance(vec, np.ndarray):
#         if norm == "l1":
#             veclen = np.sum(np.abs(vec))
#         if norm == "l2":
#             if vec.size == 0:
#                 veclen = 0.0
#             else:
#                 veclen = blas_nrm2(vec)
#         if norm == "unique":
#             veclen = np.count_nonzero(vec)
#         if veclen > 0.0:
#             if np.issubdtype(vec.dtype, np.integer):
#                 vec = vec.astype(float)
#             if return_norm:
#                 return blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen
#             else:
#                 return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
#         else:
#             if return_norm:
#                 return vec, 1.0
#             else:
#                 return vec

#     try:
#         first = next(iter(vec))  # is there at least one element?
#     except StopIteration:
#         if return_norm:
#             return vec, 1.0
#         else:
#             return vec

#     if isinstance(first, (tuple, list)) and len(first) == 2:  # gensim sparse format
#         if norm == "l1":
#             length = float(sum(abs(val) for _, val in vec))
#         if norm == "l2":
#             length = 1.0 * math.sqrt(sum(val**2 for _, val in vec))
#         if norm == "unique":
#             length = 1.0 * len(vec)
#         assert (
#             length > 0.0
#         ), "sparse documents must not contain any explicit zero entries"
#         if return_norm:
#             return ret_normalized_vec(vec, length), length
#         else:
#             return ret_normalized_vec(vec, length)
#     else:
#         raise ValueError("unknown input type")


def ret_normalized_vec(vec, length):
    """Normalize a vector in L2 (Euclidean unit norm).

    Parameters
    ----------
    vec : list of (int, number)
        Input vector in BoW format.
    length : float
        Length of vector

    Returns
    -------
    list of (int, number)
        L2-normalized vector in BoW format.

    """
    if length != 1.0:
        return [(termid, val / length) for termid, val in vec]
    else:
        return list(vec)


def cossim(vec1, vec2):
    """Get cosine similarity between two sparse vectors.

    Cosine similarity is a number between `<-1.0, 1.0>`, higher means more similar.

    Parameters
    ----------
    vec1 : list of (int, float)
        Vector in BoW format.
    vec2 : list of (int, float)
        Vector in BoW format.

    Returns
    -------
    float
        Cosine similarity between `vec1` and `vec2`.

    """
    vec1, vec2 = dict(vec1), dict(vec2)
    if not vec1 or not vec2:
        return 0.0
    vec1len = 1.0 * math.sqrt(sum(val * val for val in vec1.values()))
    vec2len = 1.0 * math.sqrt(sum(val * val for val in vec2.values()))
    assert (
        vec1len > 0.0 and vec2len > 0.0
    ), "sparse documents must not contain any explicit zero entries"
    if len(vec2) < len(vec1):
        vec1, vec2 = (
            vec2,
            vec1,
        )  # swap references so that we iterate over the shorter vector
    result = sum(value * vec2.get(index, 0.0) for index, value in vec1.items())
    result /= vec1len * vec2len  # rescale by vector lengths
    return result


def argsort(x, topn=None, reverse=False):
    """Efficiently calculate indices of the `topn` smallest elements in array `x`.

    Parameters
    ----------
    x : array_like
        Array to get the smallest element indices from.
    topn : int, optional
        Number of indices of the smallest (greatest) elements to be returned.
        If not given, indices of all elements will be returned in ascending (descending) order.
    reverse : bool, optional
        Return the `topn` greatest elements in descending order,
        instead of smallest elements in ascending order?

    Returns
    -------
    numpy.ndarray
        Array of `topn` indices that sort the array in the requested order.

    """
    x = np.asarray(x)  # unify code path for when `x` is not a np array (list, tuple...)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, "argpartition"):
        return np.argsort(x)[:topn]
    # np >= 1.8 has a fast partial argsort, use that!
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))  # resort topn into order
