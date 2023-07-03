from __future__ import annotations
from typing import Callable, Optional, Self, Type, Any, TypeVar
from typing import Tuple, Final, Generic
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
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb.api.fastapi import FastAPI
from chromadb.api.local import LocalAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import (
    # EmbeddingFunction,
    Metadata,
    ID,
    Where,
    WhereDocument,
    OneOrMany,
    Embedding,
    Include,
    GetResult,
    QueryResult,
    Document,
)

# from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, CollectionMetadata
# from chromadb.errors import ChromaError, error_types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time

from vegi_esc_api.constants import AppConstants
from vegi_esc_api.vegi_esc_repo import Vegi_ESC_Repo
from vegi_esc_api.vegi_repo import VegiRepo
import vegi_esc_api.logger as Logger


V = TypeVar('V', str, int)


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


def id_from_name(name: str):
    return re.sub(pattern=r"[^0-9A-Za-z]", repl="", string=name.replace(" ", "_"))


T = TypeVar('T')


def split_array(array: list[T], batch_size: int):
    split_list = [array[i : i + batch_size] for i in range(0, len(array), batch_size)]
    return split_list


cachedStopWords: list[str] = []


try:
    cachedStopWords = stopwords.words("english")
except LookupError:
    nltk.download('stopwords')
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
    __chromadb_client: FastAPI | LocalAPI
    
    __instance: LLM

    def __init__(
        self,
        app: Flask,
    ) -> None:
        if hasattr(type(self), "_LLM__instance") is False or not type(self).__instance:
            self._app = app
            self.chromadb_reinit()
        else:
            raise Exception(f'Cant load duplicate instance of {type(self).__name__}')

    @property
    def chroma_client(self):
        if hasattr(type(self), "_LLM__chromadb_client") is False or not LLM.__chromadb_client:
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
    def sustained_products(self):
        return self._sustained_products
    
    def _init_products_cache(
        self
    ):
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
            _vegi_products = self._vegiDbRepoConn.get_products_with_categories()
            _vegi_categories = self._vegiDbRepoConn.get_product_categories(vendor=None)
            vegi_products = VegiProductsFromJson(
                product_names=[
                    CleanProductNameCategoryTuple(
                        source_id=-1,
                        source_name="vegi",
                        product_id=p.id,
                        name=clean_words(p.name),
                        category=clean_words(c.name),
                        original_name=p.name,
                        original_category_name=c.name,
                    ) for (p, c) in _vegi_products
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
                    ) for k in _vegi_categories
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
    def getModel(cls: Type[Self], app: Flask, args: argparse.Namespace | None = None) -> Self:
        """
        Takes care of downloading, initialisation, singleton fetching and is a factory method that returns an object of this class
        """
        if hasattr(cls, "__instance") is False or not cls.__instance:
            return LLM(
                app=app,
            )
        return cls.__instance
        
    def most_similar_esc_products(self, *vegi_product_names: str):
        vegi_to_esc_product_map = self.chroma_most_similar_sustained_products(*vegi_product_names)
        if vegi_to_esc_product_map is None:
            return None
        return ({
            vegi_product_name: self._repoConn.get_esc_product_by_id(
                id=int(vegi_to_esc_product_map[vegi_product_name]["product_esc_id"]),
                source=int(vegi_to_esc_product_map[vegi_product_name]["source_id"]),
            )
            for vegi_product_name in vegi_to_esc_product_map.keys()
            if int(vegi_to_esc_product_map[vegi_product_name]["source_id"]) > 0
        })
        
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
        _still_to_create_products = list({
            id_from_name(p.name): (p, id_from_name(p.name)) for p in products if id_from_name(p.name) not in result["ids"]
        }.values())
        _to_modify_products = list({
            id_from_name(p.name): (p, id_from_name(p.name)) for p in products if id_from_name(p.name) in result["ids"]
        }.values())
        if _to_modify_products:
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
                        f"Computing batch {i} ({i*batch_sizes}/{len(_to_modify_products)})"
                    )
                    collection.update(
                        ids=[
                            id for (p, id) in batches[i]
                        ],  # unique for each doc
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
                        f"Computing batch {i} ({i*batch_sizes}/{len(_still_to_create_products)})"
                    )
                    collection.add(
                        ids=[
                            id for (p, id) in batches[i]
                        ],  # unique for each doc
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
                    if rows_calculated >= 250 or rows_calculated == len(_still_to_create_products):
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
        name: str = '',
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
        assert metadatas is not None, "chromadb QueryResult in LLM().chroma_nearest_category_vectors_to_products didnt contain metadatas"
        source = np.array([[mdi["source"] for mdi in md] if isinstance(md, list) else md["source"] for md in metadatas]).ravel()
        category = np.array([[mdi["category"] for mdi in md] if isinstance(md, list) else md["category"] for md in metadatas]).ravel()
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
        assert metadatas is not None, "chromadb QueryResult in LLM().chroma_nearest_category_vectors_to_products didnt contain metadatas"
        source = np.array([[mdi["source"] for mdi in md] if isinstance(md, list) else md["source"] for md in metadatas]).ravel()
        category = np.array([[mdi["category"] for mdi in md] if isinstance(md, list) else md["category"] for md in metadatas]).ravel()
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
        self._repoConn = Vegi_ESC_Repo(app=self._app)
        self._vegiDbRepoConn = VegiRepo(app=self._app)
        self._chroma_esc_product_collection_name = _Chroma_ESC_product_collection_name
        self._text_embedding_openai_model = "text-embedding-ada-002"
        self._init_products_cache()
        return self._run_downloads()
    
    def chromadb_check_db_for_metadata_updates(self):
        """Empties and completely resets the database. ⚠️ This is destructive and not reversible."""
        self._run_downloads()
        
    def chroma_query_vector_store(self, include: Include = ["documents"], *query_texts: str):
        collection = self.chroma_get_esc_collection(
            name=self.chroma_esc_product_collection_name,
        )
        if collection is None:
            _collections = self.chroma_client.list_collections()
            Logger.warn(
                f"Unable to get chroma esc collection: '{self.chroma_esc_product_collection_name}', possible collection names are: {pformat(_collections)}"
            )
            return None
        result = collection.query(
            query_texts=[*query_texts],
            include=include,
        ) if query_texts else collection.get(include=include)
        return result

    def chroma_most_similar_sustained_products(self, *vegi_product_names: str):
        _include: Include = ["embeddings", "documents", "distances", "metadatas"]
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
        documents = result["documents"]
        assert documents is not None, "chromadb QueryResult in LLM().chroma_most_similar_sustained_products didnt contain documents"
        metadatas = result["metadatas"]
        assert metadatas is not None, "chromadb QueryResult in LLM().chroma_most_similar_sustained_products didnt contain metadatas"
        output = {text: metadatas[i][0] for i, text in enumerate(query_texts)}
        if Logger.LOG_LEVEL >= Logger.LogLevel.verbose.value:
            for i, text in enumerate(query_texts):
                most_similiar_doc = documents[i][0]
                Logger.info(f"{text}[vegi] -> {most_similiar_doc}[sustained]")
            result_np: dict[str, np.ndarray] = {}
            for k in ["ids", *_include]:
                result_np[k] = np.array(result[k])
            Logger.info({k: result_np[k].shape for k in ["ids", *_include]})
            Logger.info(result_np["ids"])
        return output

    def chroma_most_similar_sustained_product(self, vegi_product_name: str):
        return self.chroma_most_similar_sustained_products(vegi_product_name)
