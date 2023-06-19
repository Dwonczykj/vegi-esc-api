from __future__ import annotations

import os
from flask import Flask
import requests
import json
import jsons
import cachetools.func
from vegi_esc_api.create_app import create_app

import vegi_esc_api.logger as logger
# from vegi_esc_api.models import 
from vegi_esc_api.models_wrapper import CachedSustainedItemCategory
from vegi_esc_api.sustained_mapper import SustainedVegiMapper
from vegi_esc_api.sustained_models import SustainedCategoriesList, SustainedCategory, SustainedImpact, SustainedImpactsList, SustainedProductBase, SustainedProductExplained, SustainedProductsList, SustainedSingleProductResult
from vegi_esc_api.vegi_esc_repo import Vegi_ESC_Repo, CachedItemSql, ESCSourceSql, ESCRatingSql
# from vegi_esc_api.models import CachedItemCreate

from datetime import datetime


class SustainedAPI:
    headers = {"accept": "application/json"}
    sustained_to_vegi_mapper = SustainedVegiMapper()

    def __init__(self, app: Flask) -> None:
        SustainedAPI.app = app
        SustainedAPI.vegi_esc_repo = Vegi_ESC_Repo(app=app)

    def _load_products_from_fn(self, category_id: str):
        fn = self.get_localstorage_fn(category_id)
        with open(fn, "r") as f:
            productsObjs = json.load(f)
        if not isinstance(productsObjs, list):
            raise Exception("sustainedLocalData should hold a list of products")
        return [SustainedProductBase.fromJson(c) for c in productsObjs]

    def useLocalStorage(self):
        dirname = "localstorage"
        if not os.path.exists(dirname):
            os.makedirs(os.path.dirname(dirname), exist_ok=True)

    def _checkSourceExists(self):
        sustained_source = SustainedAPI.vegi_esc_repo.get_source(source_name='sustained.com')
        if not sustained_source:
            sustained_source = SustainedAPI.vegi_esc_repo.add_source(new_source=ESCSourceSql(
                name='sustained.com',
                domain='sustained.com',
                source_type='api',
                credibility=1,
            ))
        self.sustained_source = sustained_source
        return sustained_source
    
    def get_sustained_escsource(self):
        self._checkSourceExists()
        return self.sustained_source
    
    def _fetchCategories(self):
        url = "https://api.sustained.com/choice/v1/categories"
        print(f'GET -> "{url}"')
        response = requests.get(url, headers=self.headers)
        sustainedCategoriesList = SustainedCategoriesList.fromJson(response.json())
        categories = sustainedCategoriesList.categories
        cachedItem = CachedItemSql(
            item_name="sustained_categories_list",
            item_type="category_list",
            item_source="sustained.com",
            item_json=jsons.dumps([category.toJson() for category in categories]),
            ttl_days=30,
            created_on=datetime.now()
        )
        SustainedAPI.vegi_esc_repo.add_cached_items(items=[cachedItem])
        # self.useLocalStorage()
        # # todo replace with vegi_esc_repo.write_db
        # with open('localstorage/sustained-categories.json', 'w') as f:
        #     json.dump(responseData.toJson(), f, indent=2)
        #     print('Written categories from sustained to local storage')

        return categories

    def _fetchProductsForCategory(self, category: SustainedCategory):
        # cat_id = category.id
        # fn = self.get_localstorage_fn(cat_id)
        url = category.links.products
        return self._fetchProducts(url=url, category=category)

    def _fetchProducts(self, url: str, category: SustainedCategory):
        # TODO: this needs to call the vegi_esc_repo.get_sustained_items() will have some circular class models issues to sort first htough
        products: list[SustainedProductBase] = []
        print(f'GET -> "{url}"')
        response = requests.get(url, headers=self.headers)

        responseData = SustainedProductsList.fromJson(response.json())
        products += responseData.products
        for j in range(10):
            if responseData.links and responseData.links.next:
                url = responseData.links.next
                response = requests.get(url, headers=self.headers)
                responseData = SustainedProductsList.fromJson(response.json())
                products += responseData.products
            else:
                break
        item = CachedItemSql(
            item_name=category.name,
            item_type="category",
            item_source="sustained.com",
            item_json=jsons.dumps(
                CachedSustainedItemCategory(
                    name=category.name,
                    id=category.id,
                    links=category.links,
                    products=products,
                ).toJson()
            ),
        )
        SustainedAPI.vegi_esc_repo.add_cached_items(items=[item])
        # with open(fn, 'w') as f:
        #     json.dump([r.toJson() for r in results], f, indent=2)
        # print(f'wrote cat_products to "{fn}"')
        return products

    def _fetchProduct(self, id: str):
        # id: "productid.00211dfc78bae013c19638489cc33d83"
        url = f"https://api.sustained.com/choice/v1/products/{id}"
        print(f'GET -> "{url}"')
        response = requests.get(url, headers=self.headers)
        singleItem = SustainedSingleProductResult.fromJson(response.json())
        return singleItem

    def _fetchProductImpacts(self, product: SustainedProductBase):
        if not product.links.impacts:
            raise TypeError(product.links)

        url = product.links.impacts
        print(f'GET -> "{url}"')
        response = requests.get(url, headers=self.headers)

        responseData = SustainedImpactsList.fromJson(response.json())
        impacts: list[SustainedImpact] = []
        impacts += responseData.impacts
        for j in range(10):
            if responseData.links and responseData.links.next:
                url = responseData.links.next
                response = requests.get(url, headers=self.headers)
                responseData = SustainedImpactsList.fromJson(response.json())
                impacts += responseData.impacts
            else:
                break
        productsExplained = SustainedProductExplained(product=product, impacts=impacts)
        productsExplainedForVegiDB = (
            self.sustained_to_vegi_mapper.mapSustainedProductImpactsToVegi(
                sourceProductRated=productsExplained
            )
        )
        sustained_source = self._checkSourceExists()
        for e in productsExplainedForVegiDB.explanations:
            e.source = sustained_source.id

        self.vegi_esc_repo.add_rating(
            new_rating=ESCRatingSql(
                product_name=productsExplainedForVegiDB.rating.product_name,
                product_id=productsExplainedForVegiDB.rating.product_id,
                calculated_on=productsExplainedForVegiDB.rating.calculated_on,
                rating=productsExplainedForVegiDB.rating.rating,
            ),
            explanations=productsExplainedForVegiDB.explanations,
        )
        
        return productsExplainedForVegiDB

    def get_localstorage_fn(self, category_id: str):
        return f"localstorage/sustained-{category_id}.json"

    def refresh_products_lists(
        self, category_name: str = "", refresh_categories: bool = False
    ):
        self._checkSourceExists()
        if refresh_categories:
            categories = self._fetchCategories()
            # with open('sustained.json','w') as f:
            #     json.dump({'categories':categories,'products':products}, f, indent=2)
        else:
            categories = self.get_categories()
            # self.useLocalStorage()
            # with open('localstorage/sustained-categories.json', 'r') as f:
            #     sustainedOldLocalData = json.load(f)
            #     categories = [SustainedCategory.fromJson(c) for c in sustainedOldLocalData['categories']]
            # existingProducts = sustainedOldLocalData['products'] if 'products' in sustainedOldLocalData.keys() else []

        if category_name:
            # responseData['products'] = existingProducts
            cat = next(
                (c for c in categories if c.name.lower() == category_name.lower()), None
            )
            if cat:
                products = self._fetchProductsForCategory(category=cat)
                return {"categories": [cat], "products": products}
            else:
                logger.warn(
                    f'No matching sustained category found for "{category_name}"'
                )
                return {"categories": categories, "products": []}
        else:
            n = len(categories)
            products: list[SustainedProductBase] = []
            print(f"Getting products for {n} different categories")
            for i, cat in enumerate(categories):
                print(f"Fetched {(i/n)*100:0.2}% of products from sustained API")
                products += self._fetchProductsForCategory(category=cat)

            # with open('sustained.json','w') as f:
            #     json.dump({'categories':categories,'products':products}, f, indent=2)
            return {"categories": categories, "products": products}

    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
    def get_categories(self):
        # todo replace with vegi_esc_repo.write_db
        cats_with_products = SustainedAPI.vegi_esc_repo.get_sustained_items()
        return [SustainedCategory.fromJson(c.toJson()) for c in cats_with_products]
        # self.useLocalStorage()
        # with open('localstorage/sustained-categories.json', 'r') as f:
        #     sustainedLocalData = json.load(f)
        #     categoriesObjs = sustainedLocalData['categories']
        #     if not isinstance(categoriesObjs, list):
        #         raise Exception(
        #             'sustainedLocalData should hold a list of categories')
        #     categories = [SustainedCategory.fromJson(
        #         c) for c in categoriesObjs]
        # return categories

    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
    def get_products(self, category_name: str = ""):
        cats_with_products = SustainedAPI.vegi_esc_repo.get_sustained_items()
        return [
            SustainedProductBase.fromJson(p.toJson())
            for c in cats_with_products
            for p in c.products
        ]
        # categories = self.get_categories()
        # # with open('sustained.json', 'r') as f:
        # #     sustainedLocalData = json.load(f)
        # #     products = sustainedLocalData['products']
        # if category_name:
        #     cat = next((c for c in categories if c.name.lower()
        #                == category_name.lower()), None)
        #     if not cat:
        #         raise Exception(
        #             f'No category found with name: "{category_name}"')
        #     products = self._load_products_from_fn(cat.id)
        #     return products
        #     # return [c['name'] for c in products]
        # else:
        #     return [p for c in categories for p in self._load_products_from_fn(c.id)]

    def get_product_with_impact(self, sustainedProductId: str):
        product = self._fetchProduct(id=sustainedProductId)
        productExplained = self._fetchProductImpacts(product=product)
        return productExplained

    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
    def get_category_ids(self, replace: tuple = ("", "")):
        self.useLocalStorage()
        with open("localstorage/sustained-categories.json", "r") as f:
            sustainedLocalData = json.load(f)
            categories = sustainedLocalData["categories"]
        return [
            c["id"].replace(replace[0], replace[1])
            if replace and replace[0] != ""
            else c["id"]
            for c in categories
        ]

    def get_cat_for_space_delimited_id(
        self, most_sim_cat_id: str, replace: tuple = ("", "")
    ):
        self.useLocalStorage()
        with open("localstorage/sustained-categories.json", "r") as f:
            sustainedLocalData = json.load(f)
            categories = sustainedLocalData["categories"]
        return next(
            (
                c
                for c in categories
                if (
                    c["id"].replace(replace[0], replace[1])
                    if replace and replace[0] != ""
                    else c["id"]
                ).lower()
                == most_sim_cat_id
            ),
            None,
        )

    def get_category_names(self):
        self.useLocalStorage()
        with open("localstorage/sustained-categories.json", "r") as f:
            sustainedLocalData = json.load(f)
            categories = sustainedLocalData["categories"]
        return [c["name"] for c in categories]


if __name__ == "__main__":
    # check functions work
    app, vegi_db_session = create_app(None)
    ss = SustainedAPI(app=app)
    yoghurts = ss.get_products(category_name="yoghurts")
    print(yoghurts)
    exit(0)
