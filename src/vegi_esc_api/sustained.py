from __future__ import annotations


from flask import Flask
import requests
import cachetools.func
# from vegi_esc_api.create_app import create_app
import jsons
import vegi_esc_api.logger as Logger
from vegi_esc_api.helpers import parse_measurement
from vegi_esc_api.models import ESCProductInstance
from vegi_esc_api.models_wrapper import CachedSustainedItemCategory
from vegi_esc_api.sustained_mapper import SustainedVegiMapper
from vegi_esc_api.sustained_models import (
    SustainedCategoriesList,
    SustainedCategory,
    SustainedImpact,
    SustainedImpactsList,
    SustainedProductBase,
    SustainedProductExplained,
    SustainedProductsList,
    SustainedSingleProductResult,
)
from vegi_esc_api.vegi_esc_repo import (
    Vegi_ESC_Repo,
    CachedItemSql,
    ESCSourceSql,
    ESCRatingSql,
    SUSTAINED_DOMAIN_NAME,
)

# from vegi_esc_api.models import CachedItemCreate

from datetime import datetime


class SustainedAPI:
    headers = {"accept": "application/json"}
    sustained_to_vegi_mapper = SustainedVegiMapper()

    def __init__(self, app: Flask) -> None:
        SustainedAPI.app = app
        SustainedAPI.vegi_esc_repo = Vegi_ESC_Repo(app=app)

    def _checkSourceExists(self):
        sustained_source = SustainedAPI.vegi_esc_repo.get_source(
            source_name=SUSTAINED_DOMAIN_NAME
        )
        if not sustained_source:
            sustained_source = SustainedAPI.vegi_esc_repo.add_source(
                new_source=ESCSourceSql(
                    name=SUSTAINED_DOMAIN_NAME,
                    domain=SUSTAINED_DOMAIN_NAME,
                    source_type="api",
                    credibility=1,
                )
            )
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
            item_source=SUSTAINED_DOMAIN_NAME,
            item_json=jsons.dumps([category.toJson() for category in categories]),
            ttl_days=30,
            created_on=datetime.now(),
        )
        SustainedAPI.vegi_esc_repo.add_cached_items(items=[cachedItem])
        return categories

    def _fetchProductsForCategory(self, category: SustainedCategory):
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
            item_source=SUSTAINED_DOMAIN_NAME,
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
        for product in products:
            num_units, unit_type = parse_measurement(product.pack)
            if not num_units or not unit_type:
                num_units = 100
                unit_type = "g"
            SustainedAPI.vegi_esc_repo.add_product_if_not_exists(
                name=product.name,
                product_external_id_on_source=product.id,
                source=self.get_sustained_escsource().id,
                description="",
                category=product.category,
                keyWords=[],
                imageUrl=product.image,
                ingredients="",
                packagingType="unknown",
                stockUnitsPerProduct=1,
                sizeInnerUnitValue=num_units,
                sizeInnerUnitType=unit_type,
                productBarCode="",
                supplier="",
                brandName="",
                origin="",
                finalLocation="",
                taxGroup=product.gtin,
                dateOfBirth=datetime.now(),
                finalDate=datetime.now(),
            )
        # with open(fn, 'w') as f:
        #     json.dump([r.toJson() for r in results], f, indent=2)
        # print(f'wrote cat_products to "{fn}"')
        return products

    def _fetchProduct(self, id: str):
        # id: "productid.00211dfc78bae013c19638489cc33d83"
        url = f"https://api.sustained.com/choice/v1/products/{id}"
        print(f'GET -> "{url}"')
        response = requests.get(url, headers=self.headers)
        product = SustainedSingleProductResult.fromJson(response.json())
        num_units, unit_type = parse_measurement(product.pack)
        if not num_units or not unit_type:
            num_units = 100
            unit_type = "g"
        esc_product = SustainedAPI.vegi_esc_repo.add_product_if_not_exists(
            name=product.name,
            product_external_id_on_source=product.id,
            source=self.get_sustained_escsource().id,
            description="",
            category=product.category,
            keyWords=[],
            imageUrl=product.image,
            ingredients="",
            packagingType="unknown",
            stockUnitsPerProduct=1,
            sizeInnerUnitValue=num_units,
            sizeInnerUnitType=unit_type,
            productBarCode="",
            supplier="",
            brandName="",
            origin="",
            finalLocation="",
            taxGroup=product.gtin,
            dateOfBirth=datetime.now(),
            finalDate=datetime.now(),
        )
        return product, esc_product

    def _fetchProductImpacts(
        self, product: SustainedProductBase, esc_product: ESCProductInstance
    ):
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
        productsExplained = SustainedProductExplained(
            product=product, impacts=impacts, db_product=esc_product
        )
        productsExplainedForVegiDB = (
            self.sustained_to_vegi_mapper.mapSustainedProductImpactsToVegi(
                sourceProductRated=productsExplained
            )
        )
        sustained_source = self._checkSourceExists()
        self.vegi_esc_repo.add_rating_for_product(
            new_rating=ESCRatingSql(
                product=productsExplainedForVegiDB.rating.product,
                product_name=productsExplainedForVegiDB.rating.product_name,
                product_id=productsExplainedForVegiDB.rating.product_id,
                calculated_on=productsExplainedForVegiDB.rating.calculated_on,
                rating=productsExplainedForVegiDB.rating.rating,
            ),
            explanationsCreate=productsExplainedForVegiDB.explanations,
            source=sustained_source.id,
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
        else:
            categories = self.get_categories()

        if category_name:
            cat = next(
                (c for c in categories if c.name.lower() == category_name.lower()), None
            )
            if cat:
                products = self._fetchProductsForCategory(category=cat)
                return {"categories": [cat], "products": products}
            else:
                Logger.warn(
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
        cats_with_products = SustainedAPI.vegi_esc_repo.get_sustained_categories()
        return [SustainedCategory.fromJson(c.toJson()) for c in cats_with_products]

    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
    def get_products(self, category_name: str = ""):
        products = SustainedAPI.vegi_esc_repo.get_sustained_items(
            category_name=category_name
        )
        if not products:
            ss_products: list[SustainedProductBase] = []
            sustained_cats = SustainedAPI.vegi_esc_repo.get_sustained_categories()
            for cat in sustained_cats:
                if cat.name == category_name:
                    ss_products += cat.products
            for sustained_product in ss_products:
                num_units, unit_type = parse_measurement(sustained_product.pack)
                if not num_units or not unit_type:
                    num_units = 100
                    unit_type = "g"
                esc_product = SustainedAPI.vegi_esc_repo.add_product_if_not_exists(
                    name=sustained_product.name,
                    product_external_id_on_source=sustained_product.id,
                    source=self.get_sustained_escsource().id,
                    description="",
                    category=sustained_product.category,
                    keyWords=[],
                    imageUrl=sustained_product.image,
                    ingredients="",
                    packagingType="unknown",
                    stockUnitsPerProduct=1,
                    sizeInnerUnitValue=num_units,
                    sizeInnerUnitType=unit_type,
                    productBarCode="",
                    supplier="",
                    brandName="",
                    origin="",
                    finalLocation="",
                    taxGroup=sustained_product.gtin,
                    dateOfBirth=datetime.now(),
                    finalDate=datetime.now(),
                )
                if esc_product:
                    products += [esc_product]
        return products
        # return [
        #     SustainedProductBase.fromProductSql(p.toJson())
        #     for p in products
        # ]
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
        product, esc_product = self._fetchProduct(id=sustainedProductId)
        assert (
            esc_product is not None
        ), f"esc_product cannot be None after being fetched in sustained.get_product_with_impact(sustainedProductId={sustainedProductId})"
        productExplained = self._fetchProductImpacts(
            product=product, esc_product=esc_product
        )
        return productExplained

    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
    def get_category_ids(self, replace: tuple = ("", "")):
        categories = self._fetchCategories()
        return [
            c.id.replace(replace[0], replace[1])
            if replace and replace[0] != ""
            else c.id
            for c in categories
        ]

    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
    def get_cat_for_space_delimited_id(
        self, most_sim_cat_id: str, replace: tuple = ("", "")
    ):
        categories = self._fetchCategories()
        return next(
            (
                c
                for c in categories
                if (
                    c.id.replace(replace[0], replace[1])
                    if replace and replace[0] != ""
                    else c.id
                ).lower()
                == most_sim_cat_id.lower()
            ),
            None,
        )

    def get_category_names(self):
        categories = self._fetchCategories()
        return [c.name for c in categories]

