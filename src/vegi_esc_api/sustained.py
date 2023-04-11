from dataclasses import dataclass
import os
from typing import Generic, Literal, Any, Optional, TypeVar
from typing_extensions import override
import requests
import json
import jsons
import cachetools.func
from vegi_esc_api.json_class_helpers import DataClassJsonWrapped
from vegi_esc_api.logger import info, warn, verbose



@dataclass
class SustainedProductPoints(DataClassJsonWrapped):
    pef:float
    
    
@dataclass
class SustainedCategoryLinks(DataClassJsonWrapped):
    products: str
    category: str
    next: Optional[str]


@dataclass
class SustainedProductBaseLinks(DataClassJsonWrapped):
    category: str
    impacts: str
    
@dataclass
class SustainedProductsLinks(SustainedProductBaseLinks):
    product: str

@dataclass
class SustainedProductLinks(SustainedProductBaseLinks):
    pass
        

@dataclass
class SustainedProductsListLinks(DataClassJsonWrapped):
    products: Optional[str]
    # self: Optional[str]
    next: Optional[str]
    first: Optional[str]

@dataclass
class SustainedImpactsListLinks(DataClassJsonWrapped):
    product: Optional[str]
    # self: Optional[str]
    next: Optional[str]
    first: Optional[str]
    


class SustainedCategoriesListLinks(DataClassJsonWrapped):
    categories: Optional[str]
    # self: Optional[str]
    next: Optional[str]
    first: Optional[str]

    
@dataclass
class SustainedCategory:
    id: str
    name: str
    links: SustainedCategoryLinks
    
    @classmethod
    def fromJson(cls, obj: Any):
        return jsons.load(obj, cls=cls)

    def toJson(self):
        return jsons.dump(self, strip_privates=True)

    
@dataclass
class SustainedImpact:
    id: str
    title: str
    description: str
    grade: Literal['A'] | Literal['B'] | Literal['C'] | Literal['D'] | Literal['E'] | Literal['F'] | Literal['G'] | str
    svg_icon: str
    
    @classmethod
    def fromJson(cls, obj:Any):
        return jsons.load(obj,cls=cls)
    
    def toJson(self):
        return jsons.dump(self, strip_privates=True)
    
@dataclass
class SustainedProductBase(DataClassJsonWrapped):
    id: str
    name: str
    category: str
    pack: str
    grade: Literal['A'] | Literal['B'] | Literal['C'] | Literal['D'] | Literal['E'] | Literal['F'] | Literal['G'] | str
    gtin: str
    image: str
    info_icons: list[str]
    points: SustainedProductPoints
    links: SustainedProductBaseLinks

@dataclass
class SustainedProductsResultItem(SustainedProductBase):
    links: SustainedProductsLinks
    
@dataclass
class SustainedSingleProductResult(SustainedProductBase):
    links: SustainedProductLinks
    

TP = TypeVar('TP', SustainedProductsResultItem, SustainedSingleProductResult)
@dataclass
class SustainedProductExplained(Generic[TP],DataClassJsonWrapped):
    product:SustainedProductBase
    impacts:list[SustainedImpact]
    

@dataclass
class SustainedCategoriesList(DataClassJsonWrapped):
    categories:list[SustainedCategory]
    links: SustainedCategoriesListLinks
    page: int
    page_size: int
    next_page_token: Optional[str]
    total_results: int


@dataclass
class SustainedProductsList(DataClassJsonWrapped):
    products: list[SustainedProductsResultItem]
    links: SustainedProductsListLinks
    page: int
    page_size: int
    next_page_token: Optional[str]
    total_results: int

@dataclass
class SustainedImpactsList(DataClassJsonWrapped):
    impacts:list[SustainedImpact]
    links: SustainedImpactsListLinks
    page: int
    page_size: int
    next_page_token: Optional[str]
    total_results: int


class SustainedAPI():
    
    headers = {"accept": "application/json"}
    
    def _load_products_from_fn(self, category_id:str):
        fn = self.get_localstorage_fn(category_id)
        with open(fn, 'r') as f:
            productsObjs = json.load(f)
        if not isinstance(productsObjs, list):
            raise Exception(
                'sustainedLocalData should hold a list of products')
        return [SustainedProductBase.fromJson(c) for c in productsObjs]
    
    def useLocalStorage(self):
        dirname = 'localstorage'
        if not os.path.exists(dirname):
            os.makedirs(os.path.dirname(dirname), exist_ok=True)
            # with open(self._fileName, 'w') as f:
            #     pass
    
    def _fetchCategories(self):
        url = 'https://api.sustained.com/choice/v1/categories'
        print(f'GET -> "{url}"')
        response = requests.get(url, headers=self.headers)
        responseData = SustainedCategoriesList.fromJson(response.json())
        self.useLocalStorage()
        with open('localstorage/sustained-categories.json', 'w') as f:
            json.dump(responseData.toJson(), f, indent=2)
            print('Written categories from sustained to local storage')
        return responseData.categories
    
    def _fetchProductsForCategory(self, cat:SustainedCategory):
        cat_id = cat.id
        fn = self.get_localstorage_fn(cat_id)
        url = cat.links.products
        return self._fetchProducts(url=url,fn=fn)
    
    def _fetchProducts(self, url:str, fn:str):
        results: list[SustainedProductBase] = []
        print(f'GET -> "{url}"')
        response = requests.get(
            url, headers=self.headers)
        
        responseData = SustainedProductsList.fromJson(response.json())
        results += responseData.products
        for j in range(10):
            if responseData.links and responseData.links.next:
                url = responseData.links.next
                response = requests.get(
                    url, headers=self.headers)
                responseData = SustainedProductsList.fromJson(response.json())
                results += responseData.products
            else:
                break
        with open(fn, 'w') as f:
            json.dump([r.toJson() for r in results], f, indent=2)
        print(f'wrote cat_products to "{fn}"')
        return results
    
    def _fetchProduct(self, id:str):
        # id: "productid.00211dfc78bae013c19638489cc33d83"
        url = f'https://api.sustained.com/choice/v1/products/{id}'
        print(f'GET -> "{url}"')
        response = requests.get(
            url, headers=self.headers)
        
        return SustainedSingleProductResult.fromJson(response.json())
        
    def _fetchProductImpacts(self, product: SustainedProductBase):
        if not product.links.impacts:
            raise TypeError(product.links)
    
        url = product.links.impacts
        print(f'GET -> "{url}"')
        response = requests.get(
            url, headers=self.headers)
        
        responseData = SustainedImpactsList.fromJson(response.json())
        results: list[SustainedImpact] = []
        results += responseData.impacts
        for j in range(10):
            if responseData.links and responseData.links.next:
                url = responseData.links.next
                response = requests.get(
                    url, headers=self.headers)
                responseData = SustainedImpactsList.fromJson(response.json())
                results += responseData.impacts
            else:
                break
        # with open(fn, 'w') as f:
        #     json.dump([r.toJson() for r in results], f, indent=2)
        # print(f'wrote cat_products to "{fn}"')
        return results
    
    def get_localstorage_fn(self, category_id: str):
        return f'localstorage/sustained-{category_id}.json'
    
    def refresh_products_lists(self, category_name:str='', refresh_categories:bool=False):
        if refresh_categories:
            categories = self._fetchCategories()
            # with open('sustained.json','w') as f:
            #     json.dump({'categories':categories,'products':products}, f, indent=2)
        else:
            self.useLocalStorage()
            with open('localstorage/sustained-categories.json', 'r') as f:
                sustainedOldLocalData = json.load(f)
                categories = [SustainedCategory.fromJson(c) for c in sustainedOldLocalData['categories']]
            # existingProducts = sustainedOldLocalData['products'] if 'products' in sustainedOldLocalData.keys() else []
        
        if category_name:
            # responseData['products'] = existingProducts
            cat = next(
                (c for c in categories if c.name.lower() == category_name.lower()), None)
            if cat:
                responseData = self._fetchProductsForCategory(cat=cat)
                
            else:
                warn(f'No matching sustained category found for "{category_name}"')
                return []
        else:
            
            n = len(categories)
            products:list[SustainedProductBase] = []
            print(f'Getting products for {n} different categories')
            for i,cat in enumerate(categories):
                print(f'Fetched {(i/n)*100:0.2}% of products from sustained API')
                products += self._fetchProductsForCategory(cat=cat)  
                      
            # with open('sustained.json','w') as f:
            #     json.dump({'categories':categories,'products':products}, f, indent=2)
            return {'categories': categories, 'products': products}
        
    def get_categories(self):
        self.useLocalStorage()
        with open('localstorage/sustained-categories.json', 'r') as f:
            sustainedLocalData = json.load(f)
            categoriesObjs = sustainedLocalData['categories']
            if not isinstance(categoriesObjs, list):
                raise Exception(
                    'sustainedLocalData should hold a list of categories')
            categories = [SustainedCategory.fromJson(
                c) for c in categoriesObjs]
        return categories

    def get_products(self, category_name: str = ''):
        categories = self.get_categories()
        # with open('sustained.json', 'r') as f:
        #     sustainedLocalData = json.load(f)
        #     products = sustainedLocalData['products']
        if category_name:
            cat = next((c for c in categories if c.name.lower()
                       == category_name.lower()), None)
            if not cat:
                raise Exception(
                    f'No category found with name: "{category_name}"')
            products = self._load_products_from_fn(cat.id)
            return products
            # return [c['name'] for c in products]
        else:
            return [p for c in categories for p in self._load_products_from_fn(c.id)]
        
    def get_product_with_impact(self, sustainedProductId:str):
        product = self._fetchProduct(id=sustainedProductId)
        impacts = self._fetchProductImpacts(product=product)
        return SustainedProductExplained(product=product,impacts=impacts)

    @cachetools.func.ttl_cache(maxsize=128, ttl=10 * 60)
    def get_category_ids(self, replace: tuple = ('', '')):
        self.useLocalStorage()
        with open('localstorage/sustained-categories.json', 'r') as f:
            sustainedLocalData = json.load(f)
            categories = sustainedLocalData['categories']
        return [c['id'].replace(replace[0],replace[1]) if replace and replace[0] != '' else c['id'] for c in categories]
    
    def get_cat_for_space_delimited_id(self, most_sim_cat_id: str, replace: tuple = ('', '')):
        self.useLocalStorage()
        with open('localstorage/sustained-categories.json', 'r') as f:
            sustainedLocalData = json.load(f)
            categories = sustainedLocalData['categories']
        return next((c for c in categories if (c['id'].replace(replace[0], replace[1]) if replace and replace[0] != '' else c['id']).lower() == most_sim_cat_id),None)
    
    def get_category_names(self):
        self.useLocalStorage()
        with open('localstorage/sustained-categories.json', 'r') as f:
            sustainedLocalData = json.load(f)
            categories = sustainedLocalData['categories']
        return [c['name'] for c in categories]
    
    
    
    
if __name__ == '__main__':
    # check functions work
    
    ss = SustainedAPI()
    yoghurts = ss.get_products(category_name='yoghurts')
    print(yoghurts)
    exit(0)
    
