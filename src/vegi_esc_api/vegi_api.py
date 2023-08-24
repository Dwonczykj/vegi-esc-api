from __future__ import annotations
from dataclasses import dataclass
import sys
import traceback

from flask import Flask
from datetime import datetime, timedelta
from typing import Tuple, NamedTuple, Any, Union, Self
from sqlalchemy import or_, func
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.engine.row import Row
from vegi_esc_api.vegi_repo_models import (
    # VegiESCSourceSql,
    # VegiESCExplanationSql,
    VegiESCRatingSql,
    VegiProductSql,
    VegiUserSql,
    VEGI_DB_NAMED_BIND,
    VegiUserInstance,
    VegiProductCategorySql,
    VegiProductCategoryInstance,
    VegiCategoryGroupSql,
)

from vegi_esc_api.models import (
    VegiESCRatingInstance,
    VegiVendorInstance,
    VegiProductInstance,
    # VegiESCExplanationInstance,
)
import vegi_esc_api.logger as Logger
from types import TracebackType
import asyncio
import aiohttp
from aiohttp import web
from typing import Optional, Protocol, Type
from dotenv import load_dotenv
import os
import vegi_esc_api.protocols as vegiP


async def get_users():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://jsonplaceholder.typicode.com/users') as response:
            users = await response.json()
            return users


async def get_products():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://fakestoreapi.com/products') as response:
            products = await response.json()
            return products


async def products_handler(request):
    products = await get_products()
    return web.json_response(products)


# ~ https://www.scrapingbee.com/blog/best-python-http-clients/#3-aiohttphttpsdocsaiohttporg
if os.environ.get('vegi_service_url') is None:
    load_dotenv()
# base_url = 'http://localhost:1337'
base_url = os.environ.get('vegi_service_url')
if base_url is None:
    base_url = 'https://qa-vegi.vegiapp.co.uk'


class EndPoints(Protocol):
    login = f'{base_url}/api/v1/admin/login-with-secret'
    allVendors = f'{base_url}/api/v1/vendors'
    allProducts = f'{base_url}/api/v1/products/'
    allUsers = f'{base_url}/admin/users'
    viewAccount = f'{base_url}/admin/account'
    updateESCSource = f'{base_url}/api/v1/admin/update-esc-source'
    getESCSources = f'{base_url}/api/v1/products/get-esc-sources'
    getProductRating = f'{base_url}/api/v1/products/get-product-rating'
    getProductCategoriesForVendor = f'{base_url}/api/v1/vendors/product-categories'  # vendor=vendorId
    getProduct = f'{base_url}/api/v1/products'  # productId is path parameter
    getProductOptions = f'{base_url}/api/v1/products/get-product-options/'  # itemId is path parameter

    
class VegiApiAuthenticatedSession:
    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> "VegiApiAuthenticatedSession":
        self.connection_state = 'ready'
        self._init_session()
        # await self.login()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.connection_state = 'closed'
        await self.close()

    async def close(self):
        try:
            if self.session:
                await self.session.close()
        except Exception as e:
            Logger.warn(f'VegiApiAuthenticatedSession.close errored -> "{e}"')
            pass
        
    def _init_session(self):
        if __name__ == '__main__' or os.environ.get('vegi_service_user_name') is None:
            load_dotenv()  # loads the .env file / heroku docker replicated config vars into env variables in the current shell profile ~ https://stackoverflow.com/a/34896938
        self.session = aiohttp.ClientSession(
            headers={
                'Accept': 'application/json',
                # 'origin': 'https://vegi-server.herokuapp.com',
                'api-key': os.environ.get('vegi_service_user_secret', ''),
                'api-secret': os.environ.get('vegi_service_user_secret', ''),
            }
        )
    
    async def login(self):
        if __name__ == '__main__' or os.environ.get('vegi_service_user_name') is None:
            load_dotenv()  # loads the .env file / heroku docker replicated config vars into env variables in the current shell profile ~ https://stackoverflow.com/a/34896938
        data = {
            'name': os.environ.get('vegi_service_user_name'),
            'secret': os.environ.get('vegi_service_user_secret')
        }
        self.session = aiohttp.ClientSession(
            headers={
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'origin': 'https://vegi-server.herokuapp.com',
                'api-key': os.environ.get('vegi_service_user_secret', ''),
                'api-secret': os.environ.get('vegi_service_user_secret', ''),
            }
        )
        
        async with self.session.post(EndPoints.login, params=data) as response:
            responseText = await response.text()
            if responseText != 'Unauthorized':
                self.connection_state = 'authenticated'
                # print(await response.json())
                # print(response.cookies)
                self.sessionCookies = response.cookies
                session_cookie = response.headers.get('set-cookie')
                if session_cookie is not None:
                    self.session.headers['Cookie'] = session_cookie
                # print(responseText)
            else:
                self.connection_state = 'failed'
                raise Warning(f'Unable to authenticate with vegi api [{base_url}]')
                
            # todo store session cookie from this response on the class object


@dataclass
class GetProductResponse:
    product: VegiProductInstance
    category: VegiProductCategoryInstance


class VegiApi:  # todo have same protocol as EndPoints
    async def updateESCSource(self, source: vegiP.ESCSource):
        data = source
        
    async def getESCSources(self, sourceType: vegiP.ESCSourceType):
        data = {
            'type': sourceType
        }
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(EndPoints.getESCSources, params=data) as response:
                esc_ratings = (await response.json())['sources']
                return [VegiESCRatingInstance.fromJson(r) for r in esc_ratings]
    
    async def get_vendors(self, outcode: str) -> list:
        data = {
            'outcode': outcode
        }
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(EndPoints.allVendors, params=data) as response:
                vendors = (await response.json())['vendors']
                return [VegiVendorInstance.fromJson(p) for p in vendors]
                
    async def get_products(self: Self@VegiApi, vendorId: int) -> list[VegiProductInstance]:
        data = {
            'vendorId': vendorId,
        }
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(EndPoints.allProducts, params=data) as response:
                products = (await response.json())['products']
                return [VegiProductInstance.fromJson(p) for p in products]
    
    async def get_product(self, product_id: int) -> GetProductResponse:
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(f'{EndPoints.getProduct}/{product_id}') as response:
                response = (await response.json())
                product = response['product']
                category = response['category']
                return GetProductResponse(
                    product=VegiProductInstance.fromJson(product),
                    category=VegiProductCategoryInstance.fromJson(category)
                )

    async def get_users(self: Self@VegiApi) -> list[VegiUserInstance]:
        data = {}
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(EndPoints.allUsers, headers={'Accept': 'application/json', 'origin': 'https://vegi-server.herokuapp.com'}, params=data) as response:
                content = str(response.content)
                users = (await response.json())['users']
                return [VegiUserInstance.fromJson(u) for u in users]
    
    async def view_account(self: Self@VegiApi) -> dict:
        data = {}
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(EndPoints.viewAccount, headers={'Accept': 'application/json', 'origin': 'https://vegi-server.herokuapp.com'}, params=data) as response:
                content = str(response.content)
                result = (await response.json())
                return result

    async def get_product_categories(self, vendor: int | None = None) -> list[VegiProductCategoryInstance]:
        data = {
            'vendor': vendor,
        } if vendor is not None else {}
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(EndPoints.getProductCategoriesForVendor, params=data) as response:
                product_categories = (await response.json())['productCategories']
                return [VegiProductCategoryInstance.fromJson(u) for u in product_categories]


# data = {"name": "Obi-Wan Kenobi"}

# async def main_example():
#     async with aiohttp.ClientSession() as session:
#         async with session.get('https://swapi.dev/api/starships/9/') as response:
#             print(await response.json())
#         async with session.post('https://httpbin.org/post', json=data) as response:
#             print(await response.json()) 

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main_example())


# async def main():
#     _vegiApi = VegiApi()
#     await _vegiApi.get_vendors(outcode='L1')
    

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main())
