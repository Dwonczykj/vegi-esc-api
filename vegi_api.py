from types import TracebackType
import aiohttp
import asyncio
import protocols as vegiP
from typing import Optional, Protocol, Type
from dotenv import load_dotenv
import os


# ~ https://www.scrapingbee.com/blog/best-python-http-clients/#3-aiohttphttpsdocsaiohttporg

base_url = 'https://qa-vegi.vegiapp.co.uk'
base_url = 'http://localhost:1337'

class EndPoints(Protocol):
    login = f'{base_url}/api/v1/admin/login-with-secret'
    allVendors = f'{base_url}/api/v1/vendors'
    updateESCSource = f'{base_url}/api/v1/admin/update-esc-source'
    getESCSources = f'{base_url}/api/v1/products/get-esc-sources'
    getProductRating = f'{base_url}/api/v1/products/get-product-rating'
    
class VegiApiAuthenticatedSession:
    # def __init__(self) -> None:
    #     self.session:aiohttp.ClientSession|None=None
        
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
        await self.login()
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
        except:
            pass
        

    async def login(self):
        load_dotenv()  # loads the .env file into env variables in the current shell profile
        data = {
            'name': os.getenv('vegi_service_user_name'),
            'secret': os.getenv('vegi_service_user_secret')
        }
        self.session = aiohttp.ClientSession()
        
        async with self.session.post(EndPoints.login, params=data) as response:
            responseText = await response.text()
            if responseText != 'Unauthorized':
                self.connection_state = 'authenticated'
                print(await response.json())
                print(response.cookies)
                self.sessionCookies = response.cookies
                print(responseText)
            else:
                self.connection_state = 'failed'
                raise Warning('Unable to authenticate with vegi api')
                
            # todo store session cookie from this response on the class object
    
    
class VegiApi: #todo have same protocol as EndPoints
    async def updateESCSource(self,source:vegiP.ESCSource):
        data = source
        
    
    async def getESCSources(self, sourceType: vegiP.ESCSourceType):
        data = {
            'type':sourceType
        }
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(EndPoints.getESCSources, params=data) as response:
                print(await response.json())
    
    async def getVendors(self, outcode:str):
        data = {
            'outcode':outcode
        }
        async with VegiApiAuthenticatedSession() as vApi:
            async with vApi.session.get(EndPoints.allVendors, params=data) as response:
                print(await response.json())


# data = {"name": "Obi-Wan Kenobi"}

# async def main_example():
#     async with aiohttp.ClientSession() as session:
#         async with session.get('https://swapi.dev/api/starships/9/') as response:
#             print(await response.json())
#         async with session.post('https://httpbin.org/post', json=data) as response:
#             print(await response.json()) 

# loop = asyncio.get_event_loop()
# loop.run_until_complete(main_example())


async def main():
    _vegiApi = VegiApi()
    await _vegiApi.getVendors(outcode='L1')
    

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
