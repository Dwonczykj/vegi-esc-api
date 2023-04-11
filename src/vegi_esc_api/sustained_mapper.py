from dataclasses import dataclass
from datetime import datetime
from typing import Callable
import re

from vegi_esc_api.protocols import ESCExplanation, ESCProduct, ESCRating, ESCRatingExplained
from vegi_esc_api.sustained import SustainedAPI, SustainedProductBase

@dataclass
class UnitAmount:
    units:str
    amount:float

def parseProductUnitsAmount(unitsAmountStr:str):
    pattern = r'(?P<amount>[0-9.,]+)\s?(?P<units>[A-Za-z]+)'
    statement = unitsAmountStr
    match:re.Match | None = re.match(pattern, statement)
    
    if match is None:
        return UnitAmount('g',0)
    
    md = match.groupdict()
    if 'units' in md.keys() and 'amount' in md.keys():
        units = str(md['units'])
        amount = float(md['amount'])
        return UnitAmount(units=units, amount=amount)
    else:
        return UnitAmount('g',0)
        

class SustainedVegiMapper():
    
    ss=SustainedAPI()
    validGrades = [
        'G',
        'F',
        'E',
        'D',
        'C',
        'B',
        'A',
    ]
    
    def mapSustainedProductGradeToScale(self, productGrade:str, lowerBoundOutScale:float,upperBoundOutScale:float):
        assert (sustainedGradeIndex := self.validGrades.index(productGrade)) > -1, f"Unknown product Grade received from Sustained API of [{productGrade}]"
        assert lowerBoundOutScale < upperBoundOutScale, ValueError('upperbound must be STRICTLY greater than lowerbound of out scale')
        n = len(self.validGrades)
        U = upperBoundOutScale - lowerBoundOutScale
        return ((sustainedGradeIndex / (n-1)) * U) + lowerBoundOutScale
    
    def getProductRatingWithExplanations(
        self,
        sourceProductId:str
        ):
        # convert a sustained rated product to a vegi rating object for the vegi backend to consume by request
        sourceProductRating = self.ss.get_product_with_impact(sustainedProductId=sourceProductId)
        parsedUnits = parseProductUnitsAmount(sourceProductRating.product.pack)
        return ESCRatingExplained(
            rating=ESCRating(
                calculatedOn=datetime.now(),
                product=None,
                rating=self.mapSustainedProductGradeToScale(
                    productGrade=sourceProductRating.product.grade,
                    lowerBoundOutScale=0, 
                    upperBoundOutScale=5,
                ),
                proxyForVegiProduct=ESCProduct(
                    name=sourceProductRating.product.name,
                    category=sourceProductRating.product.category,
                    vendorInternalId=sourceProductRating.product.id,
                    ingredients='',
                    productBarCode=sourceProductRating.product.gtin,
                    imageUrl=sourceProductRating.product.image,
                    sizeInnerUnitType=parsedUnits.units,
                    sizeInnerUnitValue=parsedUnits.amount,
                    description='',
                    shortDescription='',
                    basePrice=None,
                    brandName=None,
                    isAvailable=False,
                    isFeatured=False,
                    priority=None,
                    status='Active',
                    stockCount=0,
                    stockUnitsPerProduct=1,
                    supplier='',
                    taxGroup='',
                ),
                createdAt=None,
                productPublicId=None,
            ),
            explanations=[
                ESCExplanation(
                    title=x.title,
                    reasons=[x.description],
                    measure=self.mapSustainedProductGradeToScale(x.grade,0,5),
                    evidence='',
                    escsource='sustained'
                ) for x in sourceProductRating.impacts
            ],
        )