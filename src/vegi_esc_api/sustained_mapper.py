from dataclasses import dataclass
from datetime import datetime
from typing import Any
import re

from vegi_esc_api.models import ESCRatingCreate, ESCExplanationCreate
from vegi_esc_api.models_wrapper import ESCRatingExplained
from vegi_esc_api.sustained_models import (
    SustainedProductBase,
    SustainedProductExplained,
)


@dataclass
class UnitAmount:
    units: str
    amount: float

    def __init__(self, units: str, amount: float) -> None:
        self.units = units
        self.amount = amount


def parseProductUnitsAmount(unitsAmountStr: str):
    pattern = r"(?P<amount>[0-9., ]+)\s?(?P<units>[A-Za-z]+)"
    statement = unitsAmountStr
    match: re.Match | None = re.match(pattern, statement)

    if match is None:
        return UnitAmount("g", 0)

    md = match.groupdict()
    if "units" in md.keys() and "amount" in md.keys():
        units = str(md["units"])
        amount = float(md["amount"])
        return UnitAmount(units=units, amount=amount)
    else:
        return UnitAmount("g", 0)


class SustainedVegiMapper:
    validGrades = [
        "G",
        "F",
        "E",
        "D",
        "C",
        "B",
        "A",
    ]

    def mapSustainedProductGradeToScale(
        self, productGrade: str, lowerBoundOutScale: float, upperBoundOutScale: float
    ):
        assert (
            sustainedGradeIndex := self.validGrades.index(productGrade)
        ) > -1, f"Unknown product Grade received from Sustained API of [{productGrade}]"
        assert lowerBoundOutScale < upperBoundOutScale, ValueError(
            "upperbound must be STRICTLY greater than lowerbound of out scale"
        )
        n = len(self.validGrades)
        U = upperBoundOutScale - lowerBoundOutScale
        return ((sustainedGradeIndex / (n - 1)) * U) + lowerBoundOutScale

    def getProductUnits(self, product: SustainedProductBase):
        return parseProductUnitsAmount(product.pack)

    def mapSustainedProductImpactsToVegi(
        self,
        sourceProductRated: SustainedProductExplained[Any],
    ):
        ''' Create ESCRatings and Explanations from sustained API response to be saved to the DB'''
        return ESCRatingExplained(
            rating=ESCRatingCreate(
                calculated_on=datetime.now(),
                rating=self.mapSustainedProductGradeToScale(
                    productGrade=sourceProductRated.product.grade,
                    lowerBoundOutScale=0,
                    upperBoundOutScale=5,
                ),
                product=sourceProductRated.db_product.id,
                product_name=sourceProductRated.product.name,
                product_id=sourceProductRated.product.id,
                # calculatedOn=datetime.now(),
                # product=None,
                # proxyForVegiProduct=ESCProduct(
                #     name=sourceProductRated.product.name,
                #     category=sourceProductRated.product.category,
                #     vendorInternalId=sourceProductRated.product.id,
                #     ingredients='',
                #     productBarCode=sourceProductRated.product.gtin,
                #     imageUrl=sourceProductRated.product.image,
                #     sizeInnerUnitType=parsedUnits.units,
                #     sizeInnerUnitValue=parsedUnits.amount,
                #     description='',
                #     shortDescription='',
                #     basePrice=None,
                #     brandName=None,
                #     isAvailable=False,
                #     isFeatured=False,
                #     priority=None,
                #     status='Active',
                #     stockCount=0,
                #     stockUnitsPerProduct=1,
                #     supplier='',
                #     taxGroup='',
                # ),
                # createdAt=None,
                # productPublicId=None,
            ),
            explanations=[
                ESCExplanationCreate(
                    title=x.title,
                    reasons=[x.description],
                    measure=self.mapSustainedProductGradeToScale(x.grade, 0, 5),
                    evidence="",
                    # rating=-1,
                    # source=-1,  # ! FIX THIS BUG - Foreign Key not allowed!
                )
                for x in sourceProductRated.impacts
            ],
        )
