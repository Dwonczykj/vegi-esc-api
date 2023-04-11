from __future__ import annotations
from dataclasses import dataclass

from datetime import datetime
# import os
# import pandas as pd
# import numpy as np
# import re
# from typing import Annotated
# from typing import Self #! available from python 3.11
from typing import Literal, Optional, Protocol, Any

from vegi_esc_api.sustained import SustainedProductBase
from vegi_esc_api.json_class_helpers import DataClassJsonWrapped

@dataclass
class IngredientComposition(DataClassJsonWrapped):
    values: list[str]
    value: float | Literal[0]
    checks: dict[str, bool]

@dataclass
class Ingredient(DataClassJsonWrapped):
    main_ingredient: str
    embedded_ingredients: dict[int,
                               EmbeddedIngredientDetails | IngredientDetails]

@dataclass
class IngredientDetails(DataClassJsonWrapped):
    order: int
    composition: IngredientComposition
    is_embedded: bool
    ingredient: str

@dataclass
class EmbeddedIngredientDetails(DataClassJsonWrapped):
    order: int
    is_embedded: bool
    composition: IngredientComposition
    ingredient: Ingredient


ESCSourceType = Literal['api'] | Literal['database'] | Literal['webpage']


@dataclass
class ESCSource(DataClassJsonWrapped):
    name: str
    type: str | ESCSourceType
    domain: str
    credibility: float  # Annotated[float, ValueRange(0, 1)]

JsonEncodedStr = str

@dataclass
class ESCExplanation(DataClassJsonWrapped):
    title: str
    reasons: list[str]
    measure: float  # Annotated[float, ValueRange(0, 5)]
    evidence: JsonEncodedStr # json encoded string
    # escrating: Optional[int]
    escsource: Optional[str]
    
@dataclass
class ESCProduct(DataClassJsonWrapped):
    name:str
    category:str
    description:Optional[str]
    shortDescription:Optional[str]
    basePrice: Optional[float]
    imageUrl: Optional[str]
    isAvailable: Optional[bool]
    priority: Optional[float]
    isFeatured: Optional[bool]
    status: str
    ingredients: Optional[str]
    vendorInternalId: Optional[str]
    stockCount: Optional[float]
    stockUnitsPerProduct: Optional[float]
    sizeInnerUnitValue: Optional[float]
    sizeInnerUnitType: Optional[str]
    productBarCode: Optional[str]
    supplier: Optional[str]
    brandName: Optional[str]
    taxGroup: Optional[str]
    

@dataclass
class ESCRating(DataClassJsonWrapped):
    rating: float
    calculatedOn: datetime
    createdAt: Optional[datetime]
    productPublicId: Optional[str]
    product: Optional[int]
    proxyForVegiProduct: ESCProduct
    # what do we do when we have a sustained prduct that isnt in the vegi db that we are using as a proxuy for our rating?


@dataclass
class ESCRatingExplained(DataClassJsonWrapped):
    rating: ESCRating
    explanations: list[ESCExplanation]
    

@dataclass
class ESCRatingExplainedResult(ESCRatingExplained):
    original_search_term:str
    wmdistance:float
    _sustainedProduct:SustainedProductBase