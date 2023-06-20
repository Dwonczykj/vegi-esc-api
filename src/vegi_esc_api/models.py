from dataclasses import dataclass
from typing import Any, Self, Final
from datetime import datetime, timedelta
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY
import jsons
from copy import copy, deepcopy
from flask_login import UserMixin
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

from vegi_esc_api.extensions import db
from vegi_esc_api.repo_models_base import (
    _DBDataClassJsonWrapped,
    _isDBInstance,
    _isCreate,
)
from vegi_esc_api.extensions import login


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return "<User {}>".format(self.username)


def with_id(cls):
    # ~ https://stackoverflow.com/a/40610345
    id: int | None = getattr(cls, "id", None)
    if id is None:
        cls.id = -1
    return cls


# @dataclass
# class VegiESCSource:
#     name: str
#     source_type: str
#     domain: str
#     credibility: float


# @dataclass
# class VegiESCSourceCreate(VegiESCSource, _DBDataClassJsonWrapped, _isCreate):
#     def __init__(self, name: str, source_type: str, domain: str, credibility: float):
#         self.name = name
#         self.source_type = source_type
#         self.domain = domain
#         self.credibility = credibility

#     # def __repr__(self):
#     #     return f"{type(self).__name__}<name {self.name}>"


# @dataclass
# class VegiESCSourceInstance(VegiESCSource, _DBDataClassJsonWrapped, _isDBInstance):
#     def __init__(
#         self, id: int, name: str, source_type: str, domain: str, credibility: float
#     ):
#         self.id = id
#         self.name = name
#         self.source_type = source_type
#         self.domain = domain
#         self.credibility = credibility


@dataclass
class VegiESCRating:
    escRatingId: str
    rating: float
    calculatedOn: datetime
    product: int


@dataclass
class VegiESCRatingCreate(VegiESCRating, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self, escRatingId: str, rating: float, calculatedOn: datetime, product: int
    ):
        self.escRatingId = escRatingId
        self.rating = rating
        self.calculatedOn = calculatedOn
        self.product = product


@dataclass
class VegiESCRatingInstance(VegiESCRating, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self,
        id: int,
        escRatingId: str,
        rating: float,
        calculatedOn: datetime,
        product: int,
    ):
        self.id = id
        self.escRatingId = escRatingId
        self.rating = rating
        self.calculatedOn = calculatedOn
        self.product = product


# @dataclass
# class VegiESCExplanation:
#     title: str
#     measure: float
#     reasons: list[str]
#     evidence: str
#     escrating: int
#     escsource: int


# @dataclass
# class VegiESCExplanationCreate(VegiESCExplanation, _DBDataClassJsonWrapped, _isCreate):
#     def __init__(
#         self,
#         title: str,
#         measure: float,
#         reasons: list[str],
#         evidence: str,
#         escrating: int,
#         escsource: int,
#     ):
#         self.title = title
#         self.measure = measure
#         self.reasons = reasons
#         self.evidence = evidence
#         self.escrating = escrating
#         self.escsource = escsource


# @dataclass
# class VegiESCExplanationInstance(
#     VegiESCExplanation, _DBDataClassJsonWrapped, _isDBInstance
# ):
#     def __init__(
#         self,
#         id: int,
#         title: str,
#         measure: float,
#         reasons: list[str],
#         evidence: str,
#         escrating: int,
#         escsource: int,
#     ):
#         self.id = id
#         self.title = title
#         self.measure = measure
#         self.reasons = reasons
#         self.evidence = evidence
#         self.escrating = escrating
#         self.escsource = escsource


@dataclass
class VegiProduct:
    name: str
    description: str
    basePrice: float
    isAvailable: bool
    isFeatured: bool
    status: str
    ingredients: str
    stockCount: int
    supplier: str
    brandName: str
    taxGroup: str
    stockUnitsPerProduct: int
    sizeInnerUnitValue: int
    sizeInnerUnitType: str
    productBarCode: str
    vendor: int
    category: int


@dataclass
class VegiProductCreate(VegiProduct, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self,
        name: str,
        description: str,
        basePrice: float,
        isAvailable: bool,
        isFeatured: bool,
        status: str,
        ingredients: str,
        stockCount: int,
        supplier: str,
        brandName: str,
        taxGroup: str,
        stockUnitsPerProduct: int,
        sizeInnerUnitValue: int,
        sizeInnerUnitType: str,
        productBarCode: str,
        vendor: int,
        category: int,
        # calculated_on: datetime,
    ):
        self.name = name
        self.description = description
        self.basePrice = basePrice
        self.isAvailable = isAvailable
        self.isFeatured = isFeatured
        self.status = status
        self.ingredients = ingredients
        self.stockCount = stockCount
        self.supplier = supplier
        self.brandName = brandName
        self.taxGroup = taxGroup
        self.stockUnitsPerProduct = stockUnitsPerProduct
        self.sizeInnerUnitValue = sizeInnerUnitValue
        self.sizeInnerUnitType = sizeInnerUnitType
        self.productBarCode = productBarCode
        self.vendor = vendor
        self.category = category


@dataclass
class VegiProductInstance(VegiProduct, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self,
        id: int,
        name: str,
        description: str,
        basePrice: float,
        isAvailable: bool,
        isFeatured: bool,
        status: str,
        ingredients: str,
        stockCount: int,
        supplier: str,
        brandName: str,
        taxGroup: str,
        stockUnitsPerProduct: int,
        sizeInnerUnitValue: int,
        sizeInnerUnitType: str,
        productBarCode: str,
        vendor: int,
        category: int,
        # calculated_on: datetime,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.basePrice = basePrice
        self.isAvailable = isAvailable
        self.isFeatured = isFeatured
        self.status = status
        self.ingredients = ingredients
        self.stockCount = stockCount
        self.supplier = supplier
        self.brandName = brandName
        self.taxGroup = taxGroup
        self.stockUnitsPerProduct = stockUnitsPerProduct
        self.sizeInnerUnitValue = sizeInnerUnitValue
        self.sizeInnerUnitType = sizeInnerUnitType
        self.productBarCode = productBarCode
        self.vendor = vendor
        self.category = category
        
        
@dataclass
class VegiProductCategory:
    name: str
    vendor: int
    categoryGroup: int


@dataclass
class VegiProductCategoryCreate(VegiProductCategory, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self,
        name: str,
        vendor: int,
        categoryGroup: int,
    ):
        self.name = name
        self.vendor = vendor
        self.categoryGroup = categoryGroup


@dataclass
class VegiProductCategoryInstance(VegiProductCategory, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self,
        id: int,
        name: str,
        vendor: int,
        categoryGroup: int,
    ):
        self.id = id
        self.name = name
        self.vendor = vendor
        self.categoryGroup = categoryGroup
        
        
@dataclass
class VegiCategoryGroup:
    name: str
    forRestaurantItem: bool


@dataclass
class VegiCategoryGroupCreate(VegiCategoryGroup, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self,
        name: str,
        forRestaurantItem: bool,
    ):
        self.name = name
        self.forRestaurantItem = forRestaurantItem


@dataclass
class VegiCategoryGroupInstance(VegiCategoryGroup, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self,
        id: int,
        name: str,
        forRestaurantItem: bool,
    ):
        self.id = id
        self.name = name
        self.forRestaurantItem = forRestaurantItem


class VegiUser:
    email: str
    role: str
    phoneCountryCode: float
    phoneNoCountry: str
    marketingEmailContactAllowed: bool
    marketingPushContactAllowed: bool
    marketingPhoneContactAllowed: bool
    name: str
    fbUid: str
    isSuperAdmin: bool


@dataclass
class VegiUserCreate(VegiUser, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self,
        email: str,
        role: str,
        phoneCountryCode: float,
        phoneNoCountry: str,
        marketingEmailContactAllowed: bool,
        marketingPushContactAllowed: bool,
        marketingPhoneContactAllowed: bool,
        name: str,
        fbUid: str,
        isSuperAdmin: bool,
    ):
        self.email = email
        self.role = role
        self.phoneCountryCode = phoneCountryCode
        self.phoneNoCountry = phoneNoCountry
        self.marketingEmailContactAllowed = marketingEmailContactAllowed
        self.marketingPushContactAllowed = marketingPushContactAllowed
        self.marketingPhoneContactAllowed = marketingPhoneContactAllowed
        self.name = name
        self.fbUid = fbUid
        self.isSuperAdmin = isSuperAdmin


@dataclass
class VegiUserInstance(VegiUser, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self,
        id: int,
        email: str,
        role: str,
        phoneCountryCode: float,
        phoneNoCountry: str,
        marketingEmailContactAllowed: bool,
        marketingPushContactAllowed: bool,
        marketingPhoneContactAllowed: bool,
        name: str,
        fbUid: str,
        isSuperAdmin: bool,
        # calculated_on: datetime,
    ):
        self.id = id
        self.email = email
        self.role = role
        self.phoneCountryCode = phoneCountryCode
        self.phoneNoCountry = phoneNoCountry
        self.marketingEmailContactAllowed = marketingEmailContactAllowed
        self.marketingPushContactAllowed = marketingPushContactAllowed
        self.marketingPhoneContactAllowed = marketingPhoneContactAllowed
        self.name = name
        self.fbUid = fbUid
        self.isSuperAdmin = isSuperAdmin


# * ESC_DB_Classes
class ESCSource:
    name: str
    source_type: str
    domain: str
    credibility: float


@dataclass
class ESCSourceCreate(ESCSource, _DBDataClassJsonWrapped, _isCreate):
    def __init__(self, name: str, source_type: str, domain: str, credibility: float):
        self.name = name
        self.source_type = source_type
        self.domain = domain
        self.credibility = credibility


@dataclass
class ESCSourceInstance(ESCSource, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self, id: int, name: str, source_type: str, domain: str, credibility: float
    ):
        self.id = id
        self.name = name
        self.source_type = source_type
        self.domain = domain
        self.credibility = credibility


class ESCRating:
    product: int
    product_name: str
    product_id: str
    rating: float
    calculated_on: datetime


@dataclass
class ESCRatingCreate(ESCRating, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self, product: int, product_name: str, product_id: str, rating: float, calculated_on: datetime
    ):
        self.product = product
        self.product_name = product_name
        self.product_id = product_id
        self.rating = rating
        self.calculated_on = calculated_on


@dataclass
class ESCRatingInstance(ESCRating, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self,
        id: int,
        product: int,
        product_name: str,
        product_id: str,
        rating: float,
        calculated_on: datetime,
    ):
        self.id = id
        self.product = product
        self.product_name = product_name
        self.product_id = product_id
        self.rating = rating
        self.calculated_on = calculated_on


class ESCProduct:
    name: str
    product_external_id_on_source: str
    source: int
    description: str
    category: str
    keyWords: list[str]
    imageUrl: str
    ingredients: str
    packagingType: str
    stockUnitsPerProduct: int
    sizeInnerUnitValue: float
    sizeInnerUnitType: str
    productBarCode: str
    supplier: str
    brandName: str
    origin: str
    taxGroup: str
    dateOfBirth: datetime


@dataclass
class ESCProductCreate(ESCProduct, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self,
        name: str,
        product_external_id_on_source: str,
        source: int,
        description: str,
        category: str,
        keyWords: list[str],
        imageUrl: str,
        ingredients: str,
        packagingType: str,
        stockUnitsPerProduct: int,
        sizeInnerUnitValue: float,
        sizeInnerUnitType: str,
        productBarCode: str,
        supplier: str,
        brandName: str,
        origin: str,
        taxGroup: str,
        dateOfBirth: datetime,
    ):
        self.name = name
        self.product_external_id_on_source = product_external_id_on_source
        self.source = source
        self.description = description
        self.category = category
        self.keyWords = keyWords
        self.imageUrl = imageUrl
        self.ingredients = ingredients
        self.packagingType = packagingType
        self.stockUnitsPerProduct = stockUnitsPerProduct
        self.sizeInnerUnitValue = sizeInnerUnitValue
        self.sizeInnerUnitType = sizeInnerUnitType
        self.productBarCode = productBarCode
        self.supplier = supplier
        self.brandName = brandName
        self.origin = origin
        self.taxGroup = taxGroup
        self.dateOfBirth = dateOfBirth


@dataclass
class ESCProductInstance(ESCProduct, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self,
        id: int,
        name: str,
        product_external_id_on_source: str,
        source: int,
        description: str,
        category: str,
        keyWords: list[str],
        imageUrl: str,
        ingredients: str,
        packagingType: str,
        stockUnitsPerProduct: int,
        sizeInnerUnitValue: float,
        sizeInnerUnitType: str,
        productBarCode: str,
        supplier: str,
        brandName: str,
        origin: str,
        taxGroup: str,
        dateOfBirth: datetime,
    ):
        self.id = id
        self.name = name
        self.product_external_id_on_source = product_external_id_on_source
        self.source = source
        self.description = description
        self.category = category
        self.keyWords = keyWords
        self.imageUrl = imageUrl
        self.ingredients = ingredients
        self.packagingType = packagingType
        self.stockUnitsPerProduct = stockUnitsPerProduct
        self.sizeInnerUnitValue = sizeInnerUnitValue
        self.sizeInnerUnitType = sizeInnerUnitType
        self.productBarCode = productBarCode
        self.supplier = supplier
        self.brandName = brandName
        self.origin = origin
        self.taxGroup = taxGroup
        self.dateOfBirth = dateOfBirth


class ESCExplanation:
    title: str
    measure: float
    reasons: list[str]
    evidence: str
    rating: int
    source: int


@dataclass
class ESCExplanationCreate(ESCExplanation, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self,
        title: str,
        measure: float,
        reasons: list[str],
        evidence: str,
        # rating: int,
        # source: int,
    ):
        self.title = title
        self.measure = measure
        self.reasons = reasons
        self.evidence = evidence
        # self.rating = rating
        # self.source = source


@dataclass
class ESCExplanationInstance(ESCExplanation, _DBDataClassJsonWrapped, _isDBInstance):
    def __init__(
        self,
        id: int,
        title: str,
        measure: float,
        reasons: list[str],
        evidence: str,
        rating: int,
        source: int,
    ):
        self.id = id
        self.title = title
        self.measure = measure
        self.reasons = reasons
        self.evidence = evidence
        self.rating = rating
        self.source = source


class CachedItem:
    item_name: str
    item_type: str
    item_source: str
    item_json: str
    ttl_days: int
    created_on: datetime
    expires_on: datetime


@dataclass
class CachedItemCreate(CachedItem, _DBDataClassJsonWrapped, _isCreate):
    def __init__(
        self,
        item_name: str,
        item_type: str,
        item_source: str,
        item_json: str,
        ttl_days: int,
        created_on: datetime = datetime.now(),
    ):
        self.item_name = item_name
        self.item_type = item_type
        self.item_source = item_source
        self.item_json = item_json
        self.created_on = created_on
        self.ttl_days = ttl_days
        self.expires_on = self.created_on + timedelta(days=ttl_days)


@dataclass
class CachedItemInstance(CachedItem, _DBDataClassJsonWrapped, _isDBInstance):
    
    def __init__(
        self,
        id: int,
        item_name: str,
        item_type: str,
        item_source: str,
        item_json: str,
        ttl_days: int,
        created_on: datetime = datetime.now(),
    ):
        self.id = id
        self.item_name = item_name
        self.item_type = item_type
        self.item_source = item_source
        self.item_json = item_json
        self.created_on = created_on
        self.ttl_days = ttl_days
        self.expires_on = self.created_on + timedelta(days=ttl_days)


@dataclass
class ServerError:
    message: str
    code: str

    def serialize(self):
        return {
            'message': self.message,
            'code': self.code,
        }
