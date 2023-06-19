from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.sql import func
from sqlalchemy import Column
import jsons

from sqlalchemy.dialects.postgresql import JSONB, JSON

# from sqlalchemy import JSONB
from sqlalchemy import cast
from typing import Protocol

from vegi_esc_api.extensions import db
from vegi_esc_api.repo_models_base import _DBDataClassJsonWrapped, _DBFetchable
from vegi_esc_api.models import (
    VegiESCRatingInstance,
    VegiESCExplanationInstance,
    VegiESCSourceInstance,
    VegiProductInstance,
    VegiUserInstance,
    VegiCategoryGroupInstance,
    VegiProductCategoryInstance,
)


VEGI_DB_NAMED_BIND = "vegi_bind"

# @dataclass
# class _DBDataClassJsonWrapped:
#     @classmethod
#     def fromJson(cls, obj: Any):
#         return jsons.load(obj, cls=cls)

#     def toJson(self) -> dict[str, Any] | list[Any] | int | str | bool | float:
#         x: Any = jsons.dump(self, strip_privates=True)
#         return x

#     def __copy__(self: Self) -> Self:
#         return copy(self)

#     def __deepcopy__(self: Self) -> Self:
#         # copy = type(self)()
#         # memo[id(self)] = copy
#         # copy._member1 = self._member1
#         # copy._member2 = deepcopy(self._member2, memo)
#         # return copy
#         memo: dict[int, Any] = {}
#         return deepcopy(self, memo)

#     def copyWith(self: Self, **kwargs: Any) -> Self:
#         c = self.__deepcopy__()
#         for k, v in kwargs.items():
#             if hasattr(c, k):
#                 setattr(c, k, v)
#         return c


class VegiUserProto(Protocol):
    id: int
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


# * Tables (models) in vegi backend
@dataclass
class VegiUserSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
    __tablename__ = "user"

    id: int = db.Column(db.Integer, primary_key=True)
    email: str = db.Column(db.String())
    role: str = db.Column(db.String())
    phoneCountryCode: float = db.Column(db.Float())
    phoneNoCountry: str = db.Column(db.String())
    marketingEmailContactAllowed: bool = db.Column(db.Boolean())
    marketingPushContactAllowed: bool = db.Column(db.Boolean())
    marketingPhoneContactAllowed: bool = db.Column(db.Boolean())
    name: str = db.Column(db.String())
    fbUid: str = db.Column(db.String())
    isSuperAdmin: bool = db.Column(db.Boolean())
    # calculated_on: datetime = db.Column(db.DateTime, default=func.now(), nullable=False)

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
        # self.calculated_on = calculated_on

    def __repr__(self):
        return (
            f"{type(self).__name__}<id {self.id}; name {self.name}; email {self.email}>"
        )

    def serialize(self):
        return self.toJson()

    def fetch(self) -> VegiUserInstance:
        return VegiUserInstance(
            id=self.id,
            email=self.email,
            role=self.role,
            phoneCountryCode=self.phoneCountryCode,
            phoneNoCountry=self.phoneNoCountry,
            marketingEmailContactAllowed=self.marketingEmailContactAllowed,
            marketingPushContactAllowed=self.marketingPushContactAllowed,
            marketingPhoneContactAllowed=self.marketingPhoneContactAllowed,
            name=self.name,
            fbUid=self.fbUid,
            isSuperAdmin=self.isSuperAdmin,
        )


class VegiProductProto(Protocol):
    """Used ([a-zA-Z]+:\s?[a-zA-Z]+) = db\.Column\(.*\) to regex replace from Sql Class"""

    id: int
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


class VegiCategoryGroupSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
    __tablename__ = "categorygroup"
    # todo: would be nice to extend @dataclass / copy it so that it can take the generic param from Column and use that as __init__ param type annotation and then wrap Column to take a generic param.
    id = Column(db.Integer, primary_key=True)
    name = Column(db.String())
    forRestaurantItem = Column(db.Boolean())

    def __init__(
        self,
        id: int,
        name: str,
        forRestaurantItem: bool,
    ):
        self.id = id
        self.name = name
        self.forRestaurantItem = forRestaurantItem

    def __repr__(self):
        return (
            f"{type(self).__name__}<id {self.id}; name {self.name}; email {self.email}>"
        )

    def serialize(self):
        return self.toJson()

    def fetch(self):
        return VegiCategoryGroupInstance(
            id=self.id if isinstance(self.id, int) else self.id.data,
            name=self.name,
            forRestaurantItem=self.forRestaurantItem,
        )


class VegiProductCategorySql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
    __tablename__ = "productcategory"

    id: int = db.Column(db.Integer, primary_key=True)
    name: str = db.Column(db.String())
    vendor: int = db.Column(db.Integer())
    categoryGroup: int = db.Column(db.Integer())

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

        # self.calculated_on = calculated_on

    def __repr__(self):
        return (
            f"{type(self).__name__}<id {self.id}; name {self.name}; email {self.email}>"
        )

    def serialize(self):
        return self.toJson()

    def fetch(self):
        return VegiProductCategoryInstance(
            id=self.id,
            name=self.name,
            vendor=self.vendor,
            categoryGroup=self.categoryGroup,
        )


@dataclass
class VegiProductSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
    __tablename__ = "product"

    id: int = db.Column(db.Integer, primary_key=True)
    name: str = db.Column(db.String())
    description: str = db.Column(db.String())
    basePrice: float = db.Column(db.Float())
    isAvailable: bool = db.Column(db.Boolean())
    isFeatured: bool = db.Column(db.Boolean())
    status: str = db.Column(db.String())
    ingredients: str = db.Column(db.String(), nullable=True)
    stockCount: int = db.Column(db.Integer())
    supplier: str = db.Column(db.String(), nullable=True)
    brandName: str = db.Column(db.String(), nullable=True)
    taxGroup: str = db.Column(db.String(), nullable=True)
    stockUnitsPerProduct: int = db.Column(db.Integer())
    sizeInnerUnitValue: int = db.Column(db.Integer())
    sizeInnerUnitType: str = db.Column(db.String())
    productBarCode: str = db.Column(db.String(), nullable=True)
    vendor: int = db.Column(db.Integer(), nullable=True)
    category: int = db.Column(db.Integer(), nullable=True)

    # calculated_on: datetime = db.Column(db.DateTime, default=func.now(), nullable=False)

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

        # self.calculated_on = calculated_on

    def __repr__(self):
        return (
            f"{type(self).__name__}<id {self.id}; name {self.name}; email {self.email}>"
        )

    def serialize(self):
        return self.toJson()

    def fetch(self):
        return VegiProductInstance(
            id=self.id,
            name=self.name,
            description=self.description,
            basePrice=self.basePrice,
            isAvailable=self.isAvailable,
            isFeatured=self.isFeatured,
            status=self.status,
            ingredients=self.ingredients,
            stockCount=self.stockCount,
            supplier=self.supplier,
            brandName=self.brandName,
            taxGroup=self.taxGroup,
            stockUnitsPerProduct=self.stockUnitsPerProduct,
            sizeInnerUnitValue=self.sizeInnerUnitValue,
            sizeInnerUnitType=self.sizeInnerUnitType,
            productBarCode=self.productBarCode,
            vendor=self.vendor,
            category=self.category,
        )


class VegiESCRatingProto(Protocol):
    """Used ([a-zA-Z]+:\s?[a-zA-Z]+) = db\.Column\(.*\) to regex replace from Sql Class"""

    id: int
    productPublicId: str
    rating: float
    calculatedOn: datetime
    product: int


@dataclass
class VegiESCRatingSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
    __tablename__ = "escrating"

    id: int | None = db.Column(db.Integer, primary_key=True)
    productPublicId: str = db.Column(db.String())
    rating: float = db.Column(db.Float())
    calculatedOn: datetime = db.Column(db.DateTime, default=func.now(), nullable=False)
    product: int = db.Column(db.Integer())

    # Relationship with VegiESCExplanations
    # ! explanations = db.relationship('VegiESCExplanation', backref='escrating', lazy=True)

    def __init__(
        self,
        productPublicId: str,
        rating: float,
        calculatedOn: datetime,
        product: int,
        id: int | None = None,
    ):
        self.id = id
        self.productPublicId = productPublicId
        self.rating = rating
        self.calculatedOn = calculatedOn
        self.product = product

    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; rating: {self.rating}; productPublicId {self.productPublicId}>"

    def serialize(self):
        return self.toJson()

    def fetch(self):
        assert (
            self.id is not None
        ), f"{type(self).__name__} cannot have an id of None when fetching to an instance."
        return VegiESCRatingInstance(
            id=self.id,
            productPublicId=self.productPublicId,
            rating=self.rating,
            calculatedOn=self.calculatedOn,
            product=self.product,
        )


class VegiESCExplanationProto(Protocol):
    """Used ([a-zA-Z]+:\s?[a-zA-Z]+) = db\.Column\(.*\) to regex replace from Sql Class"""

    id: int
    title: str
    measure: float
    reasons: list[str]
    evidence: str
    escrating: int
    escsource: int


@dataclass
class VegiESCExplanationSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
    __tablename__ = "escexplanation"

    id: int | None = db.Column(db.Integer, primary_key=True)
    title: str = db.Column(db.String())
    measure: float = db.Column(db.Float())
    # ~ https://stackoverflow.com/a/71044396
    # NOTE from sqlalchemy.dialects.postgresql import ARRAY, JSON
    # ~ https://docs.sqlalchemy.org/en/13/core/type_basics.html
    # ~ https://docs.sqlalchemy.org/en/13/core/type_basics.html#sqlalchemy.types.JSON
    reasons: list[str] = db.Column(JSON())
    evidence: str = db.Column(JSON())
    # ~ https://stackoverflow.com/a/38793154
    escrating: int = db.Column(
        db.Integer(), db.ForeignKey("escrating.id"), nullable=False
    )
    escsource: int = db.Column(
        db.Integer(), db.ForeignKey("escsource.id"), nullable=False
    )

    def __init__(
        self,
        title: str,
        measure: float,
        reasons: list[str],
        evidence: str,
        escrating: int,
        escsource: int,
        id: int | None = None,
    ):
        self.id = id
        self.title = title
        self.measure = measure
        # self.reasons = jsons.dumps(reasons)
        # self.reasons = cast(reasons, JSONB)
        self.reasons = reasons
        self.evidence = evidence
        self.escrating = escrating
        self.escsource = escsource

    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; title {self.title}>"

    def serialize(self):
        return self.toJson()

    def fetch(self):
        assert (
            self.id is not None
        ), f"{type(self).__name__} cannot have an id of None when fetching to an instance."
        return VegiESCExplanationInstance(
            id=self.id,
            title=self.title,
            measure=self.measure,
            reasons=self.reasons,
            evidence=self.evidence,
            escrating=self.escrating,
            escsource=self.escsource,
        )


class VegiESCSourceProto(Protocol):
    """Used ([a-zA-Z]+:\s?[a-zA-Z]+) = db\.Column\(.*\) to regex replace from Sql Class"""

    id: int
    name: str
    source_type: str
    domain: str
    credibility: float


@dataclass
class VegiESCSourceSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
    __tablename__ = "escsource"

    id: int = db.Column(db.Integer, primary_key=True)
    name: str = db.Column(db.String())
    source_type: str = db.Column(db.String(), name="type")
    domain: str = db.Column(db.String())
    credibility: float = db.Column(db.Float())

    # Relationship with VegiESCExplanations
    # ! sources = db.relationship('VegiESCExplanation', backref='escsource', lazy=True)

    def __init__(
        self, id: int, name: str, source_type: str, domain: str, credibility: float
    ):
        self.id = id
        self.name = name
        self.source_type = source_type
        self.domain = domain
        self.credibility = credibility

    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; name {self.name}>"

    def serialize(self):
        return self.toJson()

    def fetch(self):
        return VegiESCSourceInstance(
            id=self.id,
            name=self.name,
            source_type=self.source_type,
            domain=self.domain,
            credibility=self.credibility,
        )
