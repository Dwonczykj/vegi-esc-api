from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
)
from sqlalchemy.ext.hybrid import hybrid_property

# from sqlalchemy import JSONB
from sqlalchemy import cast
from typing import Protocol, TypeVar

from vegi_esc_api.extensions import db
from vegi_esc_api.repo_models_base import _DBDataClassJsonWrapped, _DBFetchable
from vegi_esc_api.models import (
    VegiESCRatingInstance,
    VegiProductInstance,
    VegiUserInstance,
    VegiCategoryGroupInstance,
    VegiProductCategoryInstance,
)

T = TypeVar("T")
GColumn = Column[T] | T

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

    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int):
        self._id = id

    _id: GColumn[int] = Column("id", Integer, primary_key=True)

    @hybrid_property
    def email(self):
        return self._email

    @email.setter
    def email_setter(self, email: str):
        self._email = email

    _email: GColumn[str] = Column("email", String)

    @hybrid_property
    def role(self):
        return self._role

    @role.setter
    def role_setter(self, role: str):
        self._role = role

    _role: GColumn[str] = Column("role", String)

    @hybrid_property
    def phoneCountryCode(self):
        return self._phoneCountryCode

    @phoneCountryCode.setter
    def phoneCountryCode_setter(self, phoneCountryCode: float):
        self._phoneCountryCode = phoneCountryCode

    _phoneCountryCode: GColumn[float] = Column("phoneCountryCode", Float)

    @hybrid_property
    def phoneNoCountry(self):
        return self._phoneNoCountry

    @phoneNoCountry.setter
    def phoneNoCountry_setter(self, phoneNoCountry: str):
        self._phoneNoCountry = phoneNoCountry

    _phoneNoCountry: GColumn[str] = Column("phoneNoCountry", String)

    @hybrid_property
    def marketingEmailContactAllowed(self):
        return self._marketingEmailContactAllowed

    @marketingEmailContactAllowed.setter
    def marketingEmailContactAllowed_setter(self, marketingEmailContactAllowed: bool):
        self._marketingEmailContactAllowed = marketingEmailContactAllowed

    _marketingEmailContactAllowed: GColumn[bool] = Column(
        "marketingEmailContactAllowed", Boolean
    )

    @hybrid_property
    def marketingPushContactAllowed(self):
        return self._marketingPushContactAllowed

    @marketingPushContactAllowed.setter
    def marketingPushContactAllowed_setter(self, marketingPushContactAllowed: bool):
        self._marketingPushContactAllowed = marketingPushContactAllowed

    _marketingPushContactAllowed: GColumn[bool] = Column(
        "marketingPushContactAllowed", Boolean
    )

    @hybrid_property
    def marketingPhoneContactAllowed(self):
        return self._marketingPhoneContactAllowed

    @marketingPhoneContactAllowed.setter
    def marketingPhoneContactAllowed_setter(self, marketingPhoneContactAllowed: bool):
        self._marketingPhoneContactAllowed = marketingPhoneContactAllowed

    _marketingPhoneContactAllowed: GColumn[bool] = Column(
        "marketingPhoneContactAllowed", Boolean
    )

    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name_setter(self, name: str):
        self._name = name

    _name: GColumn[str] = Column("name", String)

    @hybrid_property
    def fbUid(self):
        return self._fbUid

    @fbUid.setter
    def fbUid_setter(self, fbUid: str):
        self._fbUid = fbUid

    _fbUid: GColumn[str] = Column("fbUid", String)

    @hybrid_property
    def isSuperAdmin(self):
        return self._isSuperAdmin

    @isSuperAdmin.setter
    def isSuperAdmin_setter(self, isSuperAdmin: bool):
        self._isSuperAdmin = isSuperAdmin

    _isSuperAdmin: GColumn[bool] = Column("isSuperAdmin", Boolean)

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

    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int):
        self._id = id

    _id: GColumn[int] = Column("id", Integer, primary_key=True)

    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name_setter(self, name: str):
        self._name = name

    _name: GColumn[str] = Column("name", String)

    @hybrid_property
    def forRestaurantItem(self):
        return self._forRestaurantItem

    @forRestaurantItem.setter
    def forRestaurantItem_setter(self, forRestaurantItem: bool):
        self._forRestaurantItem = forRestaurantItem

    _forRestaurantItem: GColumn[bool] = Column("forRestaurantItem", Boolean)

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
            f"{type(self).__name__}<id {self.id}; name {self.name}>"
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

    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int):
        self._id = id

    _id: GColumn[int] = Column("id", Integer, primary_key=True)

    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name_setter(self, name: str):
        self._name = name

    _name: GColumn[str] = Column("name", String)

    @hybrid_property
    def vendor(self):
        return self._vendor

    @vendor.setter
    def vendor_setter(self, vendor: int):
        self._vendor = vendor

    _vendor: GColumn[int] = Column("vendor", Integer)

    @hybrid_property
    def categoryGroup(self):
        return self._categoryGroup

    @categoryGroup.setter
    def categoryGroup_setter(self, categoryGroup: int):
        self._categoryGroup = categoryGroup

    _categoryGroup: GColumn[int] = Column("categoryGroup", Integer)

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

    def __repr__(self):
        return (
            f"{type(self).__name__}<id {self.id}; name {self.name}>"
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

    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int):
        self._id = id

    _id: GColumn[int] = Column("id", Integer, primary_key=True)

    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name_setter(self, name: str):
        self._name = name

    _name: GColumn[str] = Column("name", String)

    @hybrid_property
    def description(self):
        return self._description

    @description.setter
    def description_setter(self, description: str):
        self._description = description

    _description: GColumn[str] = Column("description", String)

    @hybrid_property
    def basePrice(self):
        return self._basePrice

    @basePrice.setter
    def basePrice_setter(self, basePrice: float):
        self._basePrice = basePrice

    _basePrice: GColumn[float] = Column("basePrice", Float)

    @hybrid_property
    def isAvailable(self):
        return self._isAvailable

    @isAvailable.setter
    def isAvailable_setter(self, isAvailable: bool):
        self._isAvailable = isAvailable

    _isAvailable: GColumn[bool] = Column("isAvailable", Boolean)

    @hybrid_property
    def isFeatured(self):
        return self._isFeatured

    @isFeatured.setter
    def isFeatured_setter(self, isFeatured: bool):
        self._isFeatured = isFeatured

    _isFeatured: GColumn[bool] = Column("isFeatured", Boolean)

    @hybrid_property
    def status(self):
        return self._status

    @status.setter
    def status_setter(self, status: str):
        self._status = status

    _status: GColumn[str] = Column("status", String)

    @hybrid_property
    def ingredients(self):
        return self._ingredients

    @ingredients.setter
    def ingredients_setter(self, ingredients: str):
        self._ingredients = ingredients

    _ingredients: GColumn[str] = Column("ingredients", String, nullable=True)

    @hybrid_property
    def stockCount(self):
        return self._stockCount

    @stockCount.setter
    def stockCount_setter(self, stockCount: int):
        self._stockCount = stockCount

    _stockCount: GColumn[int] = Column("stockCount", Integer)

    @hybrid_property
    def supplier(self):
        return self._supplier

    @supplier.setter
    def supplier_setter(self, supplier: str):
        self._supplier = supplier

    _supplier: GColumn[str] = Column("supplier", String, nullable=True)

    @hybrid_property
    def brandName(self):
        return self._brandName

    @brandName.setter
    def brandName_setter(self, brandName: str):
        self._brandName = brandName

    _brandName: GColumn[str] = Column("brandName", String, nullable=True)

    @hybrid_property
    def taxGroup(self):
        return self._taxGroup

    @taxGroup.setter
    def taxGroup_setter(self, taxGroup: str):
        self._taxGroup = taxGroup

    _taxGroup: GColumn[str] = Column("taxGroup", String, nullable=True)

    @hybrid_property
    def stockUnitsPerProduct(self):
        return self._stockUnitsPerProduct

    @stockUnitsPerProduct.setter
    def stockUnitsPerProduct_setter(self, stockUnitsPerProduct: int):
        self._stockUnitsPerProduct = stockUnitsPerProduct

    _stockUnitsPerProduct: GColumn[int] = Column("stockUnitsPerProduct", Integer)

    @hybrid_property
    def sizeInnerUnitValue(self):
        return self._sizeInnerUnitValue

    @sizeInnerUnitValue.setter
    def sizeInnerUnitValue_setter(self, sizeInnerUnitValue: int):
        self._sizeInnerUnitValue = sizeInnerUnitValue

    _sizeInnerUnitValue: GColumn[int] = Column("sizeInnerUnitValue", Integer)

    @hybrid_property
    def sizeInnerUnitType(self):
        return self._sizeInnerUnitType

    @sizeInnerUnitType.setter
    def sizeInnerUnitType_setter(self, sizeInnerUnitType: str):
        self._sizeInnerUnitType = sizeInnerUnitType

    _sizeInnerUnitType: GColumn[str] = Column("sizeInnerUnitType", String)

    @hybrid_property
    def productBarCode(self):
        return self._productBarCode

    @productBarCode.setter
    def productBarCode_setter(self, productBarCode: str):
        self._productBarCode = productBarCode

    _productBarCode: GColumn[str] = Column("productBarCode", String, nullable=True)

    @hybrid_property
    def vendor(self):
        return self._vendor

    @vendor.setter
    def vendor_setter(self, vendor: int):
        self._vendor = vendor

    _vendor: GColumn[int] = Column("vendor", Integer, nullable=True)

    @hybrid_property
    def category(self):
        return self._category

    @category.setter
    def category_setter(self, category: int):
        self._category = category

    _category: GColumn[int] = Column("category", Integer, nullable=True)

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

    def __repr__(self):
        return (
            f"{type(self).__name__}<id {self.id}; name {self.name}>"
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
    escRatingId: str
    rating: float
    calculatedOn: datetime
    product: int


@dataclass
class VegiESCRatingSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
    __tablename__ = "escrating"

    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int | None):
        self._id = id

    _id: GColumn[int | None] = Column("id", Integer, primary_key=True)

    @hybrid_property
    def escRatingId(self):
        return self._escRatingId

    @escRatingId.setter
    def escRatingId_setter(self, escRatingId: str):
        self._escRatingId = escRatingId

    _escRatingId: GColumn[str] = Column("escRatingId", String)

    @hybrid_property
    def rating(self):
        return self._rating

    @rating.setter
    def rating_setter(self, rating: float):
        self._rating = rating

    _rating: GColumn[float] = Column("rating", Float)

    @hybrid_property
    def calculatedOn(self):
        return self._calculatedOn

    @calculatedOn.setter
    def calculatedOn_setter(self, calculatedOn: datetime):
        self._calculatedOn = calculatedOn

    _calculatedOn: GColumn[datetime] = Column(
        "calculatedOn", DateTime, default=func.now(), nullable=False
    )

    @hybrid_property
    def product(self):
        return self._product

    @product.setter
    def product_setter(self, product: int):
        self._product = product

    _product: GColumn[int] = Column("product", Integer)

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
        self.escRatingId = productPublicId
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
            escRatingId=self.escRatingId,
            rating=self.rating,
            calculatedOn=self.calculatedOn,
            product=self.product,
        )


# class VegiESCRatingProto(Protocol):
#     """Used ([a-zA-Z]+:\s?[a-zA-Z]+) = db\.Column\(.*\) to regex replace from Sql Class"""

#     id: int
#     productPublicId: str
#     rating: float
#     calculatedOn: datetime
#     product: int


# @dataclass
# class VegiESCRatingSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
#     __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
#     __tablename__ = "escrating"

#
#     @hybrid_property
#     def id(self):
#         return self._id

#     @id.setter
#     def id_setter(self, id: int | None):
#         self._id = id

#     _id: GColumn[int | None] = Column("id", Integer, primary_key=True)
# #
#     @hybrid_property
#     def productPublicId(self):
#         return self._productPublicId

#     @productPublicId.setter
#     def productPublicId_setter(self, productPublicId: str):
#         self._productPublicId = productPublicId

#     _productPublicId: GColumn[str] = Column("productPublicId", String)
# #
#     @hybrid_property
#     def rating(self):
#         return self._rating

#     @rating.setter
#     def rating_setter(self, rating: float):
#         self._rating = rating

#     _rating: GColumn[float] = Column("rating", Float)
# #
#     @hybrid_property
#     def calculatedOn(self):
#         return self._calculatedOn

#     @calculatedOn.setter
#     def calculatedOn_setter(self, calculatedOn: datetime):
#         self._calculatedOn = calculatedOn

#     _calculatedOn: GColumn[datetime] = Column("calculatedOn", DateTime, default=func.now(), nullable=False)
# #
#     @hybrid_property
#     def product(self):
#         return self._product

#     @product.setter
#     def product_setter(self, product: int):
#         self._product = product

#     _product: GColumn[int] = Column("product", Integer)

#     # Relationship with VegiESCExplanations
#     # ! explanations = db.relationship('VegiESCExplanation', backref='escrating', lazy=True)

#     def __init__(
#         self,
#         productPublicId: str,
#         rating: float,
#         calculatedOn: datetime,
#         product: int,
#         id: int | None = None,
#     ):
#         self.id = id
#         self.productPublicId = productPublicId
#         self.rating = rating
#         self.calculatedOn = calculatedOn
#         self.product = product

#     def __repr__(self):
#         return f"{type(self).__name__}<id {self.id}; rating: {self.rating}; productPublicId {self.productPublicId}>"

#     def serialize(self):
#         return self.toJson()

#     def fetch(self):
#         assert (
#             self.id is not None
#         ), f"{type(self).__name__} cannot have an id of None when fetching to an instance."
#         return VegiESCRatingInstance(
#             id=self.id,
#             productPublicId=self.productPublicId,
#             rating=self.rating,
#             calculatedOn=self.calculatedOn,
#             product=self.product,
#         )


# class VegiESCExplanationProto(Protocol):
#     """Used ([a-zA-Z]+:\s?[a-zA-Z]+) = db\.Column\(.*\) to regex replace from Sql Class"""

#     id: int
#     title: str
#     measure: float
#     reasons: list[str]
#     evidence: str
#     escrating: int
#     escsource: int


# @dataclass
# class VegiESCExplanationSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
#     __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
#     __tablename__ = "escexplanation"

#
#     @hybrid_property
#     def id(self):
#         return self._id

#     @id.setter
#     def id_setter(self, id: int | None):
#         self._id = id

#     _id: GColumn[int | None] = Column("id", Integer, primary_key=True)
# #
#     @hybrid_property
#     def title(self):
#         return self._title

#     @title.setter
#     def title_setter(self, title: str):
#         self._title = title

#     _title: GColumn[str] = Column("title", String)
# #
#     @hybrid_property
#     def measure(self):
#         return self._measure

#     @measure.setter
#     def measure_setter(self, measure: float):
#         self._measure = measure

#     _measure: GColumn[float] = Column("measure", Float)
# #     # ~ https://stackoverflow.com/a/71044396
# #     # NOTE from sqlalchemy.dialects.postgresql import ARRAY, JSON
# #     # ~ https://docs.sqlalchemy.org/en/13/core/type_basics.html
# #     # ~ https://docs.sqlalchemy.org/en/13/core/type_basics.html#sqlalchemy.types.JSON
# #
#     @hybrid_property
#     def reasons(self):
#         return self._reasons

#     @reasons.setter
#     def reasons_setter(self, reasons: list[str]):
#         self._reasons = reasons

#     _reasons: GColumn[list[str]] = Column("reasons", JSON)
# #
#     @hybrid_property
#     def evidence(self):
#         return self._evidence

#     @evidence.setter
#     def evidence_setter(self, evidence: str):
#         self._evidence = evidence

#     _evidence: GColumn[str] = Column("evidence", JSON)
#     # ~ https://stackoverflow.com/a/38793154
#     escrating: int = db.Column(
#         db.Integer(), db.ForeignKey("escrating.id"), nullable=False
#     )
#     escsource: int = db.Column(
#         db.Integer(), db.ForeignKey("escsource.id"), nullable=False
#     )

#     def __init__(
#         self,
#         title: str,
#         measure: float,
#         reasons: list[str],
#         evidence: str,
#         escrating: int,
#         escsource: int,
#         id: int | None = None,
#     ):
#         self.id = id
#         self.title = title
#         self.measure = measure
#         # self.reasons = jsons.dumps(reasons)
#         # self.reasons = cast(reasons, JSONB)
#         self.reasons = reasons
#         self.evidence = evidence
#         self.escrating = escrating
#         self.escsource = escsource

#     def __repr__(self):
#         return f"{type(self).__name__}<id {self.id}; title {self.title}>"

#     def serialize(self):
#         return self.toJson()

#     def fetch(self):
#         assert (
#             self.id is not None
#         ), f"{type(self).__name__} cannot have an id of None when fetching to an instance."
#         return VegiESCExplanationInstance(
#             id=self.id,
#             title=self.title,
#             measure=self.measure,
#             reasons=self.reasons,
#             evidence=self.evidence,
#             escrating=self.escrating,
#             escsource=self.escsource,
#         )


# class VegiESCSourceProto(Protocol):
#     """Used ([a-zA-Z]+:\s?[a-zA-Z]+) = db\.Column\(.*\) to regex replace from Sql Class"""

#     id: int
#     name: str
#     source_type: str
#     domain: str
#     credibility: float


# @dataclass
# class VegiESCSourceSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
#     __bind_key__ = VEGI_DB_NAMED_BIND  # Replace with your named bind key
#     __tablename__ = "escsource"

#
#     @hybrid_property
#     def id(self):
#         return self._id

#     @id.setter
#     def id_setter(self, id: int):
#         self._id = id

#     _id: GColumn[int] = Column("id", Integer, primary_key=True)
# #
#     @hybrid_property
#     def name(self):
#         return self._name

#     @name.setter
#     def name_setter(self, name: str):
#         self._name = name

#     _name: GColumn[str] = Column("name", String)
# #
#     @hybrid_property
#     def source_type(self):
#         return self._source_type

#     @source_type.setter
#     def source_type_setter(self, source_type: str):
#         self._source_type = source_type

#     _source_type: GColumn[str] = Column("source_type", String, name="type")
# #
#     @hybrid_property
#     def domain(self):
#         return self._domain

#     @domain.setter
#     def domain_setter(self, domain: str):
#         self._domain = domain

#     _domain: GColumn[str] = Column("domain", String)
# #
#     @hybrid_property
#     def credibility(self):
#         return self._credibility

#     @credibility.setter
#     def credibility_setter(self, credibility: float):
#         self._credibility = credibility

#     _credibility: GColumn[float] = Column("credibility", Float)

#     # Relationship with VegiESCExplanations
#     # ! sources = db.relationship('VegiESCExplanation', backref='escsource', lazy=True)

#     def __init__(
#         self, id: int, name: str, source_type: str, domain: str, credibility: float
#     ):
#         self.id = id
#         self.name = name
#         self.source_type = source_type
#         self.domain = domain
#         self.credibility = credibility

#     def __repr__(self):
#         return f"{type(self).__name__}<id {self.id}; name {self.name}>"

#     def serialize(self):
#         return self.toJson()

#     def fetch(self):
#         return VegiESCSourceInstance(
#             id=self.id,
#             name=self.name,
#             source_type=self.source_type,
#             domain=self.domain,
#             credibility=self.credibility,
#         )
