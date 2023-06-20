from dataclasses import dataclass

from typing import Protocol, Any, Type, Self
from typing import TypeVar, Callable
from typing import List
from datetime import datetime, timedelta
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY, JSON
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property

from vegi_esc_api.extensions import db
from vegi_esc_api.repo_models_base import _DBDataClassJsonWrapped, _DBFetchable
from vegi_esc_api.models import ESCSourceInstance, ESCRatingInstance, ESCExplanationInstance, CachedItemInstance, ESCProductInstance

T = TypeVar('T')
# Column.value = _columnToValue


Base = declarative_base()


# Type alias for a list of generic elements
GColumn = Column[T] | T


ESC_DB_NAMED_BIND = "esc_bind"


class ESCSourceProto(Protocol):
    id: int
    name: str
    source_type: str
    domain: str
    credibility: float


@dataclass
class ESCSourceSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    '''
    @param credibility in [0,1]
    @param source_type in 'database'|'api'|'webpage'
    '''
    __tablename__ = "escsource"
  
    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int | None):
        self._id = id
    
    _id: GColumn[int | None] = Column("id", Integer, primary_key=True)

    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name_setter(self, name: str):
        self._name = name
    _name: GColumn[str] = Column("name", String)
    
    @hybrid_property
    def source_type(self):
        return self._source_type

    @source_type.setter
    def source_type_setter(self, source_type: str):
        self._source_type = source_type
    
    _source_type: GColumn[str] = Column("source_type", String)
    
    @hybrid_property
    def domain(self):
        return self._domain

    @domain.setter
    def domain_setter(self, domain: str):
        self._domain = domain
    
    _domain: GColumn[str] = Column("domain", String)
    
    @hybrid_property
    def credibility(self):
        return self._credibility

    @credibility.setter
    def credibility_setter(self, credibility: float):
        self._credibility = credibility
    
    _credibility: GColumn[float] = Column("credibility", Float)

    def __init__(self, name: str, source_type: str, domain: str, credibility: float, id: int | None = None):
        self._id = id
        self._name = name
        self._source_type = source_type
        self._domain = domain
        self._credibility = credibility

    def __repr__(self):
        return f"<id {self.id}; name {self.name}>"

    def serialize(self):
        return self.toJson()
        # or for more bespoke serialization
        # return {
        #     'id': self.id,
        #     'name': self.name,
        #     'source_type': self.source_type,
        #     'domain': self.domain,
        #     'credibility': self.credibility,
        # }
        
    def fetch(self):
        assert self.id is not None, f'{type(self).__name__} cannot have an id of None when fetching to an instance.'
        return ESCSourceInstance(
            id=self.id,
            name=self.name,
            source_type=self.source_type,
            domain=self.domain,
            credibility=self.credibility,
        )


class CreateESCRating:

    def __init__(
        self, product_name: str, product_id: str, rating: float, calculated_on: datetime
    ):
        self.product_name = product_name
        self.product_id = product_id
        self.rating = rating
        self.calculated_on = calculated_on

    def __repr__(self):
        return f"{type(self).__name__}<id product_name {self.product_name}>"


class ESCRatingProto(Protocol):
    id: int
    product_name: str
    product_id: str
    rating: float
    calculated_on: datetime
    

@dataclass
class ESCRatingSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __tablename__ = "escrating"

    # add property setters, getters of underlying value type
    
    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int | None):
        self._id = id
    
    _id: GColumn[int | None] = Column("id", Integer, primary_key=True)
    
    @hybrid_property
    def product(self):
        return self._product

    @product.setter
    def product_setter(self, product: int):
        self._product = product
    
    _product: GColumn[int] = Column("product", Integer)
    
    @hybrid_property
    def product_name(self):
        return self._product_name

    @product_name.setter
    def product_name_setter(self, product_name: str):
        self._product_name = product_name
    
    _product_name: GColumn[str] = Column("product_name", String)
    
    @hybrid_property
    def product_id(self):
        return self._product_id

    @product_id.setter
    def product_id_setter(self, product_id: str):
        self._product_id = product_id
    
    _product_id: GColumn[str] = Column("product_id", String)
    
    @hybrid_property
    def rating(self):
        return self._rating

    @rating.setter
    def rating_setter(self, rating: float):
        self._rating = rating
    
    _rating: GColumn[float] = Column("rating", Float)
    
    @hybrid_property
    def calculated_on(self):
        return self._calculated_on

    @calculated_on.setter
    def calculated_on_setter(self, calculated_on: datetime):
        self._calculated_on = calculated_on
    
    _calculated_on: GColumn[datetime] = Column("calculated_on", DateTime, default=func.now(), nullable=False)

    def __init__(
        self, product: int, product_name: str, product_id: str, rating: float, calculated_on: datetime, id: int | None = None
    ):
        self._id = id
        self._product = product
        self._product_name = product_name
        self._product_id = product_id
        self._rating = rating
        self._calculated_on = calculated_on

    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; product_name {self.product_name}>"

    def serialize(self):
        return self.toJson()
        
    def fetch(self):
        assert self.id is not None, f'{type(self).__name__} cannot have an id of None when fetching to an instance.'
        return ESCRatingInstance(
            id=self.id,
            product=self.product,
            product_name=self.product_name,
            product_id=self.product_id,
            rating=self.rating,
            calculated_on=self.calculated_on,
        )


class ESCExplanationProto(Protocol):
    id: int
    title: str
    measure: float
    reasons: list[str]
    evidence: str
    rating: int
    source: int
    

@dataclass
class ESCExplanationSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __tablename__ = "escexplanation"

    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int | None):
        self._id = id
    
    _id: GColumn[int | None] = Column("id", Integer, primary_key=True)
    
    @hybrid_property
    def title(self):
        return self._title

    @title.setter
    def title_setter(self, title: str):
        self._title = title
    
    _title: GColumn[str] = Column("title", String)
    
    @hybrid_property
    def measure(self):
        return self._measure

    @measure.setter
    def measure_setter(self, measure: float):
        self._measure = measure
    
    _measure: GColumn[float] = Column("measure", Float)
    # ~ https://stackoverflow.com/a/71044396
    # NOTE from sqlalchemy.dialects.postgresql import ARRAY, JSON
    # ~ https://docs.sqlalchemy.org/en/13/core/type_basics.html
    # ~ https://docs.sqlalchemy.org/en/13/core/type_basics.html#sqlalchemy.types.JSON
    
    @hybrid_property
    def reasons(self):
        return self._reasons

    @reasons.setter
    def reasons_setter(self, reasons: list[str]):
        self._reasons = reasons
    
    _reasons: GColumn[list[str]] = Column("reasons", JSON)
    
    @hybrid_property
    def evidence(self):
        return self._evidence

    @evidence.setter
    def evidence_setter(self, evidence: str):
        self._evidence = evidence
    
    _evidence: GColumn[str] = Column("evidence", JSON)

    # ~ https://stackoverflow.com/a/38793154
    
    @hybrid_property
    def rating(self):
        return self._rating

    @rating.setter
    def rating_setter(self, rating: int):
        self._rating = rating
    
    _rating: GColumn[int] = Column("rating", Integer, db.ForeignKey("escrating.id"), nullable=False)
    
    @hybrid_property
    def source(self):
        return self._source

    @source.setter
    def source_setter(self, source: int):
        self._source = source
    
    _source: GColumn[int] = Column("source", Integer, db.ForeignKey("escsource.id"), nullable=False)

    def __init__(
        self,
        title: str,
        measure: float,
        reasons: list[str],
        evidence: str,
        rating: int,
        source: int,
        id: int | None = None,
    ):
        self._id = id
        self._title = title
        self._measure = measure
        self._reasons = reasons
        self._evidence = evidence
        self._rating = rating
        self._source = source

    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; title {self.title}>"

    def serialize(self):
        return self.toJson()
        
    def fetch(self):
        assert self.id is not None, f'{type(self).__name__} cannot have an id of None when fetching to an instance.'
        return ESCExplanationInstance(
            id=self.id,
            title=self.title,
            measure=self.measure,
            reasons=self.reasons,
            evidence=self.evidence,
            rating=self.rating,
            source=self.source,
        )


class CachedItemProto(Protocol):
    id: int
    item_name: str
    item_type: str
    item_source: str
    item_json: str
    ttl_days: int
    created_on: datetime
    expires_on: datetime
    

@dataclass
class CachedItemSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __tablename__ = "cacheditem"
    
    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int | None):
        self._id = id
    
    _id: GColumn[int | None] = Column("id", Integer, primary_key=True)
    
    @hybrid_property
    def item_name(self):
        return self._item_name

    @item_name.setter
    def item_name_setter(self, item_name: str):
        self._item_name = item_name
    
    _item_name: GColumn[str] = Column("item_name", String)
    
    @hybrid_property
    def item_type(self):
        return self._item_type

    @item_type.setter
    def item_type_setter(self, item_type: str):
        self._item_type = item_type
    
    _item_type: GColumn[str] = Column("item_type", String)
    
    @hybrid_property
    def item_source(self):
        return self._item_source

    @item_source.setter
    def item_source_setter(self, item_source: str):
        self._item_source = item_source
    
    _item_source: GColumn[str] = Column("item_source", String)
    
    @hybrid_property
    def item_json(self):
        return self._item_json

    @item_json.setter
    def item_json_setter(self, item_json: str):
        self._item_json = item_json
    
    _item_json: GColumn[str] = Column("item_json", Text)
    _TTL_DEFAULT: int = 60
    
    @hybrid_property
    def ttl_days(self):
        return self._ttl_days

    @ttl_days.setter
    def ttl_days_setter(self, ttl_days: int):
        self._ttl_days = ttl_days
    
    _ttl_days: GColumn[int] = Column("ttl_days", Integer, default=_TTL_DEFAULT)
    
    @hybrid_property
    def created_on(self):
        return self._created_on

    @created_on.setter
    def created_on_setter(self, created_on: datetime):
        self._created_on = created_on
    
    _created_on: GColumn[datetime] = Column("created_on", DateTime, default=func.now(), nullable=False)
    
    @hybrid_property
    def expires_on(self):
        return self._expires_on

    @expires_on.setter
    def expires_on_setter(self, expires_on: datetime):
        self._expires_on = expires_on
    
    _expires_on: GColumn[datetime] = Column("expires_on", DateTime, default=func.now() + timedelta(days=_TTL_DEFAULT), nullable=False)

    def __init__(
        self,
        item_name: str,
        item_type: str,
        item_source: str,
        item_json: str,
        ttl_days: int = _TTL_DEFAULT,
        created_on: datetime = datetime.now(),
        id: int | None = None,
    ):
        self._id = id
        self._item_name = item_name
        self._item_type = item_type
        self._item_source = item_source
        self._item_json = item_json
        self._created_on = created_on
        self._ttl_days = ttl_days
        self._expires_on = self._created_on + timedelta(days=ttl_days)

    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; item_name {self.item_name} [{self.item_type}]>"

    def serialize(self):
        return self.toJson()
        
    def fetch(self):
        assert self.id is not None, f'{type(self).__name__} cannot have an id of None when fetching to an instance.'
        return CachedItemInstance(
            id=self.id,
            item_name=self.item_name,
            item_type=self.item_type,
            item_source=self.item_source,
            item_json=self.item_json,
            created_on=self.created_on,
            ttl_days=self.ttl_days,
        )


class ESCProductProto(Protocol):
    id: int
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
class ESCProductSql(db.Model, _DBDataClassJsonWrapped, _DBFetchable):
    __tablename__ = "product"

    @hybrid_property
    def id(self):
        return self._id

    @id.setter
    def id_setter(self, id: int | None):
        self._id = id
    
    _id: GColumn[int | None] = Column("id", Integer, primary_key=True)
    
    @hybrid_property
    def name(self):
        return self._name

    @name.setter
    def name_setter(self, name: str):
        self._name = name
    
    _name: GColumn[str] = Column("name", String)
    
    @hybrid_property
    def product_external_id_on_source(self):
        return self._product_external_id_on_source

    @product_external_id_on_source.setter
    def product_external_id_on_source_setter(self, product_external_id_on_source: str):
        self._product_external_id_on_source = product_external_id_on_source
    
    _product_external_id_on_source: GColumn[str] = Column("product_external_id_on_source", String)
    
    @hybrid_property
    def source(self):
        return self._source

    @source.setter
    def source_setter(self, source: int):
        self._source = source
    
    _source: GColumn[int] = Column("source", Integer)
    
    @hybrid_property
    def description(self):
        return self._description

    @description.setter
    def description_setter(self, description: str):
        self._description = description
    
    _description: GColumn[str] = Column("description", String)
    
    @hybrid_property
    def category(self):
        return self._category

    @category.setter
    def category_setter(self, category: str):
        self._category = category
    
    _category: GColumn[str] = Column("category", String)
    
    @hybrid_property
    def keyWords(self):
        return self._keyWords

    @keyWords.setter
    def keyWords_setter(self, keyWords: list[str]):
        self._keyWords = keyWords
    
    _keyWords: GColumn[list[str]] = Column("keyWords", JSON)
    
    @hybrid_property
    def imageUrl(self):
        return self._imageUrl

    @imageUrl.setter
    def imageUrl_setter(self, imageUrl: str):
        self._imageUrl = imageUrl
    
    _imageUrl: GColumn[str] = Column("imageUrl", String)
    
    @hybrid_property
    def ingredients(self):
        return self._ingredients

    @ingredients.setter
    def ingredients_setter(self, ingredients: str):
        self._ingredients = ingredients
    
    _ingredients: GColumn[str] = Column("ingredients", String)
    
    @hybrid_property
    def packagingType(self):
        return self._packagingType

    @packagingType.setter
    def packagingType_setter(self, packagingType: str):
        self._packagingType = packagingType
    
    _packagingType: GColumn[str] = Column("packagingType", String)
    
    @hybrid_property
    def stockUnitsPerProduct(self):
        return self._stockUnitsPerProduct

    @stockUnitsPerProduct.setter
    def stockUnitsPerProduct_setter(self, stockUnitsPerProduct: int):
        self._stockUnitsPerProduct = stockUnitsPerProduct
    
    _stockUnitsPerProduct: GColumn[int] = Column("stockUnitsPerProduct", String)
    
    @hybrid_property
    def sizeInnerUnitValue(self):
        return self._sizeInnerUnitValue

    @sizeInnerUnitValue.setter
    def sizeInnerUnitValue_setter(self, sizeInnerUnitValue: float):
        self._sizeInnerUnitValue = sizeInnerUnitValue
    
    _sizeInnerUnitValue: GColumn[float] = Column("sizeInnerUnitValue", Float)
    
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
    
    _productBarCode: GColumn[str] = Column("productBarCode", String)
    
    @hybrid_property
    def supplier(self):
        return self._supplier

    @supplier.setter
    def supplier_setter(self, supplier: str):
        self._supplier = supplier
    
    _supplier: GColumn[str] = Column("supplier", String)
    
    @hybrid_property
    def brandName(self):
        return self._brandName

    @brandName.setter
    def brandName_setter(self, brandName: str):
        self._brandName = brandName
    
    _brandName: GColumn[str] = Column("brandName", String)
    
    @hybrid_property
    def origin(self):
        return self._origin

    @origin.setter
    def origin_setter(self, origin: str):
        self._origin = origin
    
    _origin: GColumn[str] = Column("origin", String)
    
    @hybrid_property
    def taxGroup(self):
        return self._taxGroup

    @taxGroup.setter
    def taxGroup_setter(self, taxGroup: str):
        self._taxGroup = taxGroup
    
    _taxGroup: GColumn[str] = Column("taxGroup", String)
    
    @hybrid_property
    def dateOfBirth(self):
        return self._dateOfBirth

    @dateOfBirth.setter
    def dateOfBirth_setter(self, dateOfBirth: datetime):
        self._dateOfBirth = dateOfBirth
    
    _dateOfBirth: GColumn[datetime] = Column("dateOfBirth", DateTime)
    
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
        id: int | None = None,
    ):
        if id:
            self._id = id
        self._name = name
        self._product_external_id_on_source = product_external_id_on_source
        self._source = source
        self._description = description
        self._category = category
        self._keyWords = keyWords
        self._imageUrl = imageUrl
        self._ingredients = ingredients
        self._packagingType = packagingType
        self._stockUnitsPerProduct = stockUnitsPerProduct
        self._sizeInnerUnitValue = sizeInnerUnitValue
        self._sizeInnerUnitType = sizeInnerUnitType
        self._productBarCode = productBarCode
        self._supplier = supplier
        self._brandName = brandName
        self._origin = origin
        self._taxGroup = taxGroup
        self._dateOfBirth = dateOfBirth

    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; item_name {self.item_name} [{self.item_type}]>"

    def serialize(self):
        return self.toJson()
        
    def fetch(self):
        assert self.id is not None, f'{type(self).__name__} cannot have an id of None when fetching to an instance.'
        return ESCProductInstance(
            id=self.id,
            name=self.name,
            product_external_id_on_source=self.product_external_id_on_source,
            source=self.source,
            description=self.description,
            category=self.category,
            keyWords=self.keyWords,
            imageUrl=self.imageUrl,
            ingredients=self.ingredients,
            packagingType=self.packagingType,
            stockUnitsPerProduct=self.stockUnitsPerProduct,
            sizeInnerUnitValue=self.sizeInnerUnitValue,
            sizeInnerUnitType=self.sizeInnerUnitType,
            productBarCode=self.productBarCode,
            supplier=self.supplier,
            brandName=self.brandName,
            origin=self.origin,
            taxGroup=self.taxGroup,
            dateOfBirth=self.dateOfBirth,
        )
