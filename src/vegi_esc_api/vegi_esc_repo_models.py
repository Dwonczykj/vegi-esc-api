from dataclasses import dataclass

from typing import Protocol
from datetime import datetime, timedelta
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY

from vegi_esc_api.extensions import db
from vegi_esc_api.repo_models_base import _DBDataClassJsonWrapped, _DBFetchable
from vegi_esc_api.models import ESCSourceInstance, ESCRatingInstance, ESCExplanationInstance, CachedItemInstance


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

    id: int | None = db.Column(db.Integer, primary_key=True)
    name: str = db.Column(db.String())
    source_type: str = db.Column(db.String())
    domain: str = db.Column(db.String())
    credibility: float = db.Column(db.Float())

    def __init__(self, name: str, source_type: str, domain: str, credibility: float, id: int | None = None):
        self.id = id
        self.name = name
        self.source_type = source_type
        self.domain = domain
        self.credibility = credibility

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
class ESCRatingSql(db.Model, _DBDataClassJsonWrapped, CreateESCRating, _DBFetchable):
    __tablename__ = "escrating"

    id: int | None = db.Column(db.Integer, primary_key=True)
    product_name: str = db.Column(db.String())
    product_id: str = db.Column(db.String())
    rating: float = db.Column(db.Float())
    calculated_on: datetime = db.Column(db.DateTime, default=func.now(), nullable=False)

    def __init__(
        self, product_name: str, product_id: str, rating: float, calculated_on: datetime, id: int | None = None
    ):
        super().__init__(
            product_name=product_name,
            product_id=product_id,
            rating=rating,
            calculated_on=calculated_on,
        )
        self.id = id

    def __repr__(self):
        return f"{type(self).__name__}<id {self.id}; product_name {self.product_name}>"

    def serialize(self):
        return self.toJson()
        
    def fetch(self):
        assert self.id is not None, f'{type(self).__name__} cannot have an id of None when fetching to an instance.'
        return ESCRatingInstance(
            id=self.id,
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

    id: int | None = db.Column(db.Integer, primary_key=True)
    title: str = db.Column(db.String())
    measure: float = db.Column(db.Float())
    # ~ https://stackoverflow.com/a/71044396
    reasons: list[str] = db.Column(ARRAY(db.Text()))
    evidence: str = db.Column(db.Text())
    # ~ https://stackoverflow.com/a/38793154
    rating: int = db.Column(db.Integer(), db.ForeignKey("escrating.id"), nullable=False)
    source: int = db.Column(db.Integer(), db.ForeignKey("escsource.id"), nullable=False)

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
        self.id = id
        self.title = title
        self.measure = measure
        self.reasons = reasons
        self.evidence = evidence
        self.rating = rating
        self.source = source

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

    id: int | None = db.Column(db.Integer, primary_key=True)
    item_name: str = db.Column(db.String())
    item_type: str = db.Column(db.String())
    item_source: str = db.Column(db.String())
    item_json: str = db.Column(db.Text())
    _TTL_DEFAULT: int = 60
    ttl_days: int = db.Column(db.Integer(), default=_TTL_DEFAULT)
    created_on: datetime = db.Column(db.DateTime, default=func.now(), nullable=False)
    expires_on: datetime = db.Column(
        db.DateTime, default=func.now() + timedelta(days=_TTL_DEFAULT), nullable=False
    )

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
        self.id = id
        self.item_name = item_name
        self.item_type = item_type
        self.item_source = item_source
        self.item_json = item_json
        self.created_on = created_on
        self.ttl_days = ttl_days
        self.expires_on = self.created_on + timedelta(days=ttl_days)

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
