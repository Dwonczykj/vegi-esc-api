from vegi_esc_api.create_app import db

from dataclasses import dataclass
from typing import Any
from datetime import datetime, timedelta
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY
import jsons


@dataclass
class _DBDataClassJsonWrapped:
    @classmethod
    def fromJson(cls, obj: Any):
        return jsons.load(obj, cls=cls)

    def toJson(self) -> dict[str, Any] | list[Any] | int | str | bool | float:
        x: Any = jsons.dump(self, strip_privates=True)
        return x


@dataclass
class ESCSource(db.Model, _DBDataClassJsonWrapped):
    __tablename__ = "escsource"

    id: int = db.Column(db.Integer, primary_key=True)
    name: str = db.Column(db.String())
    source_type: str = db.Column(db.String())
    domain: str = db.Column(db.String())
    credibility: float = db.Column(db.Float())

    def __init__(self, name: str, source_type: str, domain: str, credibility: float):
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


@dataclass
class ESCRating(db.Model, _DBDataClassJsonWrapped):
    __tablename__ = "escrating"

    id: int = db.Column(db.Integer, primary_key=True)
    product_name: str = db.Column(db.String())
    product_id: str = db.Column(db.String())
    rating: float = db.Column(db.Float())
    calculated_on: datetime = db.Column(db.DateTime, default=func.now(), nullable=False)

    def __init__(
        self, product_name: str, product_id: str, rating: float, calculated_on: datetime
    ):
        self.product_name = product_name
        self.product_id = product_id
        self.rating = rating
        self.calculated_on = calculated_on

    def __repr__(self):
        return f"<id {self.id}; product_name {self.product_name}>"

    def serialize(self):
        return self.toJson()


@dataclass
class ESCExplanation(db.Model, _DBDataClassJsonWrapped):
    __tablename__ = "escexplanation"

    id: int = db.Column(db.Integer, primary_key=True)
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
    ):
        self.title = title
        self.measure = measure
        self.reasons = reasons
        self.evidence = evidence
        self.rating = rating
        self.source = source

    def __repr__(self):
        return f"<id {self.id}; title {self.title}>"

    def serialize(self):
        return self.toJson()


@dataclass
class CachedItem(db.Model, _DBDataClassJsonWrapped):
    __tablename__ = "cacheditem"

    id: int = db.Column(db.Integer, primary_key=True)
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
    ):
        self.item_name = item_name
        self.item_type = item_type
        self.item_source = item_source
        self.item_json = item_json
        self.created_on = created_on
        self.ttl_days = ttl_days
        self.expires_on = self.created_on + timedelta(days=ttl_days)

    def __repr__(self):
        return f"<id {self.id}; item_name {self.item_name} [{self.item_type}]>"

    def serialize(self):
        return self.toJson()
