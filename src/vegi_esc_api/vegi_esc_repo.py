from __future__ import annotations
from typing import Any, Callable
from vegi_esc_api.vegi_esc_repo_models import ESCSourceSql, ESCRatingSql, CachedItemSql, ESCExplanationSql
from vegi_esc_api.repo_models_base import appcontext
from vegi_esc_api.models import ESCSourceInstance, ESCExplanationCreate
import vegi_esc_api.logger as logger
from vegi_esc_api.extensions import db
from sqlalchemy import asc, desc, or_
from datetime import datetime, timedelta
from flask import Flask
import jsons

from vegi_esc_api.models_wrapper import CachedSustainedItemCategory


class Vegi_ESC_Repo:
    def __init__(self, app: Flask) -> None:
        Vegi_ESC_Repo.app = app

    @appcontext
    def get_sources(self, source_type: str | None = None):
        try:
            sources: list[ESCSourceSql] = (
                ESCSourceSql.query.all()
                if source_type is None
                else ESCSourceSql.query.filter(ESCSourceSql.source_type == source_type).all()
            )
            assert isinstance(sources, list)
            return [e.fetch() for e in sources]
        except Exception as e:
            logger.error(str(e))
            return []
    
    @appcontext
    def get_source(self, source_name: str) -> ESCSourceInstance | None:
        source_name = source_name.lower()
        try:
            sources: list[ESCSourceSql] = db.session\
                .query(ESCSourceSql)\
                .filter(or_(ESCSourceSql.name == source_name, ESCSourceSql.domain == source_name))\
                .all()
            assert isinstance(sources, list)
            if sources:
                return sources[0].fetch()
            return None
        except Exception as e:
            logger.error(str(e))
            return None

    @appcontext
    def get_rating(self, rating_id: int, since_time_delta: timedelta):
        ratings_calculated_after = datetime.now() - since_time_delta
        try:
            # ~ https://stackoverflow.com/a/8898533
            rating: ESCRatingSql | None = (
                ESCRatingSql.query.filter(ESCRatingSql.id == rating_id)
                .filter(ESCRatingSql.calculated_on >= ratings_calculated_after)
                .order_by(desc(ESCRatingSql.calculated_on))
                .first()
            )
            return rating.fetch() if rating else None
        except Exception as e:
            logger.error(str(e))
            return None

    @appcontext
    def get_items(self, item_source: str):
        # items_expire_after = datetime.now()
        items: list[CachedItemSql] = []
        try:
            items = (
                CachedItemSql.query
                .filter(CachedItemSql.item_source == item_source)
                .order_by(asc(CachedItemSql.created_on))
                .all()
                # .filter(CachedItem.expires_on >= items_expire_after)
            )
            assert isinstance(items, list)
            return [i.fetch() for i in items]
        except Exception as e:
            logger.error(str(e))
        assert isinstance(items, list)
        return [i.fetch() for i in items]

    @appcontext
    def get_sustained_items(self) -> list[CachedSustainedItemCategory]:
        categories_with_products_embedded = self.get_items(item_source="sustained.com")
        if not categories_with_products_embedded:
            return []
        return [
            CachedSustainedItemCategory.fromJson(jsons.loads(c.item_json))
            for c in categories_with_products_embedded
            if c.item_type == "category"
        ]

    @appcontext
    def add_source(self, new_source: ESCSourceSql):
        ''' See https://www.notion.so/vegiapp/2446a82bdee94f0cb9e6d7671a78bb06?v=a4d348fe223645ed9b5d079e1abc2b1b&pvs=4 for list of sources in vegi notion'''
        # source = ESCSource(name="Napolina", source_type="Website", domain="https://napolina.com/", credibility=0)
        db.session.add(new_source)
        logger.verbose("ESCSource created. ESCSource id={}".format(new_source.id))
        db.session.commit()
        db.session.refresh(new_source)
        return new_source.fetch()

    @appcontext
    def add_rating(self, new_rating: ESCRatingSql, explanations: list[ESCExplanationCreate]):
        # rating = ESCRating(product_name="Cannellini beans", product_id="ABC123", calculated_on=datetime.now())
        db.session.add(new_rating)
        db.session.commit()
        db.session.refresh(new_rating)
        assert new_rating.id is not None, 'a new ESC rating in vegi_esc_repo cannot have null rating id'
        for explanation in explanations:
            explanation.rating = new_rating.id
            db.session.add(ESCExplanationSql(
                title=explanation.title,
                measure=explanation.measure,
                reasons=explanation.reasons,
                evidence=explanation.evidence,
                rating=new_rating.id,
                source=explanation.source,
            ))

        db.session.commit()
        logger.verbose(
            f"ESCRating created with {len(explanations)} explanations. ESCRating id={new_rating.id}"
        )
        
        return new_rating.fetch()

    @appcontext
    def add_cached_items(self, items: list[CachedItemSql]):
        # ~ https://docs.sqlalchemy.org/en/20/orm/session_basics.html#:~:text=To%20add%20a%20list%20of%20items%20to%20the%20session%20at%20once%2C%20use%20Session.add_all()%3A
        db.session.add_all(items)
        db.session.commit()
        return self
