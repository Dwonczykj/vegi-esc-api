from vegi_esc_api.models import ESCSource, ESCRating, ESCExplanation, CachedItem
import vegi_esc_api.logger as logger
from vegi_esc_api.create_app import db
from sqlalchemy import asc, desc
from datetime import datetime, timedelta
import jsons

from vegi_esc_api.models_wrapper import CachedSustainedItemCategory


class Vegi_ESC_Repo:
    def get_sources(self, source_type: str | None = None):
        try:
            sources: list[ESCSource] = (
                ESCSource.query.all()
                if source_type
                else ESCSource.query.filter(ESCSource.source_type == source_type)
            )
            assert isinstance(sources, list)
            return [e.serialize() for e in sources]
        except Exception as e:
            logger.error(str(e))
            return []

    def get_rating(self, rating_id: int, since_time_delta: timedelta):
        ratings_calculated_after = datetime.now() - since_time_delta
        try:
            # ~ https://stackoverflow.com/a/8898533
            rating: ESCRating | None = (
                ESCRating.query.filter(ESCRating.id == rating_id)
                .filter(ESCRating.calculated_on >= ratings_calculated_after)
                .order_by(desc(ESCRating.calculated_on))
                .first()
            )
            return rating
        except Exception as e:
            logger.error(str(e))
            return None

    def get_items(self, item_source: str):
        items_expire_after = datetime.now()
        items: list[CachedItem] = []
        try:
            items = (
                CachedItem.query.filter(CachedItem.item_source == item_source)
                .filter(CachedItem.expires_on >= items_expire_after)
                .order_by(asc(CachedItem.created_on))
            )
            assert isinstance(items, list)
            return items
        except Exception as e:
            logger.error(str(e))
        assert isinstance(items, list)
        return items

    def get_sustained_items(self) -> list[CachedSustainedItemCategory]:
        categories_with_products_embedded = self.get_items(item_source="sustained.com")
        if not categories_with_products_embedded:
            return []
        return [
            CachedSustainedItemCategory.fromJson(jsons.loads(c.item_json))
            for c in categories_with_products_embedded
            if c.item_type == "category"
        ]

    def add_source(self, new_source: ESCSource):
        # source = ESCSource(name="Napolina", source_type="Website", domain="https://napolina.com/", credibility=0)
        db.session.add(new_source)
        logger.verbose("ESCSource created. ESCSource id={}".format(new_source.id))
        db.session.commit()
        return self

    def add_rating(self, new_rating: ESCRating, explanations: list[ESCExplanation]):
        # rating = ESCRating(product_name="Cannellini beans", product_id="ABC123", calculated_on=datetime.now())
        db.session.add(new_rating)
        for explanation in explanations:
            explanation.rating = new_rating.id
            db.session.add(explanation)

        db.session.commit()
        logger.verbose(
            f"ESCRating created with {len(explanations)} explanations. ESCRating id={new_rating.id}"
        )
        return self

    def add_cached_items(self, items: list[CachedItem]):
        # ~ https://docs.sqlalchemy.org/en/20/orm/session_basics.html#:~:text=To%20add%20a%20list%20of%20items%20to%20the%20session%20at%20once%2C%20use%20Session.add_all()%3A
        db.session.add_all(items)
        db.session.commit()
        return self
