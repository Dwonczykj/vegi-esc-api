from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Self
from vegi_esc_api.vegi_esc_repo_models import ESCSourceSql, ESCRatingSql, CachedItemSql, ESCExplanationSql, ESCProductSql, ESC_DB_NAMED_BIND
from vegi_esc_api.repo_models_base import appcontext
from vegi_esc_api.models import ESCSourceInstance, ESCExplanationCreate, ESCRatingInstance, ESCExplanationInstance, ESCProductInstance
import vegi_esc_api.logger as logger
from vegi_esc_api.extensions import db
from sqlalchemy import asc, desc, or_, and_
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.engine.row import Row
from datetime import datetime, timedelta
from flask import Flask
import jsons

from vegi_esc_api.models_wrapper import CachedSustainedItemCategory

SUSTAINED_DOMAIN_NAME = 'sustained.com'


@dataclass
class NewRating:
    rating: ESCRatingInstance
    explanations: list[ESCExplanationInstance]

    def serialize(self):
        return {
            "rating": self.rating.serialize(),
            "explanations": [e.serialize() for e in self.explanations],
        }


class Vegi_ESC_Repo:
    def __init__(self, app: Flask) -> None:
        Vegi_ESC_Repo.app = app
        Vegi_ESC_Repo.db_session = scoped_session(
            sessionmaker(bind=db.get_engine(ESC_DB_NAMED_BIND))
        )

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
    def get_esc_product(
        self,
        name: str,
        source: int,
    ):
        dataProduct: ESCProductSql | None = (
            Vegi_ESC_Repo.db_session.query(ESCProductSql)
            .filter(and_(ESCProductSql.name == name, ESCProductSql.source == source))
            .first()
        )
        return dataProduct.fetch() if dataProduct is not None else None

    @appcontext
    def add_product_if_not_exists(
        self: Self,
        name: str,
        product_external_id_on_source: str,
        # vendorInternalId: str,
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
        finalLocation: str,
        taxGroup: str,
        dateOfBirth: datetime,
        finalDate: datetime,
    ):  
        # Query the joined data
        dataProduct: ESCProductSql | None = (
            Vegi_ESC_Repo.db_session.query(ESCProductSql)
            .filter(and_(ESCProductSql.name == name, ESCProductSql.source == source))
            .first()
        )
        if dataProduct is None:
            try:
                new_esc_product = ESCProductSql(
                    # all params from args to function.
                    name=name,
                    product_external_id_on_source=product_external_id_on_source,
                    source=source,
                    description=description,
                    category=category,
                    keyWords=keyWords,
                    imageUrl=imageUrl,
                    ingredients=ingredients,
                    packagingType=packagingType,
                    stockUnitsPerProduct=int(stockUnitsPerProduct),
                    sizeInnerUnitValue=float(sizeInnerUnitValue),
                    sizeInnerUnitType=sizeInnerUnitType,
                    productBarCode=productBarCode,
                    supplier=supplier,
                    brandName=brandName,
                    origin=origin,
                    taxGroup=taxGroup,
                    dateOfBirth=dateOfBirth,
                )
            
                db.session.add(new_esc_product)
                db.session.commit()
                db.session.refresh(new_esc_product)
                logger.verbose(f"ESCProduct created. ESCProduct.id={new_esc_product.id}")
                dataProduct = new_esc_product
            except Exception as e:
                logger.error(e)
                return None
        return dataProduct.fetch()
    
    # @appcontext
    # def rate_product(
    #     self, 
    #     name: str,
    #     product_external_id_on_source: str,
    #     # vendorInternalId: str,
    #     source: int,
    #     description: str,
    #     category: str,
    #     keyWords: list[str],
    #     imageUrl: str,
    #     ingredients: str,
    #     packagingType: str,
    #     stockUnitsPerProduct: int,
    #     sizeInnerUnitValue: float,
    #     sizeInnerUnitType: str,
    #     productBarCode: str,
    #     supplier: str,
    #     brandName: str,
    #     origin: str,
    #     finalLocation: str,
    #     taxGroup: str,
    #     dateOfBirth: datetime,
    #     finalDate: datetime,
    # ):
        
    #     # Query the joined data
    #     dataProduct: ESCProductSql | None = (
    #         Vegi_ESC_Repo.db_session.query(ESCProductSql)
    #         .filter(and_(ESCProductSql.name == name, ESCProductSql.source == source))
    #         .first()
    #     )
    #     if dataProduct is None:
            
    #         new_esc_product = ESCProductSql(
    #             # all params from args to function.
    #             name=name,
    #             product_external_id_on_source=product_external_id_on_source,
    #             source=source,
    #             description=description,
    #             category=category,
    #             keyWords=keyWords,
    #             imageUrl=imageUrl,
    #             ingredients=ingredients,
    #             packagingType=packagingType,
    #             stockUnitsPerProduct=stockUnitsPerProduct,
    #             sizeInnerUnitValue=sizeInnerUnitValue,
    #             sizeInnerUnitType=sizeInnerUnitType,
    #             productBarCode=productBarCode,
    #             supplier=supplier,
    #             brandName=brandName,
    #             origin=origin,
    #             taxGroup=taxGroup,
    #             dateOfBirth=dateOfBirth,
    #         )
    #         db.session.add(new_esc_product)
    #         db.session.commit()
    #         db.session.refresh(new_esc_product)
    #         logger.verbose(f"ESCProduct created. ESCProduct.id={new_esc_product.id}")
    #         dataProduct = new_esc_product
        
    #     # now we check for a rating on the product
    #     dataProductRating: ESCRatingSql | None = (
    #         Vegi_ESC_Repo.db_session.query(ESCRatingSql)
    #         .filter(ESCRatingSql.product == dataProduct.id)
    #         .first()
    #     )
    #     if dataProductRating is None:
    #         # todo: create a ESCRating line by calling the rate function from the app.py... logic... which will take care of creating the rating and return the rating id for us.
    #         new_esc_rating = ESCRatingSql(
    #             product=dataProduct.id,
    #             product_id=product_source_id,
    #             product_name=name,
    #             calculated_on=datetime.now(),
    #             rating=...
    #         )
    #         db.session.add(new_esc_rating)
    #         db.session.commit()
    #         db.session.refresh(new_esc_rating)
    #         logger.verbose(f"ESCProduct created. ESCProduct.id={new_esc_rating.id}")
    #         dataProductRating = new_esc_rating
    #     data: list[Row] = (
    #         Vegi_ESC_Repo.db_session.query(ESCRatingSql, ESCExplanationSql, ESCSourceSql)
    #         .join(
    #             ESCExplanationSql,
    #             ESCExplanationSql.rating == ESCRatingSql.id,
    #         )
    #         .join(
    #             ESCSourceSql,
    #             ESCSourceSql.id == ESCExplanationSql.source,
    #         )
    #         .filter(ESCRatingSql.id == dataProductRating.id)
    #         .all()
    #     )
    #     return [
    #         {
    #             'rating': ESCRatingSql.fetch(),
    #             'explanation': ESCExplanationSql.fetch(),
    #             'source': ESCSourceSql.fetch(),
    #         } for ESCRatingSql, ESCExplanationSql, ESCSourceSql in data
    #     ]

    @appcontext
    def get_items(self, item_source: str, category_name: str) -> list[ESCProductInstance]:
        # items_expire_after = datetime.now()
        try:
            source: ESCSourceSql | None = (
                ESCSourceSql
                .query
                .filter(or_(ESCSourceSql.domain == item_source, ESCSourceSql.name == item_source))
                .first()
            )
            if not source:
                return []
            items: list[ESCProductSql] = (
                ESCProductSql.query
                .filter(and_(ESCProductSql.source == source.id, ESCProductSql.category == category_name))
                .all()
            )
            assert isinstance(items, list)
            return [i.fetch() for i in items]
        except Exception as e:
            logger.error(str(e))
            return []
    
    @appcontext
    def get_categories(self, item_source: str) -> list[CachedSustainedItemCategory]:
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
            return [
                CachedSustainedItemCategory.fromJson(jsons.loads(c.item_json))
                for c in items
                if c.item_type == "category"
            ]
        except Exception as e:
            logger.error(str(e))
            return []

    @appcontext
    def get_sustained_categories(self):
        return self.get_categories(item_source=SUSTAINED_DOMAIN_NAME)
        
    @appcontext
    def get_sustained_items(self, category_name: str):
        return self.get_items(item_source=SUSTAINED_DOMAIN_NAME, category_name=category_name)
        # if not categories_with_products_embedded:
        #     return []
        # # return [
        # #     CachedSustainedItemCategory.fromJson(jsons.loads(c.item_json))
        # #     for c in categories_with_products_embedded
        # #     if c.item_type == "category"
        # # ]
        # return [
        #     CachedSustainedItemCategory(
                
        #     )
        # ]

    @appcontext
    def add_source(self, new_source: ESCSourceSql):
        ''' See https://www.notion.so/vegiapp/2446a82bdee94f0cb9e6d7671a78bb06?v=a4d348fe223645ed9b5d079e1abc2b1b&pvs=4 for list of sources in vegi notion'''
        # source = ESCSource(name="Napolina", source_type="Website", domain="https://napolina.com/", credibility=0)
        db.session.add(new_source)
        logger.verbose("ESCSource created. ESCSource id={}".format(new_source.id))
        db.session.commit()
        db.session.refresh(new_source)
        return new_source.fetch()

    # @appcontext
    # def add_rating(
    #     self,
    #     new_rating: ESCRatingSql,
    #     explanations: list[ESCExplanationCreate],
    #     source: int
    # ):
    #     # rating = ESCRating(product_name="Cannellini beans", product_id="ABC123", calculated_on=datetime.now())
    #     db.session.add(new_rating)
    #     db.session.commit()
    #     db.session.refresh(new_rating)
    #     assert new_rating.id is not None, 'a new ESC rating in vegi_esc_repo cannot have null rating id'
    #     for explanation in explanations:
    #         db.session.add(ESCExplanationSql(
    #             title=explanation.title,
    #             measure=explanation.measure,
    #             reasons=explanation.reasons,
    #             evidence=explanation.evidence,
    #             rating=new_rating.id,
    #             source=source,
    #         ))

    #     db.session.commit()
    #     logger.verbose(
    #         f"ESCRating created with {len(explanations)} explanations. ESCRating id={new_rating.id}"
    #     )
        
    #     return new_rating.fetch()
    
    @appcontext
    def add_rating_for_product(
        self,
        # product_id: int,
        new_rating: ESCRatingSql,
        explanationsCreate: list[ESCExplanationCreate],
        source: int,
    ):
        # rating = ESCRating(product_name="Cannellini beans", product_id="ABC123", calculated_on=datetime.now())
        # Dont assert below, we want to add raitngs for products using the ratings of similar products from ssutained.
        # assert new_rating.product == product_id, "Cannot add new rating for non-matching product_id parameter"
        try:
            Vegi_ESC_Repo.db_session.add(new_rating)
            Vegi_ESC_Repo.db_session.commit()
            Vegi_ESC_Repo.db_session.refresh(new_rating)
            explanations = [
                ESCExplanationSql(
                    title=e.title,
                    measure=e.measure,
                    reasons=e.reasons,
                    evidence=e.evidence,
                    rating=new_rating.id,
                    source=source
                )
                for e in explanationsCreate
            ]
            for explanation in explanations:
                assert (
                    new_rating.id is not None
                ), "a new ESC rating in vegi_repo cannot have null rating id"
                Vegi_ESC_Repo.db_session.add(explanation)

            Vegi_ESC_Repo.db_session.commit()
            # VegiRepo.db_session.refresh(explanations)
            # logger.verbose(
            #     f"ESCRating created with {len(explanations)} explanations. ESCRating id={new_rating.id}"
            # )
            return NewRating(
                rating=new_rating.fetch(), explanations=[e.fetch() for e in explanations]
            )
        except Exception as e:
            logger.error(e)
            return None

    @appcontext
    def add_cached_items(self, items: list[CachedItemSql]):
        # ~ https://docs.sqlalchemy.org/en/20/orm/session_basics.html#:~:text=To%20add%20a%20list%20of%20items%20to%20the%20session%20at%20once%2C%20use%20Session.add_all()%3A
        db.session.add_all(items)
        db.session.commit()
        return self
