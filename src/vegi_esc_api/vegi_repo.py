from __future__ import annotations
from dataclasses import dataclass
import sys
import traceback

from flask import Flask
from datetime import datetime, timedelta
from typing import Tuple, NamedTuple, Any, Union
from sqlalchemy import or_, func
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.engine.row import Row
from vegi_esc_api.vegi_repo_models import (
    # VegiESCSourceSql,
    # VegiESCExplanationSql,
    VegiESCRatingSql,
    VegiProductSql,
    VegiUserSql,
    VEGI_DB_NAMED_BIND,
    VegiProductCategorySql,
    VegiProductCategoryInstance,
    VegiCategoryGroupSql,
    _DBFetchable,
)
from vegi_esc_api.repo_models_base import appcontext
from vegi_esc_api.models import (
    VegiESCRatingInstance,
    VegiProductInstance,
    # VegiESCExplanationInstance,
)
from vegi_esc_api.extensions import db
import vegi_esc_api.logger as Logger


# @dataclass
# class NewRating:
#     rating: VegiESCRatingInstance
#     explanations: list[VegiESCExplanationInstance]

#     def serialize(self):
#         return {
#             "rating": self.rating.serialize(),
#             "explanations": [e.serialize() for e in self.explanations],
#         }


class VegiRepo:
    def __init__(
        self,
        app: Flask,
    ):
        """
        Connect to db hosted in heroku if flask config is set to remote db connection string
        """
        VegiRepo.app = app
        VegiRepo.db_session = scoped_session(
            sessionmaker(bind=db.get_engine(VEGI_DB_NAMED_BIND))
        )
        Logger.verbose(f'{type(self).__name__} DB Session connection: {type(self).db_session} using named bind: "{VEGI_DB_NAMED_BIND}"')

    @appcontext
    def get_users(self):
        users: list[VegiUserSql] = VegiUserSql.query.all()
        assert isinstance(users, list)
        return [e.fetch() for e in users]

    @appcontext
    def get_products(self):
        products: list[VegiProductSql] = VegiProductSql.query.all()
        assert isinstance(products, list)
        return [e.fetch() for e in products]

    @appcontext
    def get_products_with_categories(self, vendor: int | None = None) -> list[Tuple[VegiProductInstance, VegiProductCategoryInstance]]:
        if vendor:
            products: list[Union[Tuple[VegiProductSql, VegiProductCategorySql], Row]] = (
                VegiRepo.db_session.query(VegiProductSql, VegiProductCategorySql)
                .join(
                    VegiProductCategorySql,
                    VegiProductSql.category == VegiProductCategorySql.id,
                )
                .filter(VegiProductSql.vendor == vendor)
                .all()
            )
        else:
            products: list[Union[Tuple[VegiProductSql, VegiProductCategorySql], Row]] = (
                VegiRepo.db_session.query(VegiProductSql, VegiProductCategorySql)
                .join(
                    VegiProductCategorySql,
                    VegiProductSql.category == VegiProductCategorySql.id,
                )
                .all()
            )
        assert isinstance(products, list)
        return [(p.fetch(), cat.fetch()) for (p, cat) in products]
    
    @appcontext
    def get_product_categories(self, vendor: int | None = None) -> list[VegiProductCategoryInstance]:
        if vendor:
            products: list[VegiProductCategorySql] = (
                VegiRepo.db_session.query(VegiProductCategorySql)
                .filter(VegiProductCategorySql.vendor == vendor)
                .all()
            )
        else:
            products: list[VegiProductCategorySql] = (
                VegiRepo.db_session.query(VegiProductCategorySql)
                .all()
            )
        assert isinstance(products, list)
        return [p.fetch() for p in products]

    @appcontext
    def get_product_category_details(self, product_id: int) -> Tuple[VegiProductInstance | None, list[VegiProductInstance] | None, VegiProductCategorySql | None]:
        try:
            dataProduct: VegiProductSql | None = (
                VegiRepo.db_session.query(VegiProductSql)
                .filter(VegiProductSql.id == product_id)
                .first()
            )
            Logger.info(dataProduct)
            if dataProduct is None:
                Logger.error(
                    f'Unable to locate product in vegi db with id: "{id}" for vegi_repo.get_product_category_details'
                )
                return (None, None, None)
            dataProductsInSameCategory: list[
                Union[Tuple[VegiProductSql, VegiProductCategorySql], Row]
            ] = (
                VegiRepo.db_session.query(VegiProductSql, VegiProductCategorySql)
                .join(
                    VegiProductCategorySql,
                    VegiProductSql.category == VegiProductCategorySql.id,
                )
                .filter(VegiProductSql.category == dataProduct.category)
                .all()
            )
            return (
                dataProduct.fetch(),
                [p.fetch() for (p, cat) in dataProductsInSameCategory],
                next((cat for (p, cat) in dataProductsInSameCategory)).fetch(),  # type: ignore
                # dataProductsInSameCategory[0]['VegiProductCategorySql'].fetch() if 'VegiProductCategorySql' in dataProductsInSameCategory[0] else None,  # type: ignore
            )
        except Exception as e:
            Logger.error(e)
            return (None, None, None)

    # @appcontext
    # def get_product_ratings(self, limit: int):
    #     limit = max(1, int(limit))
    #     # Query the joined data
    #     data = (
    #         VegiRepo.db_session.query(VegiESCRatingSql, VegiESCExplanationSql)
    #         .join(
    #             VegiESCExplanationSql,
    #             VegiESCExplanationSql.escrating == VegiESCRatingSql.id,
    #         )
    #         .limit(limit)
    #         .all()
    #     )

        # Convert the result to a list of dictionaries for JSONification
        # instead return ratings objects, what is returned here atm?
        # return [(rating.fetch(), explanation.fetch()) for rating, explanation in data]
        # result = [
        #     {
        #         # "product_id": product.id,
        #         # "product_name": product.product_name,
        #         # "client_id": client.id,
        #         # "client_name": client.name
        #         **rating.serialize(),
        #         **explanation.serialize(),
        #     } for rating, explanation in data
        # ]

        # return result

    @appcontext
    def get_products_and_ratings_from_ids(
        self, ids: list[int]
    ) -> tuple[list[VegiProductInstance], list[VegiESCRatingInstance]]:
        data = (
            VegiRepo.db_session.query(
                VegiProductSql, VegiESCRatingSql
            )
            .join(
                VegiESCRatingSql,
                VegiESCRatingSql.product == VegiProductSql.id,
                isouter=True,
            )
            .filter(VegiProductSql.id.in_(ids))  # type: ignore
            .order_by(VegiESCRatingSql.calculatedOn)
            .all()
        )
        # .filter(or_(VegiESCRating.calculatedOn.is_(None), VegiESCRating.calculatedOn >= n_days_ago))\
        # .with_entities(VegiProduct.id)\
        # .limit(limit)\
        # .distinct()\
        # product_ids = [product_id for (product_id, ) in data]
        products: list[VegiProductSql] = [
            VegiProduct for (VegiProduct, VegiESCRating, VegiESCExplanation) in data
        ]
        ratings: list[VegiESCRatingSql] = [
            VegiESCRating for (VegiProduct, VegiESCRating, VegiESCExplanation) in data
        ]
        # result = [
        #     {
        #         # "product_id": product.id,
        #         # "product_name": product.product_name,
        #         # "client_id": client.id,
        #         # "client_name": client.name
        #         **rating.serialize(),
        #         **explanation.serialize(),
        #     } for product, rating, explanation in data
        # ]

        return [p.fetch() for p in products], [r.fetch() for r in ratings]

    @appcontext
    def get_products_to_rate(self, limit: int, days_ago: int = 5) -> list[int]:
        limit = max(1, int(limit))
        days_ago = min(max(1, int(days_ago)), 180)
        # Get the date `days_ago` days ago from now
        n_days_ago = datetime.now() - timedelta(days=days_ago)

        # Query the data ordered by created_date and get distinct names
        # ? This function uses .with_entities() to only select the name column
        # ? and then orders the results by created_date.
        # ? It also uses .distinct() to get unique names.
        # users = User.query.with_entities(User.name).order_by(User.created_date).distinct().all()

        # # Extract names from the result tuples
        # names = [user.name for user in users]

        # NOTE This function uses .with_entities() to only select the id column from the products table
        # NOTE we need the .with_entities() call so that distinct() only opereates on these entities and not the whole row result
        # Query the joined data
        Logger.warn(f'"{VegiRepo.db_session.bind.url}"')
        # in this query below, we need .with_entities() to use the distinct clause
        data = (
            VegiRepo.db_session.query(
                VegiProductSql, VegiESCRatingSql
            )
            .with_entities(VegiProductSql.id, func.max(VegiESCRatingSql.calculatedOn))
            .group_by(VegiProductSql.id)
            .having(
                or_(
                    func.count(VegiESCRatingSql.id) == 0,
                    func.max(VegiESCRatingSql.calculatedOn).is_(None),
                    func.max(VegiESCRatingSql.calculatedOn)
                    <= n_days_ago,  # rating expires
                )
            )
            .join(
                VegiESCRatingSql,
                VegiESCRatingSql.product == VegiProductSql.id,
                isouter=True,
            )
            .order_by(func.max(VegiESCRatingSql.calculatedOn))
            .limit(limit)
            .all()
        )
        # data = VegiRepo.db_session\
        #     .query(VegiProduct, VegiESCRating, VegiESCExplanation)\
        #     .with_entities(VegiProduct.id)\
        #     .join(VegiESCRating, VegiESCRating.product == VegiProduct.id, isouter=True)\
        #     .join(VegiESCExplanation, VegiESCExplanation.escrating == VegiESCRating.id, isouter=True)\
        #     .filter(or_(VegiESCRating.calculatedOn.is_(None), VegiESCRating.calculatedOn >= n_days_ago))\
        #     .order_by(VegiESCRating.calculatedOn)\
        #     .limit(limit)\
        #     .distinct()\
        #     .all()
        product_ids = [product_id for (product_id, max_calc_on) in data]
        # products = [product for (product, rating, explanation) in data]
        # result = [
        #     {
        #         # "product_id": product.id,
        #         # "product_name": product.product_name,
        #         # "client_id": client.id,
        #         # "client_name": client.name
        #         **rating.serialize(),
        #         **explanation.serialize(),
        #     } for product, rating, explanation in data
        # ]

        return product_ids
    
    # @appcontext
    # def get_sources(self, source_type: str | None = None):
    #     try:
    #         sources: list[VegiESCSourceSql] = (
    #             VegiESCSourceSql.query.all()
    #             if source_type is None
    #             else VegiESCSourceSql.query.filter(
    #                 VegiESCSourceSql.source_type == source_type
    #             ).all()
    #         )
    #         assert isinstance(sources, list)
    #         return [e.fetch() for e in sources]
    #     except Exception as e:
    #         logger.error(str(e))
    #         return []

    # @appcontext
    # def add_source(self, new_source: VegiESCSourceSql):
    #     # source = ESCSource(name="Napolina", source_type="Website", domain="https://napolina.com/", credibility=0)
    #     VegiRepo.db_session.add(new_source)
    #     VegiRepo.db_session.commit()
    #     # VegiRepo.db_session.refresh(new_source)
    #     # logger.verbose("ESCSource created. ESCSource id={}".format(new_source.id))
    #     return self

    # @appcontext
    # def add_rating_for_product(
    #     self,
    #     product_id: int,
    #     new_rating: VegiESCRatingSql,
    #     explanations: list[VegiESCExplanationSql],
    # ):
    #     # rating = ESCRating(product_name="Cannellini beans", product_id="ABC123", calculated_on=datetime.now())
    #     # Dont assert below, we want to add raitngs for products using the ratings of similar products from ssutained.
    #     # assert new_rating.product == product_id, "Cannot add new rating for non-matching product_id parameter"
    #     sources = list(set((e.escsource for e in explanations)))
        
    #     try:
    #         VegiRepo.db_session.add(new_rating)
    #         VegiRepo.db_session.commit()
    #         VegiRepo.db_session.refresh(new_rating)
    #         for explanation in explanations:
    #             explanation.rating = new_rating.id
    #             assert (
    #                 new_rating.id is not None
    #             ), "a new ESC rating in vegi_repo cannot have null rating id"
    #             VegiRepo.db_session.add(explanation)

    #         VegiRepo.db_session.commit()
    #         # VegiRepo.db_session.refresh(explanations)
    #         # logger.verbose(
    #         #     f"ESCRating created with {len(explanations)} explanations. ESCRating id={new_rating.id}"
    #         # )
    #         return NewRating(
    #             rating=new_rating.fetch(), explanations=[e.fetch() for e in explanations]
    #         )
    #     except Exception as e:
    #         logger.error(e)
    #         return None

    
