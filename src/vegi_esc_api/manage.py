# # manage.py
# # i.e. conda run python -m vegi_esc_api.manage.py reset_db
# # i.e. PYTHONPATH=/Users/joey/Github_Keep/vegi-esc-api/:/Users/joey/Github_Keep/vegi-esc-api/src/:/Users/joey/Github_Keep/vegi-esc-api/src/vegi_esc_api/: python src/vegi_esc_api/manage.py reset_db

# from vegi_esc_api.create_app import create_app
# from vegi_esc_api.extensions import db
# from vegi_esc_api.models import ESCSource, ESCRating, ESCExplanation, User
# from vegi_esc_api.vegi_esc_repo import CachedItemSql, ESCRatingSql, ESCExplanationSql, ESCSourceSql
# from vegi_esc_api.models_wrapper import CachedSustainedItemCategory
# from vegi_esc_api.sustained_models import SustainedCategory, SustainedProductBase
# from flask.cli import FlaskGroup
# from datetime import datetime
# from werkzeug.security import check_password_hash
# from werkzeug.security import generate_password_hash
# import jsons
# import json


# def _create_app():
#     app, vegi_db_session = create_app()
#     return app


# cli = FlaskGroup(create_app=_create_app)


# def check_password(password_hash, password):
#     return check_password_hash(password_hash, password)


# @cli.command('populate_db')
# def populate_db():
#     "populate reseted database"
#     # user = User(username='joeyd', password_hash=generate_password_hash('joeyd'))
#     # db.session.add(user)
#     # print("User created. User id={}".format(user.id))

#     source = ESCSourceSql(name="Napolina", source_type="Website", domain="https://napolina.com/", credibility=0)
#     db.session.add(source)
#     db.session.commit()
#     db.session.refresh(source)
#     assert source.id is not None
#     print("ESCSource created. ESCSource id={}".format(source.id))
#     # TODO: Populate a ESCProductSql with product.id=1
#     rating = ESCRatingSql(
#         product=1,
#         product_name="Cannellini beans", 
#         product_id="ABC123", 
#         rating=4.5, 
#         calculated_on=datetime.now()
#     )
#     db.session.add(rating)
#     db.session.commit()
#     db.session.refresh(rating)
#     assert rating.id is not None
#     print("ESCRating created. ESCRating id={}".format(rating.id))

#     explanation1 = ESCExplanationSql(
#         title="Packaging material",
#         reasons=["aluminium can is widely recycled including from within general waste"],
#         evidence="look at the can...",
#         measure=4,
#         rating=rating.id,
#         source=source.id,
#     )
#     explanation2 = ESCExplanationSql(
#         title="vegetarian",
#         reasons=["pulses are vegetarian and sustainable source of protein"],
#         evidence="look at the can...",
#         measure=5,
#         rating=rating.id,
#         source=source.id,
#     )
#     db.session.add(explanation1)
#     db.session.add(explanation2)
#     db.session.commit()
#     print("database creation completed")


# def get_localstorage_fn(category_id: str):
#     return f"localstorage/sustained-{category_id}.json"


# @cli.command('populate_sustained_from_localstorage')
# def populate_sustained_from_localstorage():
#     "populate reseted database with sustained data from localstorage"
#     with open('localstorage/sustained-categories.json', 'r') as f:
#         sustainedLocalData = json.load(f)
#         categoriesObjs = sustainedLocalData['categories']
#         if not isinstance(categoriesObjs, list):
#             raise Exception(
#                 'sustainedLocalData should hold a list of categories')
#         categories = [SustainedCategory.fromJson(c) for c in categoriesObjs]
#     for category in categories:
#         fn = get_localstorage_fn(category_id=category.id)
#         with open(fn, "r") as f:
#             productsObjs = json.load(f)
#         if not isinstance(productsObjs, list):
#             raise Exception("sustainedLocalData should hold a list of products")
#         products = [SustainedProductBase.fromJson(c) for c in productsObjs]
#         # return [c['name'] for c in products]
#         cachedItem = CachedItemSql(
#             item_name=category.name,
#             item_type='category',
#             item_source='sustained.com',
#             # item_json=jsons.dumps({
#             #     "category": {
#             #         "name": category.name,
#             #         "links": category.links,
#             #         "id": category.id,
#             #         "products": [p.toJson() for p in products]
#             #     }
#             # })
#             item_json=jsons.dumps(
#                 CachedSustainedItemCategory(
#                     name=category.name,
#                     id=category.id,
#                     links=category.links,
#                     products=products
#                 ).toJson()
#             )
#         )
#         db.session.add(cachedItem)
#     db.session.commit()
#     print("database - Add localstorage to db completed")


# @cli.command('reset_db')
# def recreate_db():
#     """delete and reset database"""
#     db.drop_all()
#     db.create_all()
#     db.session.commit()
#     print("database reset done!")


# if __name__ == "__main__":
#     cli()

#     # run: ```shell
#     # PYTHONPATH=/Users/joey/Github_Keep/vegi-esc-api/:/Users/joey/Github_Keep/vegi-esc-api/src/:/Users/joey/Github_Keep/vegi-esc-api/src/vegi_esc_api/: python src/vegi_esc_api/manage.py reset_db
#     # # OR
#     # PYTHONPATH=/Users/joey/Github_Keep/vegi-esc-api/:/Users/joey/Github_Keep/vegi-esc-api/src/:/Users/joey/Github_Keep/vegi-esc-api/src/vegi_esc_api/: python src/vegi_esc_api/manage.py populate_sustained_from_localstorage
#     # ```
