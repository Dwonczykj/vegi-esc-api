from vegi_esc_api.views import base_app
from vegi_esc_api.config import DevelopmentConfig
from vegi_esc_api.vegi_repo_models import (
    VegiESCRatingSql,
    VegiProductSql,
    VegiUserSql,
    VEGI_DB_NAMED_BIND,
)
from sqlalchemy.orm import scoped_session, sessionmaker
import vegi_esc_api.logger as logger
from dotenv import load_dotenv
from flask import Flask
from flask.helpers import get_root_path
from flask_login import login_required
from flask_cors import CORS
from dash import Dash
import dash

cors = CORS()


def create_app(script_info=None):
    # instantiate app
    server = Flask(__name__)

    # set configs
    load_dotenv()
    # set flask app settings from environmental variables set in docker-compose // src/vegi_esc_api/config.py:10
    # app_settings = os.getenv("APP_SETTINGS") # allows us to change based on heroku vars.
    # server.config.from_object(app_settings)
    server.config.from_object(DevelopmentConfig)

    # ~ https://www.notion.so/gember/Python-Cheats-8d7b0cc6f58544ef888ea36bb5879141?pvs=4#53244313e8634abb9ef4b32b77f91c92
    server.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    if (
        "SQLALCHEMY_DATABASE_URI" not in server.config.keys()
        and "SQLALCHEMY_BINDS" not in server.config.keys()
    ):
        raise RuntimeError(
            "Either 'SQLALCHEMY_DATABASE_URI' or 'SQLALCHEMY_BINDS' must be set on the server app_settings from os.getenv(\"APP_SETTINGS\")."
        )

    # set extensions
    cors.init_app(
        server,
        resources={r"*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}},
        supports_credentials=True,
    )

    (db, migrate) = register_extensions(server)

    # must be registered after db
    register_dashapps(server)
    register_blueprints(server)

    server.register_blueprint(base_app)

    # TODO: register api
    # from app.api import api
    # api.init_app(app)

    @server.shell_context_processor
    def ctx():
        return {"app": server, "db": db}

    with server.app_context():
        # Create the scoped session
        vegi_db_session = scoped_session(
            sessionmaker(bind=db.get_engine(VEGI_DB_NAMED_BIND))
        )

        # Bind the scoped session to the model
        VegiUserSql.query = vegi_db_session.query_property()
        VegiProductSql.query = vegi_db_session.query_property()
        VegiESCRatingSql.query = vegi_db_session.query_property()
        
    return server, vegi_db_session


def register_dashapps(app):
    # from vegi_esc_api.dash_apps.plotly_dash.plot_3d_umap_layout import layout
    # from vegi_esc_api.dash_apps.plotly_dash.callbacks import register_callbacks

    # Meta tags for viewport responsiveness
    meta_viewport = {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1, shrink-to-fit=no",
    }

    # dashapp1 = dash.Dash(__name__,
    #                      server=app,
    #                      url_base_pathname='/dummy-dashboard/',
    #                      assets_folder=get_root_path(__name__) + '/dummy-dashboard/assets/',
    #                      meta_tags=[meta_viewport])

    # with app.app_context():
    #     dashapp1.title = 'Dashapp 1'
    #     dashapp1.layout = layout
    #     register_callbacks(dashapp1)

    # _protect_dashviews(dashapp1)

    from vegi_esc_api.dash_apps.plotly_dash.plot_3d_wordvec_embedding import (
        layout,
        external_stylesheets,
        register_callbacks,
    )

    dashapp2 = dash.Dash(
        __name__,
        server=app,
        url_base_pathname="/dashboard/",
        assets_folder=get_root_path(__name__) + "/dashboard/assets/",
        meta_tags=[meta_viewport],
        external_stylesheets=external_stylesheets,
    )

    with app.app_context():
        dashapp2.title = "vegi word similarity vis"
        dashapp2.layout = layout
        register_callbacks(app, dashapp2)

    _protect_dashviews(dashapp2)


def _protect_dashviews(dashapp: Dash):
    if not dashapp.server:
        return
    for view_func in dashapp.server.view_functions:
        if view_func.startswith(dashapp.config.url_base_pathname):
            dashapp.server.view_functions[view_func] = login_required(
                dashapp.server.view_functions[view_func]
            )


def register_extensions(server: Flask):
    from vegi_esc_api.extensions import db
    from vegi_esc_api.extensions import migrate
    from vegi_esc_api.extensions import login

    logger.verbose("Registering flask dbs to flask app instance...")
    db.init_app(server)
    # # Create the scoped session
    # session = scoped_session(sessionmaker(bind=db.get_engine(VEGI_DB_NAMED_BIND)))  # Replace with your named bind key
    logger.verbose("  -  ✅ Registered flask dbs")
    migrate.init_app(server, db)
    logger.verbose("  -  ✅ Migration initialised for flask dbs")
    login.init_app(server)
    logger.verbose("  -  ✅ Login for Flask App initialised")
    login.login_view = "main.login"  # type: ignore
    return (
        db,
        migrate,
    )


def register_blueprints(server: Flask):
    from vegi_esc_api.webapp import server_bp

    server.register_blueprint(server_bp)
