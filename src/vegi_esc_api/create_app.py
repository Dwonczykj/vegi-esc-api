from vegi_esc_api.views import base_app
from dotenv import load_dotenv
from flask import Flask
from flask.helpers import get_root_path
from flask_login import login_required
from flask_cors import CORS
from dash import Dash
import dash
import os

cors = CORS()


def create_app(script_info=None):

    load_dotenv()

    # set flask app settings from environmental variables set in docker-compose
    app_settings = os.getenv("APP_SETTINGS")

    # instantiate app
    server = Flask(__name__)
    
    # set configs
    server.config.from_object(app_settings)
    server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # set extensions
    cors.init_app(
        server,
        resources={r"*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}},
        supports_credentials=True
    )

    register_dashapps(server)
    (db, migrate) = \
        register_extensions(server)
    register_blueprints(server)

    server.register_blueprint(base_app)

    # TODO: register api
    # from app.api import api
    # api.init_app(app)

    @server.shell_context_processor
    def ctx():
        return {"app": server, "db": db}

    return server


def register_dashapps(app):
    from vegi_esc_api.dash_apps.plotly_dash.plot_3d_umap_layout import layout
    from vegi_esc_api.dash_apps.plotly_dash.callbacks import register_callbacks

    # Meta tags for viewport responsiveness
    meta_viewport = {
        "name": "viewport",
        "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}

    dashapp1 = dash.Dash(__name__,
                         server=app,
                         url_base_pathname='/dashboard/',
                         assets_folder=get_root_path(__name__) + '/dashboard/assets/',
                         meta_tags=[meta_viewport])

    with app.app_context():
        dashapp1.title = 'Dashapp 1'
        dashapp1.layout = layout
        register_callbacks(dashapp1)

    _protect_dashviews(dashapp1)

    # dashapp2 = dash.Dash(__name__,
    #                      server=app,
    #                      url_base_pathname='/dashboard-2/',
    #                      assets_folder=get_root_path(__name__) + '/dashboard-2/assets/',
    #                      meta_tags=[meta_viewport])

    # with app.app_context():
    #     dashapp2.title = 'Dashapp 2'
    #     dashapp2.layout = layout
    #     register_callbacks(dashapp2)

    # _protect_dashviews(dashapp2)


def _protect_dashviews(dashapp: Dash):
    for view_func in dashapp.server.view_functions:
        if view_func.startswith(dashapp.config.url_base_pathname):
            dashapp.server.view_functions[view_func] = login_required(
                dashapp.server.view_functions[view_func])


def register_extensions(server: Flask):
    from vegi_esc_api.extensions import db
    from vegi_esc_api.extensions import migrate
    from vegi_esc_api.extensions import login

    db.init_app(server)
    migrate.init_app(server, db)
    login.init_app(server)
    login.login_view = 'main.login'
    return (
        db,
        migrate,
    )


def register_blueprints(server: Flask):
    from vegi_esc_api.webapp import server_bp

    server.register_blueprint(server_bp)
