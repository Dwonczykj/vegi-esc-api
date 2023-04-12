
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os

cors = CORS()
db = SQLAlchemy()


def create_app(script_info=None):

    load_dotenv()

    # set flask app settings from environmental variables set in docker-compose
    app_settings = os.getenv("APP_SETTINGS")

    # instantiate app
    app = Flask(__name__)

    # set configs
    app.config.from_object(app_settings)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # set extensions
    cors.init_app(
        app,
        resources={r"*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}},
        supports_credentials=True
    )
    db.init_app(app)

    # TODO: register api
    # from app.api import api
    # api.init_app(app)

    @app.shell_context_processor
    def ctx():
        return {"app": app, "db": db}

    return app
