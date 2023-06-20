import os

from dotenv import load_dotenv
basedir = os.path.abspath(os.path.dirname(__file__))

load_dotenv()
config = os.environ

_defaultUri = f"postgresql://{config['DATABASE_ESC_USERNAME']}:{config['DATABASE_ESC_PASSWORD']}!@{config['DATABASE_HOST']}:{config['DATABASE_PORT']}/{config['DATABASE_ESC_DBNAME']}"


class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = config['vegi_service_user_secret']
    # SQLALCHEMY_DATABASE_URI = 'postgresql://pg_user:pg_pwd@pg_server/pg_db'
    SQLALCHEMY_DATABASE_URI = _defaultUri
    SQLALCHEMY_BINDS = {
        # 'vegi_bind': f"postgresql://@127.0.0.1/postgres"
        'vegi_bind': f"postgresql://{config['DATABASE_VEGI_USERNAME']}:{config['DATABASE_VEGI_PASSWORD']}!@{config['DATABASE_HOST']}:{config['DATABASE_PORT']}/{config['DATABASE_VEGI_DBNAME']}",
        'esc_bind': _defaultUri
    }


class ProductionConfig(Config):
    DEBUG = False


class StagingConfig(Config):
    DEVELOPMENT = True
    DEBUG = False


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
