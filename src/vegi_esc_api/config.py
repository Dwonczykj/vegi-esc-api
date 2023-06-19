import os

from dotenv import load_dotenv
basedir = os.path.abspath(os.path.dirname(__file__))

load_dotenv()
config = os.environ


class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    # SQLALCHEMY_DATABASE_URI = 'postgresql://pg_user:pg_pwd@pg_server/pg_db'
    SQLALCHEMY_DATABASE_URI = f"postgresql://{config['DATABASE_ESC_USERNAME']}:{config['DATABASE_ESC_PASSWORD']}!@{config['DATABASE_HOST']}:{config['DATABASE_PORT']}/{config['DATABASE_ESC_DBNAME']}"
    SQLALCHEMY_BINDS = {
        # 'vegi_bind': f"postgresql://@127.0.0.1/postgres"
        'vegi_bind': f"postgresql://{config['DATABASE_VEGI_USERNAME']}:{config['DATABASE_VEGI_PASSWORD']}!@{config['DATABASE_HOST']}:{config['DATABASE_PORT']}/{config['DATABASE_VEGI_DBNAME']}"
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
