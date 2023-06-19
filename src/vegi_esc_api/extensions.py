from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
# from sqlalchemy.orm import scoped_session, sessionmaker

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
