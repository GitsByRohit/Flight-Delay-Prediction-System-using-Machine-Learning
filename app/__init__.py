from flask import Flask
from app.database import init_db

def create_app():

    app = Flask(__name__)
    app.secret_key = "supersecretkey"

    # Initialize database
    init_db()

    from app.routes.auth_routes import auth_bp
    from app.routes.prediction_routes import prediction_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(prediction_bp)

    return app
