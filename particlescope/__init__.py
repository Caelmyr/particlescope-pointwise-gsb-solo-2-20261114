from pathlib import Path
import logging
from flask import Flask, jsonify, render_template
from werkzeug.exceptions import HTTPException
from .storage import Store, ConflictError
from .service import AnalysisService
from .recipes import RecipeService, ensure_builtins
from .jobs import JobService
from .api import create_api


def create_app(data_dir=None):
    root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(root / 'templates'),
        static_folder=str(root / 'static'),
    )
    app.config['MAX_CONTENT_LENGTH'] = 96 * 1024 * 1024
    app.json.ensure_ascii = False
    store = Store(data_dir or root / 'data')
    from .migration import migrate_first_version
    migrate_first_version(store, app.logger)
    service = AnalysisService(store)
    recipes = RecipeService(store)
    ensure_builtins(recipes)
    jobs = JobService(store, service, recipes)
    app.extensions['store'] = store
    app.extensions['analysis'] = service
    app.extensions['recipes'] = recipes
    app.extensions['jobs'] = jobs
    app.register_blueprint(create_api(store, service, recipes, jobs))

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.errorhandler(ConflictError)
    def conflict(error):
        return jsonify({'error': str(error), 'type': 'conflict'}), 409

    @app.errorhandler(ValueError)
    def validation(error):
        return jsonify({'error': str(error), 'type': 'validation'}), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        message = '请求体不能超过96MB' if error.code == 413 else error.description
        return jsonify({'error': message, 'type': 'http'}), error.code

    @app.errorhandler(Exception)
    def unexpected(error):
        app.logger.exception('处理请求失败')
        return jsonify({'error': '处理失败，请检查本地服务日志', 'type': 'internal'}), 500

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        if response.mimetype == 'application/json':
            response.headers['Cache-Control'] = 'no-store'
        return response

    return app
