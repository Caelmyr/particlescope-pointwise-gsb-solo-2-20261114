import os
from particlescope import create_app

app = create_app(os.environ.get('PARTICLESCOPE_DATA_DIR'))

if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get('PORT', 8894))
    print(f'ParticleScope: http://127.0.0.1:{port}', flush=True)
    serve(app, host='127.0.0.1', port=port, threads=6)
