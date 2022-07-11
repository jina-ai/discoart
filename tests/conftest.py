import os

os.environ['DISCOART_LOG_LEVEL'] = 'DEBUG'

import tempfile

import pytest

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(
    os.path.join(cur_dir, 'unit', 'array', 'docker-compose.yml')
)


@pytest.fixture(autouse=True)
def tmpfile(tmpdir):
    tmpfile = f'discoart_test_{next(tempfile._get_candidate_names())}.db'
    return tmpdir / tmpfile


@pytest.fixture(scope='session')
def set_env_vars(request):
    _old_environ = dict(os.environ)
    os.environ.update(request.param)
    yield
    os.environ.clear()
    os.environ.update(_old_environ)
