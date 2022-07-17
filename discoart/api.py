import os
from typing import Dict

os.environ['DISCOART_DISABLE_IPYTHON'] = '1'  # turn on when using from CLI

from yaml import Loader
import yaml
from . import __resources_path__


def serve(args):
    from jina import Flow
    from .executors import DiscoArtExecutor, ResultPoller

    with Flow.load_config(args.input) as f:
        f.block()


def config(args):
    with open(
        os.environ.get(
            'DISCOART_DEFAULT_PARAMETERS_YAML',
            os.path.join(__resources_path__, 'default.yml'),
        )
    ) as fp:
        args.output.write(fp.read())
        args.output.close()


def create(args):
    kwargs = yaml.load(args.input, Loader=Loader)
    from .create import create

    create(**kwargs)
