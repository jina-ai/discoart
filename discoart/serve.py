import os
import sys
from typing import Dict

from jina import Executor, requests, Flow

from . import __resources_path__


class DiscoArtExecutor(Executor):
    @requests
    def create_artworks(self, parameters: Dict, **kwargs):
        from .create import create

        return create(**parameters, **kwargs)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        if sys.argv[1] == '-i':
            _input = sys.stdin.read()
        else:
            _input = sys.argv[1]
    else:
        _input = os.path.join(__resources_path__, 'flow.yml')

    with Flow.load_config(_input):
        f.block()
