from typing import Dict

from jina import Executor, requests, Flow

from . import __resources_path__


class DiscoArtExecutor(Executor):
    @requests
    def create_artworks(self, parameters: Dict, **kwargs):
        from .create import create

        return create(**parameters, **kwargs)


if __name__ == '__main__':
    with Flow.load_config(os.path.join(__resources_path__, 'flow.yml')):
        f.block()
