import os
from typing import Dict

from jina import Executor, requests, DocumentArray


class DiscoArtExecutor(Executor):
    @requests(on='/create')
    def create_artworks(self, parameters: Dict, **kwargs):
        from .create import create

        return create(**parameters)


class ResultPoller(Executor):
    @requests(on='/result')
    def poll_results(self, parameters: Dict, **kwargs):
        path = f'{parameters["name_docarray"]}.protobuf.lz4'
        if os.path.exists(path):
            return DocumentArray.load_binary(path)
