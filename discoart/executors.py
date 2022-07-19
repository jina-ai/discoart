import multiprocessing
import os
from typing import Dict

from jina import Executor, requests, DocumentArray


class DiscoArtExecutor(Executor):
    skip_event = multiprocessing.Event()
    stop_event = multiprocessing.Event()

    @requests(on='/create')
    def create_artworks(self, parameters: Dict, **kwargs):
        from .create import create

        return create(
            skip_event=self.skip_event, stop_event=self.stop_event, **parameters
        )

    @requests(on='/skip')
    def skip_create(self, **kwargs):
        self.skip_event.set()

    @requests(on='/stop')
    def stop_create(self, **kwargs):
        self.stop_event.set()


class ResultPoller(Executor):
    @requests(on='/result')
    def poll_results(self, parameters: Dict, **kwargs):
        path = f'{parameters["name_docarray"]}.protobuf.lz4'
        if os.path.exists(path):
            return DocumentArray.load_binary(path)
