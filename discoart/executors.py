import asyncio
import os
from typing import Dict

from jina import Executor, requests, DocumentArray


class DiscoArtExecutor(Executor):
    skip_event = asyncio.Event()
    stop_event = asyncio.Event()

    @requests(on='/create')
    async def create_artworks(self, parameters: Dict, **kwargs):
        await asyncio.get_event_loop().run_in_executor(None, self._create, parameters)

    def _create(self, parameters: Dict, **kwargs):
        from .create import create

        return create(
            skip_event=self.skip_event, stop_event=self.stop_event, **parameters
        )

    @requests(on='/skip')
    async def skip_create(self, **kwargs):
        self.skip_event.set()

    @requests(on='/stop')
    async def stop_create(self, **kwargs):
        self.stop_event.set()


class ResultPoller(Executor):
    @requests(on='/result')
    def poll_results(self, parameters: Dict, **kwargs):
        from discoart.helper import get_output_dir

        path = os.path.join(
            get_output_dir(parameters['name_docarray']),
            'da.protobuf.lz4',
        )
        if os.path.exists(path):
            return DocumentArray.load_binary(path)
