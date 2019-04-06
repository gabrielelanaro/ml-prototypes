import enum
import json

import numpy as np

from tornado.websocket import WebSocketHandler
from tornado.ioloop import IOLoop


from .async_utils import async_next, AsyncStopIteration
from .model import StyleTransfer, StyleTransferResult


class InvalidTransition(Exception):
    pass


class State(enum.Enum):
    INIT = "init"
    MODEL_LOADING = "model_loading"
    MODEL_LOADED = "model_loaded"
    START_ITERATION = "start_iteration"
    END_ITERATION = "end_iteration"
    END = "end"


class StyleTransferController:
    def __init__(self, ws: WebSocketHandler):
        self._ws = ws
        self.state = State.INIT
        self._model = None

    # Transitions
    async def load_model(self):
        if self.state != State.INIT:
            raise InvalidTransition()

        await self._set_and_send_state(State.MODEL_LOADING)

        # We run model initialization in a separate thread so that
        # we don't block the tornado event loop.
        loop = IOLoop.current()
        self._model: StyleTransfer = await loop.run_in_executor(
            None, self._do_load_model
        )

        await self._set_and_send_state(State.MODEL_LOADED)

    async def request_image(self, img):
        if self.state != State.MODEL_LOADED:
            raise InvalidTransition()

        # We can't run a for loop because we risk of blocking everything
        # TODO: this could be converted to an async for loop by creating an
        # asynchronous iterator, which may not be worth the time (just syntactinc sugar)
        loop = IOLoop.current()

        iterator = iter(self._model.run_style_transfer(img, img, num_iterations=2))
        while True:
            await self._set_and_send_state(State.START_ITERATION)

            try:
                style_results: StyleTransferResult = await loop.run_in_executor(
                    None, async_next, iterator
                )
            except AsyncStopIteration:
                await self._set_and_send_state(State.END)
                break

            await self._set_and_send_state(
                State.END_ITERATION, {"iteration": style_results.iteration_no}
            )

    async def request_image_old(self, img):
        if self.state != State.MODEL_LOADED:
            raise InvalidTransition()

        # We can't run a for loop because we risk of blocking everything
        # TODO: this could be converted to an async for loop by creating a function
        # async def iterate_in_executor(executor):
        #     pass
        loop = IOLoop.current()
        iterator = iter(self._model.run_style_transfer(img, img, num_iterations=10))
        while True:
            await self._set_and_send_state(State.START_ITERATION)

            style_results: StyleTransferResult = await loop.run_in_executor(
                None, next, iterator
            )

            await self._set_and_send_state(
                State.END_ITERATION, {"iteration": style_results.iteration_no}
            )

    async def _set_and_send_state(self, state: State, data=None):
        self.state = state
        await self._ws.write_message({"state": self.state.value, "data": data})

    def _do_load_model(self) -> StyleTransfer:
        return StyleTransfer()


class StyleTransferSocket(WebSocketHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller = StyleTransferController(self)

    async def open(self):
        # TODO: We load the model once the connection is open
        # an alternative would be to make the websocket load the model later on.
        await self._controller.load_model()

    async def on_message(self, message):
        print("Received message: {}".format(message))

        message = json.loads(message)

        if message["action"] == "close":
            self.close()
        if message["action"] == "request_image":
            # TODO: Implement parsing code
            # img = _parse_image(message["image"])
            img = np.random.randint(0, 255, (512, 512, 3)).astype("float32")
            await self._controller.request_image(img)
            self.close()
        else:
            raise Exception("invalid action")

    def on_close(self):
        # TODO: maybe add some logging here, and devise some way to stop the
        # iterations maybe
        print("Closing connection")

