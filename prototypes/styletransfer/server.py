import json

import numpy as np

from tornado.websocket import WebSocketHandler

from .controller import StyleTransferController


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

