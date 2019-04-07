"""Websocket client for testing"""
import json

from tornado.websocket import websocket_connect
from tornado.ioloop import IOLoop

from .controller import State


class WebsocketClient:
    """Websocket client for testing our style transfer server"""

    def __init__(self, url: str):
        self.url = url

    async def connect(self):
        self.ws = await websocket_connect(self.url)

        while True:
            msg = await self.ws.read_message()
            print("Message received from server:\n#{}#".format(msg))
            if msg is None:
                # Con nection is closed
                print("Connection is closed")
                break

            msg = json.loads(msg)
            if msg["state"] == State.MODEL_LOADED.value:
                # TODO: send a random image
                await self.ws.write_message(json.dumps({"action": "request_image"}))
            else:
                print("MSG IS {}".format(msg["state"]))


if __name__ == "__main__":
    loop = IOLoop.instance()
    client = WebsocketClient("ws://localhost:8000/styletransfer")
    loop.spawn_callback(client.connect)
    loop.start()
