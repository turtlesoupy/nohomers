import time
import json
import aiohttp_jinja2
import aiohttp
import functools
import base64
from aiohttp import web
import pydash as py_
import copy
import uuid
import logging


def json_error(klass, error_type, message=None):
    return klass(
        text=json.dumps({"error": {"type": error_type, "message": message}}),
        content_type="application/json"
    )


def json_response(dict):
    return web.Response(
        text=json.dumps(dict),
        content_type="application/json",
    )


class Handlers:
    @property
    def routes(self):
        return [
            web.get("/", self.index),
        ]

    def __init__(self):
        pass

    async def on_startup(self, app):
        pass

    async def on_cleanup(self, app):
        pass

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        return {}
