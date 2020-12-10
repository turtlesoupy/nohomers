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
from pathlib import Path
from dataclasses import dataclass


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

@dataclass
class ContentItem:
    url: str

class ContentIndex:
    def __init__(self, manifest_path: Path, manifest_dir_url: str):
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.manifest_dir_url = manifest_dir_url
    
    def random_item(self):
        item = py_.sample(self.manifest)
        return ContentItem(
            url=f"{self.manifest_dir_url}/images/{item['image_name']}"
        )
    


class Handlers:
    @property
    def routes(self):
        return [
            web.get("/", self.index),
        ]

    def __init__(self, content_index):
        self.content_index = content_index

    async def on_startup(self, app):
        pass

    async def on_cleanup(self, app):
        pass

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        display_item = self.content_index.random_item()
        return {
            "display_item": display_item,
        }
