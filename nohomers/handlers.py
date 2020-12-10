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
from typing import Optional, Dict, Any


def json_error(klass, error_type, message=None):
    return klass(
        text=json.dumps({"error": {"type": error_type, "message": message}}),
        content_type="application/json"
    )


def json_response(d):
    return web.Response(
        text=json.dumps(d),
        content_type="application/json",
    )

@dataclass
class ContentItem:
    url: str
    key: str

    next_item_key: Optional[str]
    next_item_url: Optional[str]
    transition_url: Optional[str]

    def to_dict(self):
        return {
            "url": self.url,
            "key": self.key,
            "next_item_key": self.next_item_key,
            "next_item_url": self.next_item_url,
            "transition_url": self.transition_url,
        }
    
    @property
    def json_string(self):
        return json.dumps(self.to_dict())


class ContentIndex:
    def __init__(self, manifest_path: Path, manifest_dir_url: str):
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.manifest_dir_url = manifest_dir_url
        self.manifest_by_key = {
            e["image_name"]: i for i, e in enumerate(self.manifest)
        }

    def _content_item_from_manifest_val(self, item: Dict[str, Any]) -> ContentItem:
        transition = py_.sample(item["transitions"])
        next_item_key = transition["dest_name"]
        next_item_url = f"{self.manifest_dir_url}/images/{next_item_key}",
        transition_url = f"{self.manifest_dir_url}/videos/{transition['video_name']}"

        return ContentItem(
            url=f"{self.manifest_dir_url}/images/{item['image_name']}",
            key=item["image_name"],
            next_item_key=next_item_key,
            next_item_url=next_item_url,
            transition_url=transition_url,
        )
    
    def item_for_key(self, key) -> ContentItem:
        return self._content_item_from_manifest_val(self.manifest[self.manifest_by_key[key]])

    def random_item(self) -> ContentItem:
        item = py_.sample(self.manifest)
        return self._content_item_from_manifest_val(item)
    

class Handlers:
    @property
    def routes(self):
        return [
            web.get("/", self.index),
            web.get(r"/item/{key:[^/]+}", self.item),
        ]

    def __init__(self, content_index):
        self.content_index = content_index

    async def on_startup(self, app):
        pass

    async def on_cleanup(self, app):
        pass

    async def item(self, request):
        key = request.match_info['key']
        try:
            item = self.content_index.item_for_key(key)
        except IndexError:
            return json_error(web.HTTPBadRequest, "bad_key")
        
        return json_response(item.to_dict())

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        display_item = self.content_index.random_item()
        return {
            "display_item": display_item,
        }
