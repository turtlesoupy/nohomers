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
import urllib.parse


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
    permalink: str

    next_item_key: Optional[str]
    next_item_url: Optional[str]
    transition_url: Optional[str]

    opensea_item_url: Optional[str]

    def to_dict(self):
        return {
            "url": self.url,
            "key": self.key,
            "next_item_key": self.next_item_key,
            "next_item_url": self.next_item_url,
            "transition_url": self.transition_url,
            "permalink": self.permalink,
            "opensea_item_url": self.opensea_item_url,
        }
    
    @property
    def json_string(self):
        return json.dumps(self.to_dict())


class ContentIndex:
    def __init__(self, manifest_path: Path, manifest_dir_url: str, permalink_key="p"):
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        
        self.permalink_key = permalink_key
        self.manifest_dir_url = manifest_dir_url
        self.manifest_by_key = {
            e["name"]: i for i, e in enumerate(self.manifest)
        }
        self.manifest_by_permalink = {}

        for i, item in enumerate(self.manifest):
            try_key = item["name"][:8]
            if try_key in self.manifest_by_permalink:
                try_key = item["name"][:16]
            if try_key in self.manifest_by_permalink:
                try_key = item["name"]
            if try_key in self.manifest_by_permalink:
                raise RuntimeError(f"Unable to make permalink for {item['name']}")
            
            item["permalink"] = try_key
            self.manifest_by_permalink[try_key] = i


    def _content_item_from_manifest_val(self, item: Dict[str, Any]) -> ContentItem:
        transition = py_.sample(item["transitions"])
        next_item_key = transition["dest_name"]
        next_item_url = f"{self.manifest_dir_url}/images/{next_item_key}"
        transition_url = f"{self.manifest_dir_url}/videos/{transition['video_name']}"

        return ContentItem(
            url=f"{self.manifest_dir_url}/images/{item['name']}",
            key=item["name"],
            permalink=f"/p/{self.permalink_key}/{urllib.parse.quote(item['permalink'])}",
            next_item_key=next_item_key,
            next_item_url=next_item_url,
            transition_url=transition_url,
            opensea_item_url=item.get("opensea_url"),
        )
    
    def item_for_key(self, key) -> ContentItem:
        return self._content_item_from_manifest_val(self.manifest[self.manifest_by_key[key]])
    
    def item_for_permalink(self, permalink_key) -> ContentItem:
        return self._content_item_from_manifest_val(self.manifest[self.manifest_by_permalink[permalink_key]])

    def random_item(self) -> ContentItem:
        item = py_.sample(self.manifest)
        return self._content_item_from_manifest_val(item)
    

class Handlers:
    @property
    def routes(self):
        return [
            web.get(r"/", self.index),
            web.get(r"/p/{index_key:[^/]+}/{permalink_key:[^/]+}", self.index),
            web.get(r"/favicon.ico", self.favicon),
            web.get(r"/item/{key:[^/]+}", self.item),
        ]

    def __init__(self, content_index: ContentIndex, base_url: str, static_cdn_url_base: str):
        self.content_index = content_index
        self.base_url = base_url
        self.static_cdn_url_base = static_cdn_url_base

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

    async def favicon(self, request):
        return web.FileResponse("./static/favicons/favicon.ico")

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        if "index_key" in request.match_info and "permalink_key" in request.match_info:
            index_key = request.match_info["index_key"]
            permalink_key = request.match_info["permalink_key"]
            if index_key != self.content_index.permalink_key:
                return json_error(web.HTTPNotFound, "invalid_index_key")

            try:
                display_item = self.content_index.item_for_permalink(permalink_key)
            except KeyError:
                return json_error(web.HTTPNotFound, "invalid_permalink_key")

            og_image_url = display_item.url
            og_url = f"{self.base_url}{display_item.permalink}"
            is_root_request = False
        else:
            display_item = self.content_index.random_item()
            is_root_request = True
            og_image_url = "https://static.thisfuckeduphomerdoesnotexist.com/site_content/opengraph_image.png"
            og_url = self.base_url

        return {
            "display_item": display_item,
            "og_url": og_url,
            "og_image_url": og_image_url,
            "is_root_request": is_root_request,
            "static_cdn_url_base":  self.static_cdn_url_base,
        }
