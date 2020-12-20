import base64
import jinja2
import aiohttp_jinja2
from aiohttp import web
from pathlib import Path
from urllib.parse import quote_plus
import logging
from aiohttp.web import middleware
from .handlers import Handlers, ContentIndex
import copy


def _dev_handlers():
    logging.basicConfig(level=logging.INFO)
    content_index = ContentIndex(
        manifest_path=Path(__file__).parent.parent.parent / "resources" / "simpsons_large_cleaned_nobackground_1024_augall03_sle_res64-35-p100.manifest.json",
        manifest_dir_url="https://static.thisfuckeduphomerdoesnotexist.com/simpsons_large_cleaned_nobackground_1024_augall03_sle_res64-35-p100",
    )
    return Handlers(
        content_index=content_index,
        base_url="https://www.thisfuckeduphomerdoesnotexist.com",
    )


@middleware
async def add_www_to_url(request, handler):
    if request.url.host == "thisfuckeduphomerdoesnotexist.com":
        raise web.HTTPFound(location=str(request.url).replace("://", "://www."))

    return (await handler(request))


def app(handlers=None):
    handlers = handlers or _dev_handlers()

    my_app = web.Application(
        middlewares=[add_www_to_url]
    )
    my_app.on_startup.append(handlers.on_startup)
    my_app.on_cleanup.append(handlers.on_cleanup)

    root_path = Path(__file__).parent.parent
    my_app.add_routes(handlers.routes + [
        web.static("/static", str(root_path / "../static"))
    ])

    aiohttp_jinja2.setup(
        my_app, loader=jinja2.FileSystemLoader(str(root_path / "../templates")), filters={
            "quote_plus": quote_plus,
            "remove_period": lambda x: x.rstrip("."),
            "escape_double": lambda x: x.replace('"', r'\"'),
            "strip_quotes": lambda x: x.strip('"'),
        },
    )
    return my_app
