import base64
import jinja2
import aiohttp_jinja2
from aiohttp import web
from pathlib import Path
from urllib.parse import quote_plus
import logging
from .handlers import Handlers


def _dev_handlers():
    logging.basicConfig(level=logging.INFO)
    return Handlers()


def app(handlers=None):
    handlers = handlers or _dev_handlers()

    my_app = web.Application(
        middlewares=[]
    )
    my_app.on_startup.append(handlers.on_startup)
    my_app.on_cleanup.append(handlers.on_cleanup)

    root_path = Path(__file__).parent
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
