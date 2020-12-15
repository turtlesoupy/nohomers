from aioapns import APNs
from aiohttp import web
import argparse
import logging
from nohomers.website.app import app
from nohomers.website.handlers import Handlers, ContentIndex
import os
from pathlib import Path


class EnvDefault(argparse.Action):
    def __init__(self, env_var, required=True, default=None, **kwargs):
        if not default and env_var:
            if env_var in os.environ:
                default = os.environ[env_var]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required,
                                         **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=bool, help="Verbose logging")
    parser.add_argument(
        "--port", type=int,
        action=EnvDefault,
        env_var="PORT",
        help="Port number to listen on", 
        required=True
    )

    parser.add_argument(
        "--manifest-path",
        type=str,
        action=EnvDefault,
        env_var="MANIFEST_PATH",
        help="Manifest path location",
        default="./static/manifest.json",
    )

    parser.add_argument(
        "--manifest-dir-url",
        type=str,
        action=EnvDefault,
        env_var="MANIFEST_DIR_URL",
        help="HTTP(s) path for manifest dir",
        default="https://static.thisfuckeduphomerdoesnotexist.com/simpsons_large_cleaned_nobackground_1024_augnormal04",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        action=EnvDefault,
        env_var="BASE_URL",
        help="The absolute base url for certain callback links",
        default="https://www.thisfuckeduphomerdoesnotexist.com",
    )

    args = parser.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl)

    content_index = ContentIndex(args.manifest_path, args.manifest_dir_url)
    handlers = Handlers(content_index=content_index, base_url=args.base_url)
    my_app = app(handlers)
    web.run_app(app(handlers), port=args.port)
