from aioapns import APNs
from aiohttp import web
import argparse
import logging
from nohomers.app import app
from nohomers.handlers import Handlers
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
    parser.add_argument("--port", type=int,
                        action=EnvDefault,
                        env_var="PORT",
                        help="Port number to listen on", required=True)

    args = parser.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl)

    handlers = Handlers()
    my_app = app(handlers)
    web.run_app(app(handlers), port=args.port)
