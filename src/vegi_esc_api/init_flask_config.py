from flask import Flask
import argparse
import os
import vegi_esc_api.logger as Logger


def init_flask_config(app: Flask, args: argparse.Namespace | None = None):
    if args:
        host = args.host
        port = int(os.environ.get("PORT", int(args.port)))
    else:
        host = "0.0.0.0"
        port = int(
            os.environ.get("PORT", 5002)
        )  # allow heroku to set the path, no need for python-dotenv package to get this env variable either...
    Logger.info(f"Running app from {host}:{port}")
    Logger.info(
        f"Check the app is up by openning: `http://{host}:{port}/success/fenton` unless in docker where need to map the port from {port} to your forward port... "
    )
    # return lambda: app.run(host=host, port=port)
    # call app.run yourself
    return app, host, port
