import os
import argparse
import json
import logging
import logging.config

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "stream": "ext://sys.stdout",
            "formatter": "standard",
        }
    },
    "root": {
        "level": os.getenv("LOG_LEVEL", "INFO"),
        "handlers": ["console"],
        "propagate": True,
    },
})

from label_studio_ml.api import init_app
from model import HuggingFaceAnnotationModel

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def get_kwargs_from_config(config_path=_DEFAULT_CONFIG_PATH):
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        config = json.load(f)
    assert isinstance(config, dict)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Studio ML backend – HuggingFace Annotation")
    parser.add_argument("-p", "--port", dest="port", type=int, default=9090, help="Server port")
    parser.add_argument("--host", dest="host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument(
        "--kwargs", "--with", dest="kwargs", metavar="KEY=VAL", nargs="+",
        type=lambda kv: kv.split("="), help="Additional model initialization kwargs",
    )
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="Debug mode")
    parser.add_argument(
        "--log-level", dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
    )
    parser.add_argument("--basic-auth-user", default=os.getenv("BASIC_AUTH_USER"))
    parser.add_argument("--basic-auth-pass", default=os.getenv("BASIC_AUTH_PASS"))

    args = parser.parse_args()

    if args.log_level:
        logging.root.setLevel(args.log_level)

    def _isfloat(v):
        try:
            float(v)
            return True
        except ValueError:
            return False

    kwargs = get_kwargs_from_config()
    if args.kwargs:
        for k, v in args.kwargs:
            if v.isdigit():
                kwargs[k] = int(v)
            elif v.lower() == "true":
                kwargs[k] = True
            elif v.lower() == "false":
                kwargs[k] = False
            elif _isfloat(v):
                kwargs[k] = float(v)
            else:
                kwargs[k] = v

    app = init_app(
        model_class=HuggingFaceAnnotationModel,
        basic_auth_user=args.basic_auth_user,
        basic_auth_pass=args.basic_auth_pass,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)

else:
    # For gunicorn: gunicorn _wsgi:app
    app = init_app(model_class=HuggingFaceAnnotationModel)
