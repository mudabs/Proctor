"""Computer-vision proctoring blueprint."""

from flask import Blueprint

proctoring = Blueprint("proctoring", __name__)

from . import routes  # noqa: E402,F401
