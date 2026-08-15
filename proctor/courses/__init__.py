"""Course and assessment blueprint."""

from flask import Blueprint

courses = Blueprint("courses", __name__)

from . import routes  # noqa: E402,F401
