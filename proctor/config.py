"""Environment-backed configuration and portable application paths."""

import os
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(name, default):
    value = os.getenv(name)
    path = Path(value).expanduser() if value else APP_ROOT / default
    return path if path.is_absolute() else APP_ROOT / path


def _max_upload_size(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 5 * 1024 * 1024


class Config:
    SECRET_KEY = os.getenv("PROCTOR_SECRET_KEY", "change-me-in-production")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "PROCTOR_DATABASE_URI",
        "mysql+pymysql://proctor:proctor@db:3306/proctoring",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    PROCTOR_MODEL_DIR = _path_from_env("PROCTOR_MODEL_DIR", "models")
    PROCTOR_DATA_DIR = _path_from_env("PROCTOR_DATA_DIR", "data")
    PROCTOR_INFERENCE_DEVICE = os.getenv("PROCTOR_INFERENCE_DEVICE", "auto").lower()
    PROCTOR_LOG_LEVEL = os.getenv("PROCTOR_LOG_LEVEL", "INFO").upper()
    MAX_CONTENT_LENGTH = _max_upload_size(os.getenv("PROCTOR_MAX_UPLOAD_SIZE"))
    PROCTOR_ALLOWED_HOSTS = {
        host.strip().lower()
        for host in os.getenv("PROCTOR_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
        if host.strip()
    }
    PROCTOR_FRAME_INTERVAL_MS = int(os.getenv("PROCTOR_FRAME_INTERVAL_MS", "1000"))
    PROCTOR_FRAME_JPEG_QUALITY = int(os.getenv("PROCTOR_FRAME_JPEG_QUALITY", "75"))
    PROCTOR_FRAME_MAX_WIDTH = int(os.getenv("PROCTOR_FRAME_MAX_WIDTH", "1280"))
    PROCTOR_FRAME_MAX_HEIGHT = int(os.getenv("PROCTOR_FRAME_MAX_HEIGHT", "720"))
    PROCTOR_ENABLE_AUDIO = os.getenv("PROCTOR_ENABLE_AUDIO", "true").lower() == "true"
