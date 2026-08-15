"""Environment-backed application configuration."""

import os


class Config:
    SECRET_KEY = os.getenv("PROCTOR_SECRET_KEY", "development-only-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "PROCTOR_DATABASE_URI",
        "mysql://root:@localhost/proctoring",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
