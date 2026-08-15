# Refactoring journal

This document records the modularisation work so each change can be reviewed
against the original application behavior.

## Stage 1 — safeguards

- Added Python 3.11 CI with syntax compilation and Ruff reporting.
- Added `SECURITY.md` with private vulnerability-reporting guidance.
- Added password hashing with Werkzeug and one-time migration of legacy
  plaintext credentials after successful login.

## Stage 2 — shared application infrastructure

- Moved the SQLAlchemy and Bootstrap extension objects to
  `proctor/extensions.py`.
- Moved all SQLAlchemy models to `proctor/models.py` without changing table
  names, columns, or relationships.
- Added a shared runtime state module for the legacy detection loop.

## Stage 3 — feature modules

- Authentication routes live in `proctor/auth/routes.py`.
- Course, question, exam, quiz, timer, and result routes live in
  `proctor/courses/routes.py`.
- Proctoring routes and detection state now live in `proctor/proctoring/`.
- Administration, role management, image management, and blacklist routes now
  live in `proctor/admin/`.
- The blacklist has one owner: the administration blueprint.

## Validation policy

Every stage must pass Python compilation and `git diff --check`. Full runtime
validation requires MySQL/MariaDB credentials, the webcam, and the model files
listed in the README. Those environment-dependent checks are documented in the
corresponding pull request rather than hidden behind a failing local import.

## Target layout

```text
run.py
proctor/
  config.py
  extensions.py
  models.py
  auth/
  courses/
  proctoring/
    detection.py
  admin/
```

`app.py` remains a compatibility composition module for the current release;
`run.py` is the documented startup entrypoint. The remaining work is to move
the small set of sensor, homepage, and timer compatibility routes into a
dedicated application-factory module without changing their public URLs.
