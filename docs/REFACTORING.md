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
- Runtime stop and score buffers are centralized in `proctor/state.py`, so the
  quiz flow, sound loop, and proctoring routes share the same state.
- Each quiz resets that shared state, allowing a later attempt in the same
  process to start a fresh detection loop and score history.

## Validation policy

Every stage must pass Python compilation and `git diff --check`. The completed
refactor also passes an SQLite in-memory import smoke test and confirms the
legacy route aliases remain registered. Full runtime validation requires
MySQL/MariaDB credentials, the webcam, and the model files listed in the README.

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
