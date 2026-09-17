"""Verify deployment model assets without downloading or fabricating them."""

import os
from pathlib import Path

from proctor.inference import MODEL_FILES


model_dir = Path(os.getenv("PROCTOR_MODEL_DIR", "models"))
missing = []
for relative, purpose in MODEL_FILES.items():
    path = model_dir / relative
    status = "OK" if path.is_file() else "MISSING"
    print(f"{status}: {path} ({purpose})")
    if not path.is_file():
        missing.append(path)
if missing:
    raise SystemExit(f"Missing {len(missing)} required model asset(s). Copy/download them into {model_dir}.")
