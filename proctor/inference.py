"""Portable inference configuration and model asset validation."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def select_device(requested="auto"):
    """Return a PyTorch/Ultralytics device string without requiring CUDA."""
    requested = (requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("PROCTOR_INFERENCE_DEVICE must be auto, cpu, or cuda")
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but PyTorch is not installed")
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but is unavailable in this container")
    if requested == "cpu" or not available:
        return "cpu"
    return "cuda:0"


def log_device(requested="auto"):
    device = select_device(requested)
    gpu_name = None
    try:
        import torch
        if device.startswith("cuda"):
            gpu_name = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError):
        pass
    logger.info("Proctor inference device: %s%s", device, f" ({gpu_name})" if gpu_name else "")
    return device


MODEL_FILES = {
    "shape_predictor_68_face_landmarks.dat": "dlib facial landmarks",
    "yolov8n.pt": "YOLO object detection",
    "best_20.pt": "custom liveness detection",
    "TrainingImageLabel/Trainner.yml": "LBPH face recognition",
}


def missing_models(model_dir):
    model_dir = Path(model_dir)
    return [str(model_dir / name) for name in MODEL_FILES if not (model_dir / name).is_file()]


def require_models(model_dir, required=None):
    model_dir = Path(model_dir)
    required = required or MODEL_FILES
    missing = [str(model_dir / name) for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Required model files are missing: " + ", ".join(missing))
    return True
