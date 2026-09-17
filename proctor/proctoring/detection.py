"""Session-safe frame processing. Frames come from the student's browser."""

from datetime import datetime, timezone

import cv2
import numpy as np

from proctor.inference import select_device


def process_frame(frame_bytes, config, expected_identity=None):
    """Process one browser JPEG without opening a server camera or microphone."""
    array = cv2.imdecode(np.frombuffer(frame_bytes, dtype="uint8"), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError("The uploaded frame is not a valid image")
    max_width = config.get("PROCTOR_FRAME_MAX_WIDTH", 1280)
    max_height = config.get("PROCTOR_FRAME_MAX_HEIGHT", 720)
    height, width = array.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        array = cv2.resize(array, (int(width * scale), int(height * scale)))
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": select_device(config.get("PROCTOR_INFERENCE_DEVICE", "auto")),
        "identity": expected_identity or "Unknown",
        "faces": int(len(faces)),
        "people": int(len(faces)),
        "cellphone": "Not evaluated",
        "direction": "Not evaluated",
        "liveness": "Not evaluated",
        "lips": "Not evaluated",
        "audio": "Not evaluated; browser audio aggregate not provided",
    }


def calculate_score(*signals):
    return min(max(sum(float(value) for value in signals) / max(len(signals), 1), 0), 1)
