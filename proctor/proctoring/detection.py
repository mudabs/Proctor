"""Browser-frame inference with lazy, worker-local model loading."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from proctor.inference import select_device

logger = logging.getLogger(__name__)
_models = None
_models_lock = Lock()
_known_faces = None
_known_faces_signature = None
_known_faces_lock = Lock()


def _model_bundle(config):
    global _models
    if _models is not None:
        return _models
    with _models_lock:
        if _models is not None:
            return _models
        model_dir = Path(config["PROCTOR_MODEL_DIR"])
        device = select_device(config.get("PROCTOR_INFERENCE_DEVICE", "auto"))
        bundle = {"device": device, "object": None, "liveness": None, "dlib": None, "mesh": None}
        try:
            from ultralytics import YOLO
            object_path = model_dir / "yolov8n.pt"
            liveness_path = model_dir / "best_20.pt"
            if object_path.is_file():
                bundle["object"] = YOLO(str(object_path))
            if liveness_path.is_file():
                bundle["liveness"] = YOLO(str(liveness_path))
        except Exception:
            logger.exception("Unable to load YOLO models; continuing with classical metrics")
        try:
            import dlib
            predictor_path = model_dir / "shape_predictor_68_face_landmarks.dat"
            if predictor_path.is_file():
                bundle["dlib"] = (dlib.get_frontal_face_detector(), dlib.shape_predictor(str(predictor_path)))
        except Exception:
            logger.exception("Unable to load dlib landmarks")
        try:
            import mediapipe as mp
            bundle["mesh"] = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=5, refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except Exception:
            logger.exception("Unable to load MediaPipe face mesh")
        _models = bundle
        logger.info("Loaded inference models on %s", device)
        return bundle


def _known_face_data(config):
    global _known_faces, _known_faces_signature
    roots = [Path(config["PROCTOR_DATA_DIR"]) / "known_images"]
    static_root = Path(__file__).resolve().parents[2] / "static" / "images" / "known_images"
    if static_root.is_dir():
        roots.append(static_root)
    image_paths = sorted(
        path for root in roots if root.is_dir() for path in root.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    signature = tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in image_paths)
    if _known_faces is not None and signature == _known_faces_signature:
        return _known_faces
    with _known_faces_lock:
        if _known_faces is not None and signature == _known_faces_signature:
            return _known_faces
        try:
            import face_recognition
        except ImportError:
            return [], []
        encodings, names = [], []
        for path in image_paths:
            try:
                image = face_recognition.load_image_file(path)
                values = face_recognition.face_encodings(image)
                if values:
                    encodings.append(values[0])
                    names.append(path.stem)
            except Exception:
                logger.warning("Skipping invalid known-face image %s", path)
        _known_faces = (encodings, names)
        _known_faces_signature = signature
        return _known_faces


def _head_direction(mesh_result, width, height):
    if not mesh_result or not mesh_result.multi_face_landmarks:
        return "Not evaluated"
    landmarks = mesh_result.multi_face_landmarks[0].landmark
    points_2d, points_3d = [], []
    for index in (1, 33, 263, 61, 291, 199):
        point = landmarks[index]
        points_2d.append([point.x * width, point.y * height])
        points_3d.append([point.x * width, point.y * height, point.z * width])
    points_2d = np.asarray(points_2d, dtype=np.float64)
    points_3d = np.asarray(points_3d, dtype=np.float64)
    focal = width
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(points_3d, points_2d, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return "Not evaluated"
    angles = cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0]
    pitch, yaw = angles[0] * 360, angles[1] * 360
    if yaw < -10: return "Looking Left"
    if yaw > 10: return "Looking Right"
    if pitch < -10: return "Looking Down"
    if pitch > 10: return "Looking Up"
    return "Forward"


def process_frame(frame_bytes, config, expected_identity=None, audio_level=None):
    """Run all available metrics on one JPEG captured by a student's browser."""
    array = cv2.imdecode(np.frombuffer(frame_bytes, dtype="uint8"), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError("The uploaded frame is not a valid image")
    max_width = config.get("PROCTOR_FRAME_MAX_WIDTH", 1280)
    max_height = config.get("PROCTOR_FRAME_MAX_HEIGHT", 720)
    height, width = array.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        array = cv2.resize(array, (int(width * scale), int(height * scale)))
        height, width = array.shape[:2]
    bundle = _model_bundle(config)
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": bundle["device"], "identity": "Unknown", "faces": 0, "people": 0,
        "cellphone": "Not evaluated", "direction": "Not evaluated", "liveness": "Not evaluated",
        "lips": "Not evaluated", "audio": "Not evaluated; browser audio aggregate not provided",
    }
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    result["faces"] = result["people"] = int(len(faces))
    if bundle["object"] is not None:
        detections = bundle["object"](array, imgsz=320, conf=0.35, device=bundle["device"], verbose=False)[0]
        people, cellphone = 0, False
        for cls, confidence in zip(detections.boxes.cls.tolist(), detections.boxes.conf.tolist()):
            label = detections.names[int(cls)]
            if label == "person": people += 1
            if label == "cell phone" and confidence >= 0.35: cellphone = True
        result["people"] = people
        result["cellphone"] = "Cell Phone Detected" if cellphone else "None Detected"
    if bundle["liveness"] is not None:
        detections = bundle["liveness"](array, imgsz=320, conf=0.35, device=bundle["device"], verbose=False)[0]
        if len(detections.boxes):
            best = int(detections.boxes.conf.argmax())
            result["liveness"] = str(detections.names[int(detections.boxes.cls[best])])
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    if bundle["mesh"] is not None:
        result["direction"] = _head_direction(bundle["mesh"].process(rgb), width, height)
    if bundle["dlib"] is not None:
        detector, predictor = bundle["dlib"]
        landmarks_faces = detector(rgb, 1)
        if landmarks_faces:
            landmarks = predictor(rgb, landmarks_faces[0])
            result["lips"] = "Mouth Open" if landmarks.part(57).y - landmarks.part(51).y > 21 else "Mouth Closed"
    try:
        import face_recognition
        known_encodings, known_names = _known_face_data(config)
        encodings = face_recognition.face_encodings(rgb, face_recognition.face_locations(rgb, model="hog"))
        if encodings and known_encodings:
            matches = face_recognition.compare_faces(known_encodings, encodings[0], tolerance=0.5)
            if True in matches: result["identity"] = known_names[matches.index(True)]
        elif expected_identity and encodings:
            result["identity"] = expected_identity
    except ImportError:
        pass
    if audio_level is not None:
        level = max(0.0, min(1.0, float(audio_level)))
        result["audio"] = "Noise detected" if level >= 0.2 else "Quiet"
    return result


def calculate_score(*signals):
    return min(max(sum(float(value) for value in signals) / max(len(signals), 1), 0), 1)
