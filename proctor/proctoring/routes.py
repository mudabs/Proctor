"""Browser capture and session-isolated proctoring routes."""

import uuid
from datetime import datetime, timezone

from flask import current_app, jsonify, render_template, request, session

from . import detection
from . import proctoring
from proctor.state import sessions


def _owner_id():
    owner = session.get("user_id") or session.get("username")
    return str(owner) if owner is not None else None


def _owned_session(session_id):
    return sessions.get_owned(session_id, _owner_id())


@proctoring.route('/proctor', methods=['GET', 'POST'])
def proctor():
    return render_template('proctor.html', data={})


@proctoring.route('/proctoring/session', methods=['POST'])
def start_session():
    if _owner_id() is None:
        return jsonify({"error": "authentication required"}), 401
    session_id = uuid.uuid4().hex
    sessions.start(session_id, _owner_id())
    return jsonify({"session_id": session_id, "status": "started"}), 201


@proctoring.route('/proctoring/session/<session_id>/frame', methods=['POST'])
def upload_frame(session_id):
    if _owner_id() is None:
        return jsonify({"error": "authentication required"}), 401
    state = _owned_session(session_id)
    if state is None:
        return jsonify({"error": "session not found"}), 404
    if not request.mimetype.startswith("image/"):
        return jsonify({"error": "content type must be an image"}), 415
    frame = request.get_data(cache=False)
    if not frame:
        return jsonify({"error": "empty frame"}), 400
    now = datetime.now(timezone.utc)
    interval = current_app.config.get("PROCTOR_FRAME_INTERVAL_MS", 1000) / 1000
    if state.last_frame_at and (now - state.last_frame_at).total_seconds() < interval:
        return jsonify({"error": "frame rate exceeded"}), 429
    try:
        result = detection.process_frame(frame, current_app.config, session.get("username"))
    except ValueError as exc:
        state.errors.append(str(exc))
        return jsonify({"error": str(exc)}), 400
    state.latest_result = result
    state.last_frame_at = now
    return jsonify(result)


@proctoring.route('/proctoring/session/<session_id>/result')
def latest_result(session_id):
    if _owner_id() is None:
        return jsonify({"error": "authentication required"}), 401
    state = _owned_session(session_id)
    if state is None:
        return jsonify({"error": "session not found"}), 404
    return jsonify(state.latest_result or {"status": "waiting_for_frame"})


@proctoring.route('/proctoring/session/<session_id>/error', methods=['POST'])
def session_error(session_id):
    if _owner_id() is None:
        return jsonify({"error": "authentication required"}), 401
    state = _owned_session(session_id)
    if state is None:
        return jsonify({"error": "session not found"}), 404
    body = request.get_json(silent=True) or {}
    state.errors.append(str(body.get("error", "unknown browser error"))[:500])
    return jsonify({"status": "recorded"})


@proctoring.route('/proctoring/session/<session_id>', methods=['DELETE', 'POST'])
def stop_session(session_id):
    if _owner_id() is None:
        return jsonify({"error": "authentication required"}), 401
    if not sessions.stop(session_id, _owner_id()):
        return jsonify({"error": "session not found"}), 404
    return jsonify({"status": "stopped"})


@proctoring.route('/video')
def video():
    return jsonify({"error": "server-side webcam capture is unsupported; upload browser frames instead"}), 410


@proctoring.route('/get_objects')
def get_objects():
    return jsonify({"error": "use the authenticated session result endpoint"}), 410


@proctoring.route('/blacklist', methods=['GET', 'POST'])
def blacklist():
    return jsonify({"status": "unsupported", "message": "Website blocking requires an installed, consented client agent or browser extension."}), 410


@proctoring.route('/unblock', methods=['POST'])
def unblock():
    return jsonify({"status": "unsupported"}), 410
