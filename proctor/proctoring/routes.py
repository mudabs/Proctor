"""Proctoring HTTP routes backed by the detection module."""

from datetime import datetime

from flask import Response, jsonify, render_template, request, session

from . import proctoring
from . import detection
from proctor import state


def write_exam_session(identity, cellphone, direction, liveness, lips, num_people, num_faces):
    data = (
        f"{datetime.now().time()},{identity},{cellphone},{direction},"
        f"{liveness}, {lips}, {num_people},{num_faces},{detection.cheat}\n"
    )
    with open("./session.txt", "a") as session_file:
        session_file.write(data)


def cheating_threshold():
    cellphone_local = 0.5 if detection.cellphone == "Cell Phone Detected" else 0.1
    direction_local = {
        "Forward": 0.15,
        "Looking Left": 0.4,
        "Looking Right": 0.45,
        "Looking Up": 0.55,
        "Looking Down": 0.6,
    }.get(detection.direction, 0.1)
    liveness_local = 0.1 if detection.liveness == "real" else 0.67
    lips_local = 0.1 if detection.lips == "Mouth Closed" else 0.35
    identity_local = 0.1 if detection.identity == session.get("username") else 0.7
    people_local = 0.1 if detection.numPeople == 1 else 0.5
    faces_local = 0.1 if detection.numFaces == 1 else 0.5
    noise_local = 0.67 if state.noise > 0.2 else 0.2
    score = detection.calculate_score(
        cellphone_local,
        direction_local,
        liveness_local,
        lips_local,
        identity_local,
        people_local,
        faces_local,
        noise_local,
    )
    state.cheating_scores.append(score)
    write_exam_session(
        identity_local,
        cellphone_local,
        direction_local,
        liveness_local,
        lips_local,
        people_local,
        faces_local,
    )
    return score


@proctoring.route('/proctor', methods=['GET', 'POST'])
def proctor():
    data = {
        "cheat": detection.cheat,
        "lips": detection.lips,
        "direction": detection.direction,
        "cellphone": detection.cellphone,
        "identity": detection.identity,
        "liveness": detection.liveness,
    }
    return render_template('proctor.html', data=data)


@proctoring.route('/get_objects')
def get_objects():
    score = cheating_threshold()
    return jsonify(
        detection.cellphone,
        detection.direction,
        detection.liveness,
        detection.lips,
        detection.identity,
        detection.numPeople,
        detection.numFaces,
        score,
    )


@proctoring.route('/cheatingThreshold')
def cheatingThreshold():
    return cheating_threshold()


@proctoring.route('/video')
def video():
    detection.load_known_faces()
    return Response(
        detection.video_detection(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )
