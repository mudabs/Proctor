"""Proctoring HTTP routes backed by the detection module."""

from datetime import datetime, timedelta

from flask import Response, flash, jsonify, redirect, render_template, request, session, url_for

from . import proctoring
from . import detection
from proctor import state
from proctor.admin.routes import block_websites, unblock_websites
from proctor.extensions import db
from proctor.models import Blocked

from app import app


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
    detection.write_exam_session(
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


def get_remaining_time_in_seconds():
    expiration_time = session.get("expiration_time")
    if not expiration_time:
        return 0
    remaining_time = app.config['expiration_time'] - datetime.now()
    return remaining_time.total_seconds()


@proctoring.route('/remaining_time')
def remaining_time():
    remaining_time_in_seconds = get_remaining_time_in_seconds()
    if remaining_time_in_seconds <= 0:
        session["messages"] = "Time is up"
        return reset()
    return jsonify({'remaining_time_in_seconds': remaining_time_in_seconds})


@proctoring.route('/reset', methods=['POST'])
def reset():
    session['quiz'] = "False"
    app.config['expiration_time'] = datetime.now() + timedelta(minutes=10)
    return render_template('home.html')


@proctoring.route('/clearTimer', methods=['POST'])
def clearTimer():
    app.config['expiration_time'] = ''
    return redirect(url_for('home'))


@proctoring.route('/blacklist', methods=['GET', 'POST'])
def blacklist():
    if request.method == 'POST':
        blocked_websites = [
            "researchgate.net", "scholar.google.com", "pubmed.ncbi.nlm.nih.gov",
            "ieeexplore.ieee.org", "sciencedirect.com", "jstor.org",
            "link.springer.com", "onlinelibrary.wiley.com", "arxiv.org", "ssrn.com",
            "nature.com", "elsevier.com", "dl.acm.org", "scopus.com", "plos.org",
            "academic.oup.com", "tandfonline.com", "research.com", "researcher.com",
            "worldcat.org", "google.com", "bing.com", "yahoo.com", "duckduckgo.com",
            "baidu.com", "yandex.com", "ask.com", "ecosia.org", "startpage.com",
            "swisscows.com",
        ]
        for url in blocked_websites:
            db.session.add(Blocked(url=url))
        db.session.commit()
        unblock_websites()
        block_websites()
        flash('Websites blocked successfully', 'success')
        return redirect(url_for('blacklist'))

    blocked_urls = Blocked.query.all()
    return render_template('blacklist.html', blocked_urls=blocked_urls)


@proctoring.route('/unblock', methods=['POST'])
def unblock():
    unblock_websites()
    Blocked.query.delete()
    db.session.commit()
    flash('Websites unblocked successfully', 'success')
    return redirect(url_for('blacklist'))
