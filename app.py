from datetime import datetime, timedelta
import string
from flask import Flask, flash, render_template, session, request
import numpy as np
import os
from proctor.extensions import bootstrap, db
from proctor import state
from proctor.auth import auth
from proctor.config import Config
from proctor.admin import admin
import json
import sounddevice as sd
import numpy as np
import time as timeSound
import matplotlib.pyplot as plt
from werkzeug.security import generate_password_hash


cheat = 0
lips = ''
direction=''
cellphone=''
identity=''
liveness = ''
numPeople = 0
numFaces = 0
noise = 0

# Capturing User Image details
name=''
id=''
capture_enabled = False
image_count = 0


# def configure_app_and_access_session(app, session):
#     # Assign app.config variable
#     # Access session variable (if needed)
#     duration = session.get('duration')  # Example usage
#     if duration:
#         duration = duration * 60
#         app.config['expiration_time'] = datetime.now() + timedelta(minutes=duration)
#         session["expiration_time"] = app.config['expiration_time']
#     else:
#         app.config['expiration_time'] = datetime.now() + timedelta(minutes=0)
#         session["expiration_time"] = app.config['expiration_time']


def configure_app_and_access_session(app, session):
    if 'expiration_time' not in session:  # Set expiration time only if not already set
        app.config['expiration_time'] = datetime.now() + timedelta(minutes=1)  # Default value

    duration = session.get('duration')
    if duration:
        duration = duration * 60 + 0.25
        expiration_time = datetime.now() + timedelta(minutes=duration)
    else:
        expiration_time = datetime.now() + timedelta(minutes=0)  # Set a default if duration is not available

    session["expiration_time"] = expiration_time
    app.config['expiration_time'] = expiration_time  # Update for consistency


# Initialize Flask app
app = Flask(__name__)


app.config.from_object(Config)

db.init_app(app)
bootstrap.init_app(app)
app.register_blueprint(auth, name="auth")

# Models Loading------------------------------------------------------------------------------------------------------------------------------

from proctor.models import User, Role, Questions, Answers, CorrectAnswers, ProctorSession, Course, Enrollment, Lecturers, Exam, Quiz, QuizQuestions, Marks, QuizCompletion, UserCompletion, User_roles, Blocked, assign_role_to_user, removed_role

# Class Loading------------------------------------------------------------------------------------------------------------------------------

# /////////////////////Sound Detection

# Define the soundThreshold for sound detection
soundThreshold = 0.5

# Function to check sound level and save to file
def check_sound(indata, frames, callback_time, status):
    global noise
    volume_norm = np.linalg.norm(indata) * 2
    noise = volume_norm/2
    state.noise = noise
    if volume_norm > soundThreshold:
        print(volume_norm)
        with open('sound.txt', 'a') as file:
            file.write("{:.2f}\n".format( volume_norm ))
    else:
        print("0")
        with open('sound.txt', 'a') as file:
            file.write("0\n")


def drawSoundGraph():
    # Read data from sound.txt file
    with open("sound.txt", "r") as file:
        lines = file.readlines()

    # Convert data to float
    data = [float(line.strip()) for line in lines]

    # Generate time values (1 second intervals)
    time = [i for i in range(len(data))]

    # Plot the graph
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Sound Level')
    plt.title('Sound Level Over Time')
    plt.grid(True)
    
    plt.savefig(f'./static/graphs/SoundGraph.png')
    clearTextFile("./sound.txt")
  
def detectSound():
    # Start sound capture
    with sd.InputStream(callback=check_sound):
        while not state.stop_detection:
            timeSound.sleep(1)
          

# /////////////////////Sound Detection


@app.route('/drawGraph', methods=['GET'])
def drawGraph():
    # Read data from file
    file_path = './session.txt'
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip().split(','))
    # Extract x-axis values (time)
    x_values = [row[0] for row in data]
    x_values = [time.split(':')[1:] for time in x_values]  # Extract only minutes and seconds
    x_values = [':' .join(time) for time in x_values]  # Reconstruct time strings
    # Extract y-axis values for each column
    y_values = [[] for _ in range(8)]
    for row in data:
        for i in range(1, 9):
            y_values[i-1].append(float(row[i]))

    # Plot graphs and save to file
    for i in range(8):
        values = ["identity","cellphone","direction","liveness","lips","numPeople","numFaces","Overall Cheating"]
        if y_values[i]:  # Check if y_values[i] is not empty
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values[i])
            plt.title(f'{values[i]}')
            plt.xlabel('Time')
            plt.xticks(rotation=90)
            plt.ylabel(f'Data {i+1}')
            plt.grid(True)

            plt.savefig(f'./static/graphs/{values[i]}.png')
            print("Value - ",values[i])
            plt.close()
            clearTextFile("session.txt")
        else:
            print(f"Skipping plot for {values[i]} due to empty data.")

    print("Graphs saved successfully.")

def clearTextFile(file_path):

    # Open the file in write mode, which truncates the file
    with open(file_path, "w") as file:
        pass  # Do nothing, effectively clearing the file

@app.route('/humidity', methods=['GET'])
def humidity():
    global cheat

    data = [time() * 1000, cheat ]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response

# Sensor

@app.route('/data', methods=['GET'])
def data():
    global cheat
    Temperature = []
    for i in range(1,10):
        Temperature.append(cheat)
    data = {
        "temperature":Temperature
    }
    return data





# Sensor

try:

    from flask import (Blueprint,
                       render_template,
                       redirect, url_for, session)

    from flask import Flask, request, session, send_file
    import json
    from time import time
    from flask import Flask, render_template, make_response

except Exception as e:
    print("Some modules didnt load {}".format(e))

sensor_blueprint = Blueprint('Sensor', __name__)


@sensor_blueprint.route('/data', methods=['GET'])
def data():
    Temperature = []
    for i in range(1,10):
        Temperature.append(cheat)
    data = {
        "temperature":Temperature
    }
    return data


app.register_blueprint(sensor_blueprint, url_prefix="/Sensor")

# app.register_blueprint(result_blueprint, url_prefix="/Result")


# App

try:
    from flask import render_template

    from flask import (Blueprint,
                       render_template,
                       redirect, url_for)

    from flask import (Flask,
                       request,
                       redirect,
                       session,
                       send_file)

    from io import BytesIO
    from flask import abort, jsonify
    import io

except Exception as e:
    print("Failed to load some Modules ")

# Graph----------------------------------------------------------------------------



# Flask logic-----------------------------------------------------------------------
# Session Variable Injection
@app.context_processor
def inject_session_data():
    userid = session.get('user_id')
    username = session.get('username')
    allCourses = Course.query.all()
    teachingCourses = None
    enrollments = None

    if userid:
        teachingCourses = Course.query.filter_by(lecturerId=userid).all()
        enrollments = Enrollment.query.filter_by(user_id=userid).all()


    return dict(username=username,userid=userid, teachingCourses = teachingCourses, enrollments=enrollments, allCourses=allCourses)


# Route for the proctor page
@app.route('/')
def home():
    user = ''
    mycourses = ''
    allUsers = ''
    myEnrolledCourses =''
    allCourses=''
    if 'user_id' in session:
        mycourses = Course.query.filter_by(lecturerId=session['user_id']).all()
        myEnrolledCourses = Enrollment.query.filter_by(user_id=session['user_id']).all()
        allCourses = Course.query.all()
        user_id = session['user_id']
        allUsers = User.query.all()
        user = User.query.filter_by(id=user_id).all()
    return render_template('home.html', user=user, mycourses=mycourses, allUsers = allUsers,myEnrolledCourses = myEnrolledCourses,allCourses=allCourses)


# Obtaining Names for fetching images
def get_names():
    data = User.query.all()
    return data

# Obtaining Names for fetching images
def check_names(username):
    user = User.query.filter_by(name=username).first()
    if user:
        return True
    else:
        return False

@app.route('/edit_profile')
def edit_profile():
    user = User.query.filter_by(id=session['user_id']).one()
    images = get_images_profile()
    print("images - ", images)
    return render_template('edit_profile.html',user=user, images=images)

@app.route('/get_images_profile', methods=['GET','POST'])
def get_images_profile():
    user = User.query.filter_by(id=session['user_id']).one()
    name = user.name
    images = []
    if name:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join("static", "images","known_images")
        images_dir = os.path.join(current_dir, static_dir)

        for filename in os.listdir(images_dir):
            if filename.startswith(name):
                images.append(filename)

    return images

# Quiz Flask Logic-----------------------------------------------------------------------------------------------------------
# Questions---------------------------------------------------------------------------------------------------------------------------------------------
#query on all our questions and answer data
# @app.route('/viewQuestions/<int:course_id>', methods = ['GET', 'POST'])
# def viewQuestions(course_id):
#     # Questions data
#     questions = Questions.query.filter_by(courseId=course_id).all()
#     courses = Course.query.filter_by(id=course_id).all()
#     lecturers = Lecturers.query.all()
#     # Answers data
#     answers = {}
#     for question in questions:
#         answers[question.id] = Answers.query.filter_by(questionId=question.id).all()

#     return render_template("manageQuestions.html", questions=questions, answers=answers, courses=courses, lecturers=lecturers)

from proctor.courses import courses
from proctor.proctoring import proctoring
app.register_blueprint(courses, name="courses")
app.register_blueprint(admin, name="admin")
app.register_blueprint(proctoring, name="proctoring")


def _add_legacy_endpoint_aliases():
    """Keep existing bare endpoint names working during the package migration."""
    for rule in list(app.url_map.iter_rules()):
        if "." not in rule.endpoint:
            continue
        short_endpoint = rule.endpoint.rsplit(".", 1)[-1]
        if short_endpoint in app.view_functions:
            continue
        methods = sorted(rule.methods.difference({"HEAD", "OPTIONS"}))
        app.add_url_rule(
            rule.rule,
            endpoint=short_endpoint,
            view_func=app.view_functions[rule.endpoint],
            methods=methods,
        )


_add_legacy_endpoint_aliases()

# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    with app.app_context():  # Create the application context
        db.create_all()  # Now it can access the application context
    app.run(debug=True)
