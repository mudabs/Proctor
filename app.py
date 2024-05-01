import base64
from datetime import datetime, timedelta
import string
from threading import Thread
import threading
from flask import Flask, flash ,render_template, Response, session, request, send_file, sessions
import cv2
from matplotlib import pyplot as plt
import numpy as np
import mediapipe as mp
import time
from ultralytics import YOLO
import math
import dlib
import face_recognition
import os
from flask_bootstrap import Bootstrap
import json
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import orm
import random
from pytz import timezone
import sounddevice as sd
import numpy as np
import time as timeSound
import matplotlib.pyplot as plt


# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

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


app.secret_key = 'your_very_secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/proctoring'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # recommended for performance
app.config["SECRET_KEY"] = "mysecretkey"

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(app)

Bootstrap(app)

# Models Loading------------------------------------------------------------------------------------------------------------------------------

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    userType = db.Column(db.String(20))
    imageStatus = db.Column(db.String(20))

    def __init__(self, name, email, password,userType,imageStatus):
        self.name = name
        self.email = email
        self.password = password
        self.userType = userType
        self.imageStatus = imageStatus

    #  Define the relationship with Marks class (assuming a one-to-many relationship)
    marks = orm.relationship("Marks", backref="user")

class Role(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True)

class Questions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(500))
    answerId = db.Column(db.Integer, db.ForeignKey('answers.id'))
    userId = db.Column(db.Integer, db.ForeignKey('user.id'))
    answerType = db.Column(db.String(20))
    topic = db.Column(db.String(100))
    points = db.Column(db.Integer)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id'))

    def __init__(self,question, courseId, userId, answerType, topic,points,quizId):  # Accept both arguments
        self.question = question
        self.courseId = courseId
        self.userId = userId
        self.answerType = answerType
        self.topic = topic
        self.points = points
        self.quizId = quizId

    # def __init2__(self, answerId):  # Accept both arguments
    #     self.answerId = answerId

    def __repr__(self):
        return f"<Question {self.quizId}>"

class Answers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    answer = db.Column(db.String(255))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'))


    def __init__(self, answer, questionId):  # Accept both arguments
        self.answer = answer
        self.questionId = questionId

    def __repr__(self):
        return f"<Answer {self.answer}>"


class CorrectAnswers(db.Model):
    __tablename__ = 'correctanswers'
    id = db.Column(db.Integer, primary_key=True)
    answer = db.Column(db.String(255))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'))
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id', ondelete='CASCADE'))


    def __init__(self, answer, questionId, quizId):  # Accept both arguments
        self.answer = answer
        self.questionId = questionId
        self.quizId = quizId

    def __repr__(self):
        return f"<Answer {self.answer}>"

class ProctorSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    userId = db.Column(db.Integer, db.ForeignKey('user.id'))
    cheating = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="SET NULL"))
    percentage = db.Column(db.String(255))
    time = db.Column(db.String(255))

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    courseTitle = db.Column(db.String(255))
    lecturerId = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="SET NULL"))
    # examId = db.Column(db.Integer, db.ForeignKey('exam.id', ondelete="CASCADE"))
    courseCode = db.Column(db.String(10), unique=True)

    def __init__(self, courseTitle, lecturerId, courseCode):  # Accept both arguments
        self.courseTitle = courseTitle
        self.lecturerId = lecturerId
        self.courseCode = courseCode

    def __repr__(self):
        return f"<Answer {self.courseTitle}>"

class Enrollment(db.Model):
    __tablename__ = 'enrollments'  # Custom table name for clarity
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), primary_key=True)

    # Additional columns for enrollment specific data (optional)
    # enrolment_date = Column(DateTime, default=datetime.utcnow)
    # status = Column(String(20))  # enrolled, completed, etc.

    def __init__(self, user_id, course_id):
        self.user_id = user_id
        self.course_id = course_id

    def __repr__(self):
        return f"<Enrollment user_id: {self.user_id}, course_id: {self.course_id}>"

class Lecturers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))

    def __init__(self, name):  # Accept both arguments
        self.name = name

    def __repr__(self):
        return f"<Answer {self.name}>"

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(255))
    courseId = db.Column(db.Integer, db.ForeignKey('course.id',ondelete="CASCADE"))
    date = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Float, nullable=False)



    def __init__(self, code,courseId,date,duration):  # Accept both arguments
        self.code = code
        self.courseId = courseId
        self.date=date
        self.duration=duration

    def __repr__(self):
        return f"<Answer {self.answer_text}>"

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    courseId = db.Column(db.Integer, db.ForeignKey('course.id',ondelete="CASCADE"))
    topic = db.Column(db.String(100), nullable=False)
    totalPoints = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    duration = db.Column(db.Float, nullable=False)
    instructions = db.Column(db.String(200), nullable=False)
    proctor=db.Column(db.String(20), nullable=False)

    def __init__(self, courseId,topic,totalPoints,date,duration,instructions,proctor):  # Accept both arguments
        self.courseId = courseId
        self.topic=topic
        self.totalPoints=totalPoints
        self.date=date
        self.duration=duration
        self.instructions = instructions
        self.proctor=proctor

    def __repr__(self):
        return f"<Answer {self.id}>"

class QuizQuestions(db.Model):
    __tablename__ = 'quizquestions'
    id = db.Column(db.Integer, primary_key=True)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id',ondelete="CASCADE"))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id',ondelete="CASCADE"))

    def __init__(self, quizId,questionId):  # Accept both arguments
        self.quizId = quizId
        self.questionId=questionId

    def __repr__(self):
        return f"<Quiz Id: {self.quizId}, Question Id: {self.questionId}>"

class Marks(db.Model):
    __tablename__ = 'marks'
    id = db.Column(db.Integer, primary_key=True)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id',ondelete="CASCADE"))
    userId = db.Column(db.Integer, db.ForeignKey('user.id',ondelete="CASCADE"))
    mark = db.Column(db.Float)
    duration = db.Column(db.Integer)
    totalmark = db.Column(db.Float)

    def __init__(self, quizId,userId,mark,duration,totalmark):  # Accept both arguments
        self.quizId = quizId
        self.userId=userId
        self.mark = mark
        self.duration = duration
        self.totalmark = totalmark

    def __repr__(self):
        return f"<Quiz Id: {self.quizId}"

class QuizCompletion(db.Model):
    __tablename__ = 'quizcompletion'
    id = db.Column(db.Integer, primary_key=True)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id',ondelete="CASCADE"))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id',ondelete="CASCADE"))
    answer = db.Column(db.String(500), nullable=False)
    userId = db.Column(db.Integer, db.ForeignKey('user.id',ondelete="CASCADE"))
    status = db.Column(db.String(20), nullable=False)

    def __init__(self, quizId,userId,questionId,answer,status):  # Accept both arguments
        self.quizId = quizId
        self.questionId=questionId
        self.answer = answer
        self.userId=userId
        self.status=status

    def __repr__(self):
        return f"<Completion Id: {self.id}>"

class UserCompletion(db.Model):
    __tablename__ = 'usercompletion'
    id = db.Column(db.Integer, primary_key=True)
    quizId = db.Column(db.Integer, db.ForeignKey('quiz.id',ondelete="CASCADE"))
    userId = db.Column(db.Integer, db.ForeignKey('user.id',ondelete="CASCADE"))
    mark = db.Column(db.Float, nullable=False)
    quizStatus = db.Column(db.String(20), nullable=False)

    def __init__(self, quizId,userId,mark,quizStatus):  # Accept both arguments
        self.quizId = quizId
        self.userId=userId
        self.mark=mark
        self.quizStatus=quizStatus

    def __repr__(self):
        return f"<Completion Id: {self.id}>"

# Define a many-to-many relationship between User and Role
class User_roles(db.Model):
    user_id=db.Column( db.Integer, db.ForeignKey('user.id'))
    role_id=db.Column( db.Integer, db.ForeignKey('role.id'))
    __table_args__ = (db.PrimaryKeyConstraint('user_id', 'role_id'),)

    def __init__(self, user_id,role_id):  # Accept both arguments
        self.user_id = user_id
        self.role_id = role_id

    def __repr__(self):
        return f"<Answer {self.user_id}>"

def assign_role_to_user(userId, roleId):
    user_data = User_roles.query.filter_by(user_id=userId,role_id=roleId).first()
    if user_data:
        print("Error")
        return
    else:
        my_data = User_roles(userId, roleId)
        db.session.add(my_data)
        db.session.commit()
        return

def removed_role(userId, roleId):
    user_data = User_roles.query.filter_by(user_id=userId,role_id=roleId).first()
    if not user_data:
        print("Error")
        return
    else:
        db.session.delete(user_data)
        db.session.commit()
        return

def has_role(self, role):
    return role in self.roles

User.has_role = has_role

# Define the Blocked model
class Blocked(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(200), unique=True, nullable=False)


with app.app_context():  # Ensure we're in the application context
      db.create_all()

# Class Loading------------------------------------------------------------------------------------------------------------------------------

# Load the cascade classifier for face detection (outside of routes for efficiency)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Specify the path to save the images
save_path = os.path.join(app.root_path, 'static','images')  # Ensure path is relative to app root
os.makedirs(save_path, exist_ok=True)

# Define the path to the hosts file
hosts_path = r"C:\Windows\System32\drivers\etc\hosts"

stop_detection = False

# HeadPose Estimation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

known_faces_dir = "./static/images/known_images/"
# Load all known faces and their encodings
known_face_encodings = []
known_face_names = []

def load_known_faces():
  global known_face_encodings, known_face_names
  for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
      image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
      face_encoding = face_recognition.face_encodings(image)[0]
      known_face_encodings.append(face_encoding)
      known_face_names.append(os.path.splitext(os.path.basename(filename))[0])

me=''

def video_detection():
    global cheat
    global lips
    global direction
    global cellphone
    global identity
    global liveness
    global numPeople
    global numFaces
    global me


    confidence = 0.5
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Load both YOLO models
    model_object = YOLO("./models/yolov8n.pt")
    # model_liveness = YOLO("./models/best_20.pt")  # Path to your liveness detection model l_version_1_300.pt
    model_liveness = YOLO("./models/best_20.pt")

    # Face Recognition
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("./models/TrainingImageLabel/Trainner.yml")
    harcascadePath = "./models/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)


    # Face Recognition Arrays


    # Face Recognition Arrays

    classNames_object = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    classNames_liveness = ["real", "fake"]

    # Variables for people counting
    centroid_list = []
    count = 0

    while True:
        success, img = cap.read()

# Head Pose Estimation Opening
        # Flip the image for a selfie-view display
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)


# Face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Update person count based on detections
        centroid_list.clear()
        for (x, y, w, h) in faces:
            # Calculate centroid of the bounding box
            centroid_x = int(x + (w / 2))
            centroid_y = int(y + (h / 2))
            centroid_list.append((centroid_x, centroid_y))

        # Count people entering/leaving the frame (logic can be improved)
        if len(centroid_list) > len(centroid_list) and len(centroid_list) > 0:
            count += 1
        elif len(centroid_list) < len(centroid_list):
            count -= 1

        # numFaces = count

        # # Draw rectangles around detected people and display count
        # for (x, y) in centroid_list:
        #     cv2.rectangle(frame, (x, y), (x + 20, y + 40), (0, 255, 0), 2)
        # cv2.putText(frame, f"People Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 255, 0), 2)
# Face COUNTER


# Face Recognition


        # Initialize variables for face recognition in videos or live streams
        face_locations = []
        face_encodings = []
        face_names = []

        if not success:
            break
        else:
            # Check if faces are detected before processing

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            # rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            count = 0
            if len(face_locations) > 0:
                face_names = []
                for face_encoding in face_encodings:
                    count = count+1
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    displayName = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        displayName = known_face_names[best_match_index]

                    face_names.append(displayName)
                    if identity == me:
                        pass
                    else:
                        face_names.append("Unknown")
                    identity = displayName
            else:
                pass

        numFaces = len(face_locations)

        # Display the results
        # session["facerecognition"] = face_names
        for (top, right, bottom, left), displayName in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, displayName, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # else:
        #     pass
# Face Recognition


# Mouth Detection

        # Detect faces in the frame
        faces = detector(img)

        for face in faces:
            landmarks = predictor(img, face)

            # Extract mouth landmarks (assuming 68-point facial landmark model)
            mouth_left = landmarks.part(48).x, landmarks.part(48).y
            mouth_right = landmarks.part(54).x, landmarks.part(54).y
            mouth_top = landmarks.part(51).x, landmarks.part(51).y
            mouth_bottom = landmarks.part(57).x, landmarks.part(57).y

            # Calculate the distance between top and bottom lip to determine if mouth is open or closed
            lip_distance = mouth_bottom[1] - mouth_top[1]
            print(lip_distance)

            # Display if the mouth is open or closed based on lip distance
            if lip_distance > 21:  # You can adjust this threshold based on your needs
                cv2.putText(img, "Mouth Open", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                lips = "Mouth Open"
            else:
                cv2.putText(img, "Mouth Closed", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                lips = "Mouth Closed"

    # Mouth Detection

        # To improve performance
        img.flags.writeable = False

        # Get the result
        faceResults = face_mesh.process(img)

        # To improve performance
        img.flags.writeable = True

        # Convert the color space from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = img.shape
        face_3d = []
        face_2d = []

        if faceResults.multi_face_landmarks:
            for face_landmarks in faceResults.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360


                # See where the user's head tilting
                # Assigning cheat values based on face direction
                if y < -10:
                    text = "Looking Left"
                    cheat = 0.4
                elif y > 10:
                    text = "Looking Right"
                    cheat = 0.4
                elif x < -10:
                    text = "Looking Down"
                    cheat = 0.8
                elif x > 10:
                    text = "Looking Up"
                    cheat = 0.5
                else:
                    text = "Forward"
                    cheat = 0.15
                direction = text
                print("Cheat 1:",cheat)

                # Add the text on the image
                cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(img, str(cheat), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)


# Head Pose Estimation Closure

        # Perform object detection
        results_object = model_object(img, stream=True)
        num_people = 0
        for r in results_object:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames_object[cls]
                label = f'{class_name}{conf}'
                cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                if (class_name == "cell phone"): # Cellphone detected
                    # session["cellphone"] = "Cell Phone Detected"
                    cellphone = "Cell Phone Detected"
                else:
                    cellphone = "None Detected"


            # Person Counter
                # Check if detected object is a person (class index 0) with high confidence
                if class_name == "person" and conf > 0.5:

                    # Update count (logic can be improved for better accuracy)
                    num_people += 1

                numPeople = num_people
                print("numPeople",numPeople)
       # Perform liveness detection

        face_data = {}

        results_liveness = model_liveness(img, stream=True)
        for r in results_liveness:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                # class_name = classNames_liveness[cls]

                col = (0, 255, 0)
                if conf > confidence:
                    if classNames_liveness[cls] == 'real':
                        col = (0, 255, 0)
                    else:
                        col = (0, 0, 255)

                # Red bounding box for "fake" liveness
                # col = (0, 0, 255) if class_name == "fake" else (0, 255, 0)

                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
                label = f'{classNames_liveness[cls]}{conf}'
                liveness = classNames_liveness[cls]
                app.config["liveness"] = liveness
                cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
# /////////////////////Sound Detection

# Define the soundThreshold for sound detection
soundThreshold = 0.5

# Function to check sound level and save to file
def check_sound(indata, frames, callback_time, status):
    global noise
    volume_norm = np.linalg.norm(indata) * 2
    noise = volume_norm/2
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
    global stop_detection 
    # Start sound capture
    with sd.InputStream(callback=check_sound):
        while not stop_detection:
            timeSound.sleep(1)
          

# /////////////////////Sound Detection


@app.route('/drawGraph', methods=['GET'])
def drawGraph():
    # Read data from file
    file_path = "./session.txt"
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
@app.route('/proctor', methods=['GET', 'POST'])
def proctor():

    data = {
        "cheat": cheat,
        "lips": lips,
        "direction": direction,
        "cellphone": cellphone,
        "identity": identity,
        "liveness": liveness
    }
    return render_template('proctor.html',data=data)

@app.route('/get_objects')
def get_objects():
    global cellphone
    global direction
    global liveness
    global lips
    global identity
    global numPeople
    global numFaces
    global cheat

    if identity == session['username']:
        pass
    else:
        identity == "Unknown"

    cheat = cheatingThreshold()

    f = open("session.txt")
    print(f.readlines())
    f.close()

    print("Identity:", identity)
    return jsonify(cellphone,direction,liveness,lips,identity,numPeople, numFaces, cheat)

def write_exam_session(identity, cellphone, direction, liveness, lips, numPeople,numFaces):
# def write_exam_session():
    global cheat
    filename="./session.txt"
    # Get the current date and time
    now = datetime.now()

    # Get only the current time (hours, minutes, seconds, microseconds)
    current_time = now.time()



    data = f"{current_time},{identity},{cellphone},{direction},{liveness}, {lips}, {numPeople},{numFaces},{cheat}\n"
    #   data = f"{time},{identity},{cellphone},{direction},{liveness}, {lips}, {numPeople},{numFaces},{universalCheat},{duration}\n"

    # Open the file in append mode ('a')
    with open(filename, "a") as file:
        file.write(data)

@app.route('/cheatingThreshold')
def cheatingThreshold():
    universalCheat = 0

    global cellphone
    cellphoneLocal = 0.1
    if cellphone == "Cell Phone Detected":
        cellphoneLocal = 0.5

    global direction
    directionLocal=0.1
    if direction == "Forward":
        directionLocal = 0.15
    elif direction == "Looking Left":
        directionLocal = 0.4
    elif direction == "Looking Right":
        directionLocal = 0.45
    elif direction == "Looking Up":
        directionLocal = 0.55
    elif direction == "Looking Down":
        directionLocal = 0.6

    global liveness
    livenessLocal = 0.1
    if liveness == "real":
        livenessLocal = 0.1
    else:
        livenessLocal = 0.67

    global lips
    lipsLocal=0.1
    if lips == "Mouth Closed":
        lipsLocal = 0.1
    else:
        lipsLocal = 0.35

    global identity
    identityLocal=0.1
    if identity == session['username']:
        identityLocal = 0.1
    else:
        identity == "Unknown"
        identityLocal = 0.7

    global numPeople
    numPeopleLocal = 0.1
    if numPeople == 1:
        numPeopleLocal = 0.1
    else:
        numPeopleLocal = 0.5

    global numFaces
    numFacesLocal=0.1
    if numFaces == 1:
        numFacesLocal = 0.1
    else:
        numFacesLocal = 0.5

    global noise
    noiseLocal=0.0
    if noise > 0.2:
        noiseLocal = 0.67
    else:
        noiseLocal = 0.2


    # universalCheat = (cellphoneLocal+directionLocal+livenessLocal+lipsLocal+identityLocal+numPeopleLocal+numFacesLocal+noiseLocal)/8
    universalCheat = calculate_score(cellphoneLocal, directionLocal, livenessLocal, lipsLocal, identityLocal, numPeopleLocal, numFacesLocal, noiseLocal)
    print("universalCheat - ",universalCheat)

    # Get the current date and time
    now = datetime.now()

    # Get only the current time (hours, minutes, seconds, microseconds)
    current_time = now.time()

    # Print the current time in a specific format (e.g., HH:MM:SS)
    print(current_time.strftime("%H:%M:%S"))


    if universalCheat > 1:
        universalCheat = 1

    universalCheat=round(universalCheat,1)

    # write_exam_session(identityLocal, cellphoneLocal, directionLocal, livenessLocal, lipsLocal, numPeopleLocal,numFacesLocal,universalCheat,session['myDuration'])
    write_exam_session(identityLocal, cellphoneLocal, directionLocal, livenessLocal, lipsLocal, numPeopleLocal,numFacesLocal)
    # write_exam_session()


    return universalCheat

def calculate_score(cellphoneLocal, directionLocal, liveness, lipsLocal, identityLocal, numPeopleLocal, numFacesLocal, noiseLocal):
  """
  This function calculates a score based on input variables, with higher scores indicating a higher chance of cheating.

  Args:
      cellphoneLocal: (float) Local value for cellphone detection.
      directionLocal: (float) Local value for direction detection. (Consider adjusting its weight if not relevant)
      liveness: (float) Liveness detection value.
      lipsLocal: (float) Local value for lips detection.
      identityLocal: (float) Local value for identity detection.
      numPeopleLocal: (float) Local value for number of people detected.
      numFacesLocal: (float) Local value for number of faces detected.
      noiseLocal: (float) Local value for noise level.

  Returns:
      (float) A score between 0 and 1, with higher values indicating a higher chance of cheating.
  """

  # Weights for each variable (adjust these based on your data and priorities)
  cellphone_weight = cellphoneLocal
  direction_weight = directionLocal  # Adjust weight if direction detection is not crucial
  liveness_weight = liveness
  lips_weight = lipsLocal
  identity_weight = identityLocal
  num_people_weight = numPeopleLocal  # Lower weight as presence of others might not indicate cheating
  num_faces_weight = 0.1  # Lower weight for similar reason as num_people

  # Adjust the logic for cellphone and face detection as needed
  cellphone_penalty = 1 - cellphoneLocal  # Higher penalty for cellphone detection
  no_face_penalty = 1 - numFacesLocal  # Higher penalty for no face detected

  # Calculate the weighted sum with penalties
  score = (cellphone_weight * cellphone_penalty) + \
         (direction_weight * directionLocal) + \
         (liveness_weight * liveness) + \
         (lips_weight * lipsLocal) + \
         (identity_weight * identityLocal) + \
         (num_people_weight * numPeopleLocal) + \
         (num_faces_weight * numFacesLocal) + \
         (noiseLocal * 0.2)  # Add noise with lower weight

  # Apply a threshold function to ensure the score is within the desired range (0-1)
  score = min(score, 1)  # Cap the score at 1
#   score = max(score, 0.96)  # Set a minimum score of 0.96 (can be adjusted)

  return score


# Video Route for Proctoring -- Called in proctor.html
@app.route('/video')
def video():
    load_known_faces()
    # Create a dictionary with the variables
    return Response(video_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for the home page
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

@app.route('/resetPassword',methods=['POST'])
def resetPassword():
    user = User.query.filter_by(id=session['user_id']).one()
    if request.method == 'POST':
        if(request.form["p1"] == request.form["p2"]):
            user.password = request.form["p1"]
            try:
                db.session.commit()
            except Exception as e:
                print("Error committing changes:", e)
                db.session.rollback()  # Revert changes if error occurs

    return redirect(url_for('edit_profile'))


@app.route('/display_images_admin')
def display_images_admin():
    all_data = User.query.all()

    return render_template('display_images_admin.html', Users = all_data)

@app.route('/get_images', methods=['GET','POST'])
def get_images():
    name = request.form['user_name']
    images = []

    if name:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join("static", "images","known_images")
        images_dir = os.path.join(current_dir, static_dir)
        for filename in os.listdir(images_dir):
            if filename.startswith(name):
                images.append(filename)
    return json.dumps({"images": images})


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

@app.route('/viewAllQuestions/<int:quizId>', methods = ['GET', 'POST'])
def viewAllQuestions(quizId):

    return render_template("manageResults.html")



@app.route('/addQuestions/<courseId>',methods=['GET','POST'])
def addQuestions(courseId):
    lecturerId = session['user_id']
    num_inputs = request.args.get('num_inputs', 1, type=int)
    course = Course.query.filter_by(id=courseId).all()
    quiz = Quiz.query.filter_by(courseId = courseId).all()
    quizI = Quiz.query.filter_by(courseId = courseId).first()
    quizId=quizI
    questions = Questions.query.filter_by(quizId=quizId.id)
    answers = Answers.query.all()


    return render_template("addQuestions.html", quiz=quiz, num_inputs=num_inputs, lecturerId=lecturerId,courseId=courseId,course=course, questions=questions,answers=answers)

@app.route('/addQuestion', methods=['POST'])
def addQuestion():
    correctAnswer = ""
    if request.method == 'POST':
        question = request.form['question']
        courseId = request.form['courseId']
        userId = session['user_id']
        answerType = request.form['answerType']
        topic = request.form['topic']
        points = request.form['qpoints']
        quiz = Quiz.query.filter_by(topic=topic, courseId=courseId).first()
        if quiz is not None:
            quizId = quiz.id
        else:
            # Handle the case where no quiz is found (e.g., display an error message)
            pass

        my_data = Questions(question=question,courseId=courseId,userId=userId,answerType=answerType,topic=topic,points=points,quizId=quizId)
        db.session.add(my_data)
        db.session.commit()
        quizAllocation = QuizQuestions(quizId=quizId,questionId=my_data.id)
        db.session.add(quizAllocation)
        db.session.commit()
        answers = []
        print(my_data.id)
        if request.form["answerType"] == "Multiple Choice":

            for i in range(1, int(request.form['num_inputs']) + 1):
                ans = request.form[f'answer{i}']
                if ans:  # Check if input field is not empty
                    answer = Answers(answer=ans, questionId =  my_data.id)
                    answers.append(answer)
            db.session.add_all(answers)
            db.session.commit()


            answerPlaceHolder = request.form["checkAnswer"]
            correctAnswer = request.form[answerPlaceHolder]
            print(correctAnswer)
            answerVar = CorrectAnswers(answer = correctAnswer,questionId=my_data.id,quizId=quizId)
            db.session.add(answerVar)
            db.session.commit()


    return redirect(url_for('addQuestions',courseId=courseId))

#Create Questions
@app.route('/createQuestions', methods = ['POST'])
def createQuestions():
    if request.method == 'POST':
        question = request.form['question']
        course = request.form['courseId']
        lecturer = request.form['lecturerId']
        answer1 = request.form['answer1']
        answer2 = request.form['answer2']
        answer3 = request.form['answer3']
        answer4 = request.form['answer4']

        # Adding Question to DB
        my_data = Questions(question, course, lecturer)
        db.session.add(my_data)
        db.session.commit()

        # Adding ANswers to DB
        answers = [answer1, answer2, answer3, answer4]
        for answer_text in answers:
            my_answer = Answers(answer_text, questionId=my_data.id)  # Use my_data.id after commit
            db.session.add(my_answer)
        db.session.commit()

        flash("Question Created Successfully")
        return redirect(url_for('addQuestions',course_id=course))

#Update questions
@app.route('/updateQuestions', methods = ['GET', 'POST'])
def update():
    if request.method == 'POST':
        my_data = Questions.query.get(request.form.get('id'))
        my_data2 = Answers.query.filter_by(questionId=request.form.get('id')).all()
        course = request.form['courseId']

        my_data.question = request.form['question']
        for answer in my_data2:
            # Update individual answer based on form data
            answer.answer = request.form.get(f"answer_{answer.id}")  # Access answer-specific form field

        db.session.commit()
        flash("Question Updated Successfully")
    return redirect(url_for('viewQuestions',course_id=course))


#Delete questions
@app.route('/deleteQuestions/<id>/<course_id>', methods = ['GET', 'POST'])
def deleteQuestions(id,course_id):
    my_data = Questions.query.get(id)
    if my_data:  # Check if questions exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Questions Deleted Successfully")
    else:
        flash("Questions not found!")
    return redirect(url_for('addQuestions', courseId = course_id))
# Questions---------------------------------------------------------------------------------------------------------------------------------------------


# Lecturers---------------------------------------------------------------------------------------------------------------------------------------------
# View Lecturers
@app.route('/viewLecturers')
def viewLecturers():
    all_data = Lecturers.query.all()
    return render_template("manageLecturers.html", lecturers = all_data)


# Create Lecturers
@app.route('/createLecturers', methods = ['POST'])
def createLecturers():
    if request.method == 'POST':
        lecturer = request.form['lecturerName']

        # Adding Question to DB
        my_data = Lecturers(lecturer)
        db.session.add(my_data)
        db.session.commit()

        flash("Question Created Successfully")
        return redirect(url_for('viewLecturers'))

# update lecturers
@app.route('/updateLecturers', methods = ['GET', 'POST'])
def updateLecturers():
    if request.method == 'POST':
        my_data = Lecturers.query.get(request.form.get('id'))

        my_data.name = request.form['name']
        db.session.commit()
        flash("Lecturer Data Updated Successfully")
    return redirect(url_for('viewLecturers'))


# Delete lecturers
@app.route('/deleteLecturers/<id>/', methods=['GET', 'POST'])
def deleteLecturers(id):
    my_data = Lecturers.query.get(id)
    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Lecturer Deleted Successfully")
    else:
        flash("Lecturer not found!")
    return redirect(url_for('viewLecturers'))

# Lecturers---------------------------------------------------------------------------------------------------------------------------------------------

# Users---------------------------------------------------------------------------------------------------------------------------------------------

# User Roles
@app.route('/users')
def user_list():
    users = User.query.all()
    return render_template('roles.html', users=users)

#insert data to mysql database via html forms
@app.route('/viewUsers')
def viewUsers():
    # Questions data
    all_data = User.query.all()

    return render_template("manageUsers.html", Users = all_data)



#Create Users
@app.route('/createUsers', methods = ['POST'])
def createUsers():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        userType = request.form['userType']
        password = "default"
        imageStatus = "Unregistered"

        # Adding Users to DB
        my_data = User(name,email,password,userType,imageStatus)
        db.session.add(my_data)
        db.session.commit()

        flash("User Created Successfully")
        return redirect(url_for('display_images_admin'))

#update users
@app.route('/updateUsers', methods = ['GET', 'POST'])
def updateUsers():
    if request.method == 'POST':
        my_data = User.query.get(request.form.get('id'))

        my_data.name = request.form['name']
        my_data.regNumber = request.form['regNumber']
        db.session.commit()
        flash("User Data Updated Successfully")
    return redirect(url_for('display_images_admin'))


# Delete users
@app.route('/deleteUsers/<id>/', methods=['GET', 'POST'])
def deleteUsers(id):
    my_data = User.query.get(id)
    if my_data:  # Check if user exists before deleting
        name = my_data.name
        print("name",name)
        images = []
        if name:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            static_dir = os.path.join("static", "images","known_images")
            images_dir = os.path.join(current_dir, static_dir)

            for filename in os.listdir(images_dir):
                if filename.startswith(name):
                    newDir = os.path.join(images_dir, filename)
                    images.append(newDir)

            for image in images:
                os.remove(image)
                print("Removed file ",image)

        db.session.delete(my_data)
        db.session.commit()
        flash("User Deleted Successfully")
    else:
        flash("User not found!")
    return redirect(url_for('display_images_admin'))

# Users---------------------------------------------------------------------------------------------------------------------------------------------

# Course---------------------------------------------------------------------------------------------------------------------------------------------
#insert data to mysql database via html forms
@app.route('/viewCourses')
def viewCourses():
    # Courses data
    all_data = Course.query.all()
    lecturers = Lecturers.query.all()
    return render_template("manageCourses.html", courses = all_data, lecturers = lecturers, user =  session['username'])


@app.route('/courseAdminSide/<courseId>', methods=['GET'])
def courseAdminSide(courseId):
    courses = Course.query.filter_by(id = courseId).all()
    courseTitle=''
    for c in courses:
        courseTitle = c.courseTitle
    enrollment = Enrollment.query.filter_by(course_id = courseId).all()
    users = User.query.all()
    quiz = Quiz.query.filter_by(courseId = courseId).all()
    return render_template("courseAdminSide.html",courses = courses,enrollment=enrollment,users=users,quiz=quiz,courseTitle=courseTitle)


@app.route('/courseUsersSide/<courseId>', methods=['GET'])
def courseUsersSide(courseId):
    qq = []
    courses = Course.query.filter_by(id = courseId).all()
    enrollment = Enrollment.query.filter_by(course_id = courseId).all()
    users = User.query.all()
    quizes = Quiz.query.filter_by(courseId = courseId).all()
    marks = Marks.query.filter_by(userId=session['user_id'])
    for q in quizes:
        for mark in marks:
            if q.id == mark.quizId:
                pass
            else:
                qq.append(q)

    return render_template("courseUserSide.html",courses = courses,enrollment=enrollment,users=users,quizes=quizes,allquizes = quizes,marks=marks, userId =  session['user_id'])

#insert data to mysql database via html forms
@app.route('/viewCoursesUser')
def viewCoursesUser():
    # Courses data
    all_data = Course.query.filter_by(lecturerId=session['user_id']).all()
    lecturers = Lecturers.query.all()

    return render_template("manageCoursesUser.html", courses = all_data, lecturers = lecturers, user =  session['username'])


def generateCourseCode():
    """Generates a 6-character word with random letters and numbers."""
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(6))


#insert data to mysql database via html forms
@app.route('/createCourse', methods = ['GET','POST'])
def createCourse():
    if request.method == 'POST':
        course = request.form['courseName']
        courseCode = generateCourseCode()
        # Adding Course to DB
        my_data = Course(course,session['user_id'],courseCode)
        db.session.add(my_data)
        db.session.commit()

        flash("Course Created Successfully")
    return redirect(url_for('courseAdminSide',courseId = my_data.id))


#insert data to mysql database via html forms
@app.route('/joinCourse', methods = ['POST'])
def joinCourse():
    courseId=''
    if request.method == 'POST':
        courseCode = request.form['courseCode']
        # Adding Course to DB
        courses = Course.query.all()
        for course in courses:
            if courseCode == course.courseCode:
                courseId=course.id
                my_data = Enrollment(session['user_id'],course.id)
                db.session.add(my_data)
                db.session.commit()
                message = "You have enrolled for couse Successfully"
            else:
                message = "Course Not Found"

    flash(message)
    return redirect(url_for('courseUsersSide', courseId=courseId))


#insert data to mysql database via html forms
@app.route('/enroll', methods = ['POST'])
def enroll():
    if request.method == 'POST':
        course = request.form['course']
        student = request.form['student']
        # Adding Course to DB
        my_data = Enrollment(student,course)
        db.session.add(my_data)
        db.session.commit()

        flash("Lecturer Added to Course Successfully")
    return redirect(url_for('courseUsersSide', courseId=course))

# Unenroll Course
@app.route('/unenroll/<user_id>/<course_id>', methods=['GET', 'POST'])
def unenroll(user_id,course_id):
    my_data = Enrollment.query.get((user_id,course_id))
    # my_data = Enrollment.query.filter_by(user_id=user_id, course_id=course_id).first()

    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Unenrolled Successfully")
    else:
        flash("Error! not found!")
    return redirect(url_for('home'))

#update Course
@app.route('/updateCourse', methods = ['GET', 'POST'])
def updateCourse():
    if request.method == 'POST':
        my_data = Course.query.get(request.form.get('id'))

        my_data.courseTitle = request.form['courseName']
        my_data.lecturerId = request.form['lecturerId']
        db.session.commit()
        flash("Course Data Updated Successfully")
    return redirect(url_for('viewCourses'))


# Delete Course
@app.route('/deleteCourse/<id>/', methods=['GET', 'POST'])
def deleteCourse(id):
    my_data = Course.query.get(id)
    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Course Deleted Successfully")
    else:
        flash("Course not found!")
    return redirect(url_for('home'))
# Course---------------------------------------------------------------------------------------------------------------------------------------------



# Exam---------------------------------------------------------------------------------------------------------------------------------------------
#insert data to mysql database via html forms
@app.route('/manageResults/<quizId>')
def manageResults(quizId):
    quiz = Quiz.query.filter_by(id=quizId)
    courseId=''
    courseTitle = ''
    quizTitle = ''
    for q in quiz:
        courseId = q.courseId
        quizTitle = q.topic
    course = Course.query.filter_by(id = courseId)
    for c in course:
        courseTitle = c.courseTitle

    # users = User.query.all()
    # marks = Marks.query.filter_by(quizId = quizId)
    users = User.query.join(Marks, User.id == Marks.userId).filter(Marks.quizId == quizId).all()
    marks = Marks.query.join(User, Marks.userId == User.id).filter(Marks.quizId == quizId).all()

    return render_template("manageResults.html",courseTitle=courseTitle,quizTitle=quizTitle,users=users)

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    topic = request.args.get('topic')
    my_data = Quiz.query.filter_by(topic = topic).all()
    data=[]
    # ... query database based on selected topic ...
    if topic == "0":
        data.append({
        'points': " ",
        'dueDate': " ",
        'instructions': " ",
        'proctor': " "
        })
    else:
        for quiz in my_data:
            data.append({
                'points': quiz.totalPoints,
                'dueDate': quiz.date,
                'instructions': quiz.instructions,
                'proctor': quiz.proctor
            })
    return jsonify(data)

#insert data to mysql database via html forms
@app.route('/createExams', methods = ['POST'])
def createExams():
    if request.method == 'POST':
        code = request.form['examName']
        courseId = request.form['courseId']
        date = request.form['date']
        durationHours = request.form['durationHours']
        durationMinutes = request.form['durationMinutes']
        duration = float(durationHours) + (float(durationMinutes)/60)
        # Adding Exam to DB
        my_data = Exam(code,courseId,date,duration)
        db.session.add(my_data)
        db.session.commit()

        flash("Exam Created Successfully")
    return redirect(url_for('viewExams'))


#insert data to mysql database via html forms
@app.route('/createQuiz', methods = ['POST'])
def createQuiz():
    if request.method == 'POST':
        courseId = request.form['course']
        topic = request.form['topic']
        totalPoints = request.form['totalPoints']
        date = request.form['duedate']
        durationHours = request.form['durationHours']
        durationMinutes = request.form['durationMinutes']
        duration = float(durationHours) + (float(durationMinutes)/60)
        instructions = request.form['instructions']
        if request.form['proctor']:
            proctor = "True"
        else:
            proctor = "False"

        # Adding Quiz to DB
        my_data = Quiz(courseId,topic,totalPoints,date,duration,instructions,proctor)
        db.session.add(my_data)
        db.session.commit()

        flash("Exam Created Successfully")
    return redirect(url_for('addQuestions',courseId=courseId,topic=topic,totalPoints=totalPoints,date=date,duration=duration,instructions=instructions,proctor=proctor))


#update Exam
@app.route('/updateExams', methods = ['GET', 'POST'])
def updateExams():
    if request.method == 'POST':
        my_data = Exam.query.get(request.form.get('id'))

        my_data.courseTitle = request.form['courseName']
        my_data.lecturerId = request.form['lecturerId']
        db.session.commit()
        flash("Exam Data Updated Successfully")
    return redirect(url_for('viewExams'))


# Delete Exam
@app.route('/deleteExams/<id>/', methods=['GET', 'POST'])
def deleteExams(id):
    my_data = Exam.query.get(id)
    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Lecturer Deleted Successfully")
    else:
        flash("Lecturer not found!")
    return redirect(url_for('viewExams'))

@app.route('/setTimer/<quizId>')
def setTimer(quizId):
    configure_app_and_access_session(app, session)
    black()
    return redirect(url_for('takeQuiz',quizId=quizId))

@app.route('/takeQuiz/<quizId>')
def takeQuiz(quizId):
    global stop_detection
    quiz = Quiz.query.filter_by(id = quizId).all()
    questionLink = QuizQuestions.query.filter_by(quizId = quizId).all()
    questions = Questions.query.filter_by(quizId=quizId).all()

    answers = Answers.query.all()
    session['quiz'] = quiz[0].topic
    session['duration'] = quiz[0].duration
    session['myDuration'] = datetime.now().timestamp()

    user_timezone = session['expiration_time'].tzinfo
    numQuestions = len(questions)

    now_with_timezone = datetime.now(user_timezone)

    # Start sound detection in a separate thread
    sound_thread = threading.Thread(target=detectSound)
    sound_thread.start()


    if now_with_timezone > session['expiration_time']:
        # Handle quiz expiration (e.g., redirect to a different page, display a message)
        stop_detection = True
        drawGraph()
        drawSoundGraph()
        return redirect(url_for('home'))  # Example redirect
    
    return render_template("takeQuiz.html",quiz=quiz,quizId=quizId,questionLink=questionLink,questions=questions,answers=answers,numQuestions=numQuestions,userId = session['user_id'])

# Submit Exam
@app.route('/quizCompletion', methods=['POST'])
def quizCompletion():
    global stop_detection 
    stop_detection = True

    if request.method == 'POST':
        unblock()
        attemptedAnswers=[]
        status="0"
        totalmarks = 0

        quizId = request.form["quizId"]
        userId = request.form["userId"]

        # Marks
        mark=0
        correctAnswers = CorrectAnswers.query.filter_by(quizId=quizId).all()
        questions = Questions.query.filter_by(quizId=quizId).all()



        for question in questions:
            totalmarks = totalmarks + float(question.points)
            if question.answerType == "Multiple Choice":
                mcq = "mcq" + str(question.id)
                try:
                    answer=request.form[mcq]

                    for correctAnswer in correctAnswers:
                        if correctAnswer.questionId == question.id:

                            if answer == correctAnswer.answer:
                                status = str(question.points)
                            else:
                                status = "0"
                            mark += float(status)

                    attemptedAnswers.append(answer)
                except Exception as e:
                    print(attemptedAnswers)
            else:
                status="pending"

            now = datetime.now()
            myduration = now.timestamp() - session['myDuration']
            session['myDuration'] = int(myduration)
            my_data = QuizCompletion(quizId=quizId,userId=userId,questionId = question.id,answer=answer, status = status)
            db.session.add(my_data)
            db.session.commit()
        marks = Marks(quizId = quizId,userId = userId,mark = mark, duration = session['myDuration'], totalmark = totalmarks)
        db.session.add(marks)
        db.session.commit()

        drawGraph()
        drawSoundGraph()
        

    return redirect(url_for('userResults',quizId=quizId))

@app.route('/displayGraph/<userId>', methods=['POST'])
def displayGraph(userId):

    # if request.method == 'POST':

    return render_template('userResults.html')



@app.route('/userResults/<quizId>', methods=['GET'])
def userResults(quizId):

    quiz = Quiz.query.filter_by(id = quizId).all()
    questionLink = QuizQuestions.query.filter_by(quizId = quizId).all()
    questions = Questions.query.filter_by(quizId=quizId).all()
    answers = Answers.query.all()
    session['quiz'] = quiz[0].topic
    session['duration'] = quiz[0].duration
    numQuestions = len(questions)
    quizCompletion = QuizCompletion.query.filter_by(quizId=quizId,userId=session['user_id']).all()
    marks = Marks.query.filter_by(quizId=quizId,userId=session['user_id']).all()


    marks = Marks.query.filter_by(quizId=quizId,userId=session['user_id']).all()
    marks = Marks.query.filter_by(quizId=quizId,userId=session['user_id']).all()
    return render_template('userResults.html', quiz=quiz,quizId=quizId,questionLink=questionLink,questions=questions,answers=answers,numQuestions=numQuestions,userId = session['user_id'],quizCompletion=quizCompletion,marks=marks)




# Exam---------------------------------------------------------------------------------------------------------------------------------------------





# Login------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    global me
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        userType = request.form['userType']
        try:
            user = User.query.filter_by(email=email).one()
            if user.password == password and user.userType == userType and user.email == email:
                session['user_id'] = user.id
                session['username'] = user.name
                me = session['username']
                flash('Login successful!', 'success')
                return redirect(url_for('home'))
            else:
                flash('Invalid credentials!', 'danger')
                return render_template('login.html', error="Incorrect password or User Type.")
        except NoResultFound:
            flash('User does not exist.', 'danger')
            return render_template('login.html', error="User does not exist.")
    return render_template('login.html', error=None)
# Login------------------------------------------------------------------------------------------------------------------------------------------------

# Logout-----------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Logout successful!', 'success')
    return redirect(url_for('login'))
# Logout-----------------------------------------------------------------------------------------------------------------------------------------------


#  Register---------------------------------------------------------------------------------------------------------------------------------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    # global capture_enabled, name, id, image_count
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        userType = request.form['userType']
        imageStatus = "Registered"
        try:
            new_user = User(name=name, email=email, password=password, userType=userType,imageStatus=imageStatus)
            db.session.add(new_user)
            db.session.commit()
            session['user_id'] = new_user.id
            return redirect(url_for('home'))
        except IntegrityError:
            db.session.rollback()
            return render_template('register.html', error="User already exists.")
    return render_template('register.html', error=None)

@app.route('/captureImage', methods=['GET','POST'])
def captureImage():
    formData = request.form
    message = ''

    if request.form["name"] :
        # Check if the request came from the "Capture Image" button
        if 'imageDataURL' in formData:
            print("Hello")
            imageData = formData['imageDataURL'].split(',')[1]
            imageName = formData['name'] + '.jpg'
            imagePath = os.path.join('static/images/known_images', imageName)

            # Check if the file already exists
            if check_names(request.form['name'] == False):
                with open(os.path.join('static/images/known_images', imageName), 'wb') as f:
                    f.write(base64.decodebytes(imageData.encode()))
                    message = 'User image captured successfully'
                return jsonify({'message': message})
            else:
                return jsonify({'message': 'User image not captured'})
    else:
        message = "Input value for name"

    return jsonify({'message': message})

@app.route('/recaptureImage', methods=['GET','POST'])
def recaptureImage():
    formData = request.form
    message = ''
    print("imageData",formData)
    if 'imageDataURL' in formData:
        imageData = formData['imageDataURL'].split(',')[1]

        user = User.query.filter_by(id=session['user_id']).one()
        imageName = user.name+'.jpg'

        with open(os.path.join('static/images/known_images', imageName), 'wb') as f:
            f.write(base64.decodebytes(imageData.encode()))
            message = 'User image captured successfully'

            print(user.imageStatus)
            if (user.imageStatus == "Unregistered"):
                user.imageStatus = "Registered"
                db.session.commit()

        return jsonify({'message': message})
    return redirect(url_for('edit_image'))

#  Register---------------------------------------------------------------------------------------------------------------------------------------------

# Roles and Login---------------------------------------------------------------------------------------------------------------------------------------

@app.route('/roles', methods=['GET', 'POST'])
def manage_roles():
    if request.method == 'POST':
        role_name = request.form['name']  # Get role name from HTML form
        if role_name:
            role = Role(name=role_name)
            db.session.add(role)
            db.session.commit()
            flash(f'Role "{role.name}" created successfully!', 'success')
        else:
            flash('Please enter a role name!', 'error')
    roles = Role.query.all()
    users = User.query.all()
    return render_template('userroles.html', roles=roles,users=users)

@app.route('/assign_roles/<int:user_id>', methods=['GET'])
def assign_roles(user_id):
    user = User.query.filter_by(id=user_id).first()
    if not user:
        flash('User not found!', 'error')
        return redirect(url_for('user_list'))  # Replace with your user list route

    assigned_roles = User_roles.query.filter_by(user_id=user_id)
    roles = Role.query.all()
    return render_template('assign_roles.html', user=user, roles=roles,assigned_roles=assigned_roles)

@app.route('/assign_role/<int:user_id>', methods=['POST'])
def assign_role(user_id):
    user = User.query.get(user_id)
    role_id = request.form['role_id']
    role = Role.query.get(role_id)
    if not user or not role:
        flash('Invalid user or role!', 'error')
        return redirect(url_for('assign_roles', user_id=user_id))
    assign_role_to_user(user_id, role_id)
    flash(f'Role "{role.name}" assigned to user "{user.name}" successfully!', 'success')
    return redirect(url_for('assign_roles', user_id=user_id))

@app.route('/remove_role/<int:user_id>/<int:role_id>', methods=['GET','POST'])
def remove_role(user_id, role_id):
    role = User_roles.query.filter_by(user_id=user_id,role_id=role_id)
    user = User.query.get(user_id)
    roles = Role.query.get(role_id)
    if not role:
        flash('Invalid user or role!', 'error')
        return redirect(url_for('assign_roles', user_id=user_id))
    removed_role(user_id, role_id)
    flash(f'Role removed from user  successfully!', 'success')
    return redirect(url_for('assign_roles', user_id=user_id))

# Roles and Login---------------------------------------------------------------------------------------------------------------------------------------
@app.route('/edit_image')
def edit_image():
    user = User.query.filter_by(id=session['user_id']).one()
    return render_template('edit_image.html', user = user)

@app.route('/deleteImages', methods=['GET','POST'])
def deleteImages():
    global name , image_count, capture_enabled
    if request.method == 'POST':
        if (request.form["del"] == "True"):
            images = []
            if name:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                static_dir = os.path.join("static", "images")
                images_dir = os.path.join(current_dir, static_dir)

                for filename in os.listdir(images_dir):
                    if filename.startswith(name):
                        newDir = os.path.join(images_dir, filename)
                        images.append(newDir)

                for image in images:
                    os.remove(image)
                    print("Removed file ",image)
    return redirect(url_for('captureImageReg'))

# Exam Clock---------------------------------------------------------------------------------------------------------------------------------------------

def get_remaining_time_in_seconds():
    if session["expiration_time"]:
        remaining_time = app.config['expiration_time'] - datetime.now()
        remaining_time_in_seconds = remaining_time.total_seconds()
    return remaining_time_in_seconds

@app.route('/remaining_time')
def remaining_time():
    remaining_time_in_seconds = get_remaining_time_in_seconds()
    if remaining_time_in_seconds <= 0:
        session["messages"] = "Time is up"
        reset()
    return jsonify({'remaining_time_in_seconds': remaining_time_in_seconds})

@app.route('/reset', methods=['POST'])
def reset():

    if request.method == 'POST':
        session['quiz'] = "False"
        app.config['expiration_time'] = datetime.now() + timedelta(minutes=10)
        return render_template('home.html')

@app.route('/clearTimer', methods=['POST'])
def clearTimer():
    if request.method == 'POST':
        app.config['expiration_time'] = ''

        return redirect(url_for('home'))


# ------------------------------------------------------------------------------------------
# BlackListing Website

# Function to block websites
def block_websites():
    # List of websites to block
    blocked_websites = [blocked.url for blocked in Blocked.query.all()]
    # Get current date and time
    now = datetime.now()
    # Add blocked websites to the hosts file
    with open(hosts_path, 'a') as hosts_file:
        # hosts_file.write("\n# Blocked websites added by website blocker\n")
        hosts_file.write("\n\n")
        for website in blocked_websites:
            hosts_file.write("127.0.0.1 {}\n".format(website))
            hosts_file.write("127.0.0.1 www.{}\n".format(website))
    print("Websites blocked successfully at", now)

# Function to unblock websites
def unblock_websites():
    # Read the hosts file and remove lines related to blocked websites
    with open(hosts_path, 'r') as hosts_file:
        lines = hosts_file.readlines()
    with open(hosts_path, 'w') as hosts_file:
        for line in lines:
            if not any(website in line for website in [blocked.url for blocked in Blocked.query.all()]):
                hosts_file.write(line)
    print("Websites unblocked successfully")

# Homepage route
@app.route('/blacklist', methods=['GET', 'POST'])
def blacklist():
    if request.method == 'POST':
        # urls = request.form.get('urls')
        # if urls:
            # Split the input by newline and remove empty lines
            # urls = [url.strip() for url in urls.split('\n') if url.strip()]
        blocked_websites = [
            "researchgate.net",
            "scholar.google.com",
            "pubmed.ncbi.nlm.nih.gov",
            "ieeexplore.ieee.org",
            "sciencedirect.com",
            "jstor.org",
            "link.springer.com",
            "onlinelibrary.wiley.com",
            "arxiv.org",
            "ssrn.com",
            "nature.com",
            "elsevier.com",
            "dl.acm.org",
            "scopus.com",
            "plos.org",
            "academic.oup.com",
            "tandfonline.com",
            "research.com",
            "researcher.com",
            "worldcat.org",
            "google.com",
            "bing.com",
            "yahoo.com",
            "duckduckgo.com",
            "baidu.com",
            "yandex.com",
            "ask.com",
            "ecosia.org",
            "startpage.com",
            "swisscows.com"
        ]

        # Add the URLs to the database
        for url in blocked_websites:
            blocked = Blocked(url=url)
            db.session.add(blocked)
        db.session.commit()
        # Block websites after adding them to the database
        unblock_websites()
        block_websites()
        flash('Websites blocked successfully', 'success')
        return redirect(url_for('blacklist'))
    
    # Fetch all blocked URLs from the database
    blocked_urls = Blocked.query.all()
    return render_template('blacklist.html', blocked_urls=blocked_urls)

@app.route('/black')
def black():
    blocked_websites = [
        "researchgate.net",
        "scholar.google.com",
        "pubmed.ncbi.nlm.nih.gov",
        "ieeexplore.ieee.org",
        "sciencedirect.com",
        "jstor.org",
        "link.springer.com",
        "onlinelibrary.wiley.com",
        "arxiv.org",
        "ssrn.com",
        "nature.com",
        "elsevier.com",
        "dl.acm.org",
        "scopus.com",
        "plos.org",
        "academic.oup.com",
        "tandfonline.com",
        "research.com",
        "researcher.com",
        "worldcat.org",
        "google.com",
        "bing.com",
        "yahoo.com",
        "duckduckgo.com",
        "baidu.com",
        "yandex.com",
        "ask.com",
        "ecosia.org",
        "startpage.com",
        "swisscows.com"
    ]

    # Add the URLs to the database
    for url in blocked_websites:
        blocked = Blocked(url=url)
        db.session.add(blocked)
    db.session.commit()
    # Block websites after adding them to the database
    unblock_websites()
    block_websites()
    flash('Websites blocked successfully', 'success')
    return redirect(url_for('blacklist'))
    


# Route to unblock all websites
@app.route('/unblock', methods=['POST'])
def unblock():
    unblock_websites()
    # Delete all records from the Blocked table
    Blocked.query.delete()
    db.session.commit()
    # Unblock websites after removing them from the database
    
    flash('Websites unblocked successfully', 'success')
    return redirect(url_for('blacklist'))


































































if __name__ == '__main__':
    with app.app_context():  # Create the application context
        db.create_all()  # Now it can access the application context
    app.run(debug=True)
