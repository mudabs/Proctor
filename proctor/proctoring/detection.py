"""Computer-vision detection pipeline and shared detection state."""

import cv2
import dlib
import face_recognition
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO

from app import app
from proctor import state

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

cheat = 0
lips = ""
direction = ""
cellphone = ""
identity = ""
liveness = ""
numPeople = 0
numFaces = 0
noise = 0
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Specify the path to save the images
save_path = os.path.join(app.root_path, 'static','images')  # Ensure path is relative to app root
os.makedirs(save_path, exist_ok=True)

# Define the path to the hosts file
hosts_path = r"C:\Windows\System32\drivers\etc\hosts"


# HeadPose Estimation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

known_faces_dir = "./static/images/known_images/"
# Load all known faces and their encodings
known_face_encodings = []
known_face_names = []


def write_exam_session(identity, cellphone, direction, liveness, lips, num_people, num_faces):
    data = (
        f"{datetime.now().time()},{identity},{cellphone},{direction},"
        f"{liveness}, {lips}, {num_people},{num_faces},{cheat}\n"
    )
    with open("./session.txt", "a") as session_file:
        session_file.write(data)

def load_known_faces():
  global known_face_encodings, known_face_names
  for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
      image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
      face_encoding = face_recognition.face_encodings(image)[0]
      known_face_encodings.append(face_encoding)
      known_face_names.append(os.path.splitext(os.path.basename(filename))[0])


def video_detection():
    global cheat
    global lips
    global direction
    global cellphone
    global identity
    global liveness
    global numPeople
    global numFaces


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
                    if identity == state.me:
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


def calculate_score(cellphone_local, direction_local, liveness_local,
                    lips_local, identity_local, num_people_local,
                    num_faces_local, noise_local):
    """Calculate the bounded cheating score from detection signals."""
    cellphone_penalty = 1 - cellphone_local
    no_face_penalty = 1 - num_faces_local
    score = (
        cellphone_local * cellphone_penalty
        + direction_local * direction_local
        + liveness_local * liveness_local
        + lips_local * lips_local
        + identity_local * identity_local
        + num_people_local * num_people_local
        + 0.1 * no_face_penalty
        + noise_local * 0.2
    )
    return min(score, 1)
