# Proctor — AI-Powered Online Exam Proctoring System

A Flask web application that monitors students during online exams using real-time computer vision and machine learning to detect cheating behaviour.

## Features

### Proctoring & Detection
- **Head pose estimation** — detects when a student looks left, right, up, or down using MediaPipe Face Mesh
- **Face recognition** — verifies student identity against pre-registered face encodings
- **Liveness detection** — distinguishes real faces from photos/spoofs using a custom YOLO model
- **Object detection** — flags unauthorised items (cell phones, books, multiple people) via YOLOv8
- **Mouth/lip detection** — detects open mouth (possible whispering) using dlib 68-point facial landmarks
- **Noise detection** — monitors ambient audio via microphone with `sounddevice`
- **Cheating score** — aggregates all signals into a real-time cheating probability score

### Exam Management
- Create and manage courses, exams, and quizzes
- Add multiple-choice questions with correct answers and point values
- Set exam date, duration, and proctoring mode
- Timed quiz sessions with automatic submission
- Student result viewing with graphs

### User Management
- Role-based access: Admin, Lecturer, Student
- Student registration with face image capture (used for identity verification)
- User profile editing and image re-capture
- Blacklist/block websites during exams
- Course enrollment and management

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask 3.0 |
| Database | MySQL / MariaDB via SQLAlchemy |
| Computer Vision | OpenCV, MediaPipe, dlib, face_recognition |
| Object Detection | YOLOv8 (Ultralytics) |
| Frontend | Jinja2 templates, Bootstrap 5, jQuery |
| Audio | sounddevice |
| ML / DL | PyTorch, JAX |

## Prerequisites

- Python 3.11
- MySQL / MariaDB running locally
- A webcam
- dlib wheel (included in `models/`) — required because building from source needs CMake and a C++ compiler

## Setup

### 1. Database

Import the provided SQL dump:

```bash
mysql -u root proctoring < proctoring.sql
```

Or create the database first if it does not exist:

```sql
CREATE DATABASE proctoring;
```

### 2. Download required model files

Place the following files in the `models/` directory (not included in the repo due to size):

| File | Purpose |
|---|---|
| `shape_predictor_68_face_landmarks.dat` | dlib facial landmarks |
| `yolov8n.pt` | YOLOv8 nano — object detection |
| `best_20.pt` | Custom liveness detection model |
| `TrainingImageLabel/Trainner.yml` | LBPH face recogniser (generated after capturing student images) |

### 3. Install dependencies

```bash
pip install models/dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install -r requirements.txt
```

> The `dlib` wheel must be installed **before** the rest of the requirements because `face-recognition` depends on it.

### 4. Configure the application

In `app.py`, update the database URI if your MySQL credentials differ:

```python
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/proctoring'
```

### 5. Run

```bash
python app.py
```

The app will be available at `http://localhost:5000`.

## Project Structure

```
app.py                  # Application entry point — routes, models, detection logic
proctoring.sql          # Database schema and seed data
requirements.txt        # Python dependencies
models/                 # ML models and face detection cascades
static/
  css/                  # Bootstrap and custom styles
  js/                   # jQuery, Bootstrap, Highcharts
  images/
    known_images/       # Pre-registered student face images (populated at runtime)
  webcamjs/             # Webcam capture library
templates/              # Jinja2 HTML templates
```

## First-time Use

1. Register an admin account and log in.
2. Assign the **Admin** role via the Roles page.
3. Create courses and assign lecturers.
4. Have students register — they will be prompted to capture face images.
5. Create a quiz with proctoring enabled; students take the quiz while the webcam stream is monitored.

## Notes

- The hosts-file blacklist feature (`C:\Windows\System32\drivers\etc\hosts`) requires the application to be run with administrator privileges on Windows.
- Face images captured during registration are saved to `static/images/known_images/` and are used for identity verification during proctored exams.
