# Proctor — AI-Powered Online Exam Proctoring System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/mudabs/Proctor/actions/workflows/ci.yml/badge.svg)](https://github.com/mudabs/Proctor/actions/workflows/ci.yml)

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

> **Hardware disclaimer:** Proctoring uses PyTorch, YOLO, MediaPipe, dlib, and
> face recognition. A CUDA-enabled NVIDIA GPU is strongly recommended and may
> be required for reliable real-time performance. CPU-only systems may run very
> slowly or fail to run the full detection pipeline; CPU-only execution is not
> guaranteed.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/mudabs/Proctor.git
cd Proctor
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Create the database

Import the provided SQL dump:

```bash
mysql -u root proctoring < proctoring.sql
```

Or create the database first if it does not exist:

```sql
CREATE DATABASE proctoring;
```

### 4. Download required model files

Place the following files in the `models/` directory (not included in the repo due to size):

| File | Purpose |
|---|---|
| `shape_predictor_68_face_landmarks.dat` | dlib facial landmarks |
| `yolov8n.pt` | YOLOv8 nano — object detection |
| `best_20.pt` | Custom liveness detection model |
| `TrainingImageLabel/Trainner.yml` | LBPH face recogniser (generated after capturing student images) |

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
pip install models/dlib-19.24.1-cp311-cp311-win_amd64.whl
pip install -r requirements.txt
```

> The `dlib` wheel must be installed **before** the rest of the requirements because `face-recognition` depends on it.

### 6. Configure the application

Set the database URI and a secret key through environment variables. Windows
PowerShell:

```powershell
$env:PROCTOR_DATABASE_URI = "mysql://root:your_password@localhost/proctoring"
$env:PROCTOR_SECRET_KEY = "replace-with-a-long-random-secret"
```

macOS/Linux:

```bash
export PROCTOR_DATABASE_URI='mysql://root:your_password@localhost/proctoring'
export PROCTOR_SECRET_KEY='replace-with-a-long-random-secret'
```

The application defaults to `mysql://root:@localhost/proctoring` if
`PROCTOR_DATABASE_URI` is not set.

### 7. Run

```bash
python run.py
```

Open `http://localhost:5000` in a browser. Keep the terminal running while
using the application. To stop the server, press `Ctrl+C`.

## Project Structure

The runtime entrypoint is `run.py`; application features are organized under
the `proctor/` package. `app.py` remains as a compatibility module while the
remaining shared infrastructure is migrated.

```
app.py                  # Flask composition module and compatibility routes
run.py                  # Development/production startup entrypoint
proctor/                # Modular auth, courses, admin, and proctoring code
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
