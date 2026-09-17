# Proctor Deployment and Portability Implementation Guide

## Purpose

Implement the changes required to deploy Proctor on a Linux home server with Docker, MariaDB, optional NVIDIA GPU acceleration, Tailscale connectivity, and HTTPS access through an Ionos server.

The implementation must be performed on a feature branch, never directly on `main`. The final result must be cloneable onto a new Linux server and start through Docker Compose without requiring a Python virtual environment on the host.

## Target architecture

```text
Student browser
  - Exam UI
  - Browser webcam/microphone capture
  - HTTPS/WebSocket or HTTP frame upload
          |
          v
Ionos server
  - Public DNS
  - TLS termination/reverse proxy
  - No GPU required
          |
          v
Tailscale network
          |
          v
Linux home server
  - Docker Compose
  - Flask/Gunicorn application
  - MariaDB
  - CPU or NVIDIA GPU inference
  - Persistent uploaded images, models, and logs
```

The first deployment may run the entire stack on the Linux home server, with the Ionos server acting only as a reverse proxy. Do not require the Ionos server to have a GPU.

## Non-negotiable requirements

1. Do not modify or reset unrelated user work.
2. Do not commit passwords, API keys, Tailscale keys, TLS private keys, or other secrets.
3. Do not commit large model files unless the repository already tracks them intentionally. Provide a documented model-download or model-copy step instead.
4. Preserve the existing application features where practical.
5. Keep CPU execution available when CUDA is unavailable.
6. Use environment variables for all deployment-specific configuration.
7. Do not rely on Windows-only paths, Windows-only packages, or host-specific absolute paths.
8. Do not make remote users depend on a webcam or microphone attached to the Linux server.
9. Do not use Flask's development server in the deployed configuration.
10. Do not use process-global detection state for multiple concurrent exam sessions.

## Current repository issues to address

Inspect the current repository before editing. At minimum, account for these known issues:

- There is no Dockerfile or Docker Compose configuration.
- `requirements.txt` contains a Windows-specific absolute dlib wheel reference.
- `requirements_clean.txt` omits dlib even though `face-recognition` requires it.
- The application defaults to a MySQL database on `localhost`.
- `run.py` starts Flask with `debug=True`.
- Detection code uses relative paths such as `./models/...` and `./session.txt`.
- `detection.py` uses `math` without importing it.
- `detection.py` uses `cv2.VideoCapture(0)`, which captures the server camera rather than a remote student's camera.
- Audio capture uses `sounddevice`, which attempts to use a server-side microphone.
- The code has module/global detection variables such as `identity`, `liveness`, `numFaces`, and `numPeople`.
- The website blacklist edits a Windows hosts file and cannot control a remote student's computer.
- The repository contains `best_10.pt`, `best_20.pt`, and the dlib landmark data, but the application also expects `yolov8n.pt` and `models/TrainingImageLabel/Trainner.yml`.
- The bundled dlib wheel is Windows-only and is not usable in a Linux container.

## Branch and change management

Create or use a branch with a name such as:

```text
codex/deployment-docker-gpu
```

Before editing, verify the active branch and working tree. Do not merge, rebase, reset, or delete `main`. Keep deployment work isolated from unrelated changes.

## Phase 1: Configuration and path cleanup

### 1. Centralize configuration

Extend the configuration module so that it supports at least:

- `PROCTOR_SECRET_KEY`
- `PROCTOR_DATABASE_URI`
- `PROCTOR_MODEL_DIR`
- `PROCTOR_DATA_DIR`
- `PROCTOR_INFERENCE_DEVICE`, with values `auto`, `cpu`, or `cuda`
- `PROCTOR_LOG_LEVEL`
- `PROCTOR_MAX_UPLOAD_SIZE`
- `PROCTOR_ALLOWED_HOSTS` or equivalent host validation setting

Use safe production defaults. Never use the development secret key in production.

### 2. Replace relative paths

Build paths from the application root or configured data directory using `pathlib.Path`. Do not depend on the directory from which Gunicorn happens to be launched.

Create or validate directories at startup for:

- Model files
- Known face images
- Session/event logs
- Uploaded media, if used

Fail with a clear error identifying the missing file when a required model is unavailable.

### 3. Fix dependency declarations

Create a Linux-compatible dependency file. Remove the absolute Windows dlib wheel reference. Choose one supported Linux installation strategy:

- Build dlib in the image using system build tools; or
- Use a compatible Linux wheel from a controlled source.

Do not silently omit dlib. Pin compatible versions of Python, NumPy, OpenCV, MediaPipe, dlib, face-recognition, PyTorch, torchvision, and Ultralytics.

Keep the dependency set as small as practical. Avoid installing both `opencv-python` and `opencv-contrib-python` unless the application demonstrably requires both; LBPH face recognition requires the contrib build.

Install the CPU-compatible PyTorch packages by default. Document the optional NVIDIA/CUDA image or installation path separately if the selected PyTorch version supports the laptop GPU.

## Phase 2: Model and inference handling

### 1. Define required assets

Document the purpose, expected location, and source of every model:

- `shape_predictor_68_face_landmarks.dat`
- `yolov8n.pt`
- `best_20.pt`
- `TrainingImageLabel/Trainner.yml`

Do not invent or fabricate missing model files. Add a setup script or documented command that verifies their presence and reports exactly which files are missing.

### 2. Make device selection explicit

Add a small inference utility that selects:

- CUDA when `PROCTOR_INFERENCE_DEVICE=cuda` and CUDA is available;
- CPU when `PROCTOR_INFERENCE_DEVICE=cpu`;
- CUDA when available, otherwise CPU, when set to `auto`.

Pass the selected device into Ultralytics inference. Ensure the application does not crash merely because CUDA is unavailable.

Log the selected device at startup, including the GPU name when available.

Because the target GPU has approximately 2 GB of VRAM, use the smallest practical models, avoid unnecessary image sizes, and expose inference image size, confidence, and frame-rate settings through configuration.

### 3. Avoid loading models per request

Load models once per inference worker where safe. Do not load YOLO models inside every video request. Add clear startup errors for malformed or missing model files.

## Phase 3: Browser-based capture and session isolation

This is required for real remote proctoring.

### 1. Move webcam capture to the browser

Use browser `navigator.mediaDevices.getUserMedia()` from the exam page. The page must request camera and microphone permissions from the student. Do not use `cv2.VideoCapture(0)` for remote students.

Implement one of these transport approaches:

- Preferred initial implementation: capture periodic JPEG frames in the browser and POST them to an authenticated Flask endpoint.
- More advanced implementation: use a WebSocket connection for frames and results.

The first implementation should prioritize correctness and modest bandwidth over maximum frame rate. Make frame interval, JPEG quality, and maximum dimensions configurable.

### 2. Add authenticated session endpoints

Implement endpoints for:

- Starting a proctoring session
- Uploading a frame
- Returning the latest detection result
- Stopping a session
- Reporting session errors

Ensure that a student can only access their own session. Validate file types, frame sizes, request rates, and session ownership.

### 3. Replace global detection state

Create a session-scoped state object keyed by the authenticated exam/session ID. Detection results must not leak between users. Do not store per-student state in module globals.

Use a concurrency-safe design appropriate for the chosen deployment. If Gunicorn has multiple workers, do not rely on in-process state unless sticky sessions and documented limitations are intentional. Prefer a shared store or a single inference worker with a queue for the initial deployment.

### 4. Handle audio realistically

Either:

- Capture audio levels in the browser and send aggregate measurements; or
- Explicitly defer audio detection and disable it cleanly when no browser audio pipeline is implemented.

Do not attempt to access the Linux server microphone for remote students.

### 5. Website blocking

Do not claim that the server can edit a student's hosts file. Replace the existing feature with one of the following documented options:

- Remove it from the remote deployment and label it unsupported.
- Implement a separately installed client agent/browser extension, with explicit user consent and security documentation.
- Restrict it to a local demonstration mode.

## Phase 4: Docker implementation

Add:

- `Dockerfile`
- `.dockerignore`
- `docker-compose.yml`
- `.env.example`
- A model verification/setup script
- A production WSGI entrypoint if needed

### Dockerfile requirements

Use a Python 3.11 Linux base compatible with the pinned dependencies. Install only required OS packages, including libraries needed by OpenCV, dlib, MediaPipe, audio support if retained, and MariaDB client connectivity.

Run the application as a non-root user where practical. Set a working directory. Copy dependency files separately from application source to improve build caching. Do not copy local virtual environments, `.git`, secrets, or unnecessary large files.

The container must listen on `0.0.0.0`, not only `127.0.0.1`.

### Compose requirements

Define at least:

- `app` service
- `db` service using MariaDB or MySQL

Configure:

- Database health checks
- App startup after database readiness
- Database persistent volume
- Application data/model volume
- Published application port bound appropriately
- Environment variables from `.env`
- Restart policy suitable for the home server

Do not publish the database port publicly unless required for administration.

### GPU profile

Provide a documented optional GPU launch path, for example a Compose profile or override file. It must use the NVIDIA container runtime only when explicitly requested. The normal CPU path must remain functional.

Document these host checks:

```bash
nvidia-smi
docker run --rm --gpus all <cuda-test-image> nvidia-smi
```

If the old GPU is unsupported by the selected PyTorch/CUDA build, document the CPU fallback instead of forcing an incompatible CUDA configuration.

## Phase 5: Database and production startup

Use `PROCTOR_DATABASE_URI` to connect the Flask container to the Compose database service. Do not use `localhost` for the database from inside the app container.

Document how to:

1. Start a clean database.
2. Import `proctoring.sql`.
3. Create or rotate the application secret.
4. Back up the database.
5. Preserve uploaded face images.

Replace the development startup path with Gunicorn or another production WSGI server. Keep `run.py` usable for local development, but ensure production Compose does not enable Flask debug mode.

Add a lightweight `/health` endpoint that checks application readiness and, where appropriate, database connectivity without exposing secrets or personal data.

## Phase 6: Ionos, Tailscale, and HTTPS documentation

Add deployment documentation covering:

1. Installing Docker and Docker Compose on the Linux laptop.
2. Installing and authenticating Tailscale on both servers.
3. Restricting Tailscale access with ACLs where possible.
4. Configuring the Ionos reverse proxy to forward to the laptop's Tailscale address and application port.
5. Configuring DNS for the public hostname.
6. Terminating HTTPS at the Ionos proxy or Caddy/Nginx.
7. Forwarding WebSocket traffic if the WebSocket transport is used.
8. Ensuring the public URL is HTTPS so browsers permit camera and microphone access.
9. Keeping MariaDB private.
10. Monitoring logs and restarting failed containers.

Do not expose the Docker daemon, MariaDB, or an unauthenticated inference endpoint to the public internet.

## Phase 7: Testing and acceptance criteria

Add or update automated tests for:

- Configuration loading
- CPU device selection
- CUDA-unavailable fallback
- Missing-model diagnostics
- Health endpoint
- Database connectivity
- Session ownership and authorization
- Frame upload validation
- Independent state for two simultaneous sessions

Perform these manual checks:

1. Build from a clean checkout on Linux.
2. Start the CPU Compose configuration.
3. Import the database schema.
4. Register and authenticate a user.
5. Capture a face image from a browser.
6. Start an exam from a different browser or machine.
7. Confirm that the student's browser camera is used, not the server webcam.
8. Confirm that detection results belong to the correct student session.
9. Confirm that the app remains usable when CUDA is unavailable.
10. Test the GPU profile if the old laptop supports it.
11. Stop and restart the stack and verify that database data and uploaded images persist.
12. Access the application through the public HTTPS hostname over Tailscale/Ionos.
13. Verify that no secret, database port, debug traceback, or internal Tailscale address is exposed publicly.

The work is complete only when a fresh Linux server can follow the README and `.env.example` instructions to start the application without installing the project's Python dependencies directly on the host.

## Documentation deliverables

Update or add:

- `README.md` with quick start, local development, CPU deployment, GPU deployment, model setup, and troubleshooting.
- `.env.example` with safe placeholder values only.
- `DEPLOYMENT_IMPLEMENTATION.md` with the implementation plan and final operational notes.
- A model inventory document or section describing licensing, source, expected checksum, and storage location.
- A security/privacy section explaining webcam permissions, face data storage, session data, retention, and public exposure.

## Final git handoff

Before handing off:

1. Run tests and the documented build commands.
2. Check `git diff` and `git status`.
3. Confirm no secrets or unintended model binaries are staged.
4. Commit only the deployment-related work to the feature branch.
5. Report the branch name, commit hash, files changed, tests run, and any remaining limitations.
6. Leave `main` unchanged.

