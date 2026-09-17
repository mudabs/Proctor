# Proctor

Flask-based exam management and browser-assisted proctoring.

## Docker quick start on Linux

1. Install Docker Engine and the Compose plugin.
2. Copy `.env.example` to `.env` and replace every placeholder secret.
3. Put the assets listed in [MODEL_INVENTORY.md](MODEL_INVENTORY.md) in `models/`.
4. Build, verify, and start:

```bash
docker compose build
docker compose run --rm app python scripts/verify_models.py
docker compose up -d
```

For an empty database volume, import the schema with the credentials from `.env`:

```bash
docker compose exec -T db mariadb -u root -p"$MARIADB_ROOT_PASSWORD" proctoring < proctoring.sql
```

The app listens on `127.0.0.1:8000` by default. Use an HTTPS reverse proxy in front of it and never publish MariaDB. Readiness is available at `/health`; logs are available with `docker compose logs -f app`.

## GPU deployment

The default image uses CPU-safe PyTorch and falls back to CPU when CUDA is unavailable. For an explicit NVIDIA run, install the NVIDIA Container Toolkit and check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
docker compose run --rm app-gpu python scripts/verify_models.py
docker compose up -d db app-gpu
```

Use CPU if the older GPU cannot run the pinned CUDA/PyTorch stack.

## Development and operations

For local development, use Python 3.11 and a virtual environment, install `requirements.txt`, configure `PROCTOR_DATABASE_URI` and `PROCTOR_SECRET_KEY`, then run `python run.py`. Production uses Gunicorn through Compose and never enables Flask debug mode.

The `app_data` volume stores runtime uploads and logs. Back up it and the database volume. Rotate `PROCTOR_SECRET_KEY`, restrict storage permissions, and define a retention period for biometric data.

The exam page requests camera and microphone access from the student's browser, captures periodic JPEG frames, and sends them to an authenticated session endpoint. The server never opens a webcam or microphone. Browser audio aggregation is currently unavailable. Website blocking is unsupported remotely because a server cannot edit a student's hosts file; it would require a separately installed, consented client agent or extension.

## Tailscale, Ionos, and HTTPS

Install and authenticate Tailscale on both servers and restrict the tailnet with ACLs. Configure the Ionos reverse proxy to forward the public HTTPS hostname to the home server's Tailscale address and port 8000. Keep the Compose bind address private, point DNS to Ionos, terminate TLS at Ionos or Caddy/Nginx, and use an HTTPS public URL so browsers allow camera and microphone access. This first implementation uses HTTP frame uploads, not WebSockets.

## Troubleshooting

- If `/health` is degraded, wait for MariaDB and ensure the URI uses host `db`, not `localhost`.
- If model verification reports missing assets, copy the exact files into the configured model directory.
- If camera permission fails, use HTTPS or localhost and grant browser permission.
- Set `PROCTOR_INFERENCE_DEVICE=cpu` when CUDA is unavailable.
