# Deployment

Concrete Structural Damage Inspection & Automated Engineering Reporting System.

## What runs where

Two containers. `api` serves inference and PDF generation from the three ONNX
models; `web` serves the dashboard and proxies `/api` to the backend, so the
browser only ever talks to one origin and there is no CORS surface in
production.

Model weights are **mounted, not baked in**. Retrain a model, drop the new
`.onnx` and `_meta.json` into `models/`, restart the `api` service — that is the
whole deployment. Thresholds, class names and preprocessing constants travel in
the meta files, so nothing in the image needs to know they changed.

## First run

    cd D:\nti_project
    docker compose up --build

    Dashboard  http://localhost:5173
    API docs   http://localhost:8000/docs
    Health     http://localhost:8000/health

`/health` reports each model individually. A `degraded` status means the service
started but one or more models did not load — check that its `_meta.json` and
`.onnx` are both present under `models/`.

## Before the first build

Three files must be copied into place; the notebooks produce them but do not
install them:

    09_FastAPI_Backend.py      →  src/backend/main.py
    08_Severity_Assessment.py  →  src/engine/severity.py
    InspectionDashboard.jsx    →  src/frontend/src/InspectionDashboard.jsx

And these must exist from notebooks 04-06:

    models/classification/classifier_meta.json   + the .onnx
    models/detection/detection_meta.json         + the .onnx
    models/segmentation/segmentation_meta.json   + the .onnx

## GPU

The compose file requests one NVIDIA device and the API image is built on the
CUDA 12.1 runtime. That needs the NVIDIA Container Toolkit on the host; on
Windows this means Docker Desktop with WSL2 and a current driver.

To deploy without a GPU, set `use_gpu=False` in this script and re-run it. The
image drops to `python:3.11-slim`, `onnxruntime` replaces `onnxruntime-gpu`, and
the device reservation disappears from the compose file. Inference is roughly
4-8x slower — fine for single uploads, not for batch processing.

## Development without Docker

Two terminals:

    # backend
    cd src
    uvicorn backend.main:app --reload --port 8000

    # frontend
    cd src/frontend
    npm install
    npm run dev

The Vite dev server proxies `/api` to the backend with the same rewrite nginx
uses, so `VITE_API_BASE=/api` is correct in both environments.

## Configuration

Set on the `api` service in `docker-compose.yml`:

| Variable | Default | Purpose |
|---|---|---|
| `MODELS_DIR` | `/data/models` | Where the ONNX models and meta files live |
| `REPORTS_DIR` | `/data/reports` | Generated PDFs |
| `MAX_UPLOAD_MB` | `25` | Rejects larger uploads with a 413 |
| `MAX_IMAGE_DIM` | `4096` | Larger images are downscaled before inference |
| `REPORT_TTL_HOURS` | `24` | Age at which stored PDFs are purged on startup |
| `ALLOWED_ORIGINS` | localhost | CORS allow-list (unused behind the nginx proxy) |
| `ORGANISATION` | `Structural Inspection Unit` | Printed on every report |

## Operational notes worth knowing

**One uvicorn worker, deliberately.** ONNX Runtime sessions are not fork-safe,
and each additional worker loads its own copy of all three models into VRAM. On
a 6 GB card that is the difference between working and an OOM at the second
concurrent request. To scale, run more containers behind a load balancer rather
than more workers in one.

**Reports are transient.** PDFs older than `REPORT_TTL_HOURS` are deleted at
startup. If inspection reports are records you must retain, back the `reports/`
volume with real storage and raise the TTL — the container is not an archive.

**The inspection cache is in-memory and capped at 64 entries.** It exists so the
dashboard can request a PDF without re-uploading the image. Restarting the API
invalidates it; the dashboard falls back to a fresh inspection, which is slower
but correct.

**No authentication is included.** This system produces documents that carry
engineering weight. Put it behind your organisation's SSO or a reverse proxy
with access control before exposing it beyond localhost.

**Every response carries the disclaimer.** It is on the API responses, on the
dashboard, and on every page of every PDF. Keep it there — the system is
decision support, and a report that reads as a structural determination is a
liability rather than a deliverable.
