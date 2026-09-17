"""
=============================================================================
10_FRONTEND_AND_DEPLOYMENT.py
Concrete Structural Damage Inspection System — Phase 4, Stage 3
=============================================================================
Writes the full deployment surface into D:\nti_project and verifies the tree is
coherent before you build anything.

Running this produces:
  src/backend/Dockerfile          GPU-capable API image (falls back to CPU)
  src/frontend/Dockerfile         Vite build → nginx static serve
  src/frontend/nginx.conf         SPA routing + API proxy
  docker-compose.yml              both services, models mounted read-only
  .dockerignore                   keeps the 55k-image dataset out of the context
  src/frontend/package.json       Vite + React + Tailwind
  src/frontend/vite.config.js
  src/frontend/tailwind.config.js
  src/frontend/postcss.config.js
  src/frontend/index.html
  src/frontend/src/main.jsx
  src/frontend/src/index.css
  DEPLOYMENT.md                   run instructions and the pre-flight checklist

InspectionDashboard.jsx is delivered separately — drop it into
src/frontend/src/ before building.

Run:  python 10_Frontend_and_Deployment.py
=============================================================================
"""

# %%
# =============================================================================
# CELL 1 — CONFIG & TREE VERIFICATION
# =============================================================================
from __future__ import annotations

import json
import shutil
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DeployCfg:
    project_root: Path = Path(r"D:\nti_project")
    api_port: int = 8000
    web_port: int = 5173
    organisation: str = "Structural Inspection Unit"
    use_gpu: bool = True


DCFG = DeployCfg()
ROOT = DCFG.project_root
BACKEND = ROOT / "src" / "backend"
ENGINE = ROOT / "src" / "engine"
FRONTEND = ROOT / "src" / "frontend"

for d in (BACKEND, ENGINE, FRONTEND / "src", ROOT / "reports"):
    d.mkdir(parents=True, exist_ok=True)


def write(path: Path, content: str, label: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}  {label}")


print("=" * 78)
print("PRE-FLIGHT — what the deployment expects to find")
print("=" * 78)

required = {
    "classification ONNX": ROOT / "models" / "classification",
    "detection ONNX": ROOT / "models" / "detection",
    "segmentation ONNX": ROOT / "models" / "segmentation",
}
metas = {
    "classifier_meta.json": ROOT / "models" / "classification" / "classifier_meta.json",
    "detection_meta.json": ROOT / "models" / "detection" / "detection_meta.json",
    "segmentation_meta.json": ROOT / "models" / "segmentation" / "segmentation_meta.json",
}
sources = {
    "src/backend/main.py": BACKEND / "main.py",
    "src/engine/severity.py": ENGINE / "severity.py",
    "src/frontend/src/InspectionDashboard.jsx": FRONTEND / "src" / "InspectionDashboard.jsx",
}

missing = []
for label, p in {**required, **metas, **sources}.items():
    ok = p.exists()
    onnx_note = ""
    if p.is_dir():
        n = len(list(p.glob("*.onnx")))
        ok = n > 0
        onnx_note = f" ({n} .onnx)"
    print(f"  [{'✓' if ok else '✗'}] {label}{onnx_note}")
    if not ok:
        missing.append(label)

if missing:
    print(f"\n  {len(missing)} item(s) missing. The files below will still be "
          f"written, but `docker compose up` will fail until you:")
    print("    · copy 09_FastAPI_Backend.py  → src/backend/main.py")
    print("    · copy 08_Severity_Assessment.py → src/engine/severity.py")
    print("    · copy InspectionDashboard.jsx → src/frontend/src/")
    print("    · run notebooks 04-06 to produce the ONNX models")
else:
    print("\n  All prerequisites present.")


# %%
# =============================================================================
# CELL 2 — BACKEND DOCKERFILE
# =============================================================================
# Base image choice: the CUDA runtime image is ~2 GB larger but lets
# onnxruntime-gpu actually use the card. Set DCFG.use_gpu=False for a slim CPU
# image — inference is roughly 4-8x slower but perfectly usable for single
# uploads, and it deploys anywhere.
# =============================================================================
if DCFG.use_gpu:
    base_stage = (
        "FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04\n"
        "\n"
        "ENV DEBIAN_FRONTEND=noninteractive \\\n"
        "    PYTHONUNBUFFERED=1 \\\n"
        "    PYTHONDONTWRITEBYTECODE=1\n"
        "\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "        python3.11 python3-pip python3.11-dev \\\n"
        "        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \\\n"
        "    && ln -sf /usr/bin/python3.11 /usr/bin/python \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
    )
    ort_pkg = "onnxruntime-gpu==1.19.2"
else:
    base_stage = (
        "FROM python:3.11-slim-bookworm\n"
        "\n"
        "ENV PYTHONUNBUFFERED=1 \\\n"
        "    PYTHONDONTWRITEBYTECODE=1\n"
        "\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
    )
    ort_pkg = "onnxruntime==1.19.2"

requirements = f"""\
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
pydantic==2.10.4
{ort_pkg}
numpy==1.26.4
opencv-python-headless==4.10.0.84
pillow==11.0.0
reportlab==4.2.5
"""
write(BACKEND / "requirements.txt", requirements, "(pinned — reproducible builds)")

backend_dockerfile = f"""\
# Concrete Structural Damage Inspection — API
# Built for the zero-drift contract: model weights are MOUNTED, never baked in,
# so retraining does not require an image rebuild.
{base_stage}
WORKDIR /app

# Dependencies first — this layer caches across source changes.
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \\
    && pip install --no-cache-dir -r /app/requirements.txt

# Application source. engine/ must sit beside backend/ so `from engine.severity
# import ...` resolves.
COPY engine/ /app/engine/
COPY backend/ /app/backend/

# Models and reports arrive as volumes at runtime (see docker-compose.yml).
ENV PROJECT_ROOT=/data \\
    MODELS_DIR=/data/models \\
    REPORTS_DIR=/data/reports \\
    PYTHONPATH=/app

RUN useradd -m -u 1000 inspector && mkdir -p /data/reports \\
    && chown -R inspector:inspector /app /data
USER inspector

EXPOSE {DCFG.api_port}

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \\
    CMD curl -fsS http://localhost:{DCFG.api_port}/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "{DCFG.api_port}", \\
     "--workers", "1", "--timeout-keep-alive", "65"]
"""
write(ROOT / "src" / "Dockerfile.backend", backend_dockerfile,
      "(single worker — ONNX sessions are not fork-safe)")


# %%
# =============================================================================
# CELL 3 — FRONTEND DOCKERFILE + NGINX
# =============================================================================
frontend_dockerfile = f"""\
# Concrete Structural Damage Inspection — Dashboard
# Stage 1: build the Vite bundle. Stage 2: serve it from nginx, which also
# proxies /api to the backend so the browser never needs a second origin.

FROM node:20-alpine AS build
WORKDIR /build

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend/ ./
# Built as a relative path: nginx proxies /api, so no host name is compiled in
# and the same image works in dev, staging and production.
ENV VITE_API_BASE=/api
RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /build/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
    CMD wget -qO- http://localhost/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
"""
write(ROOT / "src" / "Dockerfile.frontend", frontend_dockerfile)

nginx_conf = f"""\
server {{
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    # Structural photographs are large; the API guards its own limit at 25 MB.
    client_max_body_size 32M;

    gzip on;
    gzip_types text/css application/javascript application/json image/svg+xml;
    gzip_min_length 1024;

    # Hashed assets are immutable; index.html must never be cached or users get
    # a stale bundle pointing at deleted chunks after a deploy.
    location /assets/ {{
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}
    location = /index.html {{
        add_header Cache-Control "no-store";
    }}

    # SPA fallback.
    location / {{
        try_files $uri $uri/ /index.html;
    }}

    # API proxy. Inference on a large image can take a while on CPU, so the
    # read timeout is generous — a 504 mid-inspection is worse than a slow one.
    location /api/ {{
        proxy_pass http://api:{DCFG.api_port}/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 15s;
        proxy_send_timeout 180s;
        proxy_read_timeout 180s;
        proxy_buffering off;
    }}
}}
"""
write(FRONTEND / "nginx.conf", nginx_conf)


# %%
# =============================================================================
# CELL 4 — DOCKER COMPOSE
# =============================================================================
gpu_block = """
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
""" if DCFG.use_gpu else ""

compose = f"""\
# Concrete Structural Damage Inspection System
#
#   docker compose up --build
#   dashboard → http://localhost:{DCFG.web_port}
#   API docs  → http://localhost:{DCFG.api_port}/docs
#
# Models are mounted read-only from ./models. Retraining a model and restarting
# the api service is enough to deploy it — no rebuild, because the *_meta.json
# files carry the thresholds and preprocessing constants.

services:
  api:
    build:
      context: ./src
      dockerfile: Dockerfile.backend
    image: concrete-inspection-api:1.0.0
    container_name: inspection-api
    restart: unless-stopped
    ports:
      - "{DCFG.api_port}:{DCFG.api_port}"
    volumes:
      - ./models:/data/models:ro
      - ./reports:/data/reports
    environment:
      PROJECT_ROOT: /data
      MODELS_DIR: /data/models
      REPORTS_DIR: /data/reports
      ALLOWED_ORIGINS: "http://localhost:{DCFG.web_port},http://localhost"
      ORGANISATION: "{DCFG.organisation}"
      MAX_UPLOAD_MB: "25"
      REPORT_TTL_HOURS: "24"
{gpu_block}
  web:
    build:
      context: ./src
      dockerfile: Dockerfile.frontend
    image: concrete-inspection-web:1.0.0
    container_name: inspection-web
    restart: unless-stopped
    ports:
      - "{DCFG.web_port}:80"
    depends_on:
      api:
        condition: service_healthy
"""
write(ROOT / "docker-compose.yml", compose)

dockerignore = """\
# The raw dataset is 55,348 images — never send it to the build daemon.
data/
data_processed/
notebooks/
audit_results/
reports/
models/
**/__pycache__/
**/*.pyc
**/.ipynb_checkpoints/
**/node_modules/
**/dist/
.git/
.venv/
*.pt
*.pth
*.onnx
"""
write(ROOT / ".dockerignore", dockerignore, "(keeps the build context small)")


# %%
# =============================================================================
# CELL 5 — FRONTEND SCAFFOLD
# =============================================================================
package_json = {
    "name": "concrete-inspection-dashboard",
    "private": True,
    "version": "1.0.0",
    "type": "module",
    "scripts": {
        "dev": f"vite --port {DCFG.web_port}",
        "build": "vite build",
        "preview": "vite preview",
    },
    "dependencies": {
        "react": "^18.3.1",
        "react-dom": "^18.3.1",
    },
    "devDependencies": {
        "@vitejs/plugin-react": "^4.3.4",
        "vite": "^6.0.7",
        "tailwindcss": "^3.4.17",
        "postcss": "^8.4.49",
        "autoprefixer": "^10.4.20",
    },
}
write(FRONTEND / "package.json", json.dumps(package_json, indent=2))

write(FRONTEND / "vite.config.js", f"""\
import {{ defineConfig }} from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({{
  plugins: [react()],
  server: {{
    port: {DCFG.web_port},
    // Dev-mode proxy mirrors the nginx rule, so VITE_API_BASE=/api works in
    // both environments and there is one less difference between them.
    proxy: {{
      "/api": {{
        target: "http://localhost:{DCFG.api_port}",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\\/api/, ""),
      }},
    }},
  }},
  build: {{ outDir: "dist", sourcemap: false }},
}});
""")

write(FRONTEND / "tailwind.config.js", """\
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};
""")

write(FRONTEND / "postcss.config.js", """\
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
};
""")

write(FRONTEND / "index.html", """\
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Concrete damage inspection</title>
    <meta name="description"
          content="Automated structural damage assessment for concrete infrastructure." />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
""")

write(FRONTEND / "src" / "main.jsx", """\
import React from "react";
import ReactDOM from "react-dom/client";
import InspectionDashboard from "./InspectionDashboard.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <InspectionDashboard />
  </React.StrictMode>
);
""")

write(FRONTEND / "src" / "index.css", """\
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  html {
    -webkit-font-smoothing: antialiased;
  }
  /* Measurements are read in columns; tabular figures keep them aligned. */
  .tabular-nums {
    font-variant-numeric: tabular-nums;
  }
  /* Keyboard focus must stay visible — inspectors work gloved, on tablets,
     and sometimes entirely by keyboard on a laptop in a vehicle. */
  :focus-visible {
    outline: 2px solid #1c1917;
    outline-offset: 2px;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
""")


# %%
# =============================================================================
# CELL 6 — DEPLOYMENT.md
# =============================================================================
deployment_md = f"""\
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

    cd {ROOT}
    docker compose up --build

    Dashboard  http://localhost:{DCFG.web_port}
    API docs   http://localhost:{DCFG.api_port}/docs
    Health     http://localhost:{DCFG.api_port}/health

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
    uvicorn backend.main:app --reload --port {DCFG.api_port}

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
| `ORGANISATION` | `{DCFG.organisation}` | Printed on every report |

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
"""
write(ROOT / "DEPLOYMENT.md", deployment_md)


# %%
# =============================================================================
# CELL 7 — FINAL TREE SUMMARY
# =============================================================================
print("\n" + "=" * 78)
print("DEPLOYMENT SURFACE WRITTEN")
print("=" * 78)

expected = [
    ROOT / "docker-compose.yml",
    ROOT / ".dockerignore",
    ROOT / "DEPLOYMENT.md",
    ROOT / "src" / "Dockerfile.backend",
    ROOT / "src" / "Dockerfile.frontend",
    BACKEND / "requirements.txt",
    FRONTEND / "nginx.conf",
    FRONTEND / "package.json",
    FRONTEND / "vite.config.js",
    FRONTEND / "tailwind.config.js",
    FRONTEND / "postcss.config.js",
    FRONTEND / "index.html",
    FRONTEND / "src" / "main.jsx",
    FRONTEND / "src" / "index.css",
]
for p in expected:
    print(f"  [{'✓' if p.exists() else '✗'}] {p.relative_to(ROOT)}")

print(f"\nGPU deployment: {'enabled' if DCFG.use_gpu else 'disabled (CPU image)'}")
print(f"\nNext:\n  cd {ROOT}\n  docker compose up --build")
print(f"  dashboard → http://localhost:{DCFG.web_port}")
print("\nRead DEPLOYMENT.md first — three source files still need copying into "
      "place, and it lists exactly which.")
print("\n✅ Notebook 10 complete. Phase 4 finished.")