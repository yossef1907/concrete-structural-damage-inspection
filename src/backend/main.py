"""
=============================================================================
09_FASTAPI_BACKEND.py   →   deploy as  src/backend/main.py
Concrete Structural Damage Inspection System — Phase 4, Stage 1
=============================================================================
Unified inference API. Loads all three ONNX models once at startup using their
*_meta.json contracts, so preprocessing, thresholds and class names come from
the files the training notebooks wrote — never from constants retyped here.
That is the whole point of the zero-drift contract: if a model is retrained,
this service picks up the new thresholds without a code change.

Endpoints
  GET  /health              liveness + which models are loaded
  GET  /models              the full model contracts (versions, thresholds)
  POST /predict/classify    patch-level damage presence
  POST /predict/detect      scene-level bounding boxes
  POST /predict/segment     pixel mask (+ PNG overlay)
  POST /inspect/full        detect + segment + severity, one pass
  POST /reports/pdf         engineering inspection PDF

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
      (from src/backend, with severity.py importable — see IMPORTS below)

Prereqs: pip install fastapi uvicorn[standard] python-multipart onnxruntime-gpu
                     opencv-python pillow reportlab
=============================================================================
"""

# %%
# =============================================================================
# CELL 1 — IMPORTS & APPLICATION CONFIG
# =============================================================================
from __future__ import annotations

import io
import os
import json
import time
import uuid
import base64
import logging
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Any

import numpy as np
import cv2
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field

# The severity engine from Notebook 08. In production both files live in
# src/engine/, and this import becomes `from engine.severity import ...`.
try:
    from engine.severity import (
        SeverityEngine, ScaleReference, render_overlay, assessment_to_json)
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
    from severity import (  # type: ignore
        SeverityEngine, ScaleReference, render_overlay, assessment_to_json)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("inspection-api")


class Settings:
    """Environment-overridable so the Docker image needs no rebuild to repoint."""
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", r"D:\nti_project"))
    MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))
    REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "reports")))

    CLS_META = MODELS_DIR / "classification" / "classifier_meta.json"
    DET_META = MODELS_DIR / "detection" / "detection_meta.json"
    SEG_META = MODELS_DIR / "segmentation" / "segmentation_meta.json"

    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
    MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "4096"))
    REPORT_TTL_HOURS = int(os.getenv("REPORT_TTL_HOURS", "24"))
    ALLOWED_ORIGINS = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    ORGANISATION = os.getenv("ORGANISATION", "Structural Inspection Unit")

    API_VERSION = "1.0.0"


SETTINGS = Settings()
SETTINGS.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DISCLAIMER = (
    "This report is produced by an automated decision-support system. It is not "
    "a structural engineering determination and does not certify the condition, "
    "safety, or load capacity of any structure. All findings require verification "
    "by a certified structural engineer before any maintenance, repair, access, "
    "or occupancy decision is made.")


# %%
# =============================================================================
# CELL 2 — MODEL REGISTRY (loaded once, at startup)
# =============================================================================
class ModelRegistry:
    """Owns the three ONNX sessions and the contracts that drive them."""

    def __init__(self):
        self.cls: Optional[dict] = None
        self.det: Optional[dict] = None
        self.seg: Optional[dict] = None
        self.engine = SeverityEngine()
        self.providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                          if ort.get_device() == "GPU" else ["CPUExecutionProvider"])
        self.loaded_at: Optional[str] = None

    @staticmethod
    def _session(path: str, providers) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        return ort.InferenceSession(path, sess_options=opts, providers=providers)

    def _load_one(self, meta_path: Path, kind: str) -> Optional[dict]:
        if not meta_path.exists():
            log.warning("%s contract missing at %s — endpoint will 503", kind, meta_path)
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        onnx_path = meta.get("onnx_path")
        if not onnx_path or not Path(onnx_path).exists():
            # Fall back to a sibling .onnx if the absolute path moved (Docker).
            sibling = next(iter(meta_path.parent.glob("*.onnx")), None)
            if sibling is None:
                log.warning("%s ONNX graph not found — endpoint will 503", kind)
                return None
            onnx_path = str(sibling)
            log.info("%s: meta path stale, using %s", kind, sibling.name)

        sess = self._session(onnx_path, self.providers)
        log.info("%s loaded: %s (%s)", kind, Path(onnx_path).name,
                 sess.get_providers()[0])
        return {"meta": meta, "session": sess,
                "input_name": sess.get_inputs()[0].name,
                "onnx_path": onnx_path}

    def load_all(self):
        t0 = time.time()
        self.cls = self._load_one(SETTINGS.CLS_META, "classification")
        self.det = self._load_one(SETTINGS.DET_META, "detection")
        self.seg = self._load_one(SETTINGS.SEG_META, "segmentation")
        self.loaded_at = datetime.now(timezone.utc).isoformat()
        log.info("model registry ready in %.1fs", time.time() - t0)

    def require(self, kind: str) -> dict:
        m = getattr(self, kind)
        if m is None:
            raise HTTPException(
                status_code=503,
                detail=f"The {kind} model is not loaded. Check that its "
                       f"_meta.json and .onnx file are present in MODELS_DIR.")
        return m

    @property
    def status(self) -> dict:
        return {
            "classification": self.cls is not None,
            "detection": self.det is not None,
            "segmentation": self.seg is not None,
        }


REGISTRY = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting inspection API v%s", SETTINGS.API_VERSION)
    REGISTRY.load_all()
    _purge_expired_reports()
    yield
    log.info("shutting down")


app = FastAPI(
    title="Concrete Structural Damage Inspection API",
    description=("Automated damage classification, detection, segmentation and "
                 "severity assessment for concrete infrastructure. " + DISCLAIMER),
    version=SETTINGS.API_VERSION,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=SETTINGS.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# %%
# =============================================================================
# CELL 3 — IMAGE INTAKE & SHARED PREPROCESSING
# =============================================================================
async def read_image(file: UploadFile) -> tuple[np.ndarray, dict]:
    """Decode an upload to RGB uint8 with size and type guards."""
    raw = await file.read()
    size_mb = len(raw) / 1024 / 1024
    if size_mb > SETTINGS.MAX_UPLOAD_MB:
        raise HTTPException(413, f"Image is {size_mb:.1f} MB; the limit is "
                                 f"{SETTINGS.MAX_UPLOAD_MB} MB.")
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")

    arr = np.frombuffer(raw, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(
            400, "Could not decode that file as an image. Supported formats: "
                 "JPEG, PNG, BMP, TIFF, WebP.")

    h, w = bgr.shape[:2]
    if max(h, w) > SETTINGS.MAX_IMAGE_DIM:
        scale = SETTINGS.MAX_IMAGE_DIM / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        log.info("downscaled %dx%d → %dx%d", w, h, bgr.shape[1], bgr.shape[0])

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    info = {"filename": file.filename or "upload",
            "original_size": {"width": w, "height": h},
            "processed_size": {"width": rgb.shape[1], "height": rgb.shape[0]},
            "size_mb": round(size_mb, 3)}
    return rgb, info


def encode_png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "Failed to encode the result image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# %%
# =============================================================================
# CELL 4 — CLASSIFICATION INFERENCE
# =============================================================================
def run_classification(rgb: np.ndarray, threshold: Optional[float] = None) -> dict:
    m = REGISTRY.require("cls")
    meta, sess = m["meta"], m["session"]
    size = int(meta["img_size"])
    mean = np.array(meta["normalization"]["mean"], np.float32)
    std = np.array(meta["normalization"]["std"], np.float32)
    thr = float(threshold if threshold is not None else meta["threshold_f1_optimal"])
    names = meta["class_names"]

    # Mirrors Notebook 04's eval transform exactly: resize short side to
    # 1.14*size, then centre-crop. Any deviation here silently shifts the
    # operating point away from the validated one.
    resize_to = int(round(size * 1.14))
    h, w = rgb.shape[:2]
    scale = resize_to / min(h, w)
    img = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    img = img[y0:y0 + size, x0:x0 + size]

    x = ((img.astype(np.float32) / 255.0) - mean) / std
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    t0 = time.time()
    logits = sess.run(None, {m["input_name"]: x})[0]
    prob = float(_softmax(logits.astype(np.float32))[0, 1])

    return {
        "damage_probability": round(prob, 4),
        "is_damaged": bool(prob >= thr),
        "predicted_class": names[int(prob >= thr)],
        "threshold": thr,
        "threshold_options": {
            "f1_optimal": meta.get("threshold_f1_optimal"),
            "high_recall": meta.get("threshold_high_recall"),
        },
        "model": {"architecture": meta.get("arch"), "input_size": size},
        "inference_ms": round((time.time() - t0) * 1000, 2),
    }


# %%
# =============================================================================
# CELL 5 — DETECTION INFERENCE (letterbox + class-wise NMS, ported from NB05)
# =============================================================================
def _letterbox(img: np.ndarray, size: int, pad: int = 114):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad, np.uint8)
    dw, dh = (size - nw) // 2, (size - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, dw, dh


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def run_detection(rgb: np.ndarray, conf: Optional[float] = None,
                  iou: Optional[float] = None) -> dict:
    m = REGISTRY.require("det")
    meta, sess = m["meta"], m["session"]
    size = int(meta["input"]["shape"][2])
    pad = int(meta["input"].get("letterbox_pad_value", 114))
    classes = {int(k): v for k, v in meta["classes"].items()}
    thr = float(conf if conf is not None else meta["thresholds"]["conf_default"])
    iou_thr = float(iou if iou is not None else meta["thresholds"]["iou_nms"])
    max_det = int(meta["thresholds"].get("max_det", 300))

    H, W = rgb.shape[:2]
    lb, r, dw, dh = _letterbox(rgb, size, pad)
    x = np.transpose(lb.astype(np.float32) / 255.0, (2, 0, 1))[None]

    t0 = time.time()
    out = sess.run(None, {m["input_name"]: x})[0]
    pred = np.squeeze(out, 0).T                      # (anchors, 4+nc)

    scores_all = pred[:, 4:]
    cls_ids = scores_all.argmax(1)
    cls_scores = scores_all.max(1)
    keep_mask = cls_scores >= thr
    pred, cls_ids, cls_scores = pred[keep_mask], cls_ids[keep_mask], cls_scores[keep_mask]

    detections = []
    if len(pred):
        xc, yc, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        xyxy = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)
        xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - dw) / r).clip(0, W)
        xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - dh) / r).clip(0, H)

        kept = []
        for c in np.unique(cls_ids):
            idx = np.where(cls_ids == c)[0]
            kept.extend(idx[_nms(xyxy[idx], cls_scores[idx], iou_thr)])
        kept = sorted(kept, key=lambda i: -cls_scores[i])[:max_det]

        frame = float(W * H)
        for i in kept:
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            detections.append({
                "class_id": int(cls_ids[i]),
                "class_name": classes.get(int(cls_ids[i]), str(int(cls_ids[i]))),
                "confidence": round(float(cls_scores[i]), 4),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_norm": [round(x1 / W, 6), round(y1 / H, 6),
                              round(x2 / W, 6), round(y2 / H, 6)],
                "area_px": round(area, 2),
                "area_ratio": round(area / frame, 6),
            })

    by_class: dict[str, int] = {}
    for d in detections:
        by_class[d["class_name"]] = by_class.get(d["class_name"], 0) + 1

    return {
        "image_size": {"width": int(W), "height": int(H)},
        "num_detections": len(detections),
        "by_class": by_class,
        "detections": detections,
        "thresholds": {"conf": thr, "iou": iou_thr, "max_det": max_det},
        "inference_ms": round((time.time() - t0) * 1000, 2),
    }


# %%
# =============================================================================
# CELL 6 — SEGMENTATION INFERENCE
# =============================================================================
def _remove_small_blobs(mask_u8: np.ndarray, min_px: int) -> np.ndarray:
    if min_px <= 0:
        return mask_u8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    out = np.zeros_like(mask_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out[labels == i] = 255
    return out


def run_segmentation(rgb: np.ndarray, threshold: Optional[float] = None,
                     min_blob_px: Optional[int] = None) -> tuple[np.ndarray, dict]:
    m = REGISTRY.require("seg")
    meta, sess = m["meta"], m["session"]
    size = int(meta["input"]["shape"][2])
    mean = np.array(meta["input"]["mean"], np.float32)
    std = np.array(meta["input"]["std"], np.float32)
    thr = float(threshold if threshold is not None
                else meta["thresholds"]["probability"])
    minblob = int(min_blob_px if min_blob_px is not None
                  else meta["thresholds"]["min_blob_px"])

    H, W = rgb.shape[:2]
    small = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = ((small.astype(np.float32) / 255.0) - mean) / std
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    t0 = time.time()
    logits = sess.run(None, {m["input_name"]: x})[0][0, 0]
    prob = 1.0 / (1.0 + np.exp(-logits))
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    mask = _remove_small_blobs((prob >= thr).astype(np.uint8) * 255, minblob)

    n_regions, _, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    damaged = int((mask > 0).sum())
    total = int(mask.size)

    summary = {
        "image_size": {"width": int(W), "height": int(H)},
        "damaged_pixels": damaged,
        "total_pixels": total,
        "coverage_ratio": round(damaged / max(total, 1), 6),
        "coverage_percent": round(100.0 * damaged / max(total, 1), 4),
        "num_regions": int(max(n_regions - 1, 0)),
        "threshold": thr,
        "min_blob_px": minblob,
        "inference_ms": round((time.time() - t0) * 1000, 2),
    }
    return mask, summary


# %%
# =============================================================================
# CELL 7 — RESPONSE SCHEMAS
# =============================================================================
class HealthResponse(BaseModel):
    status: str
    api_version: str
    models_loaded: dict
    execution_provider: str
    models_loaded_at: Optional[str] = None
    uptime_seconds: float


class ScaleInput(BaseModel):
    mm_per_pixel: Optional[float] = Field(None, gt=0)
    reference_length_mm: Optional[float] = Field(None, gt=0)
    reference_length_px: Optional[float] = Field(None, gt=0)
    camera_distance_mm: Optional[float] = Field(None, gt=0)
    focal_length_px: Optional[float] = Field(None, gt=0)


_START_TIME = time.time()


# %%
# =============================================================================
# CELL 8 — ENDPOINTS
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Liveness probe. Reports which models actually loaded, so a partially
    provisioned container is visible instead of failing on first request."""
    ready = all(REGISTRY.status.values())
    return HealthResponse(
        status="ready" if ready else "degraded",
        api_version=SETTINGS.API_VERSION,
        models_loaded=REGISTRY.status,
        execution_provider=REGISTRY.providers[0],
        models_loaded_at=REGISTRY.loaded_at,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@app.get("/models", tags=["system"])
async def models():
    """The three model contracts, including thresholds and validated metrics —
    what the dashboard shows on its model-information panel."""
    def strip(m):
        if m is None:
            return None
        meta = dict(m["meta"])
        meta.pop("training", None)      # verbose, not useful over the wire
        return meta
    return {"classification": strip(REGISTRY.cls),
            "detection": strip(REGISTRY.det),
            "segmentation": strip(REGISTRY.seg)}


@app.post("/predict/classify", tags=["inference"])
async def predict_classify(
    file: UploadFile = File(..., description="Concrete surface image"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    high_recall: bool = Query(False, description="Use the recall-biased threshold"),
):
    """Patch-level damage presence.

    Note this model was trained on close-up patch imagery. On a wide scene shot
    it answers a different question than /predict/detect and the two should not
    be treated as cross-checks of each other.
    """
    rgb, info = await read_image(file)
    if high_recall and threshold is None:
        meta = REGISTRY.require("cls")["meta"]
        threshold = meta.get("threshold_high_recall", meta["threshold_f1_optimal"])
    result = run_classification(rgb, threshold)
    return JSONResponse({"image": info, "classification": result,
                         "disclaimer": DISCLAIMER})


@app.post("/predict/detect", tags=["inference"])
async def predict_detect(
    file: UploadFile = File(...),
    conf: Optional[float] = Query(None, ge=0.0, le=1.0),
    iou: Optional[float] = Query(None, ge=0.0, le=1.0),
    return_overlay: bool = Query(False),
):
    """Scene-level damage localisation. Coordinates are original-image pixels."""
    rgb, info = await read_image(file)
    result = run_detection(rgb, conf, iou)
    payload: dict[str, Any] = {"image": info, "detection": result,
                               "disclaimer": DISCLAIMER}
    if return_overlay:
        payload["overlay_png_base64"] = encode_png_b64(
            render_overlay(rgb, None, result["detections"], None))
    return JSONResponse(payload)


@app.post("/predict/segment", tags=["inference"])
async def predict_segment(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    min_blob_px: Optional[int] = Query(None, ge=0),
    return_mask: bool = Query(True, description="Return the binary mask as PNG"),
    return_overlay: bool = Query(False),
):
    """Pixel-level damage geometry."""
    rgb, info = await read_image(file)
    mask, summary = run_segmentation(rgb, threshold, min_blob_px)

    payload: dict[str, Any] = {"image": info, "segmentation": summary,
                               "disclaimer": DISCLAIMER}
    if return_mask:
        payload["mask_png_base64"] = encode_png_b64(
            cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
    if return_overlay:
        payload["overlay_png_base64"] = encode_png_b64(
            render_overlay(rgb, mask, None, None))
    return JSONResponse(payload)


@app.post("/inspect/full", tags=["inference"])
async def inspect_full(
    file: UploadFile = File(...),
    conf: Optional[float] = Form(None),
    seg_threshold: Optional[float] = Form(None),
    mm_per_pixel: Optional[float] = Form(None),
    reference_length_mm: Optional[float] = Form(None),
    reference_length_px: Optional[float] = Form(None),
    camera_distance_mm: Optional[float] = Form(None),
    focal_length_px: Optional[float] = Form(None),
    structure_id: str = Form(""),
    inspector: str = Form(""),
    location: str = Form(""),
    return_overlay: bool = Form(True),
):
    """End-to-end inspection: detect → segment → severity assessment.

    Supply a scale reference to get millimetre dimensions. Without one the
    severity grade is relative (RDSI) and every physical field returns null —
    deliberately, because pixel dimensions are not comparable between images
    taken at different distances.
    """
    rgb, info = await read_image(file)
    inspection_id = uuid.uuid4().hex[:12]
    t0 = time.time()

    detection = run_detection(rgb, conf, None)
    mask, seg_summary = run_segmentation(rgb, seg_threshold, None)

    scale = ScaleReference(
        mm_per_pixel=mm_per_pixel,
        reference_length_mm=reference_length_mm,
        reference_length_px=reference_length_px,
        camera_distance_mm=camera_distance_mm,
        focal_length_px=focal_length_px,
    )
    assessment = REGISTRY.engine.assess(
        mask=mask,
        detections=detection["detections"],
        image_shape=(rgb.shape[0], rgb.shape[1]),
        scale=scale,
        image_id=info["filename"],
    )

    payload: dict[str, Any] = {
        "inspection_id": inspection_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": {"structure_id": structure_id, "inspector": inspector,
                     "location": location, **info},
        "detection": detection,
        "segmentation": seg_summary,
        "assessment": assessment,
        "total_ms": round((time.time() - t0) * 1000, 2),
        "disclaimer": DISCLAIMER,
    }
    if return_overlay:
        overlay = render_overlay(rgb, mask, detection["detections"], assessment)
        payload["overlay_png_base64"] = encode_png_b64(overlay)

    # Cache for /reports/pdf so the client need not re-upload the image.
    _cache_inspection(inspection_id, rgb, mask, payload)
    return JSONResponse(payload)


# %%
# =============================================================================
# CELL 9 — INSPECTION CACHE (backs the two-step inspect → report flow)
# =============================================================================
_INSPECTION_CACHE: dict[str, dict] = {}
_CACHE_LIMIT = 64


def _cache_inspection(iid: str, rgb: np.ndarray, mask: np.ndarray, payload: dict):
    if len(_INSPECTION_CACHE) >= _CACHE_LIMIT:
        oldest = min(_INSPECTION_CACHE, key=lambda k: _INSPECTION_CACHE[k]["at"])
        _INSPECTION_CACHE.pop(oldest, None)
    _INSPECTION_CACHE[iid] = {"rgb": rgb, "mask": mask, "payload": payload,
                              "at": time.time()}


def _purge_expired_reports():
    """Reports are engineering records, not permanent storage — the volume that
    holds them is expected to be backed up elsewhere."""
    cutoff = time.time() - SETTINGS.REPORT_TTL_HOURS * 3600
    removed = 0
    for p in SETTINGS.REPORTS_DIR.glob("*.pdf"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        log.info("purged %d expired report(s)", removed)


# %%
# =============================================================================
# CELL 10 — PDF ENGINEERING REPORT (ReportLab)
# =============================================================================
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm as MM
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    PageBreak, KeepTogether)

GRADE_PDF_COLOURS = {
    "No damage detected": colors.HexColor("#4E8C57"),
    "Low": colors.HexColor("#3D7EA6"),
    "Medium": colors.HexColor("#C4901F"),
    "High": colors.HexColor("#C4601F"),
    "Critical": colors.HexColor("#A62B2B"),
}
INK = colors.HexColor("#1C1C1C")
RULE = colors.HexColor("#B9B4A9")
SOFT = colors.HexColor("#F2F0EB")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold",
                                fontSize=19, leading=23, textColor=INK,
                                alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=ss["Normal"], fontName="Helvetica",
                              fontSize=10, leading=14,
                              textColor=colors.HexColor("#5A5A5A")),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                             fontSize=12.5, leading=16, textColor=INK,
                             spaceBefore=13, spaceAfter=5),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=9.5, leading=13.5, textColor=INK),
        "small": ParagraphStyle("sm", parent=ss["Normal"], fontName="Helvetica",
                                fontSize=8, leading=11,
                                textColor=colors.HexColor("#6A6A6A")),
        "cell": ParagraphStyle("c", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=8.5, leading=11.5, textColor=INK),
    }


def _kv_table(rows, widths=(52 * MM, 118 * MM)):
    t = Table(rows, colWidths=list(widths), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, RULE),
        ("BACKGROUND", (0, 0), (0, -1), SOFT),
    ]))
    return t


def _grade_banner(grade: str, rdsi: float, basis: str, S):
    colour = GRADE_PDF_COLOURS.get(grade, colors.grey)
    txt = Paragraph(
        f'<font color="white" size="15"><b>{grade}</b></font><br/>'
        f'<font color="white" size="8.5">Severity index {rdsi:.1f} / 100 · '
        f'graded on {basis} criteria</font>', S["body"])
    t = Table([[txt]], colWidths=[170 * MM], rowHeights=[20 * MM])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colour),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _np_to_flowable(rgb: np.ndarray, max_w_mm: float = 170) -> RLImage:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "Failed to encode the report image.")
    bio = io.BytesIO(buf.tobytes())
    h, w = rgb.shape[:2]
    disp_w = max_w_mm * MM
    return RLImage(bio, width=disp_w, height=disp_w * h / w)


def build_pdf_report(out_path: Path, rgb: np.ndarray, mask: np.ndarray,
                     payload: dict) -> Path:
    S = _styles()
    a = payload["assessment"]
    meta = payload["metadata"]
    geom, sev, phys = a["geometry"], a["severity"], a["physical"]
    rec = a["recommendations"]

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=20 * MM, rightMargin=20 * MM,
        topMargin=18 * MM, bottomMargin=18 * MM,
        title=f"Structural Inspection {payload['inspection_id']}",
        author=SETTINGS.ORGANISATION)

    F: list = []
    F.append(Paragraph("Structural damage inspection", S["title"]))
    F.append(Paragraph(
        f"{SETTINGS.ORGANISATION} · Report {payload['inspection_id']} · "
        f"{datetime.now().strftime('%d %B %Y, %H:%M')}", S["sub"]))
    F.append(Spacer(1, 8))
    F.append(_grade_banner(sev["grade"], sev["rdsi"], sev["grading_basis"], S))
    F.append(Spacer(1, 10))

    F.append(Paragraph("Inspection record", S["h2"]))
    F.append(_kv_table([
        ["Structure", meta.get("structure_id") or "Not recorded"],
        ["Location", meta.get("location") or "Not recorded"],
        ["Inspector", meta.get("inspector") or "Not recorded"],
        ["Source image", meta.get("filename", "—")],
        ["Image resolution", f"{meta['processed_size']['width']} × "
                             f"{meta['processed_size']['height']} px"],
        ["Assessment confidence", sev["assessment_confidence"].title()],
    ]))

    F.append(Paragraph("Annotated image", S["h2"]))
    overlay = render_overlay(rgb, mask, payload["detection"]["detections"], a)
    F.append(_np_to_flowable(overlay))
    F.append(Paragraph(
        "Red shading marks segmented damage pixels; yellow boxes mark detected "
        "defect regions with model confidence.", S["small"]))
    F.append(Spacer(1, 8))

    F.append(Paragraph("Measured geometry", S["h2"]))
    geo_rows = [
        ["Damaged surface", f"{geom['damaged_area_px']:,} px "
                            f"({geom['surface_coverage_percent']:.3f}% of frame)"],
        ["Separate defect regions", f"{geom['num_regions']}"],
        ["Longest defect span", f"{geom['max_defect_span_px']:.1f} px "
                                f"({geom['max_defect_span_ratio']:.1%} of diagonal)"],
        ["Maximum width", f"{geom['max_width_px']:.2f} px"],
        ["Representative width", f"{geom['mean_width_px']:.2f} px"],
        ["Detections", f"{payload['detection']['num_detections']} "
                       f"({', '.join(f'{k}: {v}' for k, v in payload['detection']['by_class'].items()) or 'none'})"],
        ["Defect spread",
         f"{a['detection_summary']['bbox_coverage_ratio']:.1%} of frame "
         f"covered by defect boxes"],
    ]
    F.append(_kv_table(geo_rows))

    F.append(Paragraph("Physical dimensions", S["h2"]))
    if phys["scale_available"]:
        F.append(_kv_table([
            ["Scale source", phys["scale_method"].replace("_", " ").title()],
            ["Scale factor", f"{phys['mm_per_pixel']:.5f} mm per pixel"],
            ["Damaged area", f"{phys['damaged_area_mm2']:,.1f} mm² "
                             f"({phys['damaged_area_cm2']:,.2f} cm²)"],
            ["Maximum crack width", f"{phys['max_crack_width_mm']:.2f} mm"],
            ["Representative width", f"{phys['mean_crack_width_mm']:.2f} mm"],
            ["Longest defect span", f"{phys['max_defect_span_mm']:,.1f} mm"],
        ]))
        if phys.get("caveat"):
            F.append(Spacer(1, 4))
            F.append(Paragraph(phys["caveat"], S["small"]))
    else:
        F.append(Paragraph(
            "No scale reference was supplied with this image, so no physical "
            "dimensions can be reported. The severity grade above is a relative "
            "index describing how much of this frame is affected. To obtain "
            "millimetre measurements, re-photograph the defect with a ruler or a "
            "target of known size in the same plane as the surface.", S["body"]))

    F.append(PageBreak())

    F.append(Paragraph("How this grade was reached", S["h2"]))
    for r in sev["reasoning"]:
        F.append(Paragraph(f"• {r}", S["body"]))
        F.append(Spacer(1, 2))
    if sev["confidence_notes"]:
        F.append(Spacer(1, 6))
        F.append(Paragraph("Qualifications on this assessment:", S["body"]))
        for n in sev["confidence_notes"]:
            F.append(Paragraph(f"• {n}", S["small"]))

    F.append(Paragraph("Recommended action", S["h2"]))
    F.append(_kv_table([
        ["Priority", rec["priority"]],
        ["Re-inspection interval", rec["inspection_interval"]],
    ]))
    F.append(Spacer(1, 6))
    for i, action in enumerate(rec["actions"], 1):
        F.append(Paragraph(f"{i}. {action}", S["body"]))
        F.append(Spacer(1, 2))

    regions = geom.get("largest_regions", [])
    if regions:
        F.append(Paragraph("Defect inventory", S["h2"]))
        head = ["#", "Area (px)", "Position (x, y)", "Size (w × h)", "Elongation"]
        rows = [[Paragraph(f"<b>{h}</b>", S["cell"]) for h in head]]
        for r in regions[:10]:
            x, y, w, h = r["bbox_xywh"]
            rows.append([
                Paragraph(str(r["region_id"]), S["cell"]),
                Paragraph(f"{r['area_px']:,}", S["cell"]),
                Paragraph(f"{r['centroid_px'][0]:.0f}, {r['centroid_px'][1]:.0f}", S["cell"]),
                Paragraph(f"{w} × {h}", S["cell"]),
                Paragraph(f"{r['elongation']:.1f}×", S["cell"]),
            ])
        t = Table(rows, colWidths=[12 * MM, 30 * MM, 40 * MM, 40 * MM, 30 * MM],
                  hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), SOFT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
            ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ]))
        F.append(t)
        F.append(Paragraph(
            "Elongation is the ratio of the long side to the short side of each "
            "region: high values indicate linear cracking, low values indicate "
            "patch-type damage such as spalling.", S["small"]))

    F.append(Paragraph("Basis and limitations", S["h2"]))
    det_name = (REGISTRY.det["meta"].get("base_weights", "—")
                if REGISTRY.det else "—")
    seg_name = (REGISTRY.seg["meta"].get("architecture", "—")
                if REGISTRY.seg else "—")
    F.append(_kv_table([
        ["Detection model", det_name],
        ["Segmentation model", seg_name],
        ["Severity engine", f"v{a['engine_version']}"],
        ["API version", SETTINGS.API_VERSION],
    ]))
    F.append(Spacer(1, 8))
    F.append(Paragraph(DISCLAIMER, S["body"]))

    def _chrome(canvas, doc_):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(20 * MM, 14 * MM, A4[0] - 20 * MM, 14 * MM)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#6A6A6A"))
        canvas.drawString(20 * MM, 9.5 * MM,
                          f"Report {payload['inspection_id']} · "
                          f"{SETTINGS.ORGANISATION} · Decision support only")
        canvas.drawRightString(A4[0] - 20 * MM, 9.5 * MM, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(F, onFirstPage=_chrome, onLaterPages=_chrome)
    return out_path


@app.post("/reports/pdf", tags=["reporting"])
async def reports_pdf(
    inspection_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    structure_id: str = Form(""),
    inspector: str = Form(""),
    location: str = Form(""),
    mm_per_pixel: Optional[float] = Form(None),
    reference_length_mm: Optional[float] = Form(None),
    reference_length_px: Optional[float] = Form(None),
):
    """Generate a downloadable engineering inspection PDF.

    Two ways to call it:
      * pass `inspection_id` from a previous /inspect/full response (no re-upload)
      * pass `file` to run a fresh inspection and report in one step
    """
    if inspection_id and inspection_id in _INSPECTION_CACHE:
        cached = _INSPECTION_CACHE[inspection_id]
        rgb, mask, payload = cached["rgb"], cached["mask"], dict(cached["payload"])
        # Allow the report step to fill in record fields left blank at inspection.
        for k, v in (("structure_id", structure_id), ("inspector", inspector),
                     ("location", location)):
            if v:
                payload["metadata"][k] = v
    elif file is not None:
        rgb, info = await read_image(file)
        detection = run_detection(rgb, None, None)
        mask, seg_summary = run_segmentation(rgb, None, None)
        assessment = REGISTRY.engine.assess(
            mask=mask, detections=detection["detections"],
            image_shape=(rgb.shape[0], rgb.shape[1]),
            scale=ScaleReference(mm_per_pixel=mm_per_pixel,
                                 reference_length_mm=reference_length_mm,
                                 reference_length_px=reference_length_px),
            image_id=info["filename"])
        payload = {
            "inspection_id": uuid.uuid4().hex[:12],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {"structure_id": structure_id, "inspector": inspector,
                         "location": location, **info},
            "detection": detection, "segmentation": seg_summary,
            "assessment": assessment,
        }
    else:
        raise HTTPException(
            400, "Provide either an inspection_id from a previous /inspect/full "
                 "call, or an image file to inspect and report in one step.")

    out = SETTINGS.REPORTS_DIR / f"inspection_{payload['inspection_id']}.pdf"
    build_pdf_report(out, rgb, mask, payload)
    log.info("report written: %s", out.name)

    return FileResponse(
        path=str(out), media_type="application/pdf", filename=out.name,
        headers={"X-Inspection-Id": payload["inspection_id"],
                 "X-Severity-Grade": payload["assessment"]["severity"]["grade"]})


@app.get("/reports/{inspection_id}", tags=["reporting"])
async def fetch_report(inspection_id: str):
    """Re-download a previously generated report."""
    path = SETTINGS.REPORTS_DIR / f"inspection_{inspection_id}.pdf"
    if not path.exists():
        raise HTTPException(
            404, f"No report found for {inspection_id}. Reports are kept for "
                 f"{SETTINGS.REPORT_TTL_HOURS} hours; generate a new one.")
    return FileResponse(str(path), media_type="application/pdf", filename=path.name)


# %%
# =============================================================================
# CELL 11 — ERROR HANDLING & LOCAL RUN
# =============================================================================
@app.exception_handler(Exception)
async def unhandled(request, exc):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "The server could not complete that request.",
                 "detail": str(exc), "path": str(request.url.path)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False,
                log_level="info")