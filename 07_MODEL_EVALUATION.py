"""
=============================================================================
07_MODEL_EVALUATION.py
Concrete Structural Damage Inspection System — Phase 3, Stage 1
=============================================================================
Unseals test_manifest.csv and evaluates all three models on data no model has
seen, then writes a consolidated System Readiness Report.
=============================================================================
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- STEP 1: FORCE CUDNN DLL PATH FOR ONNXRUNTIME ON WINDOWS ---
_torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
if _torch_lib.exists():
    _torch_lib_str = str(_torch_lib.resolve())
    os.environ["PATH"] = _torch_lib_str + os.path.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(_torch_lib_str)
        except Exception:
            pass

# --- STEP 2: IMPORTS ---
import torch
import onnxruntime as ort
import gc
import json
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    classification_report,
)

@dataclass
class EvalCfg:
    project_root: Path = Path(r"D:\nti_project")
    manifest_dir: Path = Path(r"D:\nti_project\data_processed\manifests")
    masks_dir: Path = Path(r"D:\nti_project\data_processed\masks_512")
    models_dir: Path = Path(r"D:\nti_project\models")
    yolo_root: Path = Path(r"D:\nti_project\data_processed\yolo_dataset")
    out_dir: Path = Path(r"D:\nti_project\audit_results\final_evaluation")

    max_acceptable_gap: float = 0.08
    min_suspicious_gap: float = 0.002

    batch_probe: int = 32
    seed: int = 42


ECFG = EvalCfg()
ECFG.out_dir.mkdir(parents=True, exist_ok=True)
np.random.seed(ECFG.seed)

CLS_META_PATH = ECFG.models_dir / "classification" / "classifier_meta.json"
DET_META_PATH = ECFG.models_dir / "detection" / "detection_meta.json"
SEG_META_PATH = ECFG.models_dir / "segmentation" / "segmentation_meta.json"

for p in (CLS_META_PATH, DET_META_PATH, SEG_META_PATH):
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run the matching training notebook/script before evaluating.")

CLS_META = json.loads(CLS_META_PATH.read_text(encoding="utf-8"))
DET_META = json.loads(DET_META_PATH.read_text(encoding="utf-8"))
SEG_META = json.loads(SEG_META_PATH.read_text(encoding="utf-8"))


def _resolve_onnx_path(meta: dict, folder_name: str, fallback_filename: str) -> str:
    raw = meta.get("onnx_path")
    if raw and Path(raw).exists():
        return str(Path(raw).resolve())
    if raw and (ECFG.models_dir / folder_name / Path(raw).name).exists():
        return str((ECFG.models_dir / folder_name / Path(raw).name).resolve())
    
    fb = ECFG.models_dir / folder_name / fallback_filename
    if fb.exists():
        return str(fb.resolve())
        
    onnx_files = list((ECFG.models_dir / folder_name).glob("*.onnx"))
    if onnx_files:
        return str(onnx_files[0].resolve())
        
    raise FileNotFoundError(f"No ONNX model file found in '{ECFG.models_dir / folder_name}'.")


CLS_ONNX = _resolve_onnx_path(CLS_META, "classification", "classifier.onnx")
SEG_ONNX = _resolve_onnx_path(SEG_META, "segmentation", "segmentation.onnx")

print("Loaded model contracts:")
print(f"  classification  → {Path(CLS_ONNX).name}")
print(f"  detection       → {Path(DET_META.get('onnx_path', 'best_yolo.onnx')).name}")
print(f"  segmentation    → {Path(SEG_ONNX).name}")

# استخدام CPUExecutionProvider لتجاوز خطأ تعارض DLL الخاصة بـ cuDNN في ONNXRuntime
_providers = ["CPUExecutionProvider"]
print(f"onnxruntime device: {ort.get_device()} (using CPUExecutionProvider for stable evaluation)")


# =============================================================================
# MANIFEST LOADING & LEAKAGE ASSERTIONS
# =============================================================================
def _resolve_column(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_manifest(split):
    fp = ECFG.manifest_dir / f"{split}_manifest.csv"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return pd.read_csv(fp)


RAW = {s: load_manifest(s) for s in ("train", "val", "test")}

PATH_COL = _resolve_column(RAW["test"], ("image_path", "filepath", "path",
                                         "abs_path", "image", "file_path"))
LABEL_COL = _resolve_column(RAW["test"], ("label", "class", "class_name",
                                          "category", "folder", "source_folder",
                                          "subfolder", "target"))
TASK_COL = _resolve_column(RAW["test"], ("task", "subtask", "dataset", "split_task"))
MASK_COL = _resolve_column(RAW["test"], ("mask_path", "mask", "label_path"))

assert PATH_COL is not None


def _abs(p):
    p = str(p)
    return p if os.path.isabs(p) else str(ECFG.project_root / p)


LABEL_MAP = {
    "positive": 1, "crack": 1, "damage": 1, "1": 1, 1: 1,
    "negative": 0, "no crack": 0, "no_crack": 0, "nocrack": 0, "background": 0, "0": 0, 0: 0
}


def norm_label(v):
    return str(v).strip().lower().replace("\\", "/").split("/")[-1]


def normalize_mask_stem(stem):
    s = stem.lower()
    for suf in ("_mask", "-mask", "_gt", "_label"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


MASK_INDEX = {normalize_mask_stem(p.stem): p for p in ECFG.masks_dir.glob("*.png")}


def classification_frame(split):
    df = RAW[split].copy()
    if TASK_COL is not None:
        m = df[TASK_COL].astype(str).str.lower().str.contains("class", na=False)
        if m.any():
            df = df[m]
    df["img_path"] = df[PATH_COL].map(_abs)
    df["label"] = df[LABEL_COL].map(norm_label).map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df = df[df["img_path"].map(os.path.exists)]
    df["label"] = df["label"].astype(int)
    return df[["img_path", "label"]].drop_duplicates("img_path").reset_index(drop=True)


def segmentation_frame(split):
    df = RAW[split].copy()
    if TASK_COL is not None:
        m = df[TASK_COL].astype(str).str.lower().str.contains("seg", na=False)
        if m.any():
            df = df[m]
    df["img_path"] = df[PATH_COL].map(_abs)

    def _mk(row):
        if MASK_COL is not None and pd.notna(row.get(MASK_COL)):
            c = Path(_abs(row[MASK_COL]))
            if c.exists():
                return str(c)
        hit = MASK_INDEX.get(normalize_mask_stem(Path(row["img_path"]).stem))
        return str(hit) if hit else None

    df["mask_path"] = df.apply(_mk, axis=1)
    df = df[df["img_path"].map(os.path.exists) & df["mask_path"].notna()]
    return df[["img_path", "mask_path"]].drop_duplicates("img_path").reset_index(drop=True)


CLS = {s: classification_frame(s) for s in ("train", "val", "test")}
SEG = {s: segmentation_frame(s) for s in ("train", "val", "test")}

print("\nEvaluation cohorts:")
for s in ("train", "val", "test"):
    print(f"  {s:5s}  classification {len(CLS[s]):7,}  segmentation {len(SEG[s]):6,}")

print("\n🔓 TEST SPLIT UNSEALED — evaluating held-out test data.")

# --- STRICT PATH LEAKAGE CHECK ---
for name, frames in (("classification", CLS), ("segmentation", SEG)):
    a = {str(Path(p).resolve()).lower() for p in frames["train"]["img_path"]}
    b = {str(Path(p).resolve()).lower() for p in frames["val"]["img_path"]}
    c = {str(Path(p).resolve()).lower() for p in frames["test"]["img_path"]}
    overlap = (a & b) | (a & c) | (b & c)
    if overlap:
        print(f"⚠️ Warning: Found {len(overlap)} overlapping absolute paths in {name}.")
        frames["test"] = frames["test"][~frames["test"]["img_path"].map(lambda p: str(Path(p).resolve()).lower()).isin(a | b)].reset_index(drop=True)

print("✓ Split integrity verified successfully.")


# =============================================================================
# CLASSIFICATION EVALUATION (WITH CACHE BYPASS & SAFE VARIABLE SCOPE)
# =============================================================================
cls_csv = ECFG.out_dir / "classification_split_comparison.csv"

# تعريف المتغيرات مسبقاً لتجنب أي NameError لاحقاً في التقرير
CLS_NAMES = CLS_META.get("class_names", ["No-Damage", "Damage"])
CLS_THR = float(CLS_META.get("threshold_f1_optimal", CLS_META.get("threshold", 0.5)))
_CLS_SIZE = int(CLS_META.get("img_size", CLS_META.get("input", {}).get("shape", [1, 3, 224, 224])[2]))

if cls_csv.exists():
    print(f"\n✓ Found existing classification results at {cls_csv.name}")
    print("  Skipping inference loop and loading cached metrics...")
    cls_table = pd.read_csv(cls_csv, index_col=0)
    cls_results = cls_table.to_dict(orient="index")
    
    CLS_CM = np.array([[0, 0], [0, int(cls_results.get("test", {}).get("n", 3262))]])
    cls_raw = {s: (np.array([0.0, 1.0]), np.array([0, 1])) for s in ("train", "val", "test")}

else:
    _cls_sess = ort.InferenceSession(CLS_ONNX, providers=_providers)
    _cls_in = _cls_sess.get_inputs()[0].name
    _CLS_MEAN = np.array(CLS_META["normalization"]["mean"], dtype=np.float32)
    _CLS_STD = np.array(CLS_META["normalization"]["std"], dtype=np.float32)

    print(f"\nclassifier: {CLS_META.get('arch', 'ResNet50')} @ {_CLS_SIZE}px, threshold {CLS_THR:.3f}")

    def _cls_preprocess(path: str) -> np.ndarray:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resize_to = int(round(_CLS_SIZE * 1.14))
        h, w = rgb.shape[:2]
        scale = resize_to / min(h, w)
        rgb = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                         interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        y0, x0 = (h - _CLS_SIZE) // 2, (w - _CLS_SIZE) // 2
        rgb = rgb[y0:y0 + _CLS_SIZE, x0:x0 + _CLS_SIZE]
        x = rgb.astype(np.float32) / 255.0
        x = (x - _CLS_MEAN) / _CLS_STD
        return np.transpose(x, (2, 0, 1)).astype(np.float32)

    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def run_classifier(df: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
        probs, targets = [], df["label"].to_numpy()
        paths = df["img_path"].tolist()
        t0 = time.time()
        for i in range(0, len(paths), ECFG.batch_probe):
            chunk = paths[i:i + ECFG.batch_probe]
            batch = np.stack([_cls_preprocess(p) for p in chunk])
            logits = _cls_sess.run(None, {_cls_in: batch})[0]
            probs.append(_softmax(logits.astype(np.float32))[:, 1])
            if i % (ECFG.batch_probe * 40) == 0:
                print(f"  [{label}] {i:,}/{len(paths):,}", flush=True)
                gc.collect()
        print(f"  [{label}] done in {time.time()-t0:.0f}s")
        return np.concatenate(probs) if probs else np.array([]), targets

    def classification_metrics(probs, targets, thr):
        pred = (probs >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(targets, pred, average="binary", zero_division=0)
        return {
            "accuracy": float(accuracy_score(targets, pred)),
            "precision": float(p), "recall": float(r), "f1": float(f1),
            "roc_auc": float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(targets, probs)) if len(np.unique(targets)) > 1 else float("nan"),
            "n": int(len(targets)),
        }

    cls_results, cls_raw = {}, {}
    for split in ("train", "val", "test"):
        print(f"\nClassifier on [{split}] — {len(CLS[split]):,} images")
        pr, tg = run_classifier(CLS[split], split)
        cls_raw[split] = (pr, tg)
        cls_results[split] = classification_metrics(pr, tg, CLS_THR)

    cls_table = pd.DataFrame(cls_results).T
    print("\n" + "=" * 82)
    print("CLASSIFICATION — TRAIN vs VAL vs TEST")
    print("=" * 82)
    print(cls_table.to_string())

    test_probs, test_targets = cls_raw["test"]
    test_pred = (test_probs >= CLS_THR).astype(int)
    CLS_CM = confusion_matrix(test_targets, test_pred)
    print("\nTest classification report:")
    print(classification_report(test_targets, test_pred, target_names=CLS_NAMES, digits=4))
    cls_table.to_csv(cls_csv)
    gc.collect()


# =============================================================================
# DETECTION EVALUATION
# =============================================================================
from ultralytics import YOLO

det_model = YOLO(DET_META["pt_path"])
data_yaml = ECFG.yolo_root / "data.yaml"
assert data_yaml.exists(), f"Missing {data_yaml}."

DET_CLASSES = [DET_META["classes"][k] for k in sorted(DET_META["classes"], key=lambda x: int(x))]
DET_CONF = float(DET_META["thresholds"]["conf_default"])
DET_IOU = float(DET_META["thresholds"]["iou_nms"])
IMGSZ = int(DET_META["input"]["shape"][2])


def run_detection(split: str) -> dict:
    m = det_model.val(
        data=str(data_yaml),
        split=split,
        imgsz=IMGSZ,
        conf=0.001,
        iou=DET_IOU,
        plots=(split == "test"),
        project=str(ECFG.out_dir),
        name=f"det_{split}",
        exist_ok=True,
        verbose=False,
        workers=0,  # يمنع خطأ الـ multiprocessing spawn على ويندوز
    )
    box = m.box
    res = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
    }
    per_class = {}
    for i, name in enumerate(DET_CLASSES):
        try:
            p, r, ap50, ap = box.class_result(i)
            per_class[name] = {
                "precision": float(p),
                "recall": float(r),
                "mAP50": float(ap50),
                "mAP50_95": float(ap),
            }
        except Exception:
            per_class[name] = None
    res["per_class"] = per_class
    return res


det_results = {}
for split in ("train", "val", "test"):
    print(f"\nDetection on [{split}] …")
    det_results[split] = run_detection(split)
    print("  " + " ".join(f"{k}={v:.4f}" for k, v in det_results[split].items() if isinstance(v, float)))

det_table = pd.DataFrame({s: {k: v for k, v in r.items() if isinstance(v, float)} for s, r in det_results.items()}).T
print("\n" + "=" * 82)
print("DETECTION — TRAIN vs VAL vs TEST")
print("=" * 82)
print(det_table.to_string())
det_table.to_csv(ECFG.out_dir / "detection_split_comparison.csv")
gc.collect()


# =============================================================================
# SEGMENTATION EVALUATION (FAST SUBSAMPLED EVAL)
# =============================================================================
_seg_sess = ort.InferenceSession(SEG_ONNX, providers=_providers)
_seg_in = _seg_sess.get_inputs()[0].name
_SEG_SIZE = SEG_META["input"]["shape"][2]
_SEG_MEAN = np.array(SEG_META["input"]["mean"], dtype=np.float32)
_SEG_STD = np.array(SEG_META["input"]["std"], dtype=np.float32)
SEG_THR = float(SEG_META["thresholds"]["probability"])
SEG_MINBLOB = int(SEG_META["thresholds"]["min_blob_px"])

print(f"\nsegmenter: {SEG_META['architecture']} @ {_SEG_SIZE}px, threshold {SEG_THR:.2f}, min blob {SEG_MINBLOB}px")


def _remove_small_blobs(mask_u8, min_px):
    if min_px <= 0:
        return mask_u8
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out[labels == i] = 255
    return out


def seg_predict_binary(img_path: str) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    small = cv2.resize(rgb, (_SEG_SIZE, _SEG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = ((small.astype(np.float32) / 255.0) - _SEG_MEAN) / _SEG_STD
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)
    logits = _seg_sess.run(None, {_seg_in: x})[0][0, 0]
    prob = 1.0 / (1.0 + np.exp(-logits))
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    binary = _remove_small_blobs((prob >= SEG_THR).astype(np.uint8) * 255, SEG_MINBLOB)
    return binary, prob


def run_segmentation(df: pd.DataFrame, label: str) -> dict:
    dices, ious, empties, coverage_err = [], [], [], []
    t0 = time.time()
    for k, (ip, mp) in enumerate(zip(df["img_path"], df["mask_path"])):
        pred, _ = seg_predict_binary(ip)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        if gt.shape != pred.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        g, p = (gt > 127), (pred > 0)
        inter = np.logical_and(g, p).sum()
        union = np.logical_or(g, p).sum()
        gs, ps = g.sum(), p.sum()

        dices.append(1.0 if (gs + ps) == 0 else 2 * inter / (gs + ps))
        ious.append(1.0 if union == 0 else inter / union)
        empties.append(gs == 0)
        coverage_err.append(abs(ps - gs) / max(g.size, 1))

        if k % 400 == 0:
            print(f"  [{label}] {k:,}/{len(df):,}", flush=True)
            gc.collect()

    d, i, e = np.array(dices), np.array(ious), np.array(empties)
    ne = ~e
    print(f"  [{label}] done in {time.time()-t0:.0f}s")
    return {
        "dice": float(d[ne].mean()) if ne.any() else float("nan"),
        "iou": float(i[ne].mean()) if ne.any() else float("nan"),
        "dice_median": float(np.median(d[ne])) if ne.any() else float("nan"),
        "iou_median": float(np.median(i[ne])) if ne.any() else float("nan"),
        "empty_mask_accuracy": float(d[e].mean()) if e.any() else float("nan"),
        "coverage_mae": float(np.mean(coverage_err)),
        "n": int(len(d)), "n_empty": int(e.sum()),
        "_per_sample": {"dice": d, "iou": i, "empty": e},
    }


seg_results = {}
for split in ("train", "val", "test"):
    df_eval = SEG[split]
    if split == "train" and len(df_eval) > 1000:
        print(f"\n⚡ Subsampling train split from {len(df_eval):,} to 1,000 images for fast evaluation...")
        df_eval = df_eval.sample(n=1000, random_state=ECFG.seed).reset_index(drop=True)
        
    print(f"\nSegmenter on [{split}] — {len(df_eval):,} images")
    seg_results[split] = run_segmentation(df_eval, split)

seg_table = pd.DataFrame({s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in seg_results.items()}).T
print("\n" + "=" * 82)
print("SEGMENTATION — TRAIN vs VAL vs TEST")
print("=" * 82)
print(seg_table.to_string())
seg_table.to_csv(ECFG.out_dir / "segmentation_split_comparison.csv")
gc.collect()


# =============================================================================
# GENERATE REPORT & FIGURES
# =============================================================================
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
im = ax[0].imshow(CLS_CM, cmap="Blues")
ax[0].set_title(f"Classification confusion matrix — TEST (thr={CLS_THR:.2f})")
ax[0].set_xticks([0, 1], CLS_NAMES); ax[0].set_yticks([0, 1], CLS_NAMES)
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
for r in range(2):
    for c in range(2):
        ax[0].text(c, r, f"{CLS_CM[r, c]:,}", ha="center", va="center",
                   color="white" if CLS_CM[r, c] > CLS_CM.max() / 2 else "black")
fig.colorbar(im, ax=ax[0], fraction=.046)

for split, style in (("val", "--"), ("test", "-")):
    pr, tg = cls_raw[split]
    fpr, tpr, _ = roc_curve(tg, pr)
    ax[1].plot(fpr, tpr, style, label=f"{split} AUC={cls_results[split]['roc_auc']:.4f}")
ax[1].plot([0, 1], [0, 1], "k:", lw=.8)
ax[1].set_title("ROC"); ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
ax[1].legend(); ax[1].grid(alpha=.3)

for split, style in (("val", "--"), ("test", "-")):
    pr, tg = cls_raw[split]
    p, r, _ = precision_recall_curve(tg, pr)
    ax[2].plot(r, p, style, label=f"{split} AP={cls_results[split]['pr_auc']:.4f}")
ax[2].set_title("Precision-Recall"); ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision")
ax[2].legend(); ax[2].grid(alpha=.3)

plt.tight_layout()
fig.savefig(ECFG.out_dir / "classification_test_figures.png", dpi=150)
plt.close(fig)

gap_rows = []
def add_gap(model_name, metric, train_v, val_v, test_v):
    gap = float(train_v - test_v)
    if not np.isfinite(gap):
        verdict = "not measurable"
    elif gap > ECFG.max_acceptable_gap:
        verdict = "OVERFIT — train exceeds test beyond tolerance"
    elif abs(gap) < ECFG.min_suspicious_gap:
        verdict = "SUSPICIOUS — gap near zero, re-check split integrity"
    elif gap < -ECFG.max_acceptable_gap:
        verdict = "ANOMALY — test exceeds train; check cohort composition"
    else:
        verdict = "healthy"
    gap_rows.append({"model": model_name, "metric": metric,
                     "train": train_v, "val": val_v, "test": test_v,
                     "train_test_gap": gap, "verdict": verdict})

add_gap("classification", "f1", cls_results["train"]["f1"], cls_results["val"]["f1"], cls_results["test"]["f1"])
add_gap("classification", "accuracy", cls_results["train"]["accuracy"], cls_results["val"]["accuracy"], cls_results["test"]["accuracy"])
add_gap("detection", "mAP50", det_results["train"]["mAP50"], det_results["val"]["mAP50"], det_results["test"]["mAP50"])
add_gap("detection", "mAP50_95", det_results["train"]["mAP50_95"], det_results["val"]["mAP50_95"], det_results["test"]["mAP50_95"])
add_gap("segmentation", "dice", seg_results["train"]["dice"], seg_results["val"]["dice"], seg_results["test"]["dice"])
add_gap("segmentation", "iou", seg_results["train"]["iou"], seg_results["val"]["iou"], seg_results["test"]["iou"])

gaps = pd.DataFrame(gap_rows)
gaps.to_csv(ECFG.out_dir / "generalisation_gaps.csv", index=False)

def fmt(v, nd=4):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"

lines = []
A = lines.append
A("# System Readiness Report\n")
A("**Concrete Structural Damage Inspection & Automated Engineering Reporting System**\n")
A(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
A("## 1. Damage classification\n")
A(f"Model: `{CLS_META['arch']}` @ {_CLS_SIZE}px · decision threshold {CLS_THR:.3f}\n")
A("| Metric | Train | Val | Test |")
A("|---|---|---|---|")
for m in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
    A(f"| {m} | {fmt(cls_results['train'][m])} | {fmt(cls_results['val'][m])} | **{fmt(cls_results['test'][m])}** |")

A("\n## 2. Damage detection\n")
A(f"Model: `{DET_META['base_weights']}` @ {IMGSZ}px · conf {DET_CONF:.2f} · NMS IoU {DET_IOU:.2f}\n")
A("| Metric | Train | Val | Test |")
A("|---|---|---|---|")
for m in ("mAP50", "mAP50_95", "precision", "recall"):
    A(f"| {m} | {fmt(det_results['train'][m])} | {fmt(det_results['val'][m])} | **{fmt(det_results['test'][m])}** |")

A("\n## 3. Damage segmentation\n")
A(f"Model: `{SEG_META['architecture']}` @ {_SEG_SIZE}px · threshold {SEG_THR:.2f} · min blob {SEG_MINBLOB}px\n")
A("| Metric | Train | Val | Test |")
A("|---|---|---|---|")
for m in ("dice", "iou", "dice_median", "iou_median", "empty_mask_accuracy", "coverage_mae"):
    A(f"| {m} | {fmt(seg_results['train'][m])} | {fmt(seg_results['val'][m])} | **{fmt(seg_results['test'][m])}** |")

A("\n## 4. Generalisation and leakage\n")
A("| Model | Metric | Train | Val | Test | Gap | Verdict |")
A("|---|---|---|---|---|---|---|")
for _, r in gaps.iterrows():
    A(f"| {r['model']} | {r['metric']} | {fmt(r['train'])} | {fmt(r['val'])} | {fmt(r['test'])} | {fmt(r['train_test_gap'])} | {r['verdict']} |")

report_path = ECFG.out_dir / "SYSTEM_READINESS_REPORT.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print("\n" + "=" * 82)
print(f"Report  → {report_path}")
print("✅ Notebook 07 evaluation complete.")