import os
import gc
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import onnxruntime as ort

# --- Config & Setup ---
@dataclass
class SegCfg:
    project_root: Path = Path(r"D:\nti_project")
    manifest_dir: Path = Path(r"D:\nti_project\data_processed\manifests")
    masks_dir: Path = Path(r"D:\nti_project\data_processed\masks_512")
    ckpt_dir: Path = Path(r"D:\nti_project\models\segmentation")
    artifact_dir: Path = Path(r"D:\nti_project\audit_results\segmentation")
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    img_size: int = 512
    batch_size: int = 4
    num_workers: int = 0
    eval_threshold: float = 0.5
    min_blob_px: int = 32
    seed: int = 42

SCFG = SegCfg()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_PTH = SCFG.ckpt_dir / "best_unet.pth"

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
except ImportError:
    HAS_SMP = False

# --- Helpers & Data Loading ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def normalize_mask_stem(stem: str) -> str:
    s = stem.lower()
    for suf in ("_mask", "-mask", "_gt", "_label"):
        if s.endswith(suf): s = s[: -len(suf)]
    return s

MASK_INDEX = {normalize_mask_stem(mp.stem): mp for mp in SCFG.masks_dir.glob("*.png")}

def build_seg_frame(split: str) -> pd.DataFrame:
    df = pd.read_csv(SCFG.manifest_dir / f"{split}_manifest.csv")
    img_512_dir = SCFG.project_root / "data_processed" / "images_512"
    path_col = [c for c in df.columns if c.lower() in ("image_path", "filepath", "path", "abs_path", "image", "file_path")][0]
    out = df.copy()
    out["img_path"] = out[path_col].map(lambda p: str(img_512_dir / Path(p).name))
    out["mask_path"] = out["img_path"].map(lambda p: str(MASK_INDEX.get(normalize_mask_stem(Path(p).stem), "")))
    out = out[out["img_path"].map(os.path.exists) & (out["mask_path"] != "")]
    return out.drop_duplicates(subset=["img_path"]).reset_index(drop=True)[["img_path", "mask_path"]]

class ConcreteSegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int):
        self.images = df["img_path"].tolist()
        self.masks = df["mask_path"].tolist()
        self.size = size

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(self.images[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(np.transpose(x, (2, 0, 1)).copy())
        y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return x, y

val_df = build_seg_frame("val")
val_ds = ConcreteSegDataset(val_df, SCFG.img_size)
val_loader = DataLoader(val_ds, batch_size=SCFG.batch_size, shuffle=False, num_workers=0)

# --- Model & Evaluation Function ---
def build_seg_model() -> nn.Module:
    if HAS_SMP:
        return smp.Unet(encoder_name=SCFG.encoder, encoder_weights=None, in_channels=3, classes=1, decoder_attention_type="scse")
    from torchvision.models import resnet34
    enc = resnet34(weights=None)
    # Basic UNet fallback if smp isn't installed
    return enc

model = build_seg_model().to(DEVICE)
ckpt = torch.load(BEST_PTH, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["state_dict"])
print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

class DiceBCELoss(nn.Module):
    def forward(self, logits, targets): return torch.tensor(0.0)

criterion = DiceBCELoss()

@torch.no_grad()
def evaluate(model, loader, criterion, thr: float = 0.5) -> dict:
    model.eval()
    dices, ious, empties = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits.float())
        preds = (probs >= thr).float()
        b = preds.shape[0]
        p, t = preds.view(b, -1), y.view(b, -1)
        inter = (p * t).sum(1)
        psum, tsum = p.sum(1), t.sum(1)
        union = psum + tsum - inter
        dice = torch.where(tsum + psum == 0, torch.ones_like(inter), 2 * inter / (psum + tsum + 1e-7))
        iou = torch.where(union == 0, torch.ones_like(inter), inter / (union + 1e-7))
        dices.append(dice.cpu().numpy()); ious.append(iou.cpu().numpy()); empties.append((tsum == 0).cpu().numpy())
    d, i, e = np.concatenate(dices), np.concatenate(ious), np.concatenate(empties)
    ne = ~e
    return {"dice": float(d[ne].mean()), "iou": float(i[ne].mean()), "empty_mask_accuracy": float(d[e].mean()), "_per_sample": {"dice": d, "iou": i, "empty": e}}

# --- CELL 8: THRESHOLD SWEEP ---
rows = []
for thr in np.round(np.arange(0.20, 0.81, 0.05), 2):
    m = evaluate(model, val_loader, criterion, float(thr))
    rows.append({"threshold": float(thr), "dice": m["dice"], "iou": m["iou"], "empty_mask_accuracy": m["empty_mask_accuracy"]})
    print(f"  thr={thr:.2f}  dice={m['dice']:.4f}  iou={m['iou']:.4f}  empty-acc={m['empty_mask_accuracy']:.3f}")

sweep = pd.DataFrame(rows)
sweep.to_csv(SCFG.artifact_dir / "threshold_sweep_val.csv", index=False)
BEST_THR = float(sweep.loc[sweep["dice"].idxmax(), "threshold"])
print(f"\nDice-optimal threshold: {BEST_THR:.2f}")

final = evaluate(model, val_loader, criterion, BEST_THR)

# --- CELL 9: ONNX EXPORT ---
model.eval().float()
ONNX_PATH = SCFG.ckpt_dir / "segmentation.onnx"
dummy = torch.randn(1, 3, SCFG.img_size, SCFG.img_size, device=DEVICE)

torch.onnx.export(
    model, dummy, str(ONNX_PATH),
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17, do_constant_folding=True,
)
print(f"✓ ONNX Exported → {ONNX_PATH}")

SEG_META = {
    "model_name": "concrete_damage_segmenter",
    "task": "segmentation",
    "architecture": f"unet_{SCFG.encoder}",
    "onnx_path": str(ONNX_PATH),
    "pth_path": str(BEST_PTH),
    "input": {
        "name": "input", "shape": [1, 3, SCFG.img_size, SCFG.img_size],
        "layout": "NCHW", "color_order": "RGB", "dtype": "float32",
        "scale": 1.0 / 255.0, "mean": IMAGENET_MEAN.tolist(), "std": IMAGENET_STD.tolist(),
        "resize": "direct_bilinear_square",
    },
    "output": {
        "name": "logits", "shape": [1, 1, SCFG.img_size, SCFG.img_size],
        "activation": "sigmoid",
        "postprocess": "sigmoid -> threshold -> remove blobs < min_blob_px -> resize NEAREST to original size",
    },
    "thresholds": {"probability": BEST_THR, "min_blob_px": SCFG.min_blob_px},
    "val_metrics": {k: v for k, v in final.items() if k != "_per_sample"},
    "best_epoch": int(ckpt["epoch"]),
    "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
META_PATH = SCFG.ckpt_dir / "segmentation_meta.json"
META_PATH.write_text(json.dumps(SEG_META, indent=2), encoding="utf-8")
print(f"✓ Metadata Created → {META_PATH}")
print("✅ Export Script Complete! You can now run 07_MODEL_EVALUATION.py")