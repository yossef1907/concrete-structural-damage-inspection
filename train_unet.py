

from __future__ import annotations

import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import math
import time
import random
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


@dataclass
class SegCfg:
    # --- Paths -------------------------------------------------------------
    project_root: Path = Path(r"D:\nti_project")
    manifest_dir: Path = Path(r"D:\nti_project\data_processed\manifests")
    masks_dir: Path = Path(r"D:\nti_project\data_processed\masks_512")   
    ckpt_dir: Path = Path(r"D:\nti_project\models\segmentation")
    artifact_dir: Path = Path(r"D:\nti_project\audit_results\segmentation")

    # --- Manifest schema ---------------------------------------------------
    path_col_candidates: Sequence[str] = (
        "image_path", "filepath", "path", "abs_path", "image", "file_path")
    mask_col_candidates: Sequence[str] = (
        "mask_path", "mask", "label_path", "mask_file")
    task_col_candidates: Sequence[str] = ("task", "subtask", "dataset", "split_task")
    difficulty_col_candidates: Sequence[str] = ("difficulty", "level", "split_level")
    segmentation_task_keyword: str = "seg"

    # --- Model -------------------------------------------------------------
    encoder: str = "resnet34"
    encoder_weights: str = "imagenet"
    img_size: int = 512

    # --- Training (Optimized for 8GB RAM + RTX 3050 6GB on Windows) ----------
  # inside SegCfg
    batch_size: int = 4    # Changed back to 4 (fits comfortably in 6GB VRAM)
    accum_steps: int = 2   # Changed to 2 (effective batch = 8)
    num_workers: int = 0   # Keep 0 for Windows safety        # Set to 0 to prevent RAM memory leaks on Windows
    epochs: int = 80
    lr: float = 3e-4
    lr_encoder_mult: float = 0.1
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    early_stopping_patience: int = 15
    grad_clip: float = 1.0
    amp: bool = True

    # --- Loss --------------------------------------------------------------
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    bce_pos_weight: Optional[float] = None

    # --- Eval --------------------------------------------------------------
    eval_threshold: float = 0.5
    min_blob_px: int = 32

    seed: int = 42


SCFG = SegCfg()
SCFG.ckpt_dir.mkdir(parents=True, exist_ok=True)
SCFG.artifact_dir.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SCFG.seed)
np.random.seed(SCFG.seed)
torch.manual_seed(SCFG.seed)
torch.cuda.manual_seed_all(SCFG.seed)
torch.backends.cudnn.benchmark = True

print(f"torch : {torch.__version__} | device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"gpu   : {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")

try:
    import segmentation_models_pytorch as smp
    HAS_SMP = True
    print(f"smp   : {smp.__version__}")
except ImportError:
    HAS_SMP = False
    print("smp   : not installed — using built-in U-Net fallback")

import os
import pandas as pd
from pathlib import Path
from typing import Sequence, Optional

# 1. Manifest Loading Functions
def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def load_manifest(split: str) -> pd.DataFrame:
    fp = SCFG.manifest_dir / f"{split}_manifest.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Manifest not found: {fp}")
    return pd.read_csv(fp)

raw = {s: load_manifest(s) for s in ("train", "val", "test")}
PATH_COL = _resolve_column(raw["train"], SCFG.path_col_candidates)
MASK_COL = _resolve_column(raw["train"], SCFG.mask_col_candidates)
TASK_COL = _resolve_column(raw["train"], SCFG.task_col_candidates)
DIFF_COL = _resolve_column(raw["train"], SCFG.difficulty_col_candidates)

def _abs(p) -> str:
    p = str(p)
    return p if os.path.isabs(p) else str(SCFG.project_root / p)

def normalize_mask_stem(stem: str) -> str:
    s = stem.lower()
    for suf in ("_mask", "-mask", "_gt", "_label"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

MASK_INDEX: dict[str, Path] = {}
for mp in SCFG.masks_dir.glob("*.png"):
    MASK_INDEX[normalize_mask_stem(mp.stem)] = mp

def build_seg_frame(df: pd.DataFrame, split: str) -> pd.DataFrame:
    out = df.copy()
    if TASK_COL is not None:
        m = out[TASK_COL].astype(str).str.lower().str.contains(
            SCFG.segmentation_task_keyword, na=False)
        if m.any():
            out = out[m]

   
    img_512_dir = SCFG.project_root / "data_processed" / "images_512"
    out["img_path"] = out[PATH_COL].map(lambda p: str(img_512_dir / Path(p).name))

    def _mask_for(row) -> Optional[str]:
        if MASK_COL is not None and pd.notna(row.get(MASK_COL)):
            cand = Path(_abs(row[MASK_COL]))
            if cand.exists():
                return str(cand)
        hit = MASK_INDEX.get(normalize_mask_stem(Path(row["img_path"]).stem))
        return str(hit) if hit else None

    out["mask_path"] = out.apply(_mask_for, axis=1)
    out["difficulty"] = (out[DIFF_COL].astype(str).str.lower()
                         if DIFF_COL is not None else "unknown")

    out = out[out["img_path"].map(os.path.exists) & out["mask_path"].notna()]
    return out.drop_duplicates(subset=["img_path"]).reset_index(drop=True)[["img_path", "mask_path", "difficulty"]]

# 2. Build Segmentation Frames
seg = {}
for s in ("train", "val", "test"):
    seg[s] = build_seg_frame(raw[s], s)

# 3. Auto-resolve Stem Leakage
def _stems(df): 
    return {Path(p).stem.lower() for p in df["img_path"]}

tr_stems = _stems(seg["train"])
va_stems = _stems(seg["val"])

val_leak = _stems(seg["val"]) & tr_stems
test_leak = (_stems(seg["test"]) & tr_stems) | (_stems(seg["test"]) & va_stems)

if val_leak:
    seg["val"] = seg["val"][~seg["val"]["img_path"].map(lambda p: Path(p).stem.lower()).isin(val_leak)].reset_index(drop=True)

if test_leak:
    seg["test"] = seg["test"][~seg["test"]["img_path"].map(lambda p: Path(p).stem.lower()).isin(test_leak)].reset_index(drop=True)

tr, va, te = _stems(seg["train"]), _stems(seg["val"]), _stems(seg["test"])
print("==============================================================================")
print(f"✓ LEAKAGE RESOLVED — train∩val: {len(tr & va)} | train∩test: {len(tr & te)} | val∩test: {len(va & te)}")
print(f"✓ Final split counts -> train: {len(seg['train']):,} | val: {len(seg['val']):,} | test: {len(seg['test']):,}")
print("==============================================================================")

stats_rows, blank_registry = [], {}
 
for split, df in seg.items():
    pos_px = tot_px = 0
    blanks, ratios = [], []
    for mp in df["mask_path"]:
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Unreadable mask: {mp}")
        binary = (m > 127)
        p, t = int(binary.sum()), int(binary.size)
        pos_px += p
        tot_px += t
        ratios.append(p / t)
        if p == 0:
            blanks.append(str(mp))
    blank_registry[split] = blanks
    stats_rows.append({
        "split": split,
        "pairs": len(df),
        "blank_masks": len(blanks),
        "damage_pixel_ratio": pos_px / max(tot_px, 1),
        "median_ratio_nonblank": float(np.median([r for r in ratios if r > 0]))
        if any(r > 0 for r in ratios) else 0.0,
    })
 
mask_stats = pd.DataFrame(stats_rows).set_index("split").loc[["train", "val", "test"]]
mask_stats.to_csv(SCFG.artifact_dir / "mask_statistics.csv")
print("=" * 82)
print("MASK STATISTICS")
print("=" * 82)
print(mask_stats.to_string())
 
TRAIN_POS_RATIO = float(mask_stats.loc["train", "damage_pixel_ratio"])
AUTO_POS_WEIGHT = float(np.clip((1 - TRAIN_POS_RATIO) / max(TRAIN_POS_RATIO, 1e-6),
                                1.0, 20.0))
POS_WEIGHT = SCFG.bce_pos_weight if SCFG.bce_pos_weight is not None else AUTO_POS_WEIGHT
print(f"\nTrain damage pixel ratio : {TRAIN_POS_RATIO:.4%}")
print(f"BCE pos_weight (clipped) : {POS_WEIGHT:.2f}")
print(f"Blank masks — train {len(blank_registry['train'])}, "
      f"val {len(blank_registry['val'])}, test {len(blank_registry['test'])} "
      f"(kept as negatives, scored separately)")
 
if DIFF_COL is not None:
    print("\nDifficulty breakdown:")
    for split, df in seg.items():
        print(f"  {split:5s}: {df['difficulty'].value_counts().to_dict()}")
 

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_pair(img, mask, size):
    # حماية ضد الصور والماسك التالفة أو الفارغة
    if img is None or mask is None or img.size == 0 or mask.size == 0:
        img = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        return img, mask
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask


def augment_pair(img: np.ndarray, mask: np.ndarray, size: int, rng: random.Random):
    if img is None or mask is None or img.size == 0 or mask.size == 0:
        return _resize_pair(img, mask, size)

    if rng.random() < 0.7:
        h, w = img.shape[:2]
        if h > 10 and w > 10:
            scale = rng.uniform(0.65, 1.0)
            ch, cw = max(1, int(h * scale)), max(1, int(w * scale))
            y0 = rng.randint(0, max(0, h - ch))
            x0 = rng.randint(0, max(0, w - cw))
            img = img[y0:y0 + ch, x0:x0 + cw]
            mask = mask[y0:y0 + ch, x0:x0 + cw]

    img, mask = _resize_pair(img, mask, size)

    if rng.random() < 0.5:
        img, mask = np.fliplr(img), np.fliplr(mask)
    if rng.random() < 0.3:
        img, mask = np.flipud(img), np.flipud(mask)
    k = rng.choice([0, 0, 1, 2, 3])
    if k:
        img, mask = np.rot90(img, k), np.rot90(mask, k)

    if rng.random() < 0.35:
        ang = rng.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, M, (size, size), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    img = img.astype(np.float32)
    if rng.random() < 0.6:
        img = img * rng.uniform(0.75, 1.25) + rng.uniform(-22, 22)
    if rng.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.2, 1.1))
    if rng.random() < 0.2:
        img = img + np.random.normal(0, rng.uniform(2, 9), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255)

    return np.ascontiguousarray(img), np.ascontiguousarray(mask)

def to_tensor(img: np.ndarray, mask: Optional[np.ndarray] = None):
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(np.transpose(x, (2, 0, 1)).copy())
    if mask is None:
        return x
    y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
    return x, y


class ConcreteSegDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, train: bool, seed: int = 0):
        self.images = df["img_path"].tolist()
        self.masks = df["mask_path"].tolist()
        self.size = size
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        bgr = cv2.imread(self.images[i], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Unreadable image: {self.images[i]}")
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Unreadable mask: {self.masks[i]}")

        if self.train:
            rng = random.Random(self.seed * 100003 + i * 7919 + random.randint(0, 1 << 30))
            img, mask = augment_pair(img, mask, self.size, rng)
        else:
            img, mask = _resize_pair(img, mask, self.size)

        return to_tensor(img, mask)


train_ds = ConcreteSegDataset(seg["train"], SCFG.img_size, True, SCFG.seed)
val_ds = ConcreteSegDataset(seg["val"], SCFG.img_size, False)

loader_kw = dict(
    num_workers=SCFG.num_workers,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=(SCFG.num_workers > 0)
)

train_loader = DataLoader(train_ds, batch_size=SCFG.batch_size, shuffle=True,
                          drop_last=True, **loader_kw)
val_loader = DataLoader(val_ds, batch_size=SCFG.batch_size, shuffle=False,
                        drop_last=False, **loader_kw)

print(f"✓ Loaders ready | train: {len(train_ds):,} imgs ({len(train_loader):,} batches) | "
      f"val: {len(val_ds):,} imgs ({len(val_loader):,} batches)")
del raw, seg, stats_rows
import gc; gc.collect()
class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.block(x)
 
 
class ResNet34UNet(nn.Module):
    """Dependency-free U-Net with a torchvision ResNet34 encoder.
    Used only when segmentation_models_pytorch is unavailable."""
 
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights
        w = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        enc = resnet34(weights=w)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # /2   64
        self.pool = enc.maxpool                                    # /4
        self.layer1, self.layer2 = enc.layer1, enc.layer2          # /4 64, /8 128
        self.layer3, self.layer4 = enc.layer3, enc.layer4          # /16 256, /32 512
 
        self.dec4 = _DoubleConv(512 + 256, 256)
        self.dec3 = _DoubleConv(256 + 128, 128)
        self.dec2 = _DoubleConv(128 + 64, 64)
        self.dec1 = _DoubleConv(64 + 64, 32)
        self.dec0 = _DoubleConv(32, 16)
        self.head = nn.Conv2d(16, 1, 1)
 
    @staticmethod
    def _up_cat(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)
 
    def forward(self, x):
        s0 = self.stem(x)          # /2
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        d = self.dec4(self._up_cat(s4, s3))
        d = self.dec3(self._up_cat(d, s2))
        d = self.dec2(self._up_cat(d, s1))
        d = self.dec1(self._up_cat(d, s0))
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.dec0(d))
 
 
def build_seg_model() -> nn.Module:
    if HAS_SMP:
        m = smp.Unet(
            encoder_name=SCFG.encoder,
            encoder_weights=SCFG.encoder_weights,
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",   # cheap, helps thin structures
        )
        print(f"model: smp.Unet({SCFG.encoder}, scse attention)")
        return m
    print("model: built-in ResNet34UNet fallback")
    return ResNet34UNet(pretrained=True)
 
 
model = build_seg_model().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params/1e6:.2f} M")
 
 
# --- Loss --------------------------------------------------------------------
class DiceLoss(nn.Module):
    """Soft Dice on logits. `smooth` in both numerator and denominator makes an
    all-empty prediction on an all-empty target score a perfect 1.0, which is
    the behaviour we want for the blank masks."""
 
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
 
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        b = probs.shape[0]
        p = probs.view(b, -1)
        t = targets.view(b, -1)
        inter = (p * t).sum(1)
        dice = (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)
        return 1.0 - dice.mean()
 
 
class DiceBCELoss(nn.Module):
    def __init__(self, dice_w, bce_w, pos_weight):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=DEVICE))
        self.dw, self.bw = dice_w, bce_w
 
    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + self.bw * self.bce(logits, targets)
 
 
criterion = DiceBCELoss(SCFG.dice_weight, SCFG.bce_weight, POS_WEIGHT)
 
# Discriminative LRs: the pretrained encoder moves slower than the fresh decoder.
enc_params, dec_params = [], []
for n, p in model.named_parameters():
    (enc_params if ("encoder" in n or n.startswith(("stem", "layer", "pool")))
     else dec_params).append(p)
optimizer = torch.optim.AdamW(
    [{"params": enc_params, "lr": SCFG.lr * SCFG.lr_encoder_mult},
     {"params": dec_params, "lr": SCFG.lr}],
    weight_decay=SCFG.weight_decay)
print(f"encoder params: {len(enc_params)} groups | decoder params: {len(dec_params)}")
 
steps_per_epoch = math.ceil(len(train_loader) / SCFG.accum_steps)
total_steps = steps_per_epoch * SCFG.epochs
warmup_steps = steps_per_epoch * SCFG.warmup_epochs
 
 
def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
 
 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda", enabled=(SCFG.amp and DEVICE.type == "cuda"))
 

def batch_metrics(probs: torch.Tensor, targets: torch.Tensor, thr: float):
    """Returns per-sample (dice, iou, gt_is_empty) as numpy arrays."""
    preds = (probs >= thr).float()
    b = preds.shape[0]
    p = preds.view(b, -1)
    t = targets.view(b, -1)
    inter = (p * t).sum(1)
    psum, tsum = p.sum(1), t.sum(1)
    union = psum + tsum - inter
 
    dice = torch.where(tsum + psum == 0,
                       torch.ones_like(inter),
                       2 * inter / (psum + tsum + 1e-7))
    iou = torch.where(union == 0, torch.ones_like(inter), inter / (union + 1e-7))
    empty = (tsum == 0)
    return (dice.detach().cpu().numpy(),
            iou.detach().cpu().numpy(),
            empty.detach().cpu().numpy())
 
 
@torch.no_grad()
def evaluate(model, loader, criterion, thr: float = 0.5) -> dict:
    model.eval()
    losses, dices, ious, empties = [], [], [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(SCFG.amp and DEVICE.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)
        probs = torch.sigmoid(logits.float())
        d, i, e = batch_metrics(probs, y, thr)
        losses.append(loss.item() * x.size(0))
        dices.append(d); ious.append(i); empties.append(e)
 
    d = np.concatenate(dices); i = np.concatenate(ious); e = np.concatenate(empties)
    nonempty = ~e
    return {
        "loss": float(np.sum(losses) / max(len(d), 1)),
        "dice": float(d[nonempty].mean()) if nonempty.any() else float("nan"),
        "iou": float(i[nonempty].mean()) if nonempty.any() else float("nan"),
        "dice_all": float(d.mean()),
        "iou_all": float(i.mean()),
        "empty_mask_accuracy": float(d[e].mean()) if e.any() else float("nan"),
        "n_empty": int(e.sum()),
        "n_nonempty": int(nonempty.sum()),
        "threshold": thr,
        "_per_sample": {"dice": d, "iou": i, "empty": e},
    }
 
 
def train_one_epoch(epoch: int) -> dict:
    model.train()
    running, seen = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(train_loader):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(SCFG.amp and DEVICE.type == "cuda")):
            logits = model(x)
            loss = criterion(logits, y) / SCFG.accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % SCFG.accum_steps == 0:
            if SCFG.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), SCFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running += loss.item() * SCFG.accum_steps * x.size(0)
        seen += x.size(0)
        
        # Periodic RAM cleanup to prevent memory leaks on Windows
        if step % 200 == 0:
            gc.collect()
            if step % 400 == 0:
                print(f"   ep{epoch:02d} {step:4d}/{len(train_loader)} "
                      f"loss {running/max(seen,1):.4f} "
                      f"lr {optimizer.param_groups[-1]['lr']:.2e}", flush=True)

    return {"loss": running / max(seen, 1), "secs": time.time() - t0}
 

torch.cuda.empty_cache(); gc.collect()
 
BEST_PTH = SCFG.ckpt_dir / "best_unet.pth"
history, best_dice, best_epoch, patience = [], -np.inf, -1, 0
 
print("=" * 82)
print(f"TRAINING U-Net/{SCFG.encoder} @ {SCFG.img_size}px, batch {SCFG.batch_size} "
      f"x{SCFG.accum_steps} accum — monitoring val Dice (non-empty)")
print("=" * 82)
 
for ep in range(1, SCFG.epochs + 1):
    tr = train_one_epoch(ep)
    va = evaluate(model, val_loader, criterion, SCFG.eval_threshold)
 
    history.append({"epoch": ep, "train_loss": tr["loss"],
                    **{k: v for k, v in va.items() if not k.startswith("_")}})
    print(f"[ep {ep:03d}/{SCFG.epochs}] train {tr['loss']:.4f} | "
          f"val {va['loss']:.4f} dice {va['dice']:.4f} iou {va['iou']:.4f} "
          f"empty-acc {va['empty_mask_accuracy']:.3f} | {tr['secs']:.0f}s")
 
    if va["dice"] > best_dice:
        best_dice, best_epoch, patience = va["dice"], ep, 0
        torch.save({
            "epoch": ep,
            "encoder": SCFG.encoder,
            "img_size": SCFG.img_size,
            "used_smp": HAS_SMP,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_metrics": {k: v for k, v in va.items() if not k.startswith("_")},
            "config": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in asdict(SCFG).items()},
            "normalization": {"mean": IMAGENET_MEAN.tolist(),
                              "std": IMAGENET_STD.tolist()},
        }, BEST_PTH)
        print(f"    ✓ new best val Dice {best_dice:.4f} → {BEST_PTH.name}")
    else:
        patience += 1
        if patience >= SCFG.early_stopping_patience:
            print(f"    ⏹ early stop at epoch {ep} "
                  f"(best Dice {best_dice:.4f} @ ep{best_epoch})")
            break
 
hist = pd.DataFrame(history)
hist.to_csv(SCFG.artifact_dir / "training_history.csv", index=False)
print(f"\nBest val Dice {best_dice:.4f} @ epoch {best_epoch}")
 