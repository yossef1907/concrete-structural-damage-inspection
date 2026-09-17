import os
import cv2
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = Path(r"D:\nti_project")
MANIFEST_DIR = PROJECT_ROOT / "data_processed" / "manifests"
MASKS_DIR = PROJECT_ROOT / "data_processed" / "masks_binary"

OUT_IMG_DIR = PROJECT_ROOT / "data_processed" / "images_512"
OUT_MASK_DIR = PROJECT_ROOT / "data_processed" / "masks_512"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

def process_image(img_path):
    out_path = OUT_IMG_DIR / Path(img_path).name
    if not out_path.exists():
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(out_path), img)

def process_mask(mask_path):
    out_path = OUT_MASK_DIR / Path(mask_path).name
    if not out_path.exists():
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_path), mask)

if __name__ == "__main__":
    print("🚀 Starting offline resizing to 512x512...")
    
    # Process all binary masks
    all_masks = list(MASKS_DIR.glob("*.png"))
    print(f"Processing {len(all_masks):,} masks...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(process_mask, [str(m) for m in all_masks]))
        
    # Process manifest images
    for split in ("train", "val", "test"):
        manifest_fp = MANIFEST_DIR / f"{split}_manifest.csv"
        if manifest_fp.exists():
            df = pd.read_csv(manifest_fp)
            col = [c for c in df.columns if c.lower() in ("image_path", "filepath", "path", "abs_path")][0]
            imgs = [p if os.path.isabs(p) else str(PROJECT_ROOT / p) for p in df[col]]
            print(f"Processing {len(imgs):,} images from {split} split...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(process_image, imgs))

    print("✅ Pre-processing complete! New dataset saved in images_512 and masks_512.")