#!/usr/bin/env python3

"""
Builds metadata_complete.csv by combining metadata.csv with image quality metrics.

- Reads metadata from --metadata (default: data_clean/metadata.csv)
- Indexes images in --images-dir (default: data_clean/images)
- Resolves image_id -> local path (tolerating missing extensions and subfolders)
- Calculates metrics (brightness, contrast, blur, black borders, etc.)
- Merges and writes --out (default: data_clean/metadata_complete.csv)
- Writes errors (if any) to data_clean/qc_errors.csv

Dependencies:
    pip install pandas numpy pillow imagehash opencv-python-headless tqdm
"""

import argparse
import os
import sys
import hashlib
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
import imagehash
from tqdm import tqdm

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def find_image_column(df: pd.DataFrame) -> str:
    candidates = ["image_id", "image", "filename", "file", "img", "name"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find column with image names. "
        f"Try renaming to 'image_id'. Available columns: {list(df.columns)}"
    )

def index_images(images_dir: Path):
    """Returns (all_paths, filename_to_path, stem_to_paths)"""
    all_paths = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            all_paths.append(p.resolve())

    filename_to_path = {}              stem_to_paths = {}             
    for p in all_paths:
        name = p.name
        stem = p.stem
        filename_to_path[name] = p
        stem_to_paths.setdefault(stem, []).append(p)

    return all_paths, filename_to_path, stem_to_paths

def resolve_path(image_id: str, images_dir: Path, filename_to_path, stem_to_paths):
    """
    Resolves an 'image_id' to a local Path, trying:
      1) Direct path (if 'image_id' contains subfolders relative to images_dir)
      2) Exact match by filename
      3) Match by stem (without extension)
      4) Try common extensions
    """
    s = str(image_id).strip().replace("\\", "/")
    base = os.path.basename(s)

    cand = (images_dir / s).resolve()
    if cand.exists():
        return cand

    if base in filename_to_path:
        return filename_to_path[base]

    stem, ext = os.path.splitext(base)
    if ext == "" and stem in stem_to_paths:
            return stem_to_paths[stem][0]

    for e in EXTS:
        cand2 = (images_dir / (base + e)).resolve()
        if cand2.exists():
            return cand2

    return None

def qc_metrics_local(path: Path):
    """
    Minimal QC metrics (no histograms):
      - width, height
      - brightness (mean grayscale, 0–255)
      - blur_var (variance of Laplacian)
      - hue_entropy (HSV-H histogram entropy, 180 bins)
      - hair_ratio (black-hat hair detector)
      - RGB stats: per-channel mean and std (0–255)

    Returns a dict with ONLY these fields (or {'error': ...} on failure).
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
        im = Image.open(BytesIO(data)).convert("RGB")
        im = ImageOps.exif_transpose(im)

        w, h = im.size
        arr = np.array(im, dtype=np.uint8)  
        Rc = arr[..., 0].astype(np.float32)
        Gc = arr[..., 1].astype(np.float32)
        Bc = arr[..., 2].astype(np.float32)

        r_mean = float(Rc.mean()); g_mean = float(Gc.mean()); b_mean = float(Bc.mean())
        r_std  = float(Rc.std());  g_std  = float(Gc.std());  b_std  = float(Bc.std())

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        brightness = float(gray.mean())
        blur_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        Hc = hsv[..., 0]
        hhist_density, _ = np.histogram(Hc, bins=180, range=(0, 180), density=True)
        hhist_density = hhist_density + 1e-12
        hue_entropy = float(-(hhist_density * np.log2(hhist_density)).sum())

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        hair_mask = (blackhat > 10) & (gray < 90)
        hair_ratio = float(hair_mask.mean())

        return dict(
            width=w,
            height=h,
            brightness=brightness,
            blur_var=blur_var,
            hue_entropy=hue_entropy,
            hair_ratio=hair_ratio,
            r_mean=r_mean, g_mean=g_mean, b_mean=b_mean,
            r_std=r_std,   g_std=g_std,   b_std=b_std,
        )

    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(
        description="Build metadata_images.csv locally (basic color/lighting metrics)."
    )
    parser.add_argument("--images-dir", type=str, default="data_clean/images",
                        help="Root directory with images (recursive).")
    parser.add_argument("--out", type=str, default="data_clean/metadata_images.csv",
                        help="Output CSV path for metrics per image.")
    parser.add_argument("--errors-out", type=str, default=None,
                        help="CSV for errors (default: <images-dir>/../qc_errors.csv).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Number of threads for processing images.")
    parser.add_argument("--limit", type=int, default=None, help="Process only N images (debug).")
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    out_csv = Path(args.out).resolve()
    errors_csv = (
        Path(args.errors_out).resolve()
        if args.errors_out else
        (images_dir.parent / "qc_errors.csv").resolve()
    )

    if not images_dir.exists():
        print(f"[ERROR] Images directory does not exist: {images_dir}", file=sys.stderr)
        sys.exit(1)

        print(f"Indexing images in: {images_dir} (recursive)")
    all_paths, _, _ = index_images(images_dir)
    print(f"Images found: {len(all_paths)}")
    if not all_paths:
        print("[ERROR] No images found in the specified directory.", file=sys.stderr)
        sys.exit(2)

        paths_to_process = all_paths
    if args.limit is not None:
        paths_to_process = paths_to_process[:args.limit]
        print(f"[DEBUG] limit={args.limit} → processing {len(paths_to_process)} images.")

        rows, errs = [], []
    print(f"Calculating metrics with {args.workers} threads…")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(qc_metrics_local, p): p for p in paths_to_process}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            p = futures[fut]              
            try:
                res = fut.result()
            except Exception as e:
                res = {"error": str(e)}

            image_path_str = str(p)
            image_id_str = p.name

            if isinstance(res, dict) and res.get("error"):
                errs.append({
                    "image_id": image_id_str,
                    "image_path": image_path_str,
                    "error": res["error"]
                })
            else:
                row = dict(res)
                row["image_id"] = image_id_str
                row["image_path"] = image_path_str
                rows.append(row)

    print(f"Metrics OK: {len(rows)}   |   Errors: {len(errs)}")
    Q = pd.DataFrame(rows)

    if "image_path" in Q.columns:
        try:
            Q["image_relpath"] = Q["image_path"].apply(lambda s: os.path.relpath(str(s), str(images_dir)))
        except Exception:
            Q["image_relpath"] = Q["image_path"]

        out_csv.parent.mkdir(parents=True, exist_ok=True)
    Q.to_csv(out_csv, index=False)
    print(f"[OK] Written: {out_csv}")

    if errs:
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errs).to_csv(errors_csv, index=False)
        print(f"[OK] Errors saved to: {errors_csv}")

        cols_metrics = [
        "width", "height", "brightness", "blur_var",
        "hue_entropy", "hair_ratio",
        "r_mean", "g_mean", "b_mean",
        "r_std", "g_std", "b_std",
    ]
    have_metrics = [c for c in cols_metrics if c in Q.columns]
    print("\nQuick summary (first 5 rows with available metrics):")
    cols_to_show = ["image_id"]
    if "image_relpath" in Q.columns:
        cols_to_show.append("image_relpath")
    cols_to_show += have_metrics
    print(Q[cols_to_show].head().to_string(index=False))

if __name__ == "__main__":
    main()

