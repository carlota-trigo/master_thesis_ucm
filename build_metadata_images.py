#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Construye metadata_complete.csv combinando metadata.csv con métricas de calidad de imagen.

- Lee metadata de --metadata (por defecto data_clean/metadata.csv)
- Indexa imágenes en --images-dir (por defecto data_clean/images)
- Resuelve image_id -> ruta local (tolerando ausencia de extensión y subcarpetas)
- Calcula métricas (brillo, contraste, blur, bordes negros, etc.)
- Mergea y escribe --out (por defecto data_clean/metadata_complete.csv)
- Escribe errores (si los hay) en data_clean/qc_errors.csv

Dependencias:
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
        "No encuentro la columna con los nombres de imagen. "
        f"Intenta renombrar a 'image_id'. Columnas disponibles: {list(df.columns)}"
    )


def index_images(images_dir: Path):
    """Devuelve (all_paths, filename_to_path, stem_to_paths)"""
    all_paths = []
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            all_paths.append(p.resolve())

    filename_to_path = {}          # 'foo.jpg' -> Path
    stem_to_paths = {}             # 'foo'     -> [Path, ...]

    for p in all_paths:
        name = p.name
        stem = p.stem
        filename_to_path[name] = p
        stem_to_paths.setdefault(stem, []).append(p)

    return all_paths, filename_to_path, stem_to_paths


def resolve_path(image_id: str, images_dir: Path, filename_to_path, stem_to_paths):
    """
    Resuelve un 'image_id' a una ruta Path local, intentando:
      1) Ruta directa (si 'image_id' trae subcarpetas relativas a images_dir)
      2) Match exacto por nombre de archivo
      3) Match por stem (sin extensión)
      4) Probar extensiones comunes
    """
    s = str(image_id).strip().replace("\\", "/")
    base = os.path.basename(s)

    # 1) si trae subcarpetas relativas al prefijo, prueba directo
    cand = (images_dir / s).resolve()
    if cand.exists():
        return cand

    # 2) match exacto por nombre de archivo
    if base in filename_to_path:
        return filename_to_path[base]

    # 3) match por 'stem' (sin extensión)
    stem, ext = os.path.splitext(base)
    if ext == "" and stem in stem_to_paths:
        # si hay varias, elige la primera
        return stem_to_paths[stem][0]

    # 4) probar extensiones comunes
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
        # Read and open (respect EXIF orientation), force RGB
        with open(path, "rb") as f:
            data = f.read()
        im = Image.open(BytesIO(data)).convert("RGB")
        im = ImageOps.exif_transpose(im)

        w, h = im.size
        arr = np.array(im, dtype=np.uint8)  # H x W x 3

        # ---- RGB channel statistics ----
        Rc = arr[..., 0].astype(np.float32)
        Gc = arr[..., 1].astype(np.float32)
        Bc = arr[..., 2].astype(np.float32)

        r_mean = float(Rc.mean()); g_mean = float(Gc.mean()); b_mean = float(Bc.mean())
        r_std  = float(Rc.std());  g_std  = float(Gc.std());  b_std  = float(Bc.std())

        # ---- Grayscale + core QC metrics ----
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        brightness = float(gray.mean())
        blur_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ---- Hue entropy (HSV-H, 180 bins) ----
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        Hc = hsv[..., 0]
        hhist_density, _ = np.histogram(Hc, bins=180, range=(0, 180), density=True)
        hhist_density = hhist_density + 1e-12
        hue_entropy = float(-(hhist_density * np.log2(hhist_density)).sum())

        # ---- Hair ratio (black-hat morphology + dark threshold) ----
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
        description="Construye metadata_images.csv localmente (métricas básicas de color/iluminación)."
    )
    parser.add_argument("--images-dir", type=str, default="data_clean/images",
                        help="Directorio raíz con las imágenes (recursivo).")
    parser.add_argument("--out", type=str, default="data_clean/metadata_images.csv",
                        help="Ruta de salida del CSV de métricas por imagen.")
    parser.add_argument("--errors-out", type=str, default=None,
                        help="CSV para errores (por defecto: <images-dir>/../qc_errors.csv).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Número de hilos para procesar imágenes.")
    parser.add_argument("--limit", type=int, default=None, help="Procesar solo N imágenes (debug).")
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    out_csv = Path(args.out).resolve()
    errors_csv = (
        Path(args.errors_out).resolve()
        if args.errors_out else
        (images_dir.parent / "qc_errors.csv").resolve()
    )

    if not images_dir.exists():
        print(f"[ERROR] No existe el directorio de imágenes: {images_dir}", file=sys.stderr)
        sys.exit(1)

    # Indexar imágenes
    print(f"Indexando imágenes en: {images_dir} (recursivo)")
    all_paths, _, _ = index_images(images_dir)
    print(f"Imágenes encontradas: {len(all_paths)}")
    if not all_paths:
        print("[ERROR] No se encontraron imágenes en el directorio indicado.", file=sys.stderr)
        sys.exit(2)

    # Preparar lista a procesar
    paths_to_process = all_paths
    if args.limit is not None:
        paths_to_process = paths_to_process[:args.limit]
        print(f"[DEBUG] limit={args.limit} → procesando {len(paths_to_process)} imágenes.")

    # Procesar en paralelo
    rows, errs = [], []
    print(f"Calculando métricas con {args.workers} hilos…")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(qc_metrics_local, p): p for p in paths_to_process}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            p = futures[fut]  # Path of this future
            try:
                res = fut.result()
            except Exception as e:
                # Por si alguna excepción se escapa del qc_metrics_local
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
                # Asegurar claves de identificación en cada fila
                row = dict(res)
                row["image_id"] = image_id_str
                row["image_path"] = image_path_str
                rows.append(row)

    print(f"Métricas OK: {len(rows)}   |   Errores: {len(errs)}")
    Q = pd.DataFrame(rows)

    # Añadir ruta relativa (útil si hay nombres repetidos)
    if "image_path" in Q.columns:
        try:
            Q["image_relpath"] = Q["image_path"].apply(lambda s: os.path.relpath(str(s), str(images_dir)))
        except Exception:
            Q["image_relpath"] = Q["image_path"]

    # Guardar salidas
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    Q.to_csv(out_csv, index=False)
    print(f"[OK] Escrito: {out_csv}")

    if errs:
        errors_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errs).to_csv(errors_csv, index=False)
        print(f"[OK] Errores guardados en: {errors_csv}")

    # Resumen rápido (adaptado a las nuevas métricas)
    cols_metrics = [
        "width", "height", "brightness", "blur_var",
        "hue_entropy", "hair_ratio",
        "r_mean", "g_mean", "b_mean",
        "r_std", "g_std", "b_std",
    ]
    have_metrics = [c for c in cols_metrics if c in Q.columns]
    print("\nResumen rápido (primeras 5 filas con métricas disponibles):")
    cols_to_show = ["image_id"]
    if "image_relpath" in Q.columns:
        cols_to_show.append("image_relpath")
    cols_to_show += have_metrics
    print(Q[cols_to_show].head().to_string(index=False))

if __name__ == "__main__":
    main()

