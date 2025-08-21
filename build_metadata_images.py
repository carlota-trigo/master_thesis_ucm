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
    Calcula métricas de calidad (brillo, contraste, blur, bordes, vignetting, etc.) desde una ruta local.
    Devuelve dict con métricas o dict con 'error'.
    """
    try:
        size_bytes = path.stat().st_size
        with open(path, "rb") as f:
            data = f.read()
        sha256 = hashlib.sha256(data).hexdigest()

        im = Image.open(BytesIO(data)).convert("RGB")
        im = ImageOps.exif_transpose(im)  # respeta la orientación EXIF

        w, h = im.size
        arr = np.array(im)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Métricas básicas
        brightness = float(gray.mean())
        contrast   = float(gray.std())
        blur_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark_ratio   = float((gray < 30).mean())
        bright_ratio = float((gray > 225).mean())
        aspect = float(w / h) if h else np.nan

        # Bordes negros (tira de 5 px en bordes, adaptado a imágenes muy pequeñas)
        bs = 5 if min(w, h) >= 20 else max(1, min(w, h)//4)
        border = np.concatenate([
            gray[:bs, :].ravel(), gray[-bs:, :].ravel(),
            gray[:, :bs].ravel(), gray[:, -bs:].ravel()
        ])
        black_border_ratio = float((border < 10).mean()) if border.size > 0 else 0.0

        # Vigneteado (centro - esquinas)
        c_h, c_w = h // 2, w // 2
        ch, cw = max(1, min(64, h)), max(1, min(64, w))
        center = gray[max(0, c_h - ch//4):min(h, c_h + ch//4),
                      max(0, c_w - cw//4):min(w, c_w + cw//4)]
        corner_sz = max(8, min(32, min(w, h)//6))
        corners = np.concatenate([
            gray[:corner_sz, :corner_sz].ravel(),
            gray[:corner_sz, -corner_sz:].ravel(),
            gray[-corner_sz:, :corner_sz].ravel(),
            gray[-corner_sz:, -corner_sz:].ravel(),
        ])
        vignette_delta = float(center.mean() - corners.mean()) if corners.size > 0 and center.size > 0 else 0.0

        # Heurística líneas largas (reglas, marcas)
        edges = cv2.Canny(gray, 80, 200)
        min_len = max(10, int(0.5 * min(w, h)))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=min_len, maxLineGap=10)
        has_long_line = int(lines is not None and len(lines) > 0)

        # Marcadores (azules / verdes)
        b, g, r = cv2.split(arr)
        bi = b.astype(np.int32); gi = g.astype(np.int32); ri = r.astype(np.int32)
        bluish_mask   = (bi - ri > 40) & (bi - gi > 20) & (b > 120)
        greenish_mask = (gi - ri > 30) & (g > 120)
        marker_ratio = float((bluish_mask | greenish_mask).mean())

        # Perceptual hashes
        ahash = imagehash.average_hash(im).__str__()
        phash = imagehash.phash(im).__str__()

                # ===== Nitidez =====
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        tenengrad = float(np.mean(gx*gx + gy*gy))

        shift2 = gray[2:, :].astype(np.float32) - gray[:-2, :].astype(np.float32)
        brenner = float(np.mean(shift2**2))

        # Alta frecuencia (FFT): energía fuera de un radio
        f = np.fft.fft2(gray)
        mag = np.abs(np.fft.fftshift(f))
        Y, X = np.ogrid[:h, :w]
        cy, cx = h//2, w//2
        R = np.sqrt((Y-cy)**2 + (X-cx)**2)
        r0 = 0.25 * min(h, w)
        hf_energy_ratio = float(mag[R >= r0].sum() / (mag.sum() + 1e-8))

        # ===== Ruido / compresión =====
        hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
        hist = hist + 1e-12
        entropy = float(-(hist*np.log2(hist)).sum())
        p1, p5, p95, p99 = np.percentile(gray, [1, 5, 95, 99])
        dyn_range = float(p99 - p1)
        midtone_span = float(p95 - p5)

        def _jpeg_blockiness(img_gray):
            H, W = img_gray.shape
            v_b = [np.abs(img_gray[:, x] - img_gray[:, x-1]).mean() for x in range(8, W, 8)]
            h_b = [np.abs(img_gray[y, :] - img_gray[y-1, :]).mean() for y in range(8, H, 8)]
            v_w = [np.abs(img_gray[:, x+4] - img_gray[:, x+3]).mean() for x in range(0, W-8, 8)]
            h_w = [np.abs(img_gray[y+4, :] - img_gray[y+3, :]).mean() for y in range(0, H-8, 8)]
            if not (v_b and h_b and v_w and h_w):
                return 0.0
            return float(max(0.0, (np.mean(v_b + h_b) - np.mean(v_w + h_w))))
        jpeg_blockiness = _jpeg_blockiness(gray)

        # ===== Color / exposición =====
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        Hc, Sc, Vc = cv2.split(hsv)
        sat_mean = float(Sc.mean())
        sat_std  = float(Sc.std())
        low_sat_ratio  = float((Sc < 20).mean())
        high_sat_ratio = float((Sc > 230).mean())
        # reflejos especulares: muy brillante y baja saturación
        specular_ratio = float(((Vc > 245) & (Sc < 30)).mean())
        # entropía de tono
        hhist, _ = np.histogram(Hc, bins=180, range=(0, 180), density=True)
        hhist = hhist + 1e-12
        hue_entropy = float(-(hhist*np.log2(hhist)).sum())

        Bc, Gc, Rc = cv2.split(arr)
        # Hasler–Süsstrunk colorfulness
        rg = Rc.astype(np.float32) - Gc.astype(np.float32)
        yb = 0.5*(Rc.astype(np.float32) + Gc.astype(np.float32)) - Bc.astype(np.float32)
        std_rg, std_yb = rg.std(), yb.std()
        mean_rg, mean_yb = rg.mean(), yb.mean()
        colorfulness = float(np.sqrt(std_rg**2 + std_yb**2) + 0.3*np.sqrt(mean_rg**2 + mean_yb**2))
        # Gray-world deviation
        mean_rgb = np.array([Rc.mean(), Gc.mean(), Bc.mean()], dtype=np.float64)
        gw_target = mean_rgb.mean()
        grayworld_deviation = float(np.std(mean_rgb / (gw_target + 1e-8)))

        # Uniformidad de iluminación: gris muy suavizado
        blur_l = cv2.GaussianBlur(gray, (0, 0), sigmaX=15, sigmaY=15)
        illum_uniformity = float(blur_l.std() / (blur_l.mean() + 1e-8))

        # ===== Artefactos =====
        # Pelo (black-hat morfológico)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        hair_mask = (blackhat > 10) & (gray < 90)
        hair_ratio = float(hair_mask.mean())

        # Anillo dermatoscópico: borde oscuro vs área central
        bw = max(3, int(0.08 * min(w, h)))
        border_mask = np.zeros_like(gray, dtype=bool)
        border_mask[:bw, :] = True; border_mask[-bw:, :] = True
        border_mask[:, :bw] = True; border_mask[:, -bw:] = True
        center_w = max(1, int(0.30 * min(w, h)))
        center_mask = np.zeros_like(gray, dtype=bool)
        cy, cx = h//2, w//2
        y0, y1 = max(0, cy - center_w//2), min(h, cy + center_w//2)
        x0, x1 = max(0, cx - center_w//2), min(w, cx + center_w//2)
        center_mask[y0:y1, x0:x1] = True
        border_mean = float(gray[border_mask].mean()) if border_mask.any() else 0.0
        center_mean = float(gray[center_mask].mean()) if center_mask.any() else 0.0
        dermoscopy_ring_score = float((center_mean - border_mean) / (center_mean + 1e-8))

        return dict(
            image_path=str(path),
            image_id=path.name,
            width=w, height=h, aspect=aspect, size_bytes=size_bytes,
            brightness=brightness, contrast=contrast, blur_var=blur_var,
            dark_ratio=dark_ratio, bright_ratio=bright_ratio,
            black_border_ratio=black_border_ratio, vignette_delta=vignette_delta,
            sha256=sha256, ahash=ahash, phash=phash,
            has_long_line=has_long_line, marker_ratio=marker_ratio,
            hf_energy_ratio=hf_energy_ratio,
            entropy=entropy,
            dyn_range=dyn_range,
            midtone_span=midtone_span,
            jpeg_blockiness=jpeg_blockiness,
            sat_mean=sat_mean, sat_std=sat_std,
            low_sat_ratio=low_sat_ratio, high_sat_ratio=high_sat_ratio,
            specular_ratio=specular_ratio,
            hue_entropy=hue_entropy,
            colorfulness=colorfulness,
            grayworld_deviation=grayworld_deviation,
            illum_uniformity=illum_uniformity,
            hair_ratio=hair_ratio,
            dermoscopy_ring_score=dermoscopy_ring_score
        )
    except Exception as e:
        return {"image_path": str(path), "error": str(e), "image_id": path.name}


def main():
    parser = argparse.ArgumentParser(description="Construye metadata_images.csv localmente (solo métricas de imagen).")
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
            res = fut.result()
            if res.get("error"):
                errs.append({"image_id": res.get("image_id"), "image_path": res.get("image_path"), "error": res["error"]})
            else:
                rows.append(res)

    print(f"Métricas OK: {len(rows)}   |   Errores: {len(errs)}")
    Q = pd.DataFrame(rows)

    # Asegurar columnas clave y añadir ruta relativa para evitar ambigüedades por nombres repetidos
    if "image_id" not in Q.columns and "image_path" in Q.columns:
        Q["image_id"] = Q["image_path"].apply(lambda s: Path(str(s)).name)
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

    # Resumen rápido
    cols_metrics = ["width","height","size_bytes","brightness","contrast","blur_var",
                    "dark_ratio","bright_ratio","black_border_ratio","vignette_delta",
                    "has_long_line","marker_ratio"]
    have_metrics = [c for c in cols_metrics if c in Q.columns]
    print("\nResumen rápido (primeras 5 filas con métricas disponibles):")
    print(Q[ ['image_id'] + (['image_relpath'] if 'image_relpath' in Q.columns else []) + have_metrics ].head().to_string(index=False))


if __name__ == "__main__":
    main()

