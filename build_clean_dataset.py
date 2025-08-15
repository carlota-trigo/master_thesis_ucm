#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Builds data_clean/ from datasets in data_raw/.

- Deduplicates images by file content hash (cross-dataset)
- Applies ISIC2020 near-duplicate suppression via ISIC_2020_Duplicates.csv (keeps one canonical image per duplicate group)
- For MIL10K IL inputs, keeps only the further-away view per lesion (heuristic-based filtering)
- Copies unique files to data_clean/images/
- Normalizes metadata fields and standardizes diagnosis codes across datasets
- Deterministic processing order for reproducibility
- Safe filename collision handling, including against existing files from previous runs

Writes data_clean/metadata.csv with columns:
    image_id, origin_dataset, lesion, diagnosis, localization, age, sex

Origin dataset priority when identical files appear across datasets (same content hash):
    HAM1000 → ITOBOS2024 → MIL10K → ISIC2020 → ISIC2019

Usage examples:
    # Full processing (default)
    python build_clean_dataset.py
    
    # Only process images (skip metadata generation)
    python build_clean_dataset.py --images_only
    
    # Only regenerate metadata (skip image processing)
    python build_clean_dataset.py --metadata_only
    
    # Skip deduplication (assume images already exist)
    python build_clean_dataset.py --skip_dedup
"""

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -------------------- Utilities --------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def is_image(fname: str) -> bool:
    return os.path.splitext(fname.lower())[1] in IMG_EXTS

def file_hash(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_unique_name(target_dir: str, desired_name: str) -> str:
    base, ext = os.path.splitext(desired_name)
    i = 1
    final = desired_name
    while os.path.exists(os.path.join(target_dir, final)):
        i = 1
        final = f"{base}__v{i}{ext}"
    return final

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

# -------------------- Normalization helpers --------------------

def normalize_sex(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return "not_provided"
    s = str(value).strip().lower()
    if s in {"m", "male", "man"}:
        return "male"
    if s in {"f", "female", "woman"}:
        return "female"
    return "not_provided"  # unknown / other

def normalize_age(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return "not_provided"
    try:
        # Handle strings like "70.0" or numeric floats
        f = float(value)
        if pd.isna(f):
            return "not_provided"
        i = int(round(f))
        if i < 0 or i > 120:
            return "not_provided"  # discard implausible
        return str(i)
    except Exception:
        s = str(value).strip()
        return s if s else "not_provided"

def normalize_localization(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == "":
        return "not_provided"
    return str(value).strip().lower()

# -------------------- ITOBOS helpers --------------------

def has_valid_bbox(ann) -> bool:
    bbox = ann.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        _, _, w, h = bbox
        return float(w) > 0 and float(h) > 0
    except Exception:
        return False

def itobos_healthy_filenames(raw_dir: str) -> Set[str]:
    """
    Return basenames of ITOBOS train images that have 0 valid bboxes.
    IMPORTANT: normalize to basenames to match disk filenames.
    """
    labels_json = os.path.join(raw_dir, "ITOBOS2024", "train", "labels.json")
    if not os.path.isfile(labels_json):
        print("[INFO] ITOBOS: train/labels.json not found → no ITOBOS images will be copied.")
        return set()
    with open(labels_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    images = {img["id"]: img["file_name"] for img in data.get("images", [])}
    counts = {iid: 0 for iid in images}
    for ann in data.get("annotations", []):
        iid = ann.get("image_id")
        if iid in counts and has_valid_bbox(ann):
            counts[iid] = 1
    # Normalize to basenames
    healthy_basenames = {os.path.basename(images[iid]) for iid, c in counts.items() if c == 0}
    return healthy_basenames

# Unified diagnosis code set (uppercase codes)
CANON_DIAG_CODES: Set[str] = {
    # malignant
    "MEL", "BCC", "AKIEC", "SCC", "SCCKA", "MAL_OTH",
    # benign
    "NV", "BKL", "DF", "VASC", "BEN_OTH", "INF",
    # other
    "UNKNOWN", "NO_LESION",
}

# Map HAM1000 codes (lowercase) to canonical codes
HAM_TO_CANON = {
    "akiec": "AKIEC",
    "bcc": "BCC",
    "mel": "MEL",
    "bkl": "BKL",
    "df": "DF",
    "nv": "NV",
    "vasc": "VASC",
    "unknown": "UNKNOWN",
}

# Map ISIC2019 codes to canonical (AK -> AKIEC; keep SCC)
ISIC2019_TO_CANON = {
    "MEL": "MEL",
    "NV": "NV",
    "BCC": "BCC",
    "AK": "AKIEC",
    "BKL": "BKL",
    "DF": "DF",
    "VASC": "VASC",
    "SCC": "SCC",
    "UNK": "UNKNOWN",
}

# Map ISIC2020 diagnosis strings (lower) to canonical codes
ISIC2020_TO_CANON = {
    "melanoma": "MEL",
    "nevus": "NV",
    "seborrheic keratosis": "BKL",
    "solar lentigo": "BKL",
    "lentigo nos": "BKL",
    "lichenoid keratosis": "BKL",
    "blue nevus": "NV",  # if appears
    "cafe-au-lait macule": "BEN_OTH",
    "atypical melanocytic proliferation": "MAL_OTH",
    "unknown": "UNKNOWN",
}

# -------------------- ISIC2020 duplicate handling --------------------

def build_isic2020_duplicate_keep_set(raw_dir: str) -> Optional[Set[str]]:
    """
    Read ISIC_2020_Duplicates.csv and return a set of canonical stems to keep.
    If file is missing, return None to indicate no filtering.
    """
    dup_csv = os.path.join(raw_dir, "ISIC2020", "ISIC_2020_Duplicates.csv")
    if not os.path.isfile(dup_csv):
        return None

    # Union-find (disjoint sets) over image stems
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # simple lexicographic rank by id
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    df = pd.read_csv(dup_csv)
    col1 = "image_name_1" if "image_name_1" in df.columns else df.columns[0]
    col2 = "image_name_2" if "image_name_2" in df.columns else df.columns[1]
    for _, r in df.iterrows():
        a = os.path.splitext(str(r.get(col1, "")).strip())[0]
        b = os.path.splitext(str(r.get(col2, "")).strip())[0]
        if a and b:
            union(a, b)

    # Choose canonical representative per set: the lexicographically smallest id
    roots: Dict[str, List[str]] = defaultdict(list)
    for x in list(parent.keys()):
        roots[find(x)].append(x)

    keep: Set[str] = set()
    for _, members in roots.items():
        members.sort()
        keep.add(members[0])
    return keep

# -------------------- MIL10K IL helpers --------------------

def extract_il_id(abs_path: str, fallback_stem: str) -> Optional[str]:
    """
    Robustly extract IL_XXXX id from path components or filename.
    Returns uppercase IL id if found, else None.
    """
    # Look for IL_ pattern anywhere in the path
    m = re.search(r"IL_\d", abs_path, flags=re.IGNORECASE)
    if m:
        return m.group(0).upper()
    # Check components explicitly
    for part in abs_path.split(os.sep):
        if part.lower().startswith("il_"):
            return part.upper()
    # Fallback: try the filename stem if it matches the pattern
    if re.fullmatch(r"IL_\d", fallback_stem, flags=re.IGNORECASE):
        return fallback_stem.upper()
    return None

def is_closeup_path(p: str) -> bool:
    s = p.lower()
    # Heuristic: tag likely close-up views
    close_tokens = ["close", "closeup", "zoom", "detail", "crop", "patch", "magnified", "dermo"]
    far_tokens = ["overview", "global", "distant", "clinical", "wide"]
    if any(t in s for t in close_tokens):
        return True
    if any(t in s for t in far_tokens):
        return False
    # Default: unknown → treat as not close-up (prefer keeping one that is not flagged close-up)
    return False

def filter_milk10k_il_far_view(paths: List[str]) -> List[str]:
    """
    Given all MIL10K IL image paths, keep only one per IL_XXXX (prefer further-away view).
    Heuristics: if one image appears to be close-up by name and another not, keep the non-close-up.
    Otherwise, choose deterministically by lexicographic order of path.
    """
    grouped: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        lid = extract_il_id(p, stem(p))
        if lid:
            grouped[lid].append(p)
        else:
            # No IL id found: keep but group by its own path to avoid dropping it
            grouped[p].append(p)

    kept: List[str] = []
    for lid, items in grouped.items():
        items = sorted(items)
        # Prefer non-closeup among items
        non_close = [x for x in items if not is_closeup_path(x)]
        if non_close:
            kept.append(non_close[0])
        else:
            kept.append(items[0])
    return sorted(kept)

# -------------------- Dataset readers --------------------

def ham1000_meta(raw_dir: str) -> Dict[str, Dict]:
    d = os.path.join(raw_dir, "HAM1000")
    out: Dict[str, Dict] = {}

    def map_dx(dx: str) -> Tuple[str, str]:
        dx = (dx or "").strip().lower()
        if dx in {"akiec", "bcc", "mel"}:
            return "malignant", HAM_TO_CANON.get(dx, "UNKNOWN")
        if dx in {"bkl", "df", "nv", "vasc"}:
            return "benign", HAM_TO_CANON.get(dx, "UNKNOWN")
        return "unknown", "UNKNOWN"

    # Train
    meta_txt = os.path.join(d, "HAM10000_metadata.txt")
    if os.path.isfile(meta_txt):
        df = pd.read_csv(meta_txt)
        for _, r in df.iterrows():
            img = str(r.get("image_id", "")).strip()
            if not img:
                continue
            lesion, diagnosis = map_dx(r.get("dx"))
            key = img.upper()
            out[key] = {
                 "lesion": lesion,
                 "diagnosis": diagnosis,
                 "age": r.get("age", "not_provided"),
                 "sex": r.get("sex", "not_provided"),
                 "localization": r.get("localization", "not_provided"),
             }

    # Test (ISIC2018) - Handle both formats
    gt = os.path.join(d, "ISIC2018_Task3_Test_GroundTruth.csv")
    if os.path.isfile(gt):
        df = pd.read_csv(gt)
        print(f"[HAM1000] Processing ISIC2018 ground truth with columns: {list(df.columns)}")
        
        # Check the actual structure of the file
        if "image_id" in df.columns and "dx" in df.columns:
            # This is the correct format with image_id, dx, age, sex, localization
            print(f"[HAM1000] Detected correct ISIC2018 format with image_id, dx, age, sex, localization")
            
            for _, r in df.iterrows():
                imgfile = str(r.get("image_id", "")).strip()
                dx = str(r.get("dx", "")).strip()
                age = r.get("age", "")
                sex = r.get("sex", "")
                loc = r.get("localization", "")
                
                if not imgfile or not dx:
                    continue
                    
                s = os.path.splitext(imgfile)[0]
                k = s.upper()
                
                if k and k not in out:
                    # Process diagnosis and get lesion type
                    lesion, diagnosis = map_dx(dx)
                    
                    out[k] = {
                        "lesion": lesion, 
                        "diagnosis": diagnosis, 
                        "age": age, 
                        "sex": sex, 
                        "localization": loc
                    }
                                         # print(f"[HAM1000] Added image {k} with dx={dx} -> {diagnosis}, age={age}, sex={sex}, loc={loc}")
        elif "lesion_id" in df.columns and "image" in df.columns:
            # This is a lesion_id -> image mapping file (fallback)
            print(f"[HAM1000] Detected lesion_id -> image mapping format (fallback)")
            
            # First, try to load the main metadata to get diagnosis info
            main_meta = os.path.join(d, "HAM10000_metadata.txt")
            lesion_to_diagnosis = {}
            if os.path.isfile(main_meta):
                try:
                    meta_df = pd.read_csv(main_meta)
                    for _, r in meta_df.iterrows():
                        lesion_id = str(r.get("lesion_id", "")).strip()
                        dx = str(r.get("dx", "")).strip()
                        if lesion_id and dx:
                            lesion_to_diagnosis[lesion_id] = dx
                    print(f"[HAM1000] Loaded {len(lesion_to_diagnosis)} lesion_id -> diagnosis mappings")
                except Exception as e:
                    print(f"[HAM1000] Warning: Could not load main metadata: {e}")
            
            for _, r in df.iterrows():
                lesion_id = str(r.get("lesion_id", "")).strip()
                imgfile = str(r.get("image", "")).strip()
                
                if not imgfile or not lesion_id:
                    continue
                    
                s = os.path.splitext(imgfile)[0]
                k = s.upper()
                
                if k and k not in out:
                    # Try to get diagnosis from the main metadata using lesion_id
                    dx = lesion_to_diagnosis.get(lesion_id, "")
                    if dx:
                        lesion, diagnosis = map_dx(dx)
                    else:
                        lesion, diagnosis = "unknown", "UNKNOWN"
                    
                    out[k] = {"lesion": lesion, "diagnosis": diagnosis, "age": "", "sex": "", "localization": ""}
                                         # print(f"[HAM1000] Added image {k} from lesion_id {lesion_id} -> {diagnosis}")
        else:
            # This is a traditional class-based ground truth file
            print(f"[HAM1000] Detected class-based ground truth format")
            key_col = "image" if "image" in df.columns else df.columns[0]
            class_cols = [c for c in df.columns if c != key_col]
            for _, r in df.iterrows():
                imgfile = str(r.get(key_col, "")).strip()
                s = os.path.splitext(imgfile)[0]
                active = [c for c in class_cols if isinstance(r.get(c), (int, float)) and r.get(c) == 1]
                label = active[0].lower() if active else ""
                lesion, diagnosis = map_dx(label)
                k = s.upper()
                if k and k not in out:
                    out[k] = {"lesion": lesion, "diagnosis": diagnosis, "age": "", "sex": "", "localization": ""}

    return out

def isic2019_meta(raw_dir: str) -> Dict[str, Dict]:
    d = os.path.join(raw_dir, "ISIC2019")
    out: Dict[str, Dict] = {}

    gt = os.path.join(d, "ISIC_2019_Training_GroundTruth.csv")
    if os.path.isfile(gt):
        df = pd.read_csv(gt)
        key_col = "image" if "image" in df.columns else df.columns[0]
        class_cols = [c for c in df.columns if c != key_col]
        for _, r in df.iterrows():
            s = os.path.splitext(str(r.get(key_col, "")).strip())[0]
            active = [c for c in class_cols if isinstance(r.get(c), (int, float)) and r.get(c) == 1]
            dx = active[0] if active else ""
            dxU = (dx or "").upper()
            canon = ISIC2019_TO_CANON.get(dxU, "UNKNOWN")
            if canon == "UNKNOWN":
                lesion, diagnosis = "unknown", "UNKNOWN"
            elif canon in {"NV", "BKL", "DF", "VASC"}:
                lesion, diagnosis = "benign", canon
            else:
                lesion, diagnosis = "malignant", canon
            if s:
                k = s.upper()
                out.setdefault(k, {})
                out[k].update({"lesion": lesion, "diagnosis": diagnosis})

    meta = os.path.join(d, "ISIC_2019_Training_Metadata.csv")
    if os.path.isfile(meta):
        df = pd.read_csv(meta)
        for _, r in df.iterrows():
            s = os.path.splitext(str(r.get("image", "")).strip())[0]
            if not s:
                continue
            k = s.upper()
            out.setdefault(k, {})
            out[k].update({
                "age": r.get("age_approx", "not_provided"),
                "sex": r.get("sex", "not_provided"),
                "localization": r.get("anatom_site_general", "not_provided"),
            })

    return out

def isic2020_meta(raw_dir: str) -> Dict[str, Dict]:
    d = os.path.join(raw_dir, "ISIC2020")
    out: Dict[str, Dict] = {}
    gt_v2 = os.path.join(d, "ISIC_2020_Training_GroundTruth_v2.csv")
    gt = gt_v2 if os.path.isfile(gt_v2) else os.path.join(d, "ISIC_2020_Training_GroundTruth.csv")
    if not os.path.isfile(gt):
        return out
    df = pd.read_csv(gt)
    key_col = "image_name" if "image_name" in df.columns else df.columns[0]

    for _, r in df.iterrows():
        s = os.path.splitext(str(r.get(key_col, "")).strip())[0]
        if not s:
            continue
        diag_raw = str(r.get("diagnosis", "")).strip().lower()
        canon_dx = ISIC2020_TO_CANON.get(diag_raw, "UNKNOWN")
        if canon_dx == "UNKNOWN":
            lesion = "unknown"
        else:
            bm = r.get("benign_malignant", "")
            try:
                bm_val = int(bm)
                lesion = "malignant" if bm_val == 1 else "benign"
            except Exception:
                lesion = "malignant" if str(bm).strip().lower() == "malignant" else "benign"
        out[s.upper()] = {
            "lesion": lesion,
            "diagnosis": canon_dx,
            "age": r.get("age_approx", "not_provided"),
            "sex": r.get("sex", "not_provided"),
            "localization": r.get("anatom_site_general_challenge", "not_provided"),
        }
    return out

def milk10k_maps() -> Tuple[Dict[str, str], Set[str]]:
    MAP = {
        "Melanoma Invasive": "MEL",
        "Melanoma in situ": "MEL",
        "Melanoma metastasis": "MEL",
        "Basal cell carcinoma": "BCC",
        "Solar or actinic keratosis": "AKIEC",
        "Squamous cell carcinoma in situ, Bowens disease": "AKIEC",
        "Squamous cell carcinoma, Invasive": "SCCKA",
        "Keratoacanthoma": "SCCKA",
        "Nevus": "NV",
        "Nevus, NOS, Junctional": "NV",
        "Nevus, NOS, Compound": "NV",
        "Nevus, NOS, Dermal": "NV",
        "Nevus, Congenital": "NV",
        "Nevus, Combined": "NV",
        "Nevus, Recurrent or persistent": "NV",
        "Nevus, Spitz": "NV",
        "Nevus, Reed": "NV",
        "Nevus, Acral": "NV",
        "Blue nevus": "NV",
        "Nevus, Balloon cell": "NV",
        "Nevus, Spilus": "NV",
        "Nevus, BAP-1 deficient": "NV",
        "Mucosal melanotic macule": "NV",
        "Seborrheic keratosis": "BKL",
        "Solar lentigo": "BKL",
        "Lichen planus like keratosis": "BKL",
        "Clear cell acanthoma": "BKL",
        "Porokeratosis": "BKL",
        "Ink-spot lentigo": "BKL",
        "Dermatofibroma": "DF",
        "Benign soft tissue proliferations - Fibro-histiocytic": "DF",
        "Hemangioma": "VASC",
        "Angiokeratoma": "VASC",
        "Pyogenic granuloma": "VASC",
        "Hemangioma, Hobnail": "VASC",
        "Benign soft tissue proliferations - Vascular": "VASC",
        "Sebaceous hyperplasia": "BEN_OTH",
        "Trichoblastoma": "BEN_OTH",
        "Infundibular or epidermal cyst": "BEN_OTH",
        "Supernumerary nipple": "BEN_OTH",
        "Juvenile xanthogranuloma": "BEN_OTH",
        "Mastocytosis": "BEN_OTH",
        "Exogenous": "BEN_OTH",
        "Benign - Other": "BEN_OTH",
        "Collision - Only benign proliferations": "BEN_OTH",
        "Inflammatory or infectious diseases": "INF",
        "Molluscum": "INF",
        "Collision - At least one malignant proliferation": "MAL_OTH",
    }
    MAL = {"MEL", "BCC", "AKIEC", "SCCKA", "MAL_OTH"}
    return MAP, MAL

def milk10k_meta(raw_dir: str) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    d = os.path.join(raw_dir, "MILK10K")
    meta_isic: Dict[str, Dict] = {}
    meta_il: Dict[str, Dict] = {}
    MAP, MAL = milk10k_maps()

    # Files
    suppl = os.path.join(d, "MILK10k_Training_Supplement.csv")
    gt = os.path.join(d, "MILK10k_Training_GroundTruth.csv")
    meta_det = os.path.join(d, "MILK10k_Training_Metadata.csv")
    meta_csv = os.path.join(d, "metadata.csv")

    # Supplement (ISIC_* mapping)
    if os.path.isfile(suppl):
        df = pd.read_csv(suppl)
        for _, r in df.iterrows():
            isic = str(r.get("isic_id", "")).strip()
            if not isic:
                continue
            full = str(r.get("diagnosis_full", "")).strip()
            code = MAP.get(full)
            if not code:
                continue
            lesion = "malignant" if code in MAL else "benign"
            diagnosis = code
            meta_isic.setdefault(isic.upper(), {})
            meta_isic[isic.upper()].update({"lesion": lesion, "diagnosis": diagnosis})

    # GroundTruth (IL_* one-hot)
    if os.path.isfile(gt):
        df = pd.read_csv(gt)
        key = "lesion_id" if "lesion_id" in df.columns else df.columns[0]
        class_cols = [c for c in df.columns if c != key]
        for _, r in df.iterrows():
            lid = str(r.get(key, "")).strip()
            if not lid:
                continue
            active = [c for c in class_cols if isinstance(r.get(c), (int, float)) and r.get(c) == 1]
            code = active[0] if active else ""
            codeU = (code or "").upper()
            if not codeU:
                continue
            lesion = "malignant" if codeU in MAL else "benign"
            diagnosis = codeU
            meta_il.setdefault(lid.upper(), {})
            meta_il[lid.upper()].update({"lesion": lesion, "diagnosis": diagnosis})

    # Detailed metadata (age/sex/localization; also maps lesion_id <-> isic_id)
    if os.path.isfile(meta_det):
        df = pd.read_csv(meta_det)
        for _, r in df.iterrows():
            lid = str(r.get("lesion_id", "")).strip()
            isic = str(r.get("isic_id", "")).strip()
            age = r.get("age_approx", "not_provided")
            sex = r.get("sex", "not_provided")
            loc = r.get("site", "not_provided")
            if lid:
                meta_il.setdefault(lid.upper(), {})
                meta_il[lid.upper()].update({"age": age, "sex": sex, "localization": loc})
            if isic:
                meta_isic.setdefault(isic.upper(), {})
                meta_isic[isic.upper()].update({"age": age, "sex": sex, "localization": loc})

    # Fallback general metadata.csv (ISIC_* only)
    if os.path.isfile(meta_csv):
        df = pd.read_csv(meta_csv)
        for _, r in df.iterrows():
            isic = str(r.get("isic_id", "")).strip()
            if not isic:
                continue
            meta_isic.setdefault(isic.upper(), {})
            meta_isic[isic.upper()].setdefault("age", r.get("age_approx", "not_provided"))
            meta_isic[isic.upper()].setdefault("sex", r.get("sex", "not_provided"))
            meta_isic[isic.upper()].setdefault("localization", r.get("anatom_site_general", "not_provided"))

    return meta_isic, meta_il

# -------------------- Collect file paths (deterministic) --------------------

def _gather_images_recursive(start_dir: str) -> List[str]:
    files = []
    for r, _, fns in os.walk(start_dir):
        for fn in fns:
            if is_image(fn):
                files.append(os.path.join(r, fn))
    return files

def collect_paths(raw_dir: str) -> Dict[str, List[str]]:
    """
    Collect images using the expected folder names and recurse inside each of them.
    Sorting is applied for deterministic ordering.
    """
    paths = {k: [] for k in ["HAM1000", "ISIC2019", "ISIC2020", "MIL10K_ISIC", "MIL10K_IL", "ITOBOS2024"]}

    # HAM1000
    ham = os.path.join(raw_dir, "HAM1000")
    ham_subs = [
        os.path.join(ham, "HAM10000_images_part_1"),
        os.path.join(ham, "HAM10000_images_part_2"),
        os.path.join(ham, "ISIC2018_Task3_Test_Images"),
    ]
    if os.path.isdir(ham):
        any_found = False
        for sd in ham_subs:
            if os.path.isdir(sd):
                paths["HAM1000"].extend(_gather_images_recursive(sd))
                any_found = True
        if not any_found:
            paths["HAM1000"].extend(_gather_images_recursive(ham))

    # ISIC2019 (exact)
    is19 = os.path.join(raw_dir, "ISIC2019", "ISIC_2019_Training_Input")
    if os.path.isdir(is19):
        paths["ISIC2019"].extend(_gather_images_recursive(is19))

    # ISIC2020 (exact)
    is20 = os.path.join(raw_dir, "ISIC2020", "ISIC_2020_Train_Input")
    if os.path.isdir(is20):
        paths["ISIC2020"].extend(_gather_images_recursive(is20))

    # MIL10K
    mil_root = os.path.join(raw_dir, "MILK10K")
    mil_images = os.path.join(mil_root, "images")
    mil_il = os.path.join(mil_root, "MILK10k_Training_Input")
    if os.path.isdir(mil_images):
        paths["MIL10K_ISIC"].extend(_gather_images_recursive(mil_images))
    if os.path.isdir(mil_il):
        paths["MIL10K_IL"].extend(_gather_images_recursive(mil_il))

    # ITOBOS2024
    itobos_imgs = os.path.join(raw_dir, "ITOBOS2024", "train", "images")
    if os.path.isdir(itobos_imgs):
        paths["ITOBOS2024"].extend(_gather_images_recursive(itobos_imgs))

    # Deterministic ordering
    for k in paths:
        paths[k] = sorted(paths[k])

    # Debug
    print("[PATHS] Using folders:")
    print("  ISIC2019:", is19 if os.path.isdir(is19) else "NOT FOUND")
    print("  ISIC2020:", is20 if os.path.isdir(is20) else "NOT FOUND")
    print("  MILK10K images:", mil_images if os.path.isdir(mil_images) else "NOT FOUND")
    print("  MILK10K IL:", mil_il if os.path.isdir(mil_il) else "NOT FOUND")

    return paths

# -------------------- Main pipeline --------------------

def build_dataset(raw_dir: str, out_dir: str, images_only: bool = False, metadata_only: bool = False, skip_dedup: bool = False):
    images_out = os.path.join(out_dir, "images")
    ensure_dir(images_out)

    # Load per-dataset metadata
    ham = ham1000_meta(raw_dir)
    is19 = isic2019_meta(raw_dir)
    is20 = isic2020_meta(raw_dir)
    mil_isic, mil_il = milk10k_meta(raw_dir)

    # ITOBOS healthy set & metadata
    itobos_ok = itobos_healthy_filenames(raw_dir)
    itobos_meta_csv = os.path.join(raw_dir, "ITOBOS2024", "train", "metadata.csv")
    itobos_meta = {}
    if os.path.isfile(itobos_meta_csv):
        df = pd.read_csv(itobos_meta_csv)
        for _, r in df.iterrows():
            sid = str(r.get("image_id", "")).strip()
            if sid:
                 itobos_meta[sid] = {
                     "age": r.get("age_at_baseline", "not_provided"),
                     "localization": r.get("body_part", "not_provided"),
                 }

    # If metadata_only, skip image processing
    if metadata_only:
        print("[MODE] Metadata-only mode - skipping image processing")
        # Load existing metadata if available
        existing_csv = os.path.join(out_dir, "metadata.csv")
        if os.path.isfile(existing_csv):
            print(f"[INFO] Loading existing metadata from {existing_csv}")
            existing_df = pd.read_csv(existing_csv)
            rows = existing_df.to_dict('records')
        else:
            print("[ERROR] No existing metadata found. Run without --metadata_only first.")
            return
        # Initialize empty paths for metadata_only mode
        paths = {k: [] for k in ["HAM1000", "ISIC2019", "ISIC2020", "MIL10K_ISIC", "MIL10K_IL", "ITOBOS2024"]}
    else:
        # Gather paths
        paths = collect_paths(raw_dir)

    # ISIC2020 duplicates suppression
    is20_keep_set = build_isic2020_duplicate_keep_set(raw_dir)
    if is20_keep_set is not None and paths.get("ISIC2020"):
        before = len(paths["ISIC2020"])
        filtered = []
        for p in paths["ISIC2020"]:
            s = stem(p)
            if s in is20_keep_set or s not in is20_keep_set and s not in is20_keep_set:
                # keep if it's canonical or not mentioned in duplicates file at all
                if (s in is20_keep_set) or (s not in is20_keep_set and s not in is20_keep_set):
                    filtered.append(p)
        # Simpler: keep canonical stems and any stems not present in the dupe graph
        present = set()
        if is20_keep_set is not None:
            # Build set of all stems involved in the dups
            # Re-read quickly to gather both columns
            dup_csv = os.path.join(raw_dir, "ISIC2020", "ISIC_2020_Duplicates.csv")
            try:
                df_dups = pd.read_csv(dup_csv)
                c1 = "image_name_1" if "image_name_1" in df_dups.columns else df_dups.columns[0]
                c2 = "image_name_2" if "image_name_2" in df_dups.columns else df_dups.columns[1]
                for _, r in df_dups.iterrows():
                    a = os.path.splitext(str(r.get(c1, "")).strip())[0]
                    b = os.path.splitext(str(r.get(c2, "")).strip())[0]
                    if a:
                        present.add(a)
                    if b:
                        present.add(b)
            except Exception:
                pass
        # Final filter
        canon_or_unmentioned = []
        for p in paths["ISIC2020"]:
            s = stem(p)
            if (s in is20_keep_set) or (s not in present):
                canon_or_unmentioned.append(p)
        paths["ISIC2020"] = sorted(canon_or_unmentioned)
        after = len(paths["ISIC2020"])
        print(f"[ISIC2020] Duplicates filtering: {before} -> {after}")

    # MIL10K IL: keep only far view per lesion
    if paths.get("MIL10K_IL"):
        before = len(paths["MIL10K_IL"])
        paths["MIL10K_IL"] = filter_milk10k_il_far_view(paths["MIL10K_IL"])
        after = len(paths["MIL10K_IL"])
        print(f"[MIL10K_IL] Far-view selection: {before} -> {after}")

    # ---------- DEBUG / VISIBILITY ----------
    print("[SCAN] Files found per dataset:")
    for k in ["ISIC2020","ISIC2019","HAM1000","MIL10K_ISIC","MIL10K_IL","ITOBOS2024"]:
        print(f"  - {k}: {len(paths.get(k, []))} files")
    print(f"[ITOBOS] healthy (no bbox) candidates: {len(itobos_ok)}")

    # Dedup priority
    priority = ["HAM1000", "ITOBOS2024", "MIL10K_ISIC", "MIL10K_IL", "ISIC2020", "ISIC2019"]

    hash_to_meta: Dict[str, Dict] = {}   # hash -> final metadata row
    hash_to_name: Dict[str, str] = {}   # hash -> final image filename
    name_to_hash: Dict[str, str] = {}   # for handling name collisions with different contents
    copied_by_origin = defaultdict(int)

    # If skip_dedup, load existing metadata to preserve it
    if skip_dedup:
        print("[MODE] Skip dedup mode - loading existing metadata")
        existing_csv = os.path.join(out_dir, "metadata.csv")
        if os.path.isfile(existing_csv):
            existing_df = pd.read_csv(existing_csv)
            for _, row in existing_df.iterrows():
                # Create a dummy hash for existing entries
                dummy_hash = f"existing_{row['image_id']}"
                hash_to_meta[dummy_hash] = row.to_dict()
                hash_to_name[dummy_hash] = str(row['image_id'])
                name_to_hash[str(row['image_id'])] = dummy_hash
            print(f"[INFO] Loaded {len(hash_to_meta)} existing metadata entries")
        else:
            print("[WARN] No existing metadata found for skip_dedup mode")

    def maybe_copy_record(abs_path: str, origin: str):
        fn = os.path.basename(abs_path)
        s = os.path.splitext(fn)[0]
        s_upper = s.upper()
        h = file_hash(abs_path)

        # Skip if content already copied
        if h in hash_to_name:
            return

        # Build metadata based on origin
        lesion, diagnosis, age, sex, loc = "", "", "not_provided", "not_provided", "not_provided"
        if origin == "HAM1000":
            info = ham.get(s_upper) or ham.get(s) or ham.get(s.lower()) or {}
            lesion = (info.get("lesion", "") or "").lower()
            diagnosis = info.get("diagnosis", "")
            age = info.get("age", "")
            sex = info.get("sex", "")
            loc = info.get("localization", "")
        elif origin == "ISIC2019":
            info = is19.get(s_upper) or is19.get(s) or is19.get(s.lower()) or {}
            lesion = (info.get("lesion", "") or "").lower()
            diagnosis = info.get("diagnosis", "")
            age = info.get("age", "")
            sex = info.get("sex", "")
            loc = info.get("localization", "")
        elif origin == "ISIC2020":
            info = is20.get(s_upper) or is20.get(s) or is20.get(s.lower()) or {}
            lesion = (info.get("lesion", "") or "").lower()
            diagnosis = info.get("diagnosis", "")
            age = info.get("age", "")
            sex = info.get("sex", "")
            loc = info.get("localization", "")
        elif origin == "MIL10K_ISIC":
            sU = s_upper
            info = mil_isic.get(sU, {})
            lesion = (info.get("lesion", "") or "").lower()
            diagnosis = info.get("diagnosis", "")
            age = info.get("age", "")
            sex = info.get("sex", "")
            loc = info.get("localization", "")
        elif origin == "MIL10K_IL":
            lid = extract_il_id(abs_path, s) or s_upper
            info = mil_il.get(lid.upper(), {})
            lesion = (info.get("lesion", "") or "").lower()
            diagnosis = info.get("diagnosis", "")
            age = info.get("age", "")
            sex = info.get("sex", "")
            loc = info.get("localization", "")
        elif origin == "ITOBOS2024":
            # Use basenames in itobos_ok
            if fn not in itobos_ok:
                return
            lesion = "no_lesion"
            diagnosis = "NO_LESION"
            sid = s  # image_XXXX
            info = itobos_meta.get(sid, {})
            age = info.get("age", "not_provided")
            sex = "not_provided"  # unknown
            loc = info.get("localization", "not_provided")

        # normalize lesion
        allowed = {"no_lesion", "benign", "malignant", "unknown"}
        if lesion and lesion not in allowed:
            lesion = ""

        # Standardize diagnosis code globally
        def to_canon(dx: str) -> str:
            if not dx:
                return ""
            u = str(dx).strip().upper()
            # map common aliases
            if u in HAM_TO_CANON.values() or u in ISIC2019_TO_CANON.values() or u in CANON_DIAG_CODES:
                # Already canonical or recognized mapped value
                return u
            # Try reverse maps
            if u.lower() in HAM_TO_CANON:
                return HAM_TO_CANON[u.lower()]
            if u in ISIC2019_TO_CANON:
                return ISIC2019_TO_CANON[u]
            if u.lower() in ISIC2020_TO_CANON:
                return ISIC2020_TO_CANON[u.lower()]
            # Leave as uppercase string if unknown but non-empty
            return u

        diagnosis = to_canon(diagnosis)

        # Normalize demographic/location
        age = normalize_age(age)
        sex = normalize_sex(sex)
        loc = normalize_localization(loc)

        # resolve final filename (avoid overwriting existing different content)
        desired = fn
        dst = os.path.join(images_out, desired)
        if os.path.exists(dst):
            # Compare with existing file's hash
            try:
                existing_h = file_hash(dst)
                if existing_h == h:
                    # Already present; just register and skip copying
                    hash_to_name[h] = desired
                    name_to_hash[desired] = h
                    origin_out_same = (
                        "ISIC2020" if origin == "ISIC2020"
                        else "ISIC2019" if origin == "ISIC2019"
                        else "HAM1000" if origin == "HAM1000"
                        else "MIL10K" if origin.startswith("MIL10K")
                        else "ITOBOS2024"
                    )
                    hash_to_meta[h] = {
                        "image_id": desired,
                        "origin_dataset": origin_out_same,
                        "lesion": lesion,
                        "diagnosis": diagnosis,
                        "localization": loc,
                        "age": age,
                        "sex": sex,
                    }
                    return
                else:
                    # Different content: choose a unique name
                    desired = safe_unique_name(images_out, desired)
                    dst = os.path.join(images_out, desired)
            except Exception:
                # If hashing existing file fails, pick a unique name to be safe
                desired = safe_unique_name(images_out, desired)
                dst = os.path.join(images_out, desired)
        else:
            # Also avoid in-run name collision with different contents
            if desired in name_to_hash and name_to_hash[desired] != h:
                desired = safe_unique_name(images_out, desired)
                dst = os.path.join(images_out, desired)

        # copy file
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(abs_path, dst)

        # register
        hash_to_name[h] = desired
        name_to_hash[desired] = h
        origin_out = (
            "ISIC2020" if origin == "ISIC2020"
            else "ISIC2019" if origin == "ISIC2019"
            else "HAM1000" if origin == "HAM1000"
            else "MIL10K" if origin.startswith("MIL10K")
            else "ITOBOS2024"
        )
        hash_to_meta[h] = {
            "image_id": desired,
            "origin_dataset": origin_out,
            "lesion": lesion,
            "diagnosis": diagnosis,
            "localization": loc,
            "age": age,
            "sex": sex,
        }
        copied_by_origin[origin_out] = 1

        # If images_only, skip metadata generation
        if images_only:
            print("[MODE] Images-only mode - skipping metadata generation")
            print("✅ Done")
            print(f"- Images processed (metadata generation skipped)")
            print(f"- Images copied to: {images_out}")
            return

        # Iterate in priority order
        for origin in priority:
            for p in paths.get(origin, []):
                try:
                    maybe_copy_record(p, origin)
                except Exception as e:
                    print(f"[WARN] Failed processing {p}: {e}")

        # ---------- Summary ----------
        print("[COPY] Copied unique files per origin_dataset:")
        for k in ["ITOBOS2024","ISIC2020","ISIC2019","HAM1000","MIL10K"]:
            print(f"  - {k}: {copied_by_origin.get(k, 0)}")

        rows = list(hash_to_meta.values())
        ensure_dir(out_dir)
        out_csv = os.path.join(out_dir, "metadata.csv")
        pd.DataFrame(rows, columns=["image_id","origin_dataset","lesion","diagnosis","localization","age","sex"]).to_csv(
            out_csv, index=False, quoting=csv.QUOTE_MINIMAL
        )

        print("✅ Done")
        print(f"- Unique images: {len(rows)}")
        print(f"- Images copied to: {images_out}")
        print(f"- Metadata written to: {out_csv}")

    # If metadata_only, regenerate metadata from existing images
    if metadata_only:
        print("[MODE] Regenerating metadata from existing images...")
        
        # Collect all image files in the output directory
        image_files = []
        for root, dirs, files in os.walk(images_out):
            for file in files:
                if is_image(file):
                    image_files.append(os.path.join(root, file))
        
        print(f"[INFO] Found {len(image_files)} existing images")
        
        # Process each image to regenerate metadata
        hash_to_meta = {}
        for img_path in image_files:
            fn = os.path.basename(img_path)
            s = os.path.splitext(fn)[0]
            s_upper = s.upper()
            
            # Try to find metadata from all sources
            info = {}
            for source_name, source_meta in [("HAM1000", ham), ("ISIC2019", is19), ("ISIC2020", is20)]:
                source_info = source_meta.get(s_upper) or source_meta.get(s) or source_meta.get(s.lower()) or {}
                if source_info:
                    info.update(source_info)
                    break
            
            # Check MIL10K sources
            mil_info = mil_isic.get(s_upper, {}) or mil_il.get(s_upper, {})
            if mil_info:
                info.update(mil_info)
            
            # Check ITOBOS
            if fn in itobos_ok:
                itobos_info = itobos_meta.get(s, {})
                if itobos_info:
                    info.update(itobos_info)
                    info["lesion"] = "no_lesion"
                    info["diagnosis"] = "NO_LESION"
            
            # Use default values if no info found
            if not info:
                info = {
                    "lesion": "unknown",
                    "diagnosis": "UNKNOWN",
                    "age": "not_provided",
                    "sex": "not_provided",
                    "localization": "not_provided"
                }
            
            # Normalize values
            lesion = info.get("lesion", "unknown")
            diagnosis = info.get("diagnosis", "UNKNOWN")
            age = normalize_age(info.get("age", ""))
            sex = normalize_sex(info.get("sex", ""))
            loc = normalize_localization(info.get("localization", ""))
            
            # Determine origin dataset (heuristic based on filename patterns)
            origin = "UNKNOWN"
            if s_upper.startswith("ISIC_"):
                origin = "ISIC2020"  # or could be ISIC2019
            elif s_upper.startswith("HAM"):
                origin = "HAM1000"
            elif s_upper.startswith("IL_"):
                origin = "MIL10K"
            elif s_upper.startswith("IMAGE_"):
                origin = "ITOBOS2024"
            
            hash_to_meta[fn] = {
                "image_id": fn,
                "origin_dataset": origin,
                "lesion": lesion,
                "diagnosis": diagnosis,
                "localization": loc,
                "age": age,
                "sex": sex,
            }
        
        rows = list(hash_to_meta.values())
        ensure_dir(out_dir)
        out_csv = os.path.join(out_dir, "metadata.csv")
        pd.DataFrame(rows, columns=["image_id","origin_dataset","lesion","diagnosis","localization","age","sex"]).to_csv(
            out_csv, index=False, quoting=csv.QUOTE_MINIMAL
        )
        
        print("✅ Metadata regeneration complete")
        print(f"- Processed {len(rows)} images")
        print(f"- Metadata written to: {out_csv}")

# -------------------- CLI --------------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_raw = os.path.abspath(os.path.join(here, "..", "data_raw"))
    default_clean = os.path.abspath(os.path.join(here, "..", "data_clean"))

    ap = argparse.ArgumentParser(description="Build data_clean (dedupe and unified metadata).")
    ap.add_argument("--raw_dir", type=str, default=default_raw, help="Path to raw datasets (default: ../data_raw)")
    ap.add_argument("--out_dir", type=str, default=default_clean, help="Output folder (default: ../data_clean)")
    ap.add_argument("--images_only", action="store_true", help="Only process images (skip metadata generation)")
    ap.add_argument("--metadata_only", action="store_true", help="Only generate metadata (skip image processing)")
    ap.add_argument("--skip_dedup", action="store_true", help="Skip deduplication (assume images already exist)")
    args = ap.parse_args()

    # Validate arguments
    if args.images_only and args.metadata_only:
        print("❌ Error: Cannot use both --images_only and --metadata_only")
        return

    build_dataset(args.raw_dir, args.out_dir, 
                 images_only=args.images_only, 
                 metadata_only=args.metadata_only, 
                 skip_dedup=args.skip_dedup)

if __name__ == "__main__":
    main()