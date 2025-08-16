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
    image_id, origin_dataset, lesion_type, diagnosis, body_region, age, gender

Origin dataset priority when identical files appear across datasets (same content hash):
    HAM1000 → ITOBOS2024 → MIL10K → ISIC2020 → ISIC2019
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
MAL_CANON = {"MEL","BCC","AKIEC","SCC","SCCKA","MAL_OTH"}
BEN_CANON = {"NV","BKL","DF","VASC","BEN_OTH","INF"}

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
        i += 1
        final = f"{base}__v{i}{ext}"
    return final

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

# -------------------- Normalization helpers --------------------

def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):   # catches pd.NA, np.nan, NaT
            return True
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"", "na", "n/a", "none", "null", "unknown", "<na>"}:
            return True
    return False

def normalize_age(value) -> str:
    if _is_missing(value):
        return "not_provided"
    if isinstance(value, str):
        m = re.search(r"\d+(\.\d+)?", value)  # e.g. "45 years"
        if not m:
            return "not_provided"
        value = m.group(0)
    try:
        f = float(value)
    except Exception:
        return "not_provided"
    if pd.isna(f):
        return "not_provided"
    i = int(round(f))
    return str(i) if 0 <= i <= 120 else "not_provided"

def normalize_gender(value) -> str:
    if _is_missing(value): return "not_provided"
    s = str(value).strip().lower()
    if s in {"m","male","man"}: return "male"
    if s in {"f","female","woman"}: return "female"
    return "not_provided"

def normalize_localization(value) -> str:
    if _is_missing(value): return "not_provided"
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
    healthy_basenames = {os.path.basename(images[iid]) for iid, c in counts.items() if c == 0}
    return healthy_basenames

# -------------------- Canonical codes & maps --------------------

CANON_DIAG_CODES: Set[str] = {
    "MEL","BCC","AKIEC","SCC","SCCKA","MAL_OTH",
    "NV","BKL","DF","VASC","BEN_OTH","INF",
    "UNKNOWN","NO_LESION",
}

HAM_TO_CANON = {
    "akiec":"AKIEC","bcc":"BCC","mel":"MEL","bkl":"BKL","df":"DF","nv":"NV","vasc":"VASC","unknown":"UNKNOWN",
}

ISIC2019_TO_CANON = {
    "MEL":"MEL","NV":"NV","BCC":"BCC","AK":"AKIEC","BKL":"BKL","DF":"DF","VASC":"VASC","SCC":"SCC","UNK":"UNKNOWN",
}

ISIC2020_TO_CANON = {
    "melanoma":"MEL","nevus":"NV","seborrheic keratosis":"BKL","solar lentigo":"BKL","lentigo nos":"BKL",
    "lichenoid keratosis":"BKL","blue nevus":"NV","cafe-au-lait macule":"BEN_OTH",
    "atypical melanocytic proliferation":"MAL_OTH","unknown":"UNKNOWN",
}

# -------------------- ISIC2020 duplicate handling --------------------

def build_isic2020_duplicate_keep_set(raw_dir: str) -> Optional[Tuple[Set[str], Set[str]]]:
    dup_csv = os.path.join(raw_dir, "ISIC2020", "ISIC_2020_Duplicates.csv")
    if not os.path.isfile(dup_csv):
        return None
    parent: Dict[str, str] = {}
    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a: str, b: str):
        ra, rb = find(a), find(b)
        if ra == rb: return
        if ra < rb: parent[rb] = ra
        else: parent[ra] = rb

    df = pd.read_csv(dup_csv)
    col1 = "image_name_1" if "image_name_1" in df.columns else df.columns[0]
    col2 = "image_name_2" if "image_name_2" in df.columns else df.columns[1]
    present: Set[str] = set()
    for _, r in df.iterrows():
        a = os.path.splitext(str(r.get(col1, "")).strip())[0]
        b = os.path.splitext(str(r.get(col2, "")).strip())[0]
        if a: present.add(a)
        if b: present.add(b)
        if a and b: union(a, b)

    roots: Dict[str, List[str]] = defaultdict(list)
    for x in list(parent.keys()):
        roots[find(x)].append(x)
    keep: Set[str] = set()
    for _, members in roots.items():
        members.sort()
        keep.add(members[0])
    return keep, present

# -------------------- MIL10K IL helpers --------------------

def extract_il_id(abs_path: str, fallback_stem: str) -> Optional[str]:
    m = re.search(r"(IL_\d+)", abs_path, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for part in abs_path.split(os.sep):
        if part.lower().startswith("il_"):
            return part.upper()
    if re.fullmatch(r"IL_\d+", fallback_stem, flags=re.IGNORECASE):
        return fallback_stem.upper()
    return None

def is_closeup_path(p: str) -> bool:
    s = p.lower()
    close_tokens = ["close","closeup","zoom","detail","crop","patch","magnified","dermo"]
    far_tokens = ["overview","global","distant","clinical","wide"]
    if any(t in s for t in close_tokens): return True
    if any(t in s for t in far_tokens): return False
    return False

def filter_milk10k_il_far_view(paths: List[str]) -> List[str]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        lid = extract_il_id(p, stem(p))
        grouped[(lid or p)].append(p)
    kept: List[str] = []
    for _, items in grouped.items():
        items = sorted(items)
        non_close = [x for x in items if not is_closeup_path(x)]
        kept.append((non_close or items)[0])
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
                 # source column is "sex" → store as "gender"
                 "gender": r.get("sex", "not_provided"),
                 "localization": r.get("localization", "not_provided"),
             }

    # Test (ISIC2018) - Handle both formats
    gt = os.path.join(d, "ISIC2018_Task3_Test_GroundTruth.csv")
    if os.path.isfile(gt):
        df = pd.read_csv(gt)
        print(f"[HAM1000] Processing ISIC2018 ground truth with columns: {list(df.columns)}")
        
        if "image_id" in df.columns and "dx" in df.columns:
            print(f"[HAM1000] Detected correct ISIC2018 format with image_id, dx, age, sex, localization")
            for _, r in df.iterrows():
                imgfile = str(r.get("image_id", "")).strip()
                dx = str(r.get("dx", "")).strip()
                age = r.get("age", "")
                gender = r.get("sex", "")       # read as sex, store as gender
                loc = r.get("localization", "")
                if not imgfile or not dx:
                    continue
                s = os.path.splitext(imgfile)[0]
                k = s.upper()
                if k and k not in out:
                    lesion, diagnosis = map_dx(dx)
                    out[k] = {
                        "lesion": lesion, 
                        "diagnosis": diagnosis, 
                        "age": age, 
                        "gender": gender, 
                        "localization": loc
                    }
        elif "lesion_id" in df.columns and "image" in df.columns:
            print(f"[HAM1000] Detected lesion_id -> image mapping format (fallback)")
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
                    dx = lesion_to_diagnosis.get(lesion_id, "")
                    if dx:
                        lesion, diagnosis = map_dx(dx)
                    else:
                        lesion, diagnosis = "unknown", "UNKNOWN"
                    out[k] = {"lesion": lesion, "diagnosis": diagnosis, "age": "", "gender": "", "localization": ""}
        else:
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
                    out[k] = {"lesion": lesion, "diagnosis": diagnosis, "age": "", "gender": "", "localization": ""}

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
                "gender": r.get("sex", "not_provided"),   # source "sex" → stored as "gender"
                "localization": r.get("anatom_site_general", "not_provided"),
            })

    return out


def _parse_bm(bm):
    if bm is None or (isinstance(bm, float) and pd.isna(bm)):
        return None
    s = str(bm).strip().lower()
    if s in {"1","true","t","y","yes","malignant"}:
        return "malignant"
    if s in {"0","false","f","n","no","benign"}:
        return "benign"
    return None

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

        # 1) PRIMARY: benign_malignant if present
        lesion = _parse_bm(r.get("benign_malignant"))

        # 2) FALLBACK: infer from diagnosis code
        if lesion is None:
            if canon_dx in MAL_CANON:
                lesion = "malignant"
            elif canon_dx in BEN_CANON:
                lesion = "benign"
            elif canon_dx == "NO_LESION":
                lesion = "no_lesion"
            else:
                lesion = "unknown"

        out[s.upper()] = {
            "lesion": lesion,
            "diagnosis": canon_dx,
            "age": r.get("age_approx", "not_provided"),
            "gender": r.get("sex", "not_provided"),  # source "sex" → stored as "gender"
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

    # Detailed metadata (age/gender/localization; also maps lesion_id <-> isic_id)
    if os.path.isfile(meta_det):
        df = pd.read_csv(meta_det)
        for _, r in df.iterrows():
            lid = str(r.get("lesion_id", "")).strip()
            isic = str(r.get("isic_id", "")).strip()
            age = r.get("age_approx", "not_provided")
            gender = r.get("sex", "not_provided")   # source "sex"
            loc = r.get("site", "not_provided")
            if lid:
                meta_il.setdefault(lid.upper(), {})
                meta_il[lid.upper()].update({"age": age, "gender": gender, "localization": loc})
            if isic:
                meta_isic.setdefault(isic.upper(), {})
                meta_isic[isic.upper()].update({"age": age, "gender": gender, "localization": loc})

    # Fallback general metadata.csv (ISIC_* only)
    if os.path.isfile(meta_csv):
        df = pd.read_csv(meta_csv)
        for _, r in df.iterrows():
            isic = str(r.get("isic_id", "")).strip()
            if not isic:
                continue
            meta_isic.setdefault(isic.upper(), {})
            meta_isic[isic.upper()].setdefault("age", r.get("age_approx", "not_provided"))
            meta_isic[isic.upper()].setdefault("gender", r.get("sex", "not_provided"))  # source "sex"
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
    paths = {k: [] for k in ["HAM1000","ISIC2019","ISIC2020","MIL10K_ISIC","MIL10K_IL","ITOBOS2024"]}

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

    is19 = os.path.join(raw_dir, "ISIC2019", "ISIC_2019_Training_Input")
    if os.path.isdir(is19):
        paths["ISIC2019"].extend(_gather_images_recursive(is19))

    is20 = os.path.join(raw_dir, "ISIC2020", "ISIC_2020_Train_Input")
    if os.path.isdir(is20):
        paths["ISIC2020"].extend(_gather_images_recursive(is20))

    mil_root = os.path.join(raw_dir, "MILK10K")
    mil_images = os.path.join(mil_root, "images")
    mil_il = os.path.join(mil_root, "MILK10k_Training_Input")
    if os.path.isdir(mil_images):
        paths["MIL10K_ISIC"].extend(_gather_images_recursive(mil_images))
    if os.path.isdir(mil_il):
        paths["MIL10K_IL"].extend(_gather_images_recursive(mil_il))

    itobos_imgs = os.path.join(raw_dir, "ITOBOS2024", "train", "images")
    if os.path.isdir(itobos_imgs):
        paths["ITOBOS2024"].extend(_gather_images_recursive(itobos_imgs))

    for k in paths:
        paths[k] = sorted(paths[k])

    print("[PATHS] ISIC2019:", is19 if os.path.isdir(is19) else "NOT FOUND")
    print("[PATHS] ISIC2020:", is20 if os.path.isdir(is20) else "NOT FOUND")
    print("[PATHS] MILK10K images:", mil_images if os.path.isdir(mil_images) else "NOT FOUND")
    print("[PATHS] MILK10K IL:", mil_il if os.path.isdir(mil_il) else "NOT FOUND")
    return paths

# -------------------- Main pipeline --------------------

def build_dataset(raw_dir: str, out_dir: str, images_only: bool=False, metadata_only: bool=False, skip_dedup: bool=False):
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

    # If we only want metadata OR we skip dedup/copy (assuming images exist), do the regeneration path and exit
    if metadata_only or skip_dedup:
        mode = "metadata-only" if metadata_only else "skip-dedup (no copy)"
        print(f"[MODE] {mode} → regenerating metadata from existing images in {images_out}")
        regenerate_metadata(images_out, out_dir, ham, is19, is20, mil_isic, mil_il, itobos_ok, itobos_meta)
        return

    # Otherwise we may copy images (default or images_only)
    paths = collect_paths(raw_dir)

    # ISIC2020 duplicates suppression
    keep_present = build_isic2020_duplicate_keep_set(raw_dir)
    if keep_present and paths.get("ISIC2020"):
        keep_set, present = keep_present
        before = len(paths["ISIC2020"])
        paths["ISIC2020"] = sorted([
            p for p in paths["ISIC2020"]
            if (stem(p) in keep_set) or (stem(p) not in present)
        ])
        after = len(paths["ISIC2020"])
        print(f"[ISIC2020] Duplicates filtering: {before} -> {after}")

    # MIL10K IL: keep only far view per lesion
    if paths.get("MIL10K_IL"):
        before = len(paths["MIL10K_IL"])
        paths["MIL10K_IL"] = filter_milk10k_il_far_view(paths["MIL10K_IL"])
        after = len(paths["MIL10K_IL"])
        print(f"[MIL10K_IL] Far-view selection: {before} -> {after}")

    print("[SCAN] Files found per dataset]:")
    for k in ["ISIC2020","ISIC2019","HAM1000","MIL10K_ISIC","MIL10K_IL","ITOBOS2024"]:
        print(f"  - {k}: {len(paths.get(k, []))} files")
    print(f"[ITOBOS] healthy (no bbox) candidates: {len(itobos_ok)}")

    # Dedup priority
    priority = ["HAM1000","ITOBOS2024","MIL10K_ISIC","MIL10K_IL","ISIC2020","ISIC2019"]

    hash_to_meta: Dict[str, Dict] = {}
    hash_to_name: Dict[str, str] = {}
    name_to_hash: Dict[str, str] = {}
    copied_by_origin = defaultdict(int)

    def to_canon(dx: str) -> str:
        if not dx: return ""
        u = str(dx).strip().upper()
        if u in CANON_DIAG_CODES or u in HAM_TO_CANON.values() or u in ISIC2019_TO_CANON.values():
            return u
        if u.lower() in HAM_TO_CANON: return HAM_TO_CANON[u.lower()]
        if u in ISIC2019_TO_CANON: return ISIC2019_TO_CANON[u]
        if u.lower() in ISIC2020_TO_CANON: return ISIC2020_TO_CANON[u.lower()]
        return u

    def build_info(origin: str, abs_path: str, fn: str, s: str, s_upper: str) -> Tuple[str,str,str,str,str]:
        lesion, diagnosis, age, gender, loc = "", "", "not_provided", "not_provided", "not_provided"
        if origin == "HAM1000":
            info = ham.get(s_upper) or ham.get(s) or ham.get(s.lower()) or {}
        elif origin == "ISIC2019":
            info = is19.get(s_upper) or is19.get(s) or is19.get(s.lower()) or {}
        elif origin == "ISIC2020":
            info = is20.get(s_upper) or is20.get(s) or is20.get(s.lower()) or {}
        elif origin == "MIL10K_ISIC":
            info = mil_isic.get(s_upper, {})
        elif origin == "MIL10K_IL":
            lid = (extract_il_id(abs_path, s) or s_upper).upper()
            info = mil_il.get(lid, {})
        elif origin == "ITOBOS2024":
            if fn not in itobos_ok:
                return ("", "", "", "", "")  # signal skip
            info = itobos_meta.get(s, {})
            lesion, diagnosis = "no_lesion", "NO_LESION"
            age = info.get("age", "not_provided")
            gender = "not_provided"
            loc = info.get("localization", "not_provided")
            return (lesion, diagnosis, age, gender, loc)
        else:
            info = {}

        lesion = (info.get("lesion", "") or "").lower() or lesion
        diagnosis = info.get("diagnosis", "") or diagnosis
        age = info.get("age", age)
        # prefer "gender", fall back to "sex" if any legacy dict slips through
        gender = info.get("gender", info.get("sex", gender))
        loc = info.get("localization", loc)
        return (lesion, diagnosis, age, gender, loc)

    # Phase 1: copy images with dedup
    for origin in priority:
        for p in paths.get(origin, []):
            fn = os.path.basename(p)
            s = os.path.splitext(fn)[0]
            s_upper = s.upper()

            lesion, diagnosis, age, gender, loc = build_info(origin, p, fn, s, s_upper)
            if origin == "ITOBOS2024" and not lesion:
                # filtered out (non-healthy)
                continue

            # normalize values
            allowed = {"no_lesion","benign","malignant","unknown"}
            if lesion and lesion not in allowed:
                lesion = ""
            diagnosis = to_canon(diagnosis)
            age = normalize_age(age)
            gender = normalize_gender(gender)
            loc = normalize_localization(loc)

            # compute hash and dedup across datasets
            h = file_hash(p)
            if h in hash_to_name:
                continue  # already copied from a higher-priority dataset

            # resolve final filename
            desired = fn
            dst = os.path.join(images_out, desired)
            if os.path.exists(dst):
                try:
                    if file_hash(dst) == h:
                        # already present
                        hash_to_name[h] = desired
                    else:
                        desired = safe_unique_name(images_out, desired)
                except Exception:
                    desired = safe_unique_name(images_out, desired)

            # avoid in-run name clash
            if desired in name_to_hash and name_to_hash[desired] != h:
                desired = safe_unique_name(images_out, desired)

            # copy
            final_path = os.path.join(images_out, desired)
            ensure_dir(os.path.dirname(final_path))
            shutil.copy2(p, final_path)

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
                "lesion_type": lesion or "unknown",
                "diagnosis": diagnosis or "UNKNOWN",
                "body_region": loc or "not_provided",
                "age": age or "not_provided",
                "gender": gender or "not_provided",
            }
            copied_by_origin[origin_out] += 1

    print("[COPY] Copied unique files per origin_dataset:")
    for k in ["ITOBOS2024","ISIC2020","ISIC2019","HAM1000","MIL10K"]:
        print(f"  - {k}: {copied_by_origin.get(k, 0)}")
    print(f"[COPY] Total unique images: {len(hash_to_meta)}")
    print(f"- Images copied to: {images_out}")

    # If only images, stop here
    if images_only:
        print("[MODE] Images-only → skipping metadata.csv")
        print("✅ Done")
        return

    # Phase 2: write metadata
    rows = list(hash_to_meta.values())
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "metadata.csv")
    pd.DataFrame(rows, columns=["image_id","origin_dataset","lesion_type","diagnosis","body_region","age","gender"]).to_csv(
        out_csv, index=False, quoting=csv.QUOTE_MINIMAL
    )
    print("✅ Done")
    print(f"- Metadata written to: {out_csv}")

def regenerate_metadata(images_out: str, out_dir: str,
                        ham, is19, is20, mil_isic, mil_il, itobos_ok: Set[str], itobos_meta: Dict[str, Dict]):
    # scan images
    image_files = []
    for root, _, files in os.walk(images_out):
        for file in files:
            if is_image(file):
                image_files.append(os.path.join(root, file))
    print(f"[INFO] Found {len(image_files)} existing images")

    rows = []
    for img_path in sorted(image_files):
        fn = os.path.basename(img_path)
        s = os.path.splitext(fn)[0]
        sU = s.upper()

        info = {}
        origin = "UNKNOWN"

        # Try dataset-specific metadata to infer origin and fields
        if sU in ham or s in ham or s.lower() in ham:
            info.update(ham.get(sU) or ham.get(s) or ham.get(s.lower()) or {})
            origin = "HAM1000"
        elif sU in is20 or s in is20 or s.lower() in is20:
            info.update(is20.get(sU) or is20.get(s) or is20.get(s.lower()) or {})
            origin = "ISIC2020"
        elif sU in is19 or s in is19 or s.lower() in is19:
            info.update(is19.get(sU) or is19.get(s) or is19.get(s.lower()) or {})
            origin = "ISIC2019"
        elif sU in mil_isic:
            info.update(mil_isic.get(sU, {}))
            origin = "MIL10K"
        elif sU in mil_il:
            info.update(mil_il.get(sU, {}))
            origin = "MIL10K"
        elif fn in itobos_ok:
            info.update(itobos_meta.get(s, {}))
            info["lesion"] = "no_lesion"
            info["diagnosis"] = "NO_LESION"
            origin = "ITOBOS2024"

        lesion = (info.get("lesion","unknown") or "unknown")
        diagnosis = (info.get("diagnosis","UNKNOWN") or "UNKNOWN")
        age = normalize_age(info.get("age",""))
        gender = normalize_gender(info.get("gender", info.get("sex","")))
        loc = normalize_localization(info.get("localization",""))

        rows.append({
            "image_id": fn,
            "origin_dataset": origin,
            "lesion_type": lesion,
            "diagnosis": diagnosis,
            "body_region": loc,
            "age": age,
            "gender": gender,
        })

    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "metadata.csv")
    pd.DataFrame(rows, columns=["image_id","origin_dataset","lesion_type","diagnosis","body_region","age","gender"]).to_csv(
        out_csv, index=False, quoting=csv.QUOTE_MINIMAL
    )
    print("Metadata regeneration complete")
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
    ap.add_argument("--skip_dedup", action="store_true", help="Skip copying/dedup (assume images already exist); still generates metadata")
    args = ap.parse_args()

    if args.images_only and args.metadata_only:
        print("Error: Cannot use both --images_only and --metadata_only")
        return

    build_dataset(args.raw_dir, args.out_dir,
                  images_only=args.images_only,
                  metadata_only=args.metadata_only,
                  skip_dedup=args.skip_dedup)

if __name__ == "__main__":
    main()
