# src/train_simple_twohead.py
# -*- coding: utf-8 -*-
from pathlib import Path
import json, os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------------------- Repro --------------------
SEED = 999
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------- Config --------------------
DATA_DIR = Path("../data")
PREPARED_CSV = DATA_DIR / "training_prepared_data.csv"
IMAGE_PATH = DATA_DIR.joinpath("images", "images")
OUTDIR = Path("outputs/simple_twohead_b0_v2")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50                  # Increased for better convergence
LR = 1e-4                   # Reduced for better fine-tuning
WEIGHT_DECAY = 1e-4
PRETRAINED = True

# imbalance / training knobs
USE_FOCAL_COARSE = True
FOCAL_GAMMA = 2.0
# alpha for coarse focal will be calculated from class frequency

USE_SAMPLE_WEIGHTS = True          # per-example weights via effective number
CLASS_BALANCED_BETA = 0.999

USE_OVERSAMPLING = True            # moderate oversampling on coarse classes
OVERSAMPLE_WEIGHTS = {
    '0': 0.20,  # benign (reduced from 0.45)
    '1': 0.50,  # malignant (increased from 0.40)
    '2': 0.30,  # no_lesion (doubled from 0.15)
}

# Fine-grained minority class oversampling
USE_FINE_OVERSAMPLING = True
FINE_MINORITY_OVERSAMPLING = {
    'df': 5.0,      # 5x oversampling
    'vasc': 5.0,    # 5x oversampling
    'other': 10.0,  # 10x oversampling
}

USE_OOD_OE = True                  # add Outlier Exposure on unknown
LAMBDA_OE = 0.1                    # scale for OE uniform loss (fine head)

# -------------------- Labels --------------------
DX_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'scc_akiec', 'vasc', 'df', 'other', 'no_lesion']  # removed 'unknown'
LESION_TYPE_CLASSES = ["benign", "malignant", "no_lesion"]
N_DX_CLASSES = len(DX_CLASSES)
N_LESION_TYPE_CLASSES = len(LESION_TYPE_CLASSES)
DX_TO_ID = {n: i for i, n in enumerate(DX_CLASSES)}
LESION_TO_ID = {n: i for i, n in enumerate(LESION_TYPE_CLASSES)}

# -------------------- Utils --------------------
def set_seed(seed=42):
    import os, random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def class_balanced_weights(counts, beta=0.999):
    counts = np.asarray(counts, dtype=np.float64)
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / np.maximum(eff_num, 1e-8)
    w = w / w.mean()
    return w.astype(np.float32)

def calculate_focal_alpha(class_counts):
    """Calculate proper alpha values for focal loss based on class frequency."""
    total = sum(class_counts)
    alpha = [total / (len(class_counts) * count) for count in class_counts]
    return np.array(alpha, dtype=np.float32)

def calculate_class_weights(class_counts):
    """Calculate inverse frequency weights for class balancing."""
    total = sum(class_counts)
    weights = [total / (len(class_counts) * count) for count in class_counts]
    return np.array(weights, dtype=np.float32)

# -------------------- Augmentation --------------------
def build_augmenter(is_training):
    if is_training:
        # multiplicative random brightness with a darkening bias
        def random_brightness_mul(x):
            x = tf.cast(x, tf.float32)
            r = tf.random.uniform([])
            factor = tf.where(r < 0.7,  # 70% darken
                              tf.random.uniform([], 0.6, 1.0),
                              tf.random.uniform([], 1.0, 1.15))  # 30% brighten
            x = x * factor
            return tf.clip_by_value(x, 0.0, 255.0)

        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.RandomCrop(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.1),
            layers.Lambda(random_brightness_mul, name="random_brightness_bias_dark"),
        ], name="augmenter")
    else:
        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.CenterCrop(IMG_SIZE, IMG_SIZE),
        ], name="preprocessor")

# stronger photometric aug for minority
def augment_minority(img):
    # mild contrast/saturation jitter + tiny gaussian noise
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=3.0)  # in [0,255] scale
    img = tf.clip_by_value(img + noise, 0.0, 255.0)
    return img

# -------------------- Dataset --------------------
def build_dataset(df, is_training, minority_fine_ids=None, fine_oversampling=None):
    """
    Emits (image, {'fine_output':y1, 'coarse_output':y2}, {'fine_output':sw1, 'coarse_output':sw2})
    Fine labels: int32 with special values:
      -1 -> ignore (masked)
      -2 -> OOD (apply uniform OE loss in the fine loss)
    Coarse labels: int32 in [0..2]; for rows that truly lack coarse, we put 0 but weight=0.
    """
    df = df.copy()
    # --- resolve paths ---
    def resolve_path(p):
        p = str(p)
        return p if os.path.isabs(p) else str(IMAGE_PATH / p)
    df["image_path"] = df["image_path"].astype(str).apply(resolve_path)

    # ---- labels arrays ----
    df_fine = df["head1_idx"].astype("int32").values
    df_coarse = df["head2_idx"].fillna(0).astype("int32").values

    # minority-targeted aug condition: coarse==1 (malignant) or fine in rare ids
    if minority_fine_ids is None:
        minority_fine_ids = set()  # can pass e.g., {DX_TO_ID['df'], DX_TO_ID['vasc'], DX_TO_ID['other'], DX_TO_ID['no_lesion']}
    
    # Apply fine-grained oversampling if specified
    if fine_oversampling is not None and is_training:
        df = apply_fine_oversampling(df, fine_oversampling)
        # Recalculate labels after oversampling
        df_fine = df["head1_idx"].astype("int32").values
        df_coarse = df["head2_idx"].fillna(0).astype("int32").values

    img_paths = df["image_path"].tolist()

    ds = tf.data.Dataset.from_tensor_slices((img_paths, df_fine, df_coarse))
    if is_training:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)

    augmenter = build_augmenter(is_training)
    rescale = layers.Rescaling(1. / 255.0)
    normalization_layer = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2]
    )

    @tf.function
    def load_and_preprocess(path, label_fine, label_coarse):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)  # uint8 [0..255]

        # base aug
        img = augmenter(img)

        # stronger aug for minority (training only implicit via pipeline; here gate by label)
        if is_training:
            minority = tf.logical_or(
                tf.equal(label_coarse, 1),  # malignant
                tf.reduce_any(tf.equal(label_fine, tf.constant(list(minority_fine_ids), dtype=tf.int32)))
                if len(minority_fine_ids) > 0 else tf.constant(False)
            )
            img = tf.cond(minority, lambda: augment_minority(img), lambda: img)

        img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
        img = rescale(img)               # [0,1]
        img = normalization_layer(img)   # ImageNet mean/var

        labels = {"fine_output": label_fine, "coarse_output": label_coarse}
        return {"input_rgb": img}, labels

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def apply_fine_oversampling(df, fine_oversampling):
    """Apply oversampling to fine-grained minority classes."""
    df_list = [df]
    
    for class_name, multiplier in fine_oversampling.items():
        if class_name in DX_TO_ID:
            class_id = DX_TO_ID[class_name]
            minority_samples = df[df["head1_idx"] == class_id]
            if len(minority_samples) > 0:
                # Repeat samples to achieve desired multiplier
                n_repeats = int(multiplier) - 1
                if n_repeats > 0:
                    repeated_samples = pd.concat([minority_samples] * n_repeats, ignore_index=True)
                    df_list.append(repeated_samples)
    
    return pd.concat(df_list, ignore_index=True)

# -------------------- Model --------------------
def create_two_head_model(n_fine, n_coarse, img_size=IMG_SIZE, dropout=0.2):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_rgb")
    backbone = tf.keras.applications.efficientnet.EfficientNetB1(
        include_top=False,
        weights="imagenet" if PRETRAINED else None,
        input_shape=(img_size, img_size, 3),
        # include_preprocessing=False,
    )
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout, name="top_dropout")(x)
    out_fine = layers.Dense(n_fine, name="fine_output")(x)       # logits
    out_coarse = layers.Dense(n_coarse, name="coarse_output")(x) # logits
    return keras.Model(inputs={'input_rgb': inputs}, outputs=[out_fine, out_coarse], name="EffB1TwoHead")
# -------------------- Losses --------------------
def masked_sparse_ce_with_oe(y_true, y_pred):
    """
    Fine head loss:
      - y_true == -1 : ignore
      - y_true == -2 : OOD → uniform (Outlier Exposure) loss scaled by LAMBDA_OE
      - else         : standard sparse CE
    """
    y_true = tf.cast(y_true, tf.int32)
    num_classes = tf.shape(y_pred)[-1]

    mask_valid = tf.logical_and(y_true >= 0, y_true < num_classes)
    mask_ignore = tf.equal(y_true, -1)
    mask_ood = tf.equal(y_true, -2)

    # CE for valid
    y_true_safe = tf.where(mask_valid, y_true, tf.zeros_like(y_true))
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_safe, y_pred, from_logits=True)

    # Uniform OE for OOD
    log_p = tf.nn.log_softmax(y_pred, axis=-1)
    k = tf.cast(num_classes, log_p.dtype)
    oe = -tf.reduce_mean(log_p, axis=-1)  # equals CE to uniform = -mean log p
    oe = LAMBDA_OE * oe

    per_example = tf.where(mask_valid, ce, tf.zeros_like(ce))
    per_example = tf.where(mask_ood, oe, per_example)

    denom = tf.reduce_sum(tf.cast(tf.logical_or(mask_valid, mask_ood), per_example.dtype))
    return tf.math.divide_no_nan(tf.reduce_sum(per_example), denom)

def sparse_categorical_focal_loss(gamma=2.0, alpha=None):
    """
    Focal loss for coarse head. alpha can be scalar or per-class vector tensor length C.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
        # probs
        p = tf.nn.softmax(y_pred, axis=-1)
        p_t = tf.reduce_sum(y_true_oh * p, axis=-1)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        mod = tf.pow(1.0 - p_t, gamma)
        fl = mod * ce
        if alpha is not None:
            alpha_vec = tf.convert_to_tensor(alpha, dtype=fl.dtype)
            alpha_t = tf.reduce_sum(y_true_oh * alpha_vec, axis=-1)
            fl = alpha_t * fl
        return tf.reduce_mean(fl)
    return loss

# -------------------- Main --------------------
def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)

    # ---- Load CSV ----
    df = pd.read_csv(PREPARED_CSV)

    # ---- Coarse labels from text if needed ----
    if "lesion_type" in df.columns:
        lt = df["lesion_type"].astype(str).str.strip().str.lower()
        map_coarse = {"benign": 0, "malignant": 1, "no_lesion": 2}
        # keep existing numeric where valid, else fill from text
        df["head2_idx"] = pd.to_numeric(df.get("head2_idx"), errors="coerce")
        to_fill = df["head2_idx"].isna() & lt.isin(map_coarse.keys())
        df.loc[to_fill, "head2_idx"] = lt[to_fill].map(map_coarse)
    # cast + sanity
    df["head2_idx"] = pd.to_numeric(df.get("head2_idx"), errors="coerce")
    bad_coarse = df["head2_idx"].isna() | (df["head2_idx"] < 0) | (df["head2_idx"] >= N_LESION_TYPE_CLASSES)
    if bad_coarse.any():
        print(f"[coarse] dropping {int(bad_coarse.sum())} rows with invalid coarse labels")
        df = df[~bad_coarse].copy()

    # ---- Fine labels (respect your numeric mapping; mask OOR) ----
    # If you also have text dx columns and want to mark unknown as OOD:
    is_ood = np.zeros(len(df), dtype=bool)
    text_dx_col = None
    for c in ["dx_merged", "dx", "diagnosis", "fine_label"]:
        if c in df.columns:
            text_dx_col = c
            break
    if text_dx_col is not None:
        dx_txt = df[text_dx_col].astype(str).str.strip().str.lower()
        is_ood = dx_txt.eq("unknown").values
    # use existing numeric fine labels
    df["head1_idx"] = pd.to_numeric(df.get("head1_idx"), errors="coerce")
    mask_bad_fine = df["head1_idx"].isna() | (df["head1_idx"] < 0) | (df["head1_idx"] >= N_DX_CLASSES)
    df.loc[mask_bad_fine, "head1_idx"] = -1  # ignore
    # set OOD sentinel -2 if requested
    if USE_OOD_OE and is_ood.any():
        df.loc[is_ood, "head1_idx"] = -2

    df["head1_idx"] = df["head1_idx"].astype("int32")

    # ---- Stratified Split ----
    # Use stratified split to ensure representative validation set
    if "split" in df.columns:
        train_df = df[df.split == "train"].copy()
        val_df = df[df.split == "val"].copy()
        print("Using existing train/val split")
    else:
        # Create stratified split based on coarse labels
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['head2_idx'], random_state=SEED
        )
        print("Created stratified train/val split")

    # ---- Class weights from TRAIN ONLY ----
    def counts_from_labels(series, n_classes, valid_range=(0, None)):
        y = pd.to_numeric(series, errors="coerce").astype("float")
        lo = valid_range[0]; hi = valid_range[1] if valid_range[1] is not None else np.inf
        y = y[(y >= lo) & (y < (hi if hi != np.inf else np.inf))]
        y = y.dropna().astype(int).values
        counts = np.bincount(y, minlength=n_classes)
        return counts

    coarse_counts = counts_from_labels(train_df["head2_idx"], N_LESION_TYPE_CLASSES, (0, N_LESION_TYPE_CLASSES))
    fine_counts = counts_from_labels(train_df["head1_idx"], N_DX_CLASSES, (0, N_DX_CLASSES))
    print("coarse_counts:", coarse_counts)
    print("fine_counts:", fine_counts)
    print("coarse_class_distribution:", coarse_counts / coarse_counts.sum())
    print("fine_class_distribution:", fine_counts / fine_counts.sum())

    coarse_w = class_balanced_weights(coarse_counts, beta=CLASS_BALANCED_BETA)
    fine_w = class_balanced_weights(fine_counts, beta=CLASS_BALANCED_BETA)
    print("coarse_w:", coarse_w)
    print("fine_w:", fine_w)
    
    # Calculate focal loss alpha values
    coarse_alpha = calculate_focal_alpha(coarse_counts)
    fine_alpha = calculate_focal_alpha(fine_counts)
    print("coarse_alpha (for focal loss):", coarse_alpha)
    print("fine_alpha (for focal loss):", fine_alpha)

    # ---- Minority fine ids for stronger aug ----
    minority_fine_names = ["df", "vasc", "other", "no_lesion"]
    minority_fine_ids = {DX_TO_ID[n] for n in minority_fine_names if n in DX_TO_ID}

    # ---- Build datasets ----
    if USE_OVERSAMPLING:
        # Build one ds per coarse class, then mix
        ds_parts = []
        weights = []
        for c in [0, 1, 2]:
            sub = train_df[train_df["head2_idx"] == c]
            if len(sub) == 0:
                continue
            ds_c = build_dataset(sub, is_training=True, minority_fine_ids=minority_fine_ids, 
                                 fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)
            ds_parts.append(ds_c)
            weights.append(OVERSAMPLE_WEIGHTS.get(str(c), 0.0))
        # normalize weights
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        train_ds = tf.data.Dataset.sample_from_datasets(
            ds_parts, weights=weights.tolist(), stop_on_empty_dataset=False
        )
    else:
        train_ds = build_dataset(train_df, is_training=True, minority_fine_ids=minority_fine_ids,
                                 fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)

    val_ds = build_dataset(val_df, is_training=False, minority_fine_ids=minority_fine_ids)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # ---- Model ----
    model = create_two_head_model(N_DX_CLASSES, N_LESION_TYPE_CLASSES)

    # ---- Losses ----
    if USE_FOCAL_COARSE:
        # Use proper focal loss alpha based on class frequency
        coarse_loss = sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=coarse_alpha)
    else:
        coarse_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # ---- Compile ----
    # Learning rate scheduling
    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR,
        decay_steps=total_steps,
        alpha=0.01
    )
    
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
    
    model.compile(
        optimizer=optimizer,
        loss={
            "fine_output": masked_sparse_ce_with_oe,
            "coarse_output": coarse_loss,
        },
        metrics={
            "fine_output": ["sparse_categorical_accuracy"],
            "coarse_output": ["sparse_categorical_accuracy"],
        },
    )

    # ---- Callbacks ----
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(OUTDIR.joinpath("best_model.keras")),
            save_best_only=True,
            monitor="val_coarse_output_loss",  # Monitor coarse loss specifically
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(OUTDIR.joinpath("history.csv"))),
        
        # TensorBoard logging
        keras.callbacks.TensorBoard(
            log_dir=str(OUTDIR / "tensorboard_logs"),
            histogram_freq=1,  # Log histograms every epoch
            write_graph=True,   # Log the model graph
            write_images=True,  # Log model weights as images
            update_freq='epoch',  # Log at the end of each epoch
            profile_batch=0,    # Disable profiling for better performance
        ),

        # Reduce LR with more patience
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=5, min_delta=1e-4,
            cooldown=2, min_lr=1e-7, verbose=1
        ),

        # Early stopping with increased patience
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,  # Increased from 4
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        
        # Additional callback for monitoring training progress
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch}: Fine Acc={logs.get('fine_output_sparse_categorical_accuracy', 0):.4f}, Coarse Acc={logs.get('coarse_output_sparse_categorical_accuracy', 0):.4f}")
        ),
    ]

    # ---- Train ----
    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Save history and generate evaluation plots ----
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / "stats.json", "w") as f:
        json.dump(history.history, f)
    
    # Evaluate on validation set
    evaluate_model(model, val_ds, val_df, OUTDIR)
    
    print(f"Training complete. Best model and stats saved to '{OUTDIR}'.")
    print("Tip: monitor balanced metrics (macro-F1) offline for class-imbalance reporting.")
    print("Generated detailed metrics and reports.")
    print(f"\n📊 TensorBoard logs saved to: {OUTDIR / 'tensorboard_logs'}")
    print("To view TensorBoard, run: tensorboard --logdir=" + str(OUTDIR / "tensorboard_logs"))

def evaluate_model(model, val_ds, val_df, outdir):
    """Comprehensive model evaluation with class-specific metrics."""
    print("\nEvaluating model on validation set...")
    
    # Get predictions
    predictions = model.predict(val_ds, verbose=1)
    fine_preds = predictions[0]
    coarse_preds = predictions[1]
    
    # Convert to class predictions
    fine_pred_classes = np.argmax(fine_preds, axis=1)
    coarse_pred_classes = np.argmax(coarse_preds, axis=1)
    
    # Get true labels
    fine_true = val_df['head1_idx'].values
    coarse_true = val_df['head2_idx'].values
    
    # Filter out masked samples for fine evaluation
    valid_fine_mask = fine_true >= 0
    fine_true_valid = fine_true[valid_fine_mask]
    fine_pred_valid = fine_pred_classes[valid_fine_mask]
    
    # Generate classification reports
    print("\nFine-grained Classification Report:")
    fine_report = classification_report(fine_true_valid, fine_pred_valid, 
                                       target_names=DX_CLASSES, digits=4)
    print(fine_report)
    
    print("\nCoarse Classification Report:")
    coarse_report = classification_report(coarse_true, coarse_pred_classes,
                                         target_names=LESION_TYPE_CLASSES, digits=4)
    print(coarse_report)
    
    # Save reports
    with open(outdir / "fine_classification_report.txt", "w") as f:
        f.write(str(fine_report))
    with open(outdir / "coarse_classification_report.txt", "w") as f:
        f.write(str(coarse_report))
    
    # Print confusion matrices to console
    print("\nFine-grained Confusion Matrix:")
    cm_fine = confusion_matrix(fine_true_valid, fine_pred_valid)
    print(cm_fine)
    
    print("\nCoarse Confusion Matrix:")
    cm_coarse = confusion_matrix(coarse_true, coarse_pred_classes)
    print(cm_coarse)
    
    print(f"\nEvaluation complete. Reports saved to {outdir}")

if __name__ == "__main__":
    main()
