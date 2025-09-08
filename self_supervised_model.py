# src/self_supervised_model.py
# -*- coding: utf-8 -*-
"""
Self-Supervised Learning + Fine-tuning for Dermatology Classification
Implements SimCLR (Simple Contrastive Learning) followed by fine-tuning
"""

from pathlib import Path
import json, os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

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

# Self-supervised learning config
SSL_OUTDIR = Path("outputs/ssl_simclr")
FINE_TUNE_OUTDIR = Path("outputs/ssl_finetuned")

IMG_SIZE = 224
BATCH_SIZE = 32
SSL_EPOCHS = 50
FINE_TUNE_EPOCHS = 30
LR_SSL = 1e-3
LR_FINE_TUNE = 1e-4
WEIGHT_DECAY = 1e-4
PRETRAINED = True

# SimCLR specific parameters
TEMPERATURE = 0.1
PROJECTION_DIM = 128
HIDDEN_DIM = 512

# Fine-tuning config (same as first_model.py)
USE_FOCAL_COARSE = True
FOCAL_GAMMA = 2.0
USE_SAMPLE_WEIGHTS = True
CLASS_BALANCED_BETA = 0.999
USE_OVERSAMPLING = True
OVERSAMPLE_WEIGHTS = {
    '0': 0.20,  # benign
    '1': 0.50,  # malignant
    '2': 0.30,  # no_lesion
}
USE_FINE_OVERSAMPLING = True
FINE_MINORITY_OVERSAMPLING = {
    'df': 5.0,
    'vasc': 5.0,
    'other': 10.0,
}
USE_OOD_OE = True
LAMBDA_OE = 0.1

# -------------------- Labels --------------------
DX_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'scc_akiec', 'vasc', 'df', 'other', 'no_lesion']
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

# -------------------- SimCLR Data Augmentation --------------------
def simclr_augment(image, img_size=IMG_SIZE):
    """
    SimCLR-style data augmentation for contrastive learning.
    Applies two different augmentations to the same image.
    """
    # Convert to float32
    image = tf.cast(image, tf.float32)
    
    # Random brightness adjustment
    def random_brightness(x):
        r = tf.random.uniform([])
        factor = tf.where(r < 0.7,  # 70% darken
                          tf.random.uniform([], 0.6, 1.0),
                          tf.random.uniform([], 1.0, 1.15))  # 30% brighten
        x = x * factor
        return tf.clip_by_value(x, 0.0, 255.0)
    
    # Random color jitter
    def color_jitter(x):
        x = tf.image.random_contrast(x, 0.8, 1.2)
        x = tf.image.random_saturation(x, 0.8, 1.2)
        x = tf.image.random_hue(x, 0.1)
        return tf.clip_by_value(x, 0.0, 255.0)
    
    # Random Gaussian blur
    def gaussian_blur(x):
        sigma = tf.random.uniform([], 0.1, 2.0)
        return tf.image.gaussian_filter2d(x, filter_shape=[5, 5], sigma=sigma)
    
    # Apply augmentations
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_rotation(image, 0.2)
    image = random_brightness(image)
    image = color_jitter(image)
    
    # Randomly apply blur
    if tf.random.uniform([]) < 0.5:
        image = gaussian_blur(image)
    
    # Resize and crop
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, [img_size, img_size, 3])
    
    return image

def create_simclr_dataset(df, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create dataset for SimCLR training with paired augmentations.
    """
    df = df.copy()
    
    # Resolve image paths
    def resolve_path(p):
        p = str(p)
        return p if os.path.isabs(p) else str(IMAGE_PATH / p)
    df["image_path"] = df["image_path"].astype(str).apply(resolve_path)
    
    img_paths = df["image_path"].tolist()
    
    ds = tf.data.Dataset.from_tensor_slices(img_paths)
    ds = ds.shuffle(len(df), reshuffle_each_iteration=True)
    
    rescale = layers.Rescaling(1. / 255.0)
    normalization_layer = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2]
    )
    
    @tf.function
    def load_and_augment(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        
        # Create two different augmentations of the same image
        aug1 = simclr_augment(img, img_size)
        aug2 = simclr_augment(img, img_size)
        
        # Normalize both
        aug1 = rescale(aug1)
        aug1 = normalization_layer(aug1)
        aug2 = rescale(aug2)
        aug2 = normalization_layer(aug2)
        
        return aug1, aug2
    
    ds = ds.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------- SimCLR Model --------------------
def create_simclr_model(img_size=IMG_SIZE, projection_dim=PROJECTION_DIM, hidden_dim=HIDDEN_DIM):
    """
    Create SimCLR model with encoder and projection head.
    """
    # Encoder (backbone)
    encoder = tf.keras.applications.efficientnet.EfficientNetB1(
        include_top=False,
        weights="imagenet" if PRETRAINED else None,
        input_shape=(img_size, img_size, 3),
    )
    
    # Projection head
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")
    features = encoder(inputs)
    features = layers.GlobalAveragePooling2D()(features)
    
    # Projection head
    projection = layers.Dense(hidden_dim, activation="relu", name="projection_hidden")(features)
    projection = layers.Dense(projection_dim, name="projection_output")(projection)
    
    model = keras.Model(inputs=inputs, outputs=projection, name="SimCLR_Encoder")
    return model

def simclr_loss(temperature=TEMPERATURE):
    """
    SimCLR contrastive loss function.
    """
    def loss_fn(projections):
        # Normalize projections
        projections = tf.nn.l2_normalize(projections, axis=1)
        
        # Split into two views
        batch_size = tf.shape(projections)[0] // 2
        proj1 = projections[:batch_size]
        proj2 = projections[batch_size:]
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(proj1, proj2, transpose_b=True) / temperature
        
        # Create labels (diagonal elements are positive pairs)
        labels = tf.eye(batch_size)
        
        # Compute loss
        loss = tf.keras.losses.categorical_crossentropy(
            labels, similarity_matrix, from_logits=True
        )
        
        return tf.reduce_mean(loss)
    
    return loss_fn

# -------------------- Fine-tuning Dataset (reuse from first_model.py) --------------------
def build_augmenter(is_training):
    if is_training:
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

def augment_minority(img):
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=3.0)
    img = tf.clip_by_value(img + noise, 0.0, 255.0)
    return img

def build_dataset(df, is_training, minority_fine_ids=None, fine_oversampling=None):
    """
    Build dataset for fine-tuning (same as first_model.py).
    """
    df = df.copy()
    
    def resolve_path(p):
        p = str(p)
        return p if os.path.isabs(p) else str(IMAGE_PATH / p)
    df["image_path"] = df["image_path"].astype(str).apply(resolve_path)

    df_fine = df["head1_idx"].astype("int32").values
    df_coarse = df["head2_idx"].fillna(0).astype("int32").values

    if minority_fine_ids is None:
        minority_fine_ids = set()
    
    if fine_oversampling is not None and is_training:
        df = apply_fine_oversampling(df, fine_oversampling)
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
        img = tf.image.decode_jpeg(img, channels=3)

        img = augmenter(img)

        if is_training:
            minority = tf.logical_or(
                tf.equal(label_coarse, 1),  # malignant
                tf.reduce_any(tf.equal(label_fine, tf.constant(list(minority_fine_ids), dtype=tf.int32)))
                if len(minority_fine_ids) > 0 else tf.constant(False)
            )
            img = tf.cond(minority, lambda: augment_minority(img), lambda: img)

        img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
        img = rescale(img)
        img = normalization_layer(img)

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
                n_repeats = int(multiplier) - 1
                if n_repeats > 0:
                    repeated_samples = pd.concat([minority_samples] * n_repeats, ignore_index=True)
                    df_list.append(repeated_samples)
    
    return pd.concat(df_list, ignore_index=True)

# -------------------- Fine-tuning Model --------------------
def create_finetuned_model(ssl_encoder, n_fine, n_coarse, img_size=IMG_SIZE, dropout=0.2):
    """
    Create fine-tuned model using pre-trained SSL encoder.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_rgb")
    
    # Use the pre-trained encoder (without projection head)
    backbone = ssl_encoder.layers[1]  # EfficientNetB1 layer
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout, name="top_dropout")(x)
    
    out_fine = layers.Dense(n_fine, name="fine_output")(x)
    out_coarse = layers.Dense(n_coarse, name="coarse_output")(x)
    
    return keras.Model(inputs={'input_rgb': inputs}, outputs=[out_fine, out_coarse], name="SSL_FineTuned")

# -------------------- Losses (same as first_model.py) --------------------
def masked_sparse_ce_with_oe(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    num_classes = tf.shape(y_pred)[-1]

    mask_valid = tf.logical_and(y_true >= 0, y_true < num_classes)
    mask_ignore = tf.equal(y_true, -1)
    mask_ood = tf.equal(y_true, -2)

    y_true_safe = tf.where(mask_valid, y_true, tf.zeros_like(y_true))
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_safe, y_pred, from_logits=True)

    log_p = tf.nn.log_softmax(y_pred, axis=-1)
    k = tf.cast(num_classes, log_p.dtype)
    oe = -tf.reduce_mean(log_p, axis=-1)
    oe = LAMBDA_OE * oe

    per_example = tf.where(mask_valid, ce, tf.zeros_like(ce))
    per_example = tf.where(mask_ood, oe, per_example)

    denom = tf.reduce_sum(tf.cast(tf.logical_or(mask_valid, mask_ood), per_example.dtype))
    return tf.math.divide_no_nan(tf.reduce_sum(per_example), denom)

def sparse_categorical_focal_loss(gamma=2.0, alpha=None):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(y_true, depth=num_classes)
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

# -------------------- Main Functions --------------------
def train_ssl_model():
    """
    Train SimCLR model using self-supervised learning.
    """
    print("Starting SimCLR self-supervised training...")
    SSL_OUTDIR.mkdir(exist_ok=True, parents=True)
    
    # Load data
    df = pd.read_csv(PREPARED_CSV)
    
    # Use all available data for SSL (including unlabeled)
    ssl_df = df.copy()
    print(f"Using {len(ssl_df)} samples for SSL training")
    
    # Create SSL dataset
    ssl_ds = create_simclr_dataset(ssl_df)
    
    # Create SimCLR model
    ssl_model = create_simclr_model()
    
    # Compile model
    optimizer = keras.optimizers.AdamW(learning_rate=LR_SSL, weight_decay=WEIGHT_DECAY)
    ssl_model.compile(
        optimizer=optimizer,
        loss=simclr_loss(),
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(SSL_OUTDIR / "ssl_best_model.keras"),
            save_best_only=True,
            monitor="loss",
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(SSL_OUTDIR / "ssl_history.csv")),
        keras.callbacks.TensorBoard(
            log_dir=str(SSL_OUTDIR / "ssl_tensorboard_logs"),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
        ),
        keras.callbacks.EarlyStopping(
            monitor="loss",
            mode="min",
            patience=10,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    
    # Train
    print("Training SimCLR model...")
    history = ssl_model.fit(
        ssl_ds,
        epochs=SSL_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save history
    with open(SSL_OUTDIR / "ssl_stats.json", "w") as f:
        json.dump(history.history, f)
    
    print(f"SSL training complete. Model saved to '{SSL_OUTDIR}'.")
    return ssl_model

def fine_tune_model(ssl_model):
    """
    Fine-tune the SSL model on labeled data.
    """
    print("Starting fine-tuning...")
    FINE_TUNE_OUTDIR.mkdir(exist_ok=True, parents=True)
    
    # Load data
    df = pd.read_csv(PREPARED_CSV)
    
    # Process labels (same as first_model.py)
    if "lesion_type" in df.columns:
        lt = df["lesion_type"].astype(str).str.strip().str.lower()
        map_coarse = {"benign": 0, "malignant": 1, "no_lesion": 2}
        df["head2_idx"] = pd.to_numeric(df.get("head2_idx"), errors="coerce")
        to_fill = df["head2_idx"].isna() & lt.isin(map_coarse.keys())
        df.loc[to_fill, "head2_idx"] = lt[to_fill].map(map_coarse)
    
    df["head2_idx"] = pd.to_numeric(df.get("head2_idx"), errors="coerce")
    bad_coarse = df["head2_idx"].isna() | (df["head2_idx"] < 0) | (df["head2_idx"] >= N_LESION_TYPE_CLASSES)
    if bad_coarse.any():
        print(f"[coarse] dropping {int(bad_coarse.sum())} rows with invalid coarse labels")
        df = df[~bad_coarse].copy()

    # Handle fine labels
    is_ood = np.zeros(len(df), dtype=bool)
    text_dx_col = None
    for c in ["dx_merged", "dx", "diagnosis", "fine_label"]:
        if c in df.columns:
            text_dx_col = c
            break
    if text_dx_col is not None:
        dx_txt = df[text_dx_col].astype(str).str.strip().str.lower()
        is_ood = dx_txt.eq("unknown").values
    
    df["head1_idx"] = pd.to_numeric(df.get("head1_idx"), errors="coerce")
    mask_bad_fine = df["head1_idx"].isna() | (df["head1_idx"] < 0) | (df["head1_idx"] >= N_DX_CLASSES)
    df.loc[mask_bad_fine, "head1_idx"] = -1  # ignore
    
    if USE_OOD_OE and is_ood.any():
        df.loc[is_ood, "head1_idx"] = -2
    
    df["head1_idx"] = df["head1_idx"].astype("int32")

    # Split data
    if "split" in df.columns:
        train_df = df[df.split == "train"].copy()
        val_df = df[df.split == "val"].copy()
        print("Using existing train/val split")
    else:
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['head2_idx'], random_state=SEED
        )
        print("Created stratified train/val split")

    # Calculate class weights
    def counts_from_labels(series, n_classes, valid_range=(0, None)):
        y = pd.to_numeric(series, errors="coerce").astype("float")
        lo = valid_range[0]; hi = valid_range[1] if valid_range[1] is not None else np.inf
        y = y[(y >= lo) & (y < (hi if hi != np.inf else np.inf))]
        y = y.dropna().astype(int).values
        counts = np.bincount(y, minlength=n_classes)
        return counts

    coarse_counts = counts_from_labels(train_df["head2_idx"], N_LESION_TYPE_CLASSES, (0, N_LESION_TYPE_CLASSES))
    fine_counts = counts_from_labels(train_df["head1_idx"], N_DX_CLASSES, (0, N_DX_CLASSES))
    
    coarse_w = class_balanced_weights(coarse_counts, beta=CLASS_BALANCED_BETA)
    fine_w = class_balanced_weights(fine_counts, beta=CLASS_BALANCED_BETA)
    coarse_alpha = calculate_focal_alpha(coarse_counts)

    # Build datasets
    minority_fine_names = ["df", "vasc", "other", "no_lesion"]
    minority_fine_ids = {DX_TO_ID[n] for n in minority_fine_names if n in DX_TO_ID}

    if USE_OVERSAMPLING:
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

    # Create fine-tuned model
    model = create_finetuned_model(ssl_model, N_DX_CLASSES, N_LESION_TYPE_CLASSES)
    
    # Freeze backbone initially, then unfreeze gradually
    backbone = model.layers[1]
    backbone.trainable = False
    
    # Compile with frozen backbone
    if USE_FOCAL_COARSE:
        coarse_loss = sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=coarse_alpha)
    else:
        coarse_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_steps = FINE_TUNE_EPOCHS * steps_per_epoch
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR_FINE_TUNE,
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

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(FINE_TUNE_OUTDIR / "finetuned_best_model.keras"),
            save_best_only=True,
            monitor="val_coarse_output_loss",
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(FINE_TUNE_OUTDIR / "finetuned_history.csv")),
        keras.callbacks.TensorBoard(
            log_dir=str(FINE_TUNE_OUTDIR / "finetuned_tensorboard_logs"),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=10,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        # Unfreeze backbone after some epochs
        keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: unfreeze_backbone(model, epoch) if epoch == 10 else None
        ),
    ]

    # Train with frozen backbone first
    print("Training with frozen backbone...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1,
    )

    # Unfreeze backbone and continue training
    backbone.trainable = True
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR_FINE_TUNE/10, weight_decay=WEIGHT_DECAY),
        loss={
            "fine_output": masked_sparse_ce_with_oe,
            "coarse_output": coarse_loss,
        },
        metrics={
            "fine_output": ["sparse_categorical_accuracy"],
            "coarse_output": ["sparse_categorical_accuracy"],
        },
    )

    print("Training with unfrozen backbone...")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS-10,
        initial_epoch=10,
        callbacks=callbacks,
        verbose=1,
    )

    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    # Save results
    with open(FINE_TUNE_OUTDIR / "finetuned_stats.json", "w") as f:
        json.dump(combined_history, f)
    
    # Evaluate model
    evaluate_model(model, val_ds, val_df, FINE_TUNE_OUTDIR)
    
    print(f"Fine-tuning complete. Model saved to '{FINE_TUNE_OUTDIR}'.")
    return model

def unfreeze_backbone(model, epoch):
    """Unfreeze backbone after certain epoch."""
    if epoch == 10:
        backbone = model.layers[1]
        backbone.trainable = True
        print(f"Epoch {epoch}: Unfreezing backbone for fine-tuning")

def evaluate_model(model, val_ds, val_df, outdir):
    """Comprehensive model evaluation."""
    print("\nEvaluating SSL fine-tuned model...")
    
    predictions = model.predict(val_ds, verbose=1)
    fine_preds = predictions[0]
    coarse_preds = predictions[1]
    
    fine_pred_classes = np.argmax(fine_preds, axis=1)
    coarse_pred_classes = np.argmax(coarse_preds, axis=1)
    
    fine_true = val_df['head1_idx'].values
    coarse_true = val_df['head2_idx'].values
    
    valid_fine_mask = fine_true >= 0
    fine_true_valid = fine_true[valid_fine_mask]
    fine_pred_valid = fine_pred_classes[valid_fine_mask]
    
    print("\nFine-grained Classification Report:")
    fine_report = classification_report(fine_true_valid, fine_pred_valid, 
                                       target_names=DX_CLASSES, digits=4)
    print(fine_report)
    
    print("\nCoarse Classification Report:")
    coarse_report = classification_report(coarse_true, coarse_pred_classes,
                                       target_names=LESION_TYPE_CLASSES, digits=4)
    print(coarse_report)
    
    with open(outdir / "ssl_fine_classification_report.txt", "w") as f:
        f.write(str(fine_report))
    with open(outdir / "ssl_coarse_classification_report.txt", "w") as f:
        f.write(str(coarse_report))
    
    print("\nFine-grained Confusion Matrix:")
    cm_fine = confusion_matrix(fine_true_valid, fine_pred_valid)
    print(cm_fine)
    
    print("\nCoarse Confusion Matrix:")
    cm_coarse = confusion_matrix(coarse_true, coarse_pred_classes)
    print(cm_coarse)
    
    print(f"\nSSL evaluation complete. Reports saved to {outdir}")

def main():
    """
    Main function to run SSL + Fine-tuning pipeline.
    """
    print("Starting Self-Supervised Learning + Fine-tuning pipeline...")
    
    # Step 1: Train SSL model
    ssl_model = train_ssl_model()
    
    # Step 2: Fine-tune on labeled data
    finetuned_model = fine_tune_model(ssl_model)
    
    print("\n" + "="*50)
    print("SSL + Fine-tuning pipeline completed successfully!")
    print("="*50)
    print(f"SSL model saved to: {SSL_OUTDIR}")
    print(f"Fine-tuned model saved to: {FINE_TUNE_OUTDIR}")
    print("\nTo compare with baseline model, run evaluation scripts.")

if __name__ == "__main__":
    main()
