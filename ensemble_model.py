# src/ensemble_model.py
# -*- coding: utf-8 -*-
"""
Ensemble Learning with Architectural Diversity
Implements multiple backbone architectures with ensemble methods
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
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

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

# Ensemble config
ENSEMBLE_OUTDIR = Path("outputs/ensemble_models")
INDIVIDUAL_OUTDIR = Path("outputs/individual_models")
COMPARISON_OUTDIR = Path("outputs/ensemble_comparison")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
PRETRAINED = True

# Ensemble specific parameters
N_MODELS = 3  # Number of different architectures
ENSEMBLE_METHODS = ['voting', 'weighted_average', 'stacking']

# Model configurations
MODEL_CONFIGS = {
    'efficientnet': {
        'name': 'EfficientNetB1',
        'backbone': 'efficientnet',
        'augmentation_strength': 'medium',
        'learning_rate': LR,
        'weight': 1.0
    },
    'resnet': {
        'name': 'ResNet50',
        'backbone': 'resnet',
        'augmentation_strength': 'strong',
        'learning_rate': LR * 1.5,
        'weight': 1.0
    },
    'densenet': {
        'name': 'DenseNet121',
        'backbone': 'densenet',
        'augmentation_strength': 'light',
        'learning_rate': LR * 0.8,
        'weight': 1.0
    }
}

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

# -------------------- Diverse Data Augmentations --------------------
def create_diverse_augmenters():
    """
    Create different augmentation strategies for each model to increase diversity.
    """
    def light_augmenter():
        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.RandomCrop(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.05),
        ], name="light_augmenter")
    
    def medium_augmenter():
        def random_brightness_mul(x):
            x = tf.cast(x, tf.float32)
            r = tf.random.uniform([])
            factor = tf.where(r < 0.7,
                              tf.random.uniform([], 0.6, 1.0),
                              tf.random.uniform([], 1.0, 1.15))
            x = x * factor
            return tf.clip_by_value(x, 0.0, 255.0)

        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.RandomCrop(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.1),
            layers.Lambda(random_brightness_mul, name="random_brightness"),
        ], name="medium_augmenter")
    
    def strong_augmenter():
        def random_brightness_mul(x):
            x = tf.cast(x, tf.float32)
            r = tf.random.uniform([])
            factor = tf.where(r < 0.7,
                              tf.random.uniform([], 0.5, 1.0),
                              tf.random.uniform([], 1.0, 1.2))
            x = x * factor
            return tf.clip_by_value(x, 0.0, 255.0)
        
        def color_jitter(x):
            x = tf.image.random_contrast(x, 0.7, 1.3)
            x = tf.image.random_saturation(x, 0.7, 1.3)
            x = tf.image.random_hue(x, 0.1)
            return tf.clip_by_value(x, 0.0, 255.0)
        
        def gaussian_noise(x):
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=5.0)
            return tf.clip_by_value(x + noise, 0.0, 255.0)

        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.RandomCrop(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.15),
            layers.Lambda(random_brightness_mul, name="random_brightness"),
            layers.Lambda(color_jitter, name="color_jitter"),
            layers.Lambda(gaussian_noise, name="gaussian_noise"),
        ], name="strong_augmenter")
    
    return {
        'light': light_augmenter,
        'medium': medium_augmenter,
        'strong': strong_augmenter
    }

def build_augmenter(is_training, augmentation_strength='medium'):
    """Build augmenter based on strength level."""
    if not is_training:
        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.CenterCrop(IMG_SIZE, IMG_SIZE),
        ], name="preprocessor")
    
    augmenters = create_diverse_augmenters()
    return augmenters[augmentation_strength]()

def augment_minority(img):
    """Minority class augmentation (same as first_model.py)."""
    img = tf.image.random_contrast(img, 0.85, 1.15)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=3.0)
    img = tf.clip_by_value(img + noise, 0.0, 255.0)
    return img

# -------------------- Diverse Backbone Architectures --------------------
def create_efficientnet_backbone(img_size=IMG_SIZE):
    """Create EfficientNetB1 backbone."""
    return tf.keras.applications.efficientnet.EfficientNetB1(
        include_top=False,
        weights="imagenet" if PRETRAINED else None,
        input_shape=(img_size, img_size, 3),
    )

def create_resnet_backbone(img_size=IMG_SIZE):
    """Create ResNet50 backbone."""
    return tf.keras.applications.resnet.ResNet50(
        include_top=False,
        weights="imagenet" if PRETRAINED else None,
        input_shape=(img_size, img_size, 3),
    )

def create_densenet_backbone(img_size=IMG_SIZE):
    """Create DenseNet121 backbone."""
    return tf.keras.applications.densenet.DenseNet121(
        include_top=False,
        weights="imagenet" if PRETRAINED else None,
        input_shape=(img_size, img_size, 3),
    )

def create_backbone(backbone_type, img_size=IMG_SIZE):
    """Create backbone based on type."""
    backbones = {
        'efficientnet': create_efficientnet_backbone,
        'resnet': create_resnet_backbone,
        'densenet': create_densenet_backbone,
    }
    return backbones[backbone_type](img_size)

# -------------------- Two-Head Model Creation --------------------
def create_two_head_model(backbone_type, n_fine, n_coarse, img_size=IMG_SIZE, dropout=0.2):
    """
    Create two-head model with specified backbone.
    """
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input_rgb")
    backbone = create_backbone(backbone_type, img_size)
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout, name="top_dropout")(x)
    out_fine = layers.Dense(n_fine, name="fine_output")(x)
    out_coarse = layers.Dense(n_coarse, name="coarse_output")(x)
    
    model_name = f"{MODEL_CONFIGS[backbone_type]['name']}_TwoHead"
    return keras.Model(inputs={'input_rgb': inputs}, outputs=[out_fine, out_coarse], name=model_name)

# -------------------- Dataset Creation --------------------
def build_dataset(df, is_training, backbone_type, minority_fine_ids=None, fine_oversampling=None):
    """
    Build dataset with backbone-specific augmentation.
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

    # Use backbone-specific augmentation
    augmentation_strength = MODEL_CONFIGS[backbone_type]['augmentation_strength']
    augmenter = build_augmenter(is_training, augmentation_strength)
    
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

# -------------------- Loss Functions (same as first_model.py) --------------------
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

# -------------------- Individual Model Training --------------------
def train_individual_model(backbone_type, train_df, val_df, model_dir):
    """
    Train individual model with specified backbone.
    """
    print(f"\nTraining {MODEL_CONFIGS[backbone_type]['name']} model...")
    model_dir.mkdir(exist_ok=True, parents=True)
    
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
            ds_c = build_dataset(sub, is_training=True, backbone_type=backbone_type, 
                               minority_fine_ids=minority_fine_ids, 
                               fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)
            ds_parts.append(ds_c)
            weights.append(OVERSAMPLE_WEIGHTS.get(str(c), 0.0))
        weights = np.asarray(weights, dtype=np.float32)
        weights = weights / (weights.sum() + 1e-8)
        train_ds = tf.data.Dataset.sample_from_datasets(
            ds_parts, weights=weights.tolist(), stop_on_empty_dataset=False
        )
    else:
        train_ds = build_dataset(train_df, is_training=True, backbone_type=backbone_type,
                                 minority_fine_ids=minority_fine_ids,
                                 fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)

    val_ds = build_dataset(val_df, is_training=False, backbone_type=backbone_type, 
                          minority_fine_ids=minority_fine_ids)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create model
    model = create_two_head_model(backbone_type, N_DX_CLASSES, N_LESION_TYPE_CLASSES)
    
    # Compile model
    if USE_FOCAL_COARSE:
        coarse_loss = sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=coarse_alpha)
    else:
        coarse_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=MODEL_CONFIGS[backbone_type]['learning_rate'],
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
            filepath=str(model_dir / f"{backbone_type}_best_model.keras"),
            save_best_only=True,
            monitor="val_coarse_output_loss",
            mode="min",
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(model_dir / f"{backbone_type}_history.csv")),
        keras.callbacks.TensorBoard(
            log_dir=str(model_dir / f"{backbone_type}_tensorboard_logs"),
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
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save history
    with open(model_dir / f"{backbone_type}_stats.json", "w") as f:
        json.dump(history.history, f)
    
    # Evaluate model
    evaluate_individual_model(model, val_ds, val_df, model_dir, backbone_type)
    
    print(f"{MODEL_CONFIGS[backbone_type]['name']} training complete. Model saved to '{model_dir}'.")
    return model

def evaluate_individual_model(model, val_ds, val_df, outdir, backbone_type):
    """Evaluate individual model."""
    print(f"\nEvaluating {MODEL_CONFIGS[backbone_type]['name']} model...")
    
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
    
    print(f"\n{MODEL_CONFIGS[backbone_type]['name']} Fine-grained Classification Report:")
    fine_report = classification_report(fine_true_valid, fine_pred_valid, 
                                       target_names=DX_CLASSES, digits=4)
    print(fine_report)
    
    print(f"\n{MODEL_CONFIGS[backbone_type]['name']} Coarse Classification Report:")
    coarse_report = classification_report(coarse_true, coarse_pred_classes,
                                       target_names=LESION_TYPE_CLASSES, digits=4)
    print(coarse_report)
    
    with open(outdir / f"{backbone_type}_fine_classification_report.txt", "w") as f:
        f.write(str(fine_report))
    with open(outdir / f"{backbone_type}_coarse_classification_report.txt", "w") as f:
        f.write(str(coarse_report))
    
    return {
        'fine_predictions': fine_preds,
        'coarse_predictions': coarse_preds,
        'fine_pred_classes': fine_pred_classes,
        'coarse_pred_classes': coarse_pred_classes,
        'fine_true': fine_true,
        'coarse_true': coarse_true,
        'fine_true_valid': fine_true_valid,
        'fine_pred_valid': fine_pred_valid
    }

# -------------------- Ensemble Methods --------------------
def create_voting_ensemble(models, val_ds):
    """
    Create voting ensemble from multiple models.
    """
    print("\nCreating voting ensemble...")
    
    all_fine_preds = []
    all_coarse_preds = []
    
    for i, (backbone_type, model) in enumerate(models.items()):
        print(f"Getting predictions from {MODEL_CONFIGS[backbone_type]['name']}...")
        preds = model.predict(val_ds, verbose=0)
        all_fine_preds.append(preds[0])
        all_coarse_preds.append(preds[1])
    
    # Average predictions
    ensemble_fine_preds = np.mean(all_fine_preds, axis=0)
    ensemble_coarse_preds = np.mean(all_coarse_preds, axis=0)
    
    ensemble_fine_classes = np.argmax(ensemble_fine_preds, axis=1)
    ensemble_coarse_classes = np.argmax(ensemble_coarse_preds, axis=1)
    
    return {
        'fine_predictions': ensemble_fine_preds,
        'coarse_predictions': ensemble_coarse_preds,
        'fine_pred_classes': ensemble_fine_classes,
        'coarse_pred_classes': ensemble_coarse_classes
    }

def create_weighted_ensemble(models, val_ds, weights=None):
    """
    Create weighted ensemble from multiple models.
    """
    print("\nCreating weighted ensemble...")
    
    if weights is None:
        weights = [MODEL_CONFIGS[backbone_type]['weight'] for backbone_type in models.keys()]
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    all_fine_preds = []
    all_coarse_preds = []
    
    for i, (backbone_type, model) in enumerate(models.items()):
        print(f"Getting predictions from {MODEL_CONFIGS[backbone_type]['name']} (weight: {weights[i]:.3f})...")
        preds = model.predict(val_ds, verbose=0)
        all_fine_preds.append(preds[0] * weights[i])
        all_coarse_preds.append(preds[1] * weights[i])
    
    # Weighted average predictions
    ensemble_fine_preds = np.sum(all_fine_preds, axis=0)
    ensemble_coarse_preds = np.sum(all_coarse_preds, axis=0)
    
    ensemble_fine_classes = np.argmax(ensemble_fine_preds, axis=1)
    ensemble_coarse_classes = np.argmax(ensemble_coarse_preds, axis=1)
    
    return {
        'fine_predictions': ensemble_fine_preds,
        'coarse_predictions': ensemble_coarse_preds,
        'fine_pred_classes': ensemble_fine_classes,
        'coarse_pred_classes': ensemble_coarse_classes,
        'weights': weights
    }

def evaluate_ensemble(ensemble_results, val_df, ensemble_name):
    """
    Evaluate ensemble performance.
    """
    print(f"\nEvaluating {ensemble_name} ensemble...")
    
    fine_pred_classes = ensemble_results['fine_pred_classes']
    coarse_pred_classes = ensemble_results['coarse_pred_classes']
    
    fine_true = val_df['head1_idx'].values
    coarse_true = val_df['head2_idx'].values
    
    valid_fine_mask = fine_true >= 0
    fine_true_valid = fine_true[valid_fine_mask]
    fine_pred_valid = fine_pred_classes[valid_fine_mask]
    
    print(f"\n{ensemble_name} Fine-grained Classification Report:")
    fine_report = classification_report(fine_true_valid, fine_pred_valid, 
                                       target_names=DX_CLASSES, digits=4)
    print(fine_report)
    
    print(f"\n{ensemble_name} Coarse Classification Report:")
    coarse_report = classification_report(coarse_true, coarse_pred_classes,
                                       target_names=LESION_TYPE_CLASSES, digits=4)
    print(coarse_report)
    
    return {
        'fine_report': fine_report,
        'coarse_report': coarse_report,
        'fine_pred_classes': fine_pred_classes,
        'coarse_pred_classes': coarse_pred_classes,
        'fine_true_valid': fine_true_valid,
        'fine_pred_valid': fine_pred_valid,
        'coarse_true': coarse_true
    }

# -------------------- Main Functions --------------------
def train_all_models():
    """
    Train all individual models.
    """
    print("Starting ensemble training pipeline...")
    
    # Load and prepare data
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

    # Train individual models
    models = {}
    individual_results = {}
    
    for backbone_type in MODEL_CONFIGS.keys():
        model_dir = INDIVIDUAL_OUTDIR / backbone_type
        model = train_individual_model(backbone_type, train_df, val_df, model_dir)
        models[backbone_type] = model
        
        # Evaluate individual model
        val_ds = build_dataset(val_df, is_training=False, backbone_type=backbone_type)
        individual_results[backbone_type] = evaluate_individual_model(model, val_ds, val_df, model_dir, backbone_type)
    
    return models, individual_results, val_df

def create_and_evaluate_ensembles(models, val_df):
    """
    Create and evaluate different ensemble methods.
    """
    print("\nCreating and evaluating ensembles...")
    ENSEMBLE_OUTDIR.mkdir(exist_ok=True, parents=True)
    
    # Create validation dataset (use EfficientNet for consistency)
    val_ds = build_dataset(val_df, is_training=False, backbone_type='efficientnet')
    
    ensemble_results = {}
    
    # Voting ensemble
    voting_results = create_voting_ensemble(models, val_ds)
    ensemble_results['voting'] = evaluate_ensemble(voting_results, val_df, "Voting")
    
    # Weighted ensemble
    weighted_results = create_weighted_ensemble(models, val_ds)
    ensemble_results['weighted'] = evaluate_ensemble(weighted_results, val_df, "Weighted")
    
    # Save ensemble results
    for ensemble_name, results in ensemble_results.items():
        with open(ENSEMBLE_OUTDIR / f"{ensemble_name}_fine_classification_report.txt", "w") as f:
            f.write(results['fine_report'])
        with open(ENSEMBLE_OUTDIR / f"{ensemble_name}_coarse_classification_report.txt", "w") as f:
            f.write(results['coarse_report'])
    
    return ensemble_results

def main():
    """
    Main function to run ensemble training and evaluation.
    """
    print("Starting Ensemble Learning Pipeline...")
    
    # Step 1: Train all individual models
    models, individual_results, val_df = train_all_models()
    
    # Step 2: Create and evaluate ensembles
    ensemble_results = create_and_evaluate_ensembles(models, val_df)
    
    print("\n" + "="*60)
    print("ENSEMBLE LEARNING PIPELINE COMPLETED")
    print("="*60)
    print(f"Individual models saved to: {INDIVIDUAL_OUTDIR}")
    print(f"Ensemble results saved to: {ENSEMBLE_OUTDIR}")
    print("\nModels trained:")
    for backbone_type in MODEL_CONFIGS.keys():
        print(f"- {MODEL_CONFIGS[backbone_type]['name']}")
    print("\nEnsemble methods evaluated:")
    print("- Voting ensemble")
    print("- Weighted ensemble")

if __name__ == "__main__":
    main()
