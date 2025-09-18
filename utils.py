"""
Utility function. Consolidates common functions used across multiple scripts
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SEED = 999

DATA_DIR = Path("../data")
PREPARED_CSV = DATA_DIR / "training_prepared_data.csv"
IMAGE_PATH = DATA_DIR.joinpath("images", "images")

IMG_SIZE = 224
BATCH_SIZE = 64  

DX_CLASSES = sorted(['nv', 'mel', 'bkl', 'bcc', 'scc_akiec', 'vasc', 'df', 'other', 'no_lesion'])
LESION_TYPE_CLASSES = ["benign", "malignant", "no_lesion"]
N_DX_CLASSES = len(DX_CLASSES)
N_LESION_TYPE_CLASSES = len(LESION_TYPE_CLASSES)
DX_TO_ID = {n: i for i, n in enumerate(DX_CLASSES)}
LESION_TO_ID = {n: i for i, n in enumerate(LESION_TYPE_CLASSES)}

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

# REPRODUCIBILITY UTILITIES

def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# CLASS BALANCING UTILITIES

def class_balanced_weights(counts, beta=CLASS_BALANCED_BETA):
    """Calculate class-balanced weights using effective number."""
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

def counts_from_labels(series, n_classes, valid_range=(0, None)):
    """Extract class counts from label series."""
    # Handle both pandas Series and numpy arrays
    if isinstance(series, np.ndarray):
        y = series.astype("float")
    else:
        y = pd.to_numeric(series, errors="coerce").astype("float")
    
    lo = valid_range[0]
    hi = valid_range[1] if valid_range[1] is not None else np.inf
    y = y[(y >= lo) & (y < (hi if hi != np.inf else np.inf))]
    
    # Handle NaN values appropriately for both pandas and numpy
    if isinstance(y, np.ndarray):
        y = y[~np.isnan(y)].astype(int)
    else:
        y = y.dropna().astype(int).values
    
    counts = np.bincount(y, minlength=n_classes)
    return counts

# DATA AUGMENTATION UTILITIES

def build_augmenter(is_training, augmentation_strength='medium'):
    """Build data augmenter based on training mode and strength."""
    if not is_training:
        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.CenterCrop(IMG_SIZE, IMG_SIZE),
        ], name="preprocessor")
    
    if augmentation_strength == 'light':
        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.RandomCrop(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.05),
        ], name="light_augmenter")
    
    elif augmentation_strength == 'medium':
        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.RandomCrop(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.1),
        ], name="medium_augmenter")
    
    elif augmentation_strength == 'fast':
        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.RandomCrop(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal"),
        ], name="fast_augmenter")
    
    elif augmentation_strength == 'strong':
        
        def color_jitter(x):
            # Convert to float32 for image operations, then back to original dtype
            original_dtype = x.dtype
            x_float32 = tf.cast(x, tf.float32)
            x_float32 = tf.image.random_contrast(x_float32, 0.7, 1.3)
            x_float32 = tf.image.random_saturation(x_float32, 0.7, 1.3)
            x_float32 = tf.image.random_hue(x_float32, 0.1)
            return tf.cast(tf.clip_by_value(x_float32, 0.0, 255.0), original_dtype)
        
        def gaussian_noise(x):
            noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=5.0, dtype=x.dtype)
            return tf.clip_by_value(x + noise, 0.0, 255.0)

        return keras.Sequential([
            layers.Resizing(256, 256),
            layers.RandomCrop(IMG_SIZE, IMG_SIZE),
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(factor=0.15),
            layers.Lambda(color_jitter, name="color_jitter"),
            layers.Lambda(gaussian_noise, name="gaussian_noise"),
        ], name="strong_augmenter")

def augment_minority(img):
    """Apply additional augmentation for minority classes."""
    # Convert to float32 for image operations, then back to original dtype
    original_dtype = img.dtype
    img_float32 = tf.cast(img, tf.float32)
    
    img_float32 = tf.image.random_contrast(img_float32, 0.85, 1.15)
    img_float32 = tf.image.random_saturation(img_float32, 0.9, 1.1)
    
    # Convert back to original dtype for noise addition
    img = tf.cast(img_float32, original_dtype)
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=3.0, dtype=img.dtype)
    img = tf.clip_by_value(img + noise, 0.0, 255.0)
    return img

def resolve_image_path(path):
    """
    Resolve image path to absolute path, handling cross-platform compatibility.
    
    This function handles various path formats:
    - Windows absolute paths (C:\path\to\file.jpg)
    - Unix absolute paths (/path/to/file.jpg)  
    - Relative paths (../data/images/images/file.jpg)
    - Filename only (file.jpg)
    """
    p = str(path)
    
    # Check if it's a Windows path (contains drive letter like C:)
    if ':' in p and ('\\' in p or p.startswith('C:') or p.startswith('D:')):
        # Windows path - extract just the filename for cross-platform compatibility
        if '\\' in p:
            filename = p.split('\\')[-1]
        else:
            filename = p.split('/')[-1]  # fallback for forward slashes
        
        # Construct path using the standard image directory
        linux_path = str(IMAGE_PATH / filename)
        if os.path.exists(linux_path):
            return linux_path
        else:
            print(f"Warning: Could not find image file: {filename}")
            return linux_path  # Return the constructed path anyway
    
    # If it's already an absolute Unix path, check if it exists
    elif os.path.isabs(p):
        if os.path.exists(p):
            return p
        else:
            # Try to find the file in the standard image directory
            filename = os.path.basename(p)
            linux_path = str(IMAGE_PATH / filename)
            if os.path.exists(linux_path):
                return linux_path
            else:
                print(f"Warning: Could not find image file: {filename}")
                return linux_path
    
    # Check if it's already a relative path that starts with ../data/images/images/
    elif p.startswith('../data/images/images/'):
        # This is our standardized relative path format - use it as is
        if os.path.exists(p):
            return p
        else:
            print(f"Warning: Could not find image file: {p}")
            return p
    
    # For other relative paths (just filename), construct the full path
    return str(IMAGE_PATH / p)

def apply_fine_oversampling(df, fine_oversampling):
    """Apply oversampling to fine-grained minority classes."""
    df_list = [df]
    
    for class_name, multiplier in fine_oversampling.items():
        if class_name in DX_TO_ID:
            class_id = DX_TO_ID[class_name]
            minority_samples = df[df["head2_idx"] == class_id]
            if len(minority_samples) > 0:
                n_repeats = int(multiplier) - 1
                if n_repeats > 0:
                    repeated_samples = pd.concat([minority_samples] * n_repeats, ignore_index=True)
                    df_list.append(repeated_samples)
    
    return pd.concat(df_list, ignore_index=True)

def build_dataset(df, is_training=False, backbone_type='efficientnet', 
                 minority_fine_ids=None, fine_oversampling=None):
    """Build dataset for training or evaluation."""
    df = df.dropna(subset=['image_path', 'head1_idx']).copy()
    df_coarse = df['head1_idx'].astype('int32').values  # Coarse classification (lesion_type)
    df_fine = df['head2_idx'].fillna(-1).astype('int32').values  # Fine-grained diagnosis

    # Resolve image paths
    img_paths = df['image_path'].astype(str).apply(resolve_image_path).tolist()

    ds = tf.data.Dataset.from_tensor_slices((img_paths, df_coarse, df_fine))
    
    if is_training:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)

    # Apply fine oversampling if specified
    if fine_oversampling is not None and is_training:
        df = apply_fine_oversampling(df, fine_oversampling)
        df_coarse = df['head1_idx'].astype('int32').values  # Coarse classification (lesion_type)
        df_fine = df['head2_idx'].fillna(-1).astype('int32').values  # Fine-grained diagnosis
        img_paths = df['image_path'].astype(str).apply(resolve_image_path).tolist()
        ds = tf.data.Dataset.from_tensor_slices((img_paths, df_coarse, df_fine))
        if is_training:
            ds = ds.shuffle(len(df), reshuffle_each_iteration=True)

    # Build augmenter
    augmentation_strength = 'medium'  # Default
    if backbone_type == 'efficientnet':
        augmentation_strength = 'medium'  # Restored original
    elif backbone_type == 'resnet':
        augmentation_strength = 'strong'  # Restored original
    elif backbone_type == 'densenet':
        augmentation_strength = 'light'  # Restored original
    
    augmenter = build_augmenter(is_training, augmentation_strength)
    
    rescale = layers.Rescaling(1./255)
    normalization_layer = layers.Normalization(
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2]
    )

    def load_and_preprocess(path, label_coarse, label_fine):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = augmenter(img)
        
        # Apply minority augmentation if training
        if is_training and minority_fine_ids is not None:
            minority = tf.logical_or(
                tf.equal(label_coarse, 1),  # malignant
                tf.reduce_any(tf.equal(label_fine, tf.constant(list(minority_fine_ids), dtype=tf.int32)))
                if len(minority_fine_ids) > 0 else tf.constant(False)
            )
            img = tf.cond(minority, lambda: augment_minority(img), lambda: img)

        img = rescale(img)
        img = normalization_layer(img)
        return img, {"fine_output": label_fine, "coarse_output": label_coarse}

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # Cache dataset for faster subsequent epochs
    if not is_training:
        ds = ds.cache()
    return ds

# MODEL CREATION UTILITIES

def create_efficientnet_backbone(img_size=IMG_SIZE):
    """Create EfficientNetB1 backbone."""
    try:
        return tf.keras.applications.efficientnet.EfficientNetB1(
            include_top=False,
            weights="imagenet",
            input_shape=(img_size, img_size, 3),
        )
    except ValueError as e:
        if "Shape mismatch" in str(e) and "stem_conv" in str(e):
            print(f"Warning: ImageNet weights loading failed due to shape mismatch: {e}")
            print("Creating EfficientNetB1 without pre-trained weights...")
            return tf.keras.applications.efficientnet.EfficientNetB1(
                include_top=False,
                weights=None,
                input_shape=(img_size, img_size, 3),
            )
        else:
            raise e

def create_resnet_backbone(img_size=IMG_SIZE):
    """Create ResNet50 backbone."""
    return tf.keras.applications.resnet.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
    )

def create_densenet_backbone(img_size=IMG_SIZE):
    """Create DenseNet121 backbone."""
    return tf.keras.applications.densenet.DenseNet121(
        include_top=False,
        weights="imagenet",
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

def create_two_head_model(backbone_type='efficientnet', n_fine=N_DX_CLASSES, 
                         n_coarse=N_LESION_TYPE_CLASSES, img_size=IMG_SIZE, dropout=0.2):
    """Create two-head model with specified backbone."""
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")
    
    backbone = create_backbone(backbone_type, img_size)
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout, name="top_dropout")(x)

    output_coarse = layers.Dense(n_coarse, name="coarse_output")(x)  # Head 1: Coarse classification
    output_fine = layers.Dense(n_fine, name="fine_output")(x)        # Head 2: Fine-grained diagnosis

    model_name = f"{backbone_type.title()}_TwoHead"
    model = keras.Model(inputs=inputs, outputs=[output_coarse, output_fine], name=model_name)
    return model

# LOSS FUNCTIONS

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Masked sparse categorical crossentropy loss."""
    y_true = tf.cast(y_true, tf.int32)
    mask = tf.cast(tf.not_equal(y_true, -1), dtype=tf.float32)
    loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / (tf.reduce_sum(mask) + 1e-8)

@tf.keras.utils.register_keras_serializable()
def masked_sparse_ce_with_oe(y_true, y_pred):
    """
    Masked sparse categorical crossentropy with outlier exposure.
    
    This loss function handles three types of labels:
    - Valid labels (>= 0): Use standard cross-entropy loss
    - Masked labels (-1): Ignore in loss calculation  
    - OOD labels (-2): Use outlier exposure loss to encourage uniform predictions
    """
    y_true = tf.cast(y_true, tf.int32)
    num_classes = tf.shape(y_pred)[-1]

    # Create masks for different label types
    mask_valid = tf.logical_and(y_true >= 0, y_true < num_classes)  # Normal samples
    mask_ood = tf.equal(y_true, -2)  # Out-of-distribution samples

    # Calculate standard cross-entropy for valid samples
    y_true_safe = tf.where(mask_valid, y_true, tf.zeros_like(y_true))
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_safe, y_pred, from_logits=True)

    # Calculate outlier exposure loss (encourages uniform distribution)
    log_p = tf.nn.log_softmax(y_pred, axis=-1)
    oe = LAMBDA_OE * (-tf.reduce_mean(log_p, axis=-1))

    # Apply appropriate loss based on label type
    per_example = tf.where(mask_ood, oe, tf.where(mask_valid, ce, tf.zeros_like(ce)))

    # Calculate average loss over valid and OOD samples only
    denom = tf.reduce_sum(tf.cast(tf.logical_or(mask_valid, mask_ood), per_example.dtype))
    return tf.math.divide_no_nan(tf.reduce_sum(per_example), denom)

@tf.keras.utils.register_keras_serializable()
def sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=None):
    """Sparse categorical focal loss."""
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

# MODEL LOADING UTILITIES

def load_individual_model(model_path, backbone_type='efficientnet'):
    """Load individual model from path."""
    try:
        # First try to load the complete model directly
        if model_path.exists():
            # Load model with custom objects for loss functions
            custom_objects = {
                'masked_sparse_ce_with_oe': masked_sparse_ce_with_oe,
                'sparse_categorical_focal_loss': sparse_categorical_focal_loss
            }
            model = keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)
            print(f"✓ Loaded complete model from {model_path}")
            return model
        else:
            print(f"✗ Model file not found: {model_path}")
            return None
    except Exception as e:
        print(f"✗ Failed to load model from {model_path}: {e}")
        # Fallback: try creating model and loading weights
        try:
            print(f"Attempting fallback: creating model and loading weights...")
            model = create_two_head_model(backbone_type, N_DX_CLASSES, N_LESION_TYPE_CLASSES)
            model.load_weights(str(model_path))
            print(f"✓ Loaded model weights from {model_path}")
            return model
        except Exception as e2:
            print(f"✗ Fallback also failed: {e2}")
            return None

def load_ensemble_models(individual_dir):
    """Load individual models for ensemble."""
    ensemble_models = {}
    backbone_types = ['efficientnet', 'resnet', 'densenet']
    
    for backbone_type in backbone_types:
        # Try both naming conventions used by ensemble_model.py
        model_paths = [
            individual_dir / f"{backbone_type}" / f"{backbone_type}_best_model.keras",  # ensemble_model.py saves as this
            # individual_dir / f"{backbone_type}_best_model.keras"  # alternative naming
        ]
        
        model = None
        for model_path in model_paths:
            if model_path.exists():
                model = load_individual_model(model_path, backbone_type)
                if model is not None:
                    break
        
        if model is not None:
            ensemble_models[backbone_type] = model
    
    return ensemble_models

# PREDICTION UTILITIES

def get_predictions_and_labels(model, dataset):
    """Get predictions and labels from a model."""
    all_labels_h1, all_labels_h2 = [], []
    all_logits_h1, all_logits_h2 = [], []

    for images, labels in dataset:
        logits_h1, logits_h2 = model.predict_on_batch(images)
        all_logits_h1.append(logits_h1)
        all_logits_h2.append(logits_h2)

        all_labels_h1.append(labels['coarse_output'].numpy())
        all_labels_h2.append(labels['fine_output'].numpy())
        
    all_logits_h1 = np.concatenate(all_logits_h1, axis=0)
    all_logits_h2 = np.concatenate(all_logits_h2, axis=0)
    all_labels_h1 = np.concatenate(all_labels_h1, axis=0)
    all_labels_h2 = np.concatenate(all_labels_h2, axis=0)

    return all_labels_h1, all_logits_h1, all_labels_h2, all_logits_h2

def create_voting_ensemble(models_dict, dataset):
    """Create voting ensemble from multiple models."""
    all_fine_preds = []
    all_coarse_preds = []
    
    for backbone_type, model in models_dict.items():
        # print(f"Getting predictions from {backbone_type}...")
        preds = model.predict(dataset, verbose=0)
        all_coarse_preds.append(preds[0])
        all_fine_preds.append(preds[1])
    
    # Average predictions
    ensemble_fine_preds = np.mean(all_fine_preds, axis=0)
    ensemble_coarse_preds = np.mean(all_coarse_preds, axis=0)
    
    # CORRECTED: Return in correct order [coarse, fine] to match individual models
    return ensemble_coarse_preds, ensemble_fine_preds

def calculate_model_weights(models_dict, val_dataset):
    """Calculate performance-based weights for ensemble models."""
    weights = {}
    
    for backbone_type, model in models_dict.items():
        # Get predictions on validation set
        labels_h1, logits_h1, labels_h2, logits_h2 = get_predictions_and_labels(model, val_dataset)
        
        # Calculate performance metrics
        preds_h1 = np.argmax(logits_h1, axis=1)
        preds_h2 = np.argmax(logits_h2, axis=1)
        
        # Use balanced accuracy as the primary metric for weighting
        from sklearn.metrics import balanced_accuracy_score
        
        # Calculate balanced accuracy for both heads
        valid_mask_h1 = labels_h1 >= 0
        valid_mask_h2 = labels_h2 >= 0
        
        if valid_mask_h1.sum() > 0:
            coarse_acc = balanced_accuracy_score(labels_h1[valid_mask_h1], preds_h1[valid_mask_h1])
        else:
            coarse_acc = 0.0
            
        if valid_mask_h2.sum() > 0:
            fine_acc = balanced_accuracy_score(labels_h2[valid_mask_h2], preds_h2[valid_mask_h2])
        else:
            fine_acc = 0.0
        
        # Combined performance score (weighted average of both heads)
        combined_score = 0.6 * coarse_acc + 0.4 * fine_acc
        weights[backbone_type] = max(combined_score, 0.1)  # Minimum weight to avoid zero
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    return weights

def create_weighted_ensemble(models_dict, dataset, weights=None, val_dataset=None):
    """Create weighted ensemble from multiple models."""
    if weights is None:
        if val_dataset is not None:
            # Calculate performance-based weights
            weights = calculate_model_weights(models_dict, val_dataset)
            print(f"Calculated performance-based weights: {weights}")
        else:
            # Fallback to equal weights
            weights = {k: 1.0/len(models_dict) for k in models_dict.keys()}
            print(f"Using equal weights: {weights}")
    
    all_fine_preds = []
    all_coarse_preds = []
    
    for backbone_type, model in models_dict.items():
        weight = weights[backbone_type]
        print(f"Getting predictions from {backbone_type} (weight: {weight:.3f})...")
        preds = model.predict(dataset, verbose=0)
        all_coarse_preds.append(preds[0] * weight)
        all_fine_preds.append(preds[1] * weight)
    
    # Weighted average predictions
    ensemble_fine_preds = np.sum(all_fine_preds, axis=0)
    ensemble_coarse_preds = np.sum(all_coarse_preds, axis=0)
    
    # CORRECTED: Return in correct order [coarse, fine] to match individual models
    return ensemble_coarse_preds, ensemble_fine_preds

def get_ensemble_predictions(ensemble_models, dataset, method='voting', val_dataset=None):
    """Get ensemble predictions."""
    if method == 'voting':
        coarse_preds, fine_preds = create_voting_ensemble(ensemble_models, dataset)
    elif method == 'weighted':
        coarse_preds, fine_preds = create_weighted_ensemble(ensemble_models, dataset, val_dataset=val_dataset)
    else:
        raise ValueError("Method must be 'voting' or 'weighted'")
    
    # Get labels from first model (all should be the same)
    first_model = list(ensemble_models.values())[0]
    labels_h1, _, labels_h2, _ = get_predictions_and_labels(first_model, dataset)
    
    # CORRECTED: Return in correct order [coarse, fine] to match individual models
    return labels_h1, coarse_preds, labels_h2, fine_preds

# EVALUATION UTILITIES

def calculate_metrics(labels, preds, class_names):
    """Calculate comprehensive metrics."""
    # Filter out masked samples (label == -1)
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_preds = preds[valid_mask]
    
    if len(valid_labels) == 0:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'macro_f1': 0.0,
            'weighted_f1': 0.0
        }
    
    accuracy = accuracy_score(valid_labels, valid_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average=None, zero_division=0
    )
    
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        valid_labels, valid_preds, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': f1
    }

def plot_confusion_matrix(labels, preds, class_names, title):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def get_msp_scores(logits):
    """Get Maximum Softmax Probability scores for OOD detection."""
    softmax_probs = tf.nn.softmax(logits, axis=1).numpy()
    return np.max(softmax_probs, axis=1)

# DATA PROCESSING UTILITIES

def process_labels(df):
    """Process and clean labels in dataframe."""
    df = df.copy()
    
    # Process coarse labels
    if "lesion_type" in df.columns:
        lt = df["lesion_type"].astype(str).str.strip().str.lower()
        map_coarse = {"benign": 0, "malignant": 1, "no_lesion": 2}
        df["head1_idx"] = pd.to_numeric(df.get("head1_idx"), errors="coerce")
        to_fill = df["head1_idx"].isna() & lt.isin(map_coarse.keys())
        df.loc[to_fill, "head1_idx"] = lt[to_fill].map(map_coarse)
    
    df["head1_idx"] = pd.to_numeric(df.get("head1_idx"), errors="coerce")
    bad_coarse = df["head1_idx"].isna() | (df["head1_idx"] < 0) | (df["head1_idx"] >= N_LESION_TYPE_CLASSES)
    if bad_coarse.any():
        print(f"[coarse] dropping {int(bad_coarse.sum())} rows with invalid coarse labels")
        df = df[~bad_coarse].copy()

    # Handle fine labels
    is_ood = np.zeros(len(df), dtype=bool)
    if 'diagnosis_grouped' in df.columns:
        is_ood = df['diagnosis_grouped'].astype(str).str.strip().str.lower().eq("unknown").values
    
    df["head2_idx"] = pd.to_numeric(df.get("head2_idx"), errors="coerce")
    mask_bad_fine = df["head2_idx"].isna() | (df["head2_idx"] < 0) | (df["head2_idx"] >= N_DX_CLASSES)
    df.loc[mask_bad_fine, "head2_idx"] = -1  # ignore
    
    if USE_OOD_OE and is_ood.any():
        df.loc[is_ood, "head2_idx"] = -2  # OOD assigned to head2_idx (fine head)
    
    df["head2_idx"] = df["head2_idx"].astype("int32")
    
    return df

# CALLBACK UTILITIES

def create_callbacks(model_dir, backbone_type, monitor="val_coarse_output_loss"):
    """Create standard callbacks for training."""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / f"{backbone_type}_best_model.keras"),
            save_best_only=True,
            monitor=monitor,
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
    return callbacks

# VISUALIZATION UTILITIES

def plot_model_comparison(comparison_df, all_metrics, models_config):
    """Plot comprehensive model comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Fine-grained accuracy comparison
    models = comparison_df['Model'].tolist()
    fine_acc = comparison_df['Fine Accuracy'].tolist()
    fine_f1 = comparison_df['Fine F1'].tolist()
    coarse_acc = comparison_df['Coarse Accuracy'].tolist()
    coarse_f1 = comparison_df['Coarse F1'].tolist()

    colors = [models_config.get(key, {}).get('color', '#666666') for key in all_metrics.keys()]

    axes[0, 0].bar(models, fine_acc, color=colors, alpha=0.7)
    axes[0, 0].set_title('Fine-grained Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(models, fine_f1, color=colors, alpha=0.7)
    axes[0, 1].set_title('Fine-grained F1 Score Comparison')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(models, coarse_acc, color=colors, alpha=0.7)
    axes[1, 0].set_title('Coarse Accuracy Comparison')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(models, coarse_f1, color=colors, alpha=0.7)
    axes[1, 1].set_title('Coarse F1 Score Comparison')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_ood_detection(all_predictions, models_config):
    """Plot OOD detection analysis."""
    plt.figure(figsize=(10,8))

    # Plot MSP distributions for each model
    for i, (model_key, predictions) in enumerate(all_predictions.items()):
        model_name = models_config[model_key]['name']
        
        id_msp_scores = get_msp_scores(predictions['id_logits_h1'])
        ood_msp_scores = get_msp_scores(predictions['ood_logits_h1'])
        
        plt.subplot(3, 2, i+1)
        # Use consistent viridis colors: ID = green-blue, OOD = dark purple
        plt.hist(id_msp_scores, bins=30, alpha=0.7, label='ID', color='#35b779', density=True)
        plt.hist(ood_msp_scores, bins=30, alpha=0.7, label='OOD', color='#440154', density=True)
        plt.title(f'{model_name}')
        plt.xlabel('Maximum Softmax Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# FILE UTILITIES

def save_results(output_dir, comparison_df, all_metrics, ood_results, summary_report):
    """Save comprehensive results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save comparison table
    comparison_df.to_csv(output_dir / "model_comparison_table.csv", index=False)
    print(f"✓ Saved comparison table to: {output_dir / 'model_comparison_table.csv'}")
    
    # Save detailed metrics
    with open(output_dir / "detailed_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Saved detailed metrics to: {output_dir / 'detailed_metrics.json'}")
    
    # Save OOD results
    with open(output_dir / "ood_detection_results.json", "w") as f:
        json.dump(ood_results, f, indent=2)
    print(f"✓ Saved OOD detection results to: {output_dir / 'ood_detection_results.json'}")
    
    # Save summary report
    with open(output_dir / "summary_report.md", "w") as f:
        f.write(summary_report)
    print(f"✓ Saved summary report to: {output_dir / 'summary_report.md'}")

# INITIALIZATION

# Set seed on import
set_seed(SEED)
