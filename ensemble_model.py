"""
Ensemble Learning with Architectural Diversity
Implements multiple backbone architectures with ensemble methods

Usage:
    python ensemble_model.py                    # Train individual models from scratch
    python ensemble_model.py --load-existing   # Load existing individual models
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import argparse
import utils
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

utils.set_seed(utils.SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth configuration failed: {e}")

DATA_DIR = utils.DATA_DIR
PREPARED_CSV = utils.PREPARED_CSV
IMAGE_PATH = utils.IMAGE_PATH
IMG_SIZE = utils.IMG_SIZE
BATCH_SIZE = utils.BATCH_SIZE

# Ensemble config
ENSEMBLE_OUTDIR = Path("outputs/ensemble_models")
INDIVIDUAL_OUTDIR = Path("outputs/individual_models")

EPOCHS = 15  # Reduced for faster training
LR = 1e-5  # Match base model learning rate
WEIGHT_DECAY = 1e-4

# Model configurations
MODEL_CONFIGS = {
    'efficientnet': {
        'name': 'EfficientNetB1',
        'backbone': 'efficientnet',
        'learning_rate': LR,
        'weight': 1.0
    },
    'resnet': {
        'name': 'ResNet50',
        'backbone': 'resnet',
        'learning_rate': LR * 1.5,
        'weight': 1.0
    },
    'densenet': {
        'name': 'DenseNet121',
        'backbone': 'densenet',
        'learning_rate': LR * 0.8,
        'weight': 1.0
    }
}

USE_FOCAL_COARSE = utils.USE_FOCAL_COARSE
FOCAL_GAMMA = utils.FOCAL_GAMMA
USE_SAMPLE_WEIGHTS = utils.USE_SAMPLE_WEIGHTS
CLASS_BALANCED_BETA = utils.CLASS_BALANCED_BETA
USE_OVERSAMPLING = utils.USE_OVERSAMPLING
OVERSAMPLE_WEIGHTS = utils.OVERSAMPLE_WEIGHTS
USE_FINE_OVERSAMPLING = utils.USE_FINE_OVERSAMPLING
FINE_MINORITY_OVERSAMPLING = utils.FINE_MINORITY_OVERSAMPLING
USE_OOD_OE = utils.USE_OOD_OE
LAMBDA_OE = utils.LAMBDA_OE

# -------------------- Labels --------------------
DX_CLASSES = utils.DX_CLASSES
LESION_TYPE_CLASSES = utils.LESION_TYPE_CLASSES
N_DX_CLASSES = utils.N_DX_CLASSES
N_LESION_TYPE_CLASSES = utils.N_LESION_TYPE_CLASSES
DX_TO_ID = utils.DX_TO_ID
LESION_TO_ID = utils.LESION_TO_ID

# -------------------- Individual Model Training --------------------
def train_individual_model(backbone_type, train_df, val_df, model_dir):
    """
    Train individual model with specified backbone.
    """
    print(f"\nTraining {MODEL_CONFIGS[backbone_type]['name']} model...")
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Calculate class weights using utils
    coarse_counts = utils.counts_from_labels(train_df["head1_idx"], N_LESION_TYPE_CLASSES, (0, N_LESION_TYPE_CLASSES))
    coarse_alpha = utils.calculate_focal_alpha(coarse_counts)

    # Build datasets using utils
    minority_fine_names = ["df", "vasc", "other", "no_lesion"]
    minority_fine_ids = {DX_TO_ID[n] for n in minority_fine_names if n in DX_TO_ID}

    ds_parts = []
    weights = []
    for c in range(N_LESION_TYPE_CLASSES):
        sub = train_df[train_df["head1_idx"] == c]
        if len(sub) == 0:
            continue
        ds_c = utils.build_dataset(sub, is_training=True, backbone_type=backbone_type, 
                            minority_fine_ids=minority_fine_ids, 
                            fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)
        ds_parts.append(ds_c)
        weights.append(OVERSAMPLE_WEIGHTS.get(str(c), 0.0))
    weights = np.asarray(weights, dtype=np.float32)
    wsum = weights.sum()
    if not ds_parts:
        raise ValueError("Oversampling enabled but no per-class datasets were built.")
    if wsum <= 1e-8:
        weights = np.full(len(ds_parts), 1.0 / len(ds_parts), dtype=np.float32)
    else:
        weights = weights / wsum
    train_ds = tf.data.Dataset.sample_from_datasets(
        ds_parts, weights=weights.tolist(), stop_on_empty_dataset=False
    )

    val_ds = utils.build_dataset(val_df, is_training=False, backbone_type=backbone_type, 
                          minority_fine_ids=minority_fine_ids)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create model using utils
    model = utils.create_two_head_model(backbone_type, N_DX_CLASSES, N_LESION_TYPE_CLASSES)
    
    print(f"\n{MODEL_CONFIGS[backbone_type]['name']} Model Summary:")
    model.summary()
    
    coarse_loss = utils.sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=coarse_alpha)

    import math
    steps_per_epoch = max(1, math.ceil(len(train_df) / BATCH_SIZE))
    total_steps = max(1, EPOCHS * steps_per_epoch)
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=MODEL_CONFIGS[backbone_type]['learning_rate'],
        decay_steps=total_steps,
        alpha=0.01
    )
    
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
    
    model.compile(
        optimizer=optimizer,
        loss={
            "coarse_output": coarse_loss,
            "fine_output": utils.masked_sparse_ce_with_oe,
        },
        metrics={
            "coarse_output": ["sparse_categorical_accuracy"],
            "fine_output": ["sparse_categorical_accuracy"],
        },
    )

    # Callbacks using utils
    callbacks = utils.create_callbacks(model_dir, backbone_type)
    
    # Add progress callback for detailed logging
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{EPOCHS} - {MODEL_CONFIGS[backbone_type]['name']}")
            print(f"{'='*60}")
            
        def on_epoch_end(self, epoch, logs=None):
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"- Coarse Loss: {logs.get('coarse_output_loss', 0):.4f}")
            print(f"- Coarse Accuracy: {logs.get('coarse_output_sparse_categorical_accuracy', 0):.4f}")
            print(f"- Fine Loss: {logs.get('fine_output_loss', 0):.4f}")
            print(f"- Fine Accuracy: {logs.get('fine_output_sparse_categorical_accuracy', 0):.4f}")
            print(f"- Total Loss: {logs.get('loss', 0):.4f}")
            if 'val_loss' in logs:
                print(f"- Val Loss: {logs.get('val_loss', 0):.4f}")
                print(f"- Val Coarse Acc: {logs.get('val_coarse_output_sparse_categorical_accuracy', 0):.4f}")
                print(f"- Val Fine Acc: {logs.get('val_fine_output_sparse_categorical_accuracy', 0):.4f}")
    
    callbacks.append(ProgressCallback())

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
    )

    # Save history
    with open(model_dir / f"{backbone_type}_stats.json", "w") as f:
        json.dump(history.history, f)
    
    # Save the trained model
    model.save(model_dir / f"{backbone_type}_model.keras")
    
    print(f"{MODEL_CONFIGS[backbone_type]['name']} training complete. Model saved to '{model_dir}'.")
    return model

# -------------------- Ensemble Methods --------------------
# Use utils ensemble functions instead of custom implementations

def create_stacking_ensemble(models, train_df, val_df):
    """Create a stacking ensemble that learns to combine model predictions."""
    print("\nCreating stacking ensemble...")
    
    # Get predictions from all models on TRAINING set for meta-training
    # Build dataset per model to ensure correct preprocessing
    all_coarse_preds = []
    all_fine_preds = []
    for backbone_type, model in models.items():
        print(f"Getting training predictions from {backbone_type} for stacking...")
        train_ds = utils.build_dataset(train_df, is_training=False, backbone_type=backbone_type)
        preds = model.predict(train_ds, verbose=0)
        all_coarse_preds.append(preds[0])  # Coarse predictions
        all_fine_preds.append(preds[1])  # Fine predictions
        
    
    # Stack predictions: shape (n_samples, n_models * n_classes)
    stacked_fine = np.concatenate(all_fine_preds, axis=1)
    stacked_coarse = np.concatenate(all_coarse_preds, axis=1)
    
    # Safety assertion: predictions must align 1:1 with DataFrame rows
    expected = len(train_df)
    assert stacked_fine.shape[0] == expected, f"Fine predictions shape {stacked_fine.shape[0]} != expected {expected}"
    assert stacked_coarse.shape[0] == expected, f"Coarse predictions shape {stacked_coarse.shape[0]} != expected {expected}"
        
    # Create meta-model
    fine_input = keras.Input(shape=(stacked_fine.shape[1],), name='fine_features')
    coarse_input = keras.Input(shape=(stacked_coarse.shape[1],), name='coarse_features')
    
    # Meta-model layers for fine-grained classification
    fine_meta = keras.layers.Dense(128, activation='relu')(fine_input)
    fine_meta = keras.layers.Dropout(0.3)(fine_meta)
    fine_meta = keras.layers.Dense(64, activation='relu')(fine_meta)
    fine_meta = keras.layers.Dropout(0.2)(fine_meta)
    fine_output = keras.layers.Dense(N_DX_CLASSES, name='fine_output')(fine_meta)
    
    # Meta-model layers for coarse classification
    coarse_meta = keras.layers.Dense(64, activation='relu')(coarse_input)
    coarse_meta = keras.layers.Dropout(0.3)(coarse_meta)
    coarse_meta = keras.layers.Dense(32, activation='relu')(coarse_meta)
    coarse_meta = keras.layers.Dropout(0.2)(coarse_meta)
    coarse_output = keras.layers.Dense(N_LESION_TYPE_CLASSES, name='coarse_output')(coarse_meta)
    
    meta_model = keras.Model(
        inputs=[coarse_input, fine_input],
        outputs=[coarse_output, fine_output]
    )
    
    return meta_model, stacked_coarse, stacked_fine

def train_stacking_ensemble(meta_model, stacked_coarse_train, stacked_fine_train, train_df, val_df, models):
    """Train the stacking ensemble meta-model on training data and validate on validation data."""
    print("\nTraining stacking ensemble meta-model...")
    
    # Prepare TRAINING targets
    coarse_true_train = train_df['head1_idx'].astype('int32').values
    fine_true_train = train_df['head2_idx'].fillna(-1).astype('int32').values
    
    
    # Create valid mask for fine-grained labels (training)
    mask_train = fine_true_train >= 0
    fine_true_train_valid = fine_true_train[mask_train]
    coarse_true_train_valid = coarse_true_train[mask_train]
    stacked_fine_train_valid = stacked_fine_train[mask_train]
    stacked_coarse_train_valid = stacked_coarse_train[mask_train]
    
    # Prepare VALIDATION targets for validation during training
    coarse_true_val = val_df['head1_idx'].astype('int32').values
    fine_true_val = val_df['head2_idx'].fillna(-1).astype('int32').values
    
    # Create valid mask for fine-grained labels (validation)
    mask_val = fine_true_val >= 0
    coarse_true_val_valid = coarse_true_val[mask_val]
    fine_true_val_valid = fine_true_val[mask_val]
    
    # Get validation predictions from all models for validation during meta-training
    val_fine_preds = []
    val_coarse_preds = []
    
    for backbone_type, model in models.items():
        print(f"Getting validation predictions from {backbone_type} for meta-validation...")
        val_ds = utils.build_dataset(val_df, is_training=False, backbone_type=backbone_type)
        val_preds = model.predict(val_ds, verbose=0)
        val_fine_preds.append(val_preds[1])
        val_coarse_preds.append(val_preds[0])
    
    # Stack validation predictions
    stacked_fine_val = np.concatenate(val_fine_preds, axis=1)
    stacked_coarse_val = np.concatenate(val_coarse_preds, axis=1)
    
    # Apply validation mask
    stacked_fine_val_valid = stacked_fine_val[mask_val]
    stacked_coarse_val_valid = stacked_coarse_val[mask_val]
    
    # Calculate class weights for meta-model (using training data)
    coarse_counts = utils.counts_from_labels(coarse_true_train_valid, N_LESION_TYPE_CLASSES, (0, N_LESION_TYPE_CLASSES))
    coarse_alpha = utils.calculate_focal_alpha(coarse_counts)
    
    # Compile meta-model
    coarse_loss = utils.sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=coarse_alpha)
    
    # Add cosine decay learning rate schedule for meta-model
    import math
    meta_steps_per_epoch = max(1, math.ceil(len(stacked_coarse_train_valid) / BATCH_SIZE))
    meta_total_steps = max(1, 50 * meta_steps_per_epoch)  # 50 epochs for meta-model
    
    meta_lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR/5,
        decay_steps=meta_total_steps,
        alpha=0.01
    )
    
    meta_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=meta_lr_schedule, weight_decay=WEIGHT_DECAY),
        loss={
            "coarse_output": coarse_loss,
            "fine_output": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        },
        metrics={
            "coarse_output": ["sparse_categorical_accuracy"],
            "fine_output": ["sparse_categorical_accuracy"],
        },
    )
    
    # Create callbacks for meta-model
    meta_model_dir = ENSEMBLE_OUTDIR / "stacking_ensemble"
    meta_model_dir.mkdir(exist_ok=True, parents=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(meta_model_dir / "stacking_best_model.keras"),
            save_best_only=True,
            monitor="val_coarse_output_loss",
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,  # Increased patience for meta-model convergence
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-8,
            verbose=1,
        ),
    ]
    
    # Train meta-model on training data with validation on validation data
    history = meta_model.fit(
        [stacked_coarse_train_valid, stacked_fine_train_valid],
        [coarse_true_train_valid, fine_true_train_valid],
        validation_data=([stacked_coarse_val_valid, stacked_fine_val_valid], 
                        [coarse_true_val_valid, fine_true_val_valid]),
        epochs=75,  # Increased epochs for better meta-model convergence
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save meta-model history
    with open(meta_model_dir / "stacking_history.json", "w") as f:
        json.dump(history.history, f)
    
    print(f"Stacking ensemble trained and saved to: {meta_model_dir}")
    return meta_model, meta_model_dir

# Removed evaluate_ensemble function - evaluation will be done in separate script

# -------------------- Model Loading Functions --------------------

def load_existing_models(train_df):
    """Load all existing individual models."""
    models = {}
    
    # Calculate focal alpha like in base model (needed for recompilation)
    coarse_counts = utils.counts_from_labels(train_df["head1_idx"], N_LESION_TYPE_CLASSES, (0, N_LESION_TYPE_CLASSES))
    coarse_alpha = utils.calculate_focal_alpha(coarse_counts)
    
    for backbone_type in MODEL_CONFIGS.keys():
        model_dir = INDIVIDUAL_OUTDIR / backbone_type
        model_path = model_dir / f"{backbone_type}_best_model.keras"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading {MODEL_CONFIGS[backbone_type]['name']} from {model_path}")
        
        # Load model with custom objects for loss functions (matching base model approach)
        custom_objects = {
            'masked_sparse_ce_with_oe': utils.masked_sparse_ce_with_oe,
            'sparse_categorical_focal_loss': utils.sparse_categorical_focal_loss
        }
        
        try:
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
            models[backbone_type] = model
            print(f"✓ {MODEL_CONFIGS[backbone_type]['name']} loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load {backbone_type} with custom objects: {e}")
            print("Trying to load without custom objects...")
            try:
                # Try loading without custom objects (for inference only)
                model = keras.models.load_model(model_path, compile=False)
                # Recompile with custom objects for training (matching individual model training approach)
                if USE_FOCAL_COARSE:
                    coarse_loss = utils.sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=coarse_alpha)
                else:
                    coarse_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                
                # Use the same optimizer setup as individual model training
                import math
                steps_per_epoch = max(1, math.ceil(len(train_df) / BATCH_SIZE))
                total_steps = max(1, EPOCHS * steps_per_epoch)
                
                lr_schedule = keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=MODEL_CONFIGS[backbone_type]['learning_rate'],
                    decay_steps=total_steps,
                    alpha=0.01
                )
                
                optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
                
                model.compile(
                    optimizer=optimizer,
                    loss={
                        "coarse_output": coarse_loss,
                        "fine_output": utils.masked_sparse_ce_with_oe,
                    },
                    metrics={
                        "coarse_output": ["sparse_categorical_accuracy"],
                        "fine_output": ["sparse_categorical_accuracy"],
                    },
                )
                models[backbone_type] = model
                print(f"✓ {MODEL_CONFIGS[backbone_type]['name']} loaded and recompiled successfully")
            except Exception as e2:
                raise RuntimeError(f"Failed to load {backbone_type} model: {e2}")
    
    return models

# -------------------- Main Functions --------------------

# Removed create_and_evaluate_ensembles function - evaluation will be done in separate script

def main():
    """
    Main function to run ensemble training.
    Can either train individual models from scratch or load existing ones.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ensemble Learning Pipeline')
    parser.add_argument('--load-existing', action='store_true',
                       help='Load existing individual models instead of training from scratch')
    args = parser.parse_args()
    
    if args.load_existing:
        print("Starting Ensemble Learning Pipeline (Loading Existing Models)...")
    else:
        print("Starting Ensemble Learning Pipeline (Training from Scratch)...")
    
    # Create output directories
    ENSEMBLE_OUTDIR.mkdir(exist_ok=True, parents=True)
    INDIVIDUAL_OUTDIR.mkdir(exist_ok=True, parents=True)
    
    # Load and prepare data using utils
    df = pd.read_csv(PREPARED_CSV)
    processed_df = utils.process_labels(df)

    # Split data - Use proper 3-way split
    train_df = processed_df[processed_df.split == "train"].copy()
    val_df = processed_df[processed_df.split == "val"].copy()
    print("Using existing train/val/test split")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Load or train individual models
    models = {}
    
    if args.load_existing:
        # Load existing individual models
        print("\nLoading existing individual models...")
        try:
            models = load_existing_models(train_df)
            print(f"\nLoaded {len(models)} individual models:")
            for backbone_type, model in models.items():
                print(f"- {MODEL_CONFIGS[backbone_type]['name']}")
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Please train individual models first by running:")
            print("python ensemble_model.py")
            return
    else:
        # Train individual models from scratch
        print("\nTraining individual models from scratch...")
        
        for backbone_type in MODEL_CONFIGS.keys():
            model_dir = INDIVIDUAL_OUTDIR / backbone_type
            
            # Check if EfficientNet model already exists (from base model training)
            if backbone_type == 'efficientnet':
                base_model_path = Path("outputs/base_model/simple_twohead_best_model.keras")
                if base_model_path.exists():
                    print(f"\nLoading existing EfficientNetB1 model from {base_model_path}")
                    try:
                        # Load model with custom objects for loss functions
                        custom_objects = {
                            'masked_sparse_ce_with_oe': utils.masked_sparse_ce_with_oe,
                            'sparse_categorical_focal_loss': utils.sparse_categorical_focal_loss
                        }
                        model = keras.models.load_model(base_model_path, custom_objects=custom_objects)
                        # Copy to ensemble directory for consistency
                        model_dir.mkdir(exist_ok=True, parents=True)
                        model.save(str(model_dir / "efficientnet_best_model.keras"))
                        print(f"EfficientNetB1 model loaded and saved to {model_dir}")
                        models[backbone_type] = model
                        continue
                    except Exception as e:
                        print(f"Failed to load existing model: {e}")
                        print("Falling back to training from scratch...")
            
            # Train other models or EfficientNet if loading failed
            model = train_individual_model(backbone_type, train_df, val_df, model_dir)
            models[backbone_type] = model
        
        print(f"\nTrained {len(models)} individual models:")
        for backbone_type, model in models.items():
            print(f"- {MODEL_CONFIGS[backbone_type]['name']}")
        
    # Create and train stacking ensemble   
    meta_model, stacked_coarse, stacked_fine = create_stacking_ensemble(models, train_df, val_df)
    _, stacking_dir = train_stacking_ensemble(meta_model, stacked_coarse, stacked_fine, train_df, val_df, models)
    
    print(f"Individual models saved to: {INDIVIDUAL_OUTDIR}")
    print(f"Stacking ensemble saved to: {stacking_dir}")

if __name__ == "__main__":
    main()
