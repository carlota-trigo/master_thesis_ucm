import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import utils
from pathlib import Path
import json, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

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
OUTDIR = Path("outputs/base_model")

EPOCHS = 25
LR = 1e-5
WEIGHT_DECAY = 1e-4
PRETRAINED = True

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

def main():
    OUTDIR.mkdir(exist_ok=True, parents=True)
    
    df = pd.read_csv(PREPARED_CSV)
    processed_df = utils.process_labels(df)

    train_df = processed_df[processed_df.split == "train"].copy()
    val_df = processed_df[processed_df.split == "val"].copy()
    print("Using existing train/val/test split")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print("Note: Test set will be used later for unbiased evaluation")
   
    minority_fine_names = ["df", "vasc", "other", "no_lesion"]
    minority_fine_ids = {DX_TO_ID[n] for n in minority_fine_names if n in DX_TO_ID}

    ds_parts = []
    weights = []
    for c in [0, 1, 2]:
        sub = train_df[train_df["head1_idx"] == c]
        if len(sub) == 0:
            continue
        ds_c = utils.build_dataset(sub, is_training=True, backbone_type='efficientnet', minority_fine_ids=minority_fine_ids, 
                            fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)
        ds_parts.append(ds_c)
        weights.append(OVERSAMPLE_WEIGHTS.get(str(c), 0.0))
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    train_ds = tf.data.Dataset.sample_from_datasets(
        ds_parts, weights=weights.tolist(), stop_on_empty_dataset=False
    )

    val_ds = utils.build_dataset(val_df, is_training=False, backbone_type='efficientnet', minority_fine_ids=minority_fine_ids)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create model using utils
    model = utils.create_two_head_model('efficientnet', N_DX_CLASSES, N_LESION_TYPE_CLASSES)
    print(model.summary())

    steps_per_epoch = len(train_df) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR,
        decay_steps=total_steps,
        alpha=0.01
    )
    
    optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
    
    # Calculate class weights and focal loss alpha
    coarse_counts = utils.counts_from_labels(train_df["head1_idx"], N_LESION_TYPE_CLASSES, (0, N_LESION_TYPE_CLASSES))
    coarse_alpha = utils.calculate_focal_alpha(coarse_counts)
    
    coarse_loss = utils.sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=coarse_alpha)
    
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

    # Callbacks
    callbacks = utils.create_callbacks(OUTDIR, 'simple_twohead')
    
    # Add custom progress callback for detailed tracking
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{EPOCHS}")
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

    # Calculate steps for better progress tracking
    steps_per_epoch = len(train_df) // BATCH_SIZE
    validation_steps = len(val_df) // BATCH_SIZE
    
    print(f"\nTraining Configuration:")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Validation steps: {validation_steps}")
    print(f"- Total training steps: {steps_per_epoch * EPOCHS}")
    print(f"- Batch size: {BATCH_SIZE}")
    print(f"- Learning rate: {LR}")
    print(f"- Mixed precision: Enabled")
    print(f"- Early stopping: Enabled (patience=10)")
    print(f"\nStarting training...\n")

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Save history
    with open(OUTDIR / "stats.json", "w") as f:
        json.dump(history.history, f)
    
    print(f"Training complete. Model saved to '{OUTDIR}'.")

if __name__ == "__main__":
    main()