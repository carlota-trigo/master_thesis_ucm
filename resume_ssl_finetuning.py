# resume_ssl_finetuning.py
# Resume fine-tuning with already trained SSL model
# -*- coding: utf-8 -*-
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
# GPU memory optimization
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Fix CuDNN version mismatch issues
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Check if we should force CPU mode (from environment or restart)
FORCE_CPU_MODE = os.environ.get('CUDA_VISIBLE_DEVICES', '') == '-1'

if FORCE_CPU_MODE:
    print("\n" + "="*60)
    print("CPU-ONLY MODE ENABLED")
    print("="*60)
    print("Running in CPU-only mode to avoid CuDNN issues")
    print("Training will be slower but more stable")
    print("="*60)

# Fallback options for CuDNN issues
try:
    import tensorflow as tf
    # Test if CuDNN is working
    tf.config.experimental.list_physical_devices('GPU')
except Exception as e:
    print(f"CuDNN initialization warning: {e}")
    print("Will attempt to continue with fallback options...")

import utils
from pathlib import Path
import json, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# -------------------- Repro --------------------
utils.set_seed(utils.SEED)

# Configure GPU for optimal performance
print("="*60)
print("GPU CONFIGURATION")
print("="*60)

# List all available devices
print("Available devices:")
for device in tf.config.list_physical_devices():
    print(f"  - {device}")

# Configure GPU memory growth and mixed precision with CuDNN fallback
gpus = None
use_gpu = False

# Skip GPU configuration if CPU-only mode is forced
if FORCE_CPU_MODE:
    print("\nCPU-only mode: Skipping GPU configuration")
    gpus = None
    use_gpu = False
else:
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"\nFound {len(gpus)} GPU(s):")
            try:
                for i, gpu in enumerate(gpus):
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"  GPU {i}: {gpu.name} - Memory growth enabled")
                    
                    # Set memory limit to prevent OOM
                    try:
                        memory_limit = tf.config.experimental.get_device_details(gpu).get('device_memory_limit', 22000)
                        tf.config.experimental.set_memory_limit(gpu, int(memory_limit * MAX_GPU_MEMORY_FRACTION))
                        print(f"  GPU {i}: Memory limit set to {MAX_GPU_MEMORY_FRACTION*100}%")
                    except:
                        print(f"  GPU {i}: Could not set memory limit")
                
                # Test CuDNN by creating a simple operation
                try:
                    with tf.device('/GPU:0'):
                        test_tensor = tf.ones((1, 1, 1, 1))
                        _ = tf.nn.conv2d(test_tensor, tf.ones((1, 1, 1, 1)), strides=1, padding='SAME')
                    print("  CuDNN test: SUCCESS")
                    use_gpu = True
                except Exception as cudnn_error:
                    print(f"  CuDNN test: FAILED - {cudnn_error}")
                    print("  CuDNN version mismatch detected. Disabling GPU acceleration.")
                    # Disable GPU to force CPU training
                    try:
                        tf.config.experimental.set_visible_devices([], 'GPU')
                        # Force CPU-only mode
                        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                        print("  GPU disabled - falling back to CPU training")
                    except:
                        pass
                    gpus = None
                    use_gpu = False
                
                if use_gpu:
                    # Enable mixed precision for better GPU performance
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                    print("  Mixed precision enabled (float16)")
                    print("  Using GPU:0 as default device")
                
            except RuntimeError as e:
                print(f"GPU configuration failed: {e}")
                print("Falling back to CPU training")
                use_gpu = False
        else:
            print("\nNo GPU found. Training will use CPU.")
            print("Note: CPU training will be significantly slower.")
            use_gpu = False
    except Exception as e:
        print(f"GPU detection failed: {e}")
        print("Falling back to CPU training")
        use_gpu = False

# -------------------- Config --------------------
# Use constants from utils
DATA_DIR = utils.DATA_DIR
PREPARED_CSV = utils.PREPARED_CSV
IMAGE_PATH = utils.IMAGE_PATH
IMG_SIZE = utils.IMG_SIZE
BATCH_SIZE = utils.BATCH_SIZE

# Self-supervised learning config
SSL_OUTDIR = Path("outputs/ssl_simclr")
FINE_TUNE_OUTDIR = Path("outputs/ssl_finetuned")

FINE_TUNE_EPOCHS = 25  # Match base model epochs
LR_FINE_TUNE = 1e-5  # Match base model learning rate
WEIGHT_DECAY = 1e-4

# Optimize batch size based on available hardware
BATCH_SIZE_GPU = BATCH_SIZE  # Default value
if use_gpu and not FORCE_CPU_MODE:
    # Use conservative batch size to avoid OOM errors
    BATCH_SIZE_GPU = 64  # Reduced from 256 to prevent memory issues
    print(f"GPU detected: Using conservative batch size {BATCH_SIZE_GPU} to prevent OOM")
else:
    # CPU training - use optimized batch size
    BATCH_SIZE_GPU = 64  # Optimized batch size for CPU
    print(f"CPU training: Using batch size {BATCH_SIZE_GPU}")

# Memory optimization settings
USE_GRADIENT_CHECKPOINTING = True
MAX_GPU_MEMORY_FRACTION = 0.8  # Use only 80% of GPU memory

# Use constants from utils
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

# -------------------- Fine-tuning Model --------------------
def create_finetuned_model(ssl_encoder, n_fine, n_coarse, img_size=IMG_SIZE, dropout=0.2):
    inputs = keras.Input(shape=(img_size, img_size, 3), name="input")
    # get the backbone by name - handle both EfficientNet and ResNet50
    try:
        backbone = ssl_encoder.get_layer("efficientnetb1")
    except:
        try:
            backbone = ssl_encoder.get_layer("resnet50")
        except:
            # Fallback: get the first layer that's not Input
            backbone = ssl_encoder.layers[1]
    
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout, name="top_dropout")(x)
    out_coarse = layers.Dense(n_coarse, name="coarse_output")(x)
    out_fine   = layers.Dense(n_fine,   name="fine_output")(x)
    return keras.Model(inputs=inputs, outputs=[out_coarse, out_fine], name="SSL_FineTuned")

# -------------------- Memory Management --------------------
def cleanup_memory():
    """Clean up GPU and system memory."""
    try:
        import gc
        gc.collect()
        
        if use_gpu and gpus:
            # Clear GPU cache
            tf.keras.backend.clear_session()
            print("GPU memory cleanup completed")
        else:
            print("CPU memory cleanup completed")
    except Exception as e:
        print(f"Memory cleanup error: {e}")

# -------------------- Main Function --------------------
def main():
    """
    Main function to resume fine-tuning with already trained SSL model.
    """
    global BATCH_SIZE_GPU  # Make BATCH_SIZE_GPU accessible within function
    
    print("Resuming SSL Fine-tuning pipeline...")
    
    # ==================== LOAD TRAINED SSL MODEL ====================
    print("\n" + "="*60)
    print("LOADING TRAINED SSL MODEL")
    print("="*60)
    
    # Check if SSL model exists
    ssl_model_path = SSL_OUTDIR / "ssl_simclr_best_model.keras"
    if not ssl_model_path.exists():
        print(f"Error: SSL model not found at {ssl_model_path}")
        print("Please run the SSL training first.")
        return
    
    try:
        # Load the trained SSL model with custom objects
        # We need to define the custom loss function for loading - EXACT COPY from original
        TEMPERATURE = 0.1  # Same as original
        
        @tf.keras.utils.register_keras_serializable()
        def simclr_loss(temperature=TEMPERATURE):
            def loss_fn(y_true, y_pred):
                # SimCLR doesn't use y_true, but we need to accept it for Keras compatibility
                z = tf.nn.l2_normalize(y_pred, axis=1)
                n = tf.shape(z)[0] // 2
                z1, z2 = z[:n], z[n:]
                z = tf.concat([z1, z2], axis=0)                  # (2n,d)
                sim = tf.matmul(z, z, transpose_b=True) / temperature  # (2n,2n)

                # mask self-similarity
                mask = tf.eye(2*n, dtype=tf.bool)
                sim = tf.where(mask, tf.fill(tf.shape(sim), -1e9), sim)


                # positives: i <-> i+n
                pos = tf.concat([tf.range(n, 2*n), tf.range(0, n)], axis=0)  # (2n,)
                loss = tf.keras.losses.sparse_categorical_crossentropy(pos, sim, from_logits=True)
                return tf.reduce_mean(loss)
            return loss_fn
        
        # Register the custom loss function
        custom_objects = {'loss_fn': simclr_loss()}
        
        ssl_model = keras.models.load_model(ssl_model_path, custom_objects=custom_objects)
        print(f"Successfully loaded SSL model from {ssl_model_path}")
        print(f"SSL model summary:")
        ssl_model.summary()
    except Exception as e:
        print(f"Error loading SSL model: {e}")
        print("Trying alternative loading method...")
        
        # Alternative: Load just the weights and reconstruct the model
        try:
            # Import the model creation function from the original script
            from self_supervised_model import create_simclr_model
            
            # Create a new model with the same architecture
            ssl_model = create_simclr_model()
            
            # Load the weights
            ssl_model.load_weights(str(ssl_model_path).replace('.keras', '_weights.h5'))
            print(f"Successfully loaded SSL model weights")
            print(f"SSL model summary:")
            ssl_model.summary()
        except Exception as e2:
            print(f"Alternative loading also failed: {e2}")
            print("Please check if the SSL model files exist:")
            print(f"  Model file: {ssl_model_path}")
            print(f"  Weights file: {str(ssl_model_path).replace('.keras', '_weights.h5')}")
            return
    
    # Clean memory before fine-tuning
    print("Cleaning memory before fine-tuning...")
    cleanup_memory()
    
    # ==================== STEP 2: SUPERVISED FINE-TUNING ====================
    print("\n" + "="*60)
    print("STEP 2: SUPERVISED FINE-TUNING")
    print("="*60)
    
    FINE_TUNE_OUTDIR.mkdir(exist_ok=True, parents=True)
    
    # Load and process data using utils
    try:
        df = pd.read_csv(PREPARED_CSV)
        print(f"Loaded dataset with {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: Could not find {PREPARED_CSV}")
        print("Please ensure the data file exists and path is correct.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    processed_df = utils.process_labels(df)

    # Split data - Use proper 3-way split (test set will be used later for evaluation)
    if "split" in processed_df.columns:
        train_df = processed_df[processed_df.split == "train"].copy()
        val_df = processed_df[processed_df.split == "val"].copy()
        # test_df will be used later in evaluation script
        print("Using existing train/val/test split")
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")
        print("Note: Test set will be used later for unbiased evaluation")
    else:
        # Fallback: Create 3-way split if no split column exists
        train_val_df, test_df = train_test_split(
            processed_df, test_size=0.15, stratify=processed_df['head1_idx'], random_state=utils.SEED
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.2, stratify=train_val_df['head1_idx'], random_state=utils.SEED
        )
        print("Created stratified train/val/test split")
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print("Note: Test set will be used later for unbiased evaluation")

    # Calculate class weights using utils
    coarse_counts = utils.counts_from_labels(train_df["head1_idx"], N_LESION_TYPE_CLASSES, (0, N_LESION_TYPE_CLASSES))    
    coarse_alpha = utils.calculate_focal_alpha(coarse_counts)

    # Build datasets using utils
    minority_fine_names = ["df", "vasc", "other", "no_lesion"]
    minority_fine_ids = {DX_TO_ID[n] for n in minority_fine_names if n in DX_TO_ID}

    if USE_OVERSAMPLING:
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
    else:
        train_ds = utils.build_dataset(train_df, is_training=True, backbone_type='efficientnet', minority_fine_ids=minority_fine_ids,
                                 fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)

    val_ds = utils.build_dataset(val_df, is_training=False, backbone_type='efficientnet', minority_fine_ids=minority_fine_ids)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create fine-tuned model
    model = create_finetuned_model(ssl_model, N_DX_CLASSES, N_LESION_TYPE_CLASSES)
    
    # Freeze backbone initially, then unfreeze gradually
    backbone = model.layers[1]
    backbone.trainable = False
    
    # Compile with frozen backbone
    if USE_FOCAL_COARSE:
        coarse_loss = utils.sparse_categorical_focal_loss(gamma=FOCAL_GAMMA, alpha=coarse_alpha)
    else:
        coarse_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    steps_per_epoch = len(train_df) // BATCH_SIZE_GPU
    if steps_per_epoch == 0:
        steps_per_epoch = 1  # Minimum 1 step per epoch
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
            "coarse_output": coarse_loss,
            "fine_output": utils.masked_sparse_ce_with_oe,
        },
        metrics={
            "coarse_output": ["sparse_categorical_accuracy"],
            "fine_output": ["sparse_categorical_accuracy"],
        },
    )
    print(model.summary())
    
    # Callbacks using utils
    callbacks = utils.create_callbacks(FINE_TUNE_OUTDIR, 'ssl_finetuned')
    
    # Add GPU monitoring callback with memory cleanup
    class GPUMonitorCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if use_gpu and gpus:
                try:
                    # Get GPU memory info
                    gpu_details = tf.config.experimental.get_device_details(gpus[0])
                    print(f"\nGPU Memory Info:")
                    print(f"  Device: {gpus[0].name}")
                    print(f"  Memory limit: {gpu_details.get('device_memory_limit', 'Unknown')}")
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Clear TensorFlow cache
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    print(f"GPU monitoring error: {e}")
            else:
                # CPU training - just do basic cleanup
                try:
                    import gc
                    gc.collect()
                except:
                    pass
                    
        def on_batch_end(self, batch, logs=None):
            # Periodic memory cleanup during training
            if batch % 50 == 0:  # Every 50 batches
                try:
                    import gc
                    gc.collect()
                except:
                    pass

    # Add custom progress callback for detailed tracking
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1}/{FINE_TUNE_EPOCHS}")
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
    callbacks.append(GPUMonitorCallback())

    # Calculate steps for better progress tracking
    steps_per_epoch = len(train_df) // BATCH_SIZE_GPU
    if steps_per_epoch == 0:
        steps_per_epoch = 1  # Minimum 1 step per epoch
    validation_steps = len(val_df) // BATCH_SIZE_GPU
    if validation_steps == 0:
        validation_steps = 1  # Minimum 1 validation step
    
    print(f"\nTraining Configuration:")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Validation steps: {validation_steps}")
    print(f"- Total training steps: {steps_per_epoch * FINE_TUNE_EPOCHS}")
    print(f"- Batch size: {BATCH_SIZE_GPU}")
    print(f"- Learning rate: {LR_FINE_TUNE}")
    if FORCE_CPU_MODE:
        print(f"- Hardware: CPU (forced due to CuDNN issues)")
        print(f"- Mixed precision: Disabled (CPU mode)")
    else:
        print(f"- Hardware: {'GPU (RTX 4090)' if use_gpu else 'CPU'}")
        print(f"- Mixed precision: {'Enabled (float16)' if use_gpu else 'Disabled (CPU)'}")
    print(f"- Early stopping: Enabled (patience=10)")
    print(f"\nStarting training...\n")

    # Train with frozen backbone first
    print("Training with frozen backbone...")
    try:
        cleanup_memory()  # Clean memory before training
        
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=10,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )
    except Exception as e:
        print(f"Frozen backbone training failed: {e}")
        
        # Check if it's a CuDNN/GPU issue
        if "DNN library initialization failed" in str(e) or "CuDNN" in str(e):
            print("Detected CuDNN/GPU issue during fine-tuning. Restarting with CPU-only mode...")
            # Set environment variable to force CPU mode
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Please restart the script with: CUDA_VISIBLE_DEVICES=-1 python resume_ssl_finetuning.py")
            return
            
        else:
            print("Attempting memory cleanup and retry...")
            cleanup_memory()
            
            # Reduce batch size and retry
            BATCH_SIZE_GPU = 32
            # Rebuild datasets with smaller batch size
            train_ds = utils.build_dataset(train_df, is_training=True, backbone_type='efficientnet', minority_fine_ids=minority_fine_ids,
                                     fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)
            val_ds = utils.build_dataset(val_df, is_training=False, backbone_type='efficientnet', minority_fine_ids=minority_fine_ids)
            steps_per_epoch = len(train_df) // BATCH_SIZE_GPU
            validation_steps = len(val_df) // BATCH_SIZE_GPU
            
            print(f"Retrying with batch size {BATCH_SIZE_GPU}")
            history1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=10,
                callbacks=callbacks,
                verbose=1,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
            )

    # Unfreeze backbone and continue training
    backbone.trainable = True
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR_FINE_TUNE/10, weight_decay=WEIGHT_DECAY),
        loss={
            "coarse_output": coarse_loss,
            "fine_output": utils.masked_sparse_ce_with_oe,
        },
        metrics={
            "coarse_output": ["sparse_categorical_accuracy"],
            "fine_output": ["sparse_categorical_accuracy"],
        },
    )

    print("Training with unfrozen backbone...")
    cleanup_memory()  # Clean memory before unfrozen training
    
    try:
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS-10,
            initial_epoch=10,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )
    except Exception as e:
        print(f"Unfrozen backbone training failed: {e}")
        
        # Check if it's a CuDNN/GPU issue
        if "DNN library initialization failed" in str(e) or "CuDNN" in str(e):
            print("Detected CuDNN/GPU issue during unfrozen training. Restarting with CPU-only mode...")
            # Set environment variable to force CPU mode
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Please restart the script with: CUDA_VISIBLE_DEVICES=-1 python resume_ssl_finetuning.py")
            return
            
        else:
            print("Attempting memory cleanup and retry...")
            cleanup_memory()
            
            # Further reduce batch size if needed
            BATCH_SIZE_GPU = 16  # Even smaller batch size
            train_ds = utils.build_dataset(train_df, is_training=True, backbone_type='efficientnet', minority_fine_ids=minority_fine_ids,
                                     fine_oversampling=FINE_MINORITY_OVERSAMPLING if USE_FINE_OVERSAMPLING else None)
            val_ds = utils.build_dataset(val_df, is_training=False, backbone_type='efficientnet', minority_fine_ids=minority_fine_ids)
            steps_per_epoch = len(train_df) // BATCH_SIZE_GPU
            validation_steps = len(val_df) // BATCH_SIZE_GPU
            
            print(f"Retrying with batch size {BATCH_SIZE_GPU}")
            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=FINE_TUNE_EPOCHS-10,
                initial_epoch=10,
                callbacks=callbacks,
                verbose=1,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
            )

    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    # Save results
    with open(FINE_TUNE_OUTDIR / "stats.json", "w") as f:
        json.dump(combined_history, f)
    
    print(f"\nFine-tuning complete! Model saved to '{FINE_TUNE_OUTDIR}'")
    print("Note: Use a separate evaluation script with the test set for unbiased evaluation")
    
    # ==================== COMPLETION ====================
    print("\n" + "="*60)
    print("SSL FINE-TUNING PIPELINE COMPLETE")
    print("="*60)
    print("✅ Fine-tuning completed")
    print("✅ Model saved and ready for evaluation")
    print("📝 Next step: Run evaluation script on test set")
    print("="*60)

if __name__ == "__main__":
    main()
