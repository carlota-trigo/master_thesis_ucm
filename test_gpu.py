#!/usr/bin/env python3
"""Test script to verify GPU availability and configuration."""

import os
import tensorflow as tf
import numpy as np

print("="*60)
print("GPU AVAILABILITY TEST")
print("="*60)

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# List all available devices
print("\nAvailable devices:")
devices = tf.config.list_physical_devices()
for device in devices:
    print(f"  - {device}")

# Check GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"\nGPU devices found: {len(gpus)}")

if gpus:
    print("GPU details:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # Test GPU computation
    print("\nTesting GPU computation...")
    try:
        with tf.device('/GPU:0'):
            # Create a simple tensor operation
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"  Matrix multiplication result: {c.numpy()}")
            print("  ✓ GPU computation successful!")
            
        # Test mixed precision
        print("\nTesting mixed precision...")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("  ✓ Mixed precision enabled")
        
    except Exception as e:
        print(f"  ✗ GPU test failed: {e}")
else:
    print("\nNo GPU available. Will use CPU for training.")
    print("Note: CPU training will be significantly slower.")

# Check CUDA availability
print(f"\nCUDA built: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available()}")

print("\n" + "="*60)
