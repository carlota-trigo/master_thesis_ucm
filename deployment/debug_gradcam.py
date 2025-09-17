# -*- coding: utf-8 -*-
"""
Debug script to test GradCAM layer detection
"""

import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import utils
    print("✓ Successfully imported utils module")
except ImportError as e:
    print(f"⚠ Warning: Could not import utils: {e}")
    sys.exit(1)

# Load the SSL model
SSL_MODEL_PATH = Path('../outputs/ssl_finetuned/ssl_finetuned_best_model.keras')

if not SSL_MODEL_PATH.exists():
    print(f"❌ SSL model not found: {SSL_MODEL_PATH}")
    sys.exit(1)

print(f"Loading SSL model from: {SSL_MODEL_PATH}")
ssl_model = tf.keras.models.load_model(SSL_MODEL_PATH, compile=False)
print("✓ SSL model loaded successfully")

# Print model summary
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
ssl_model.summary()

print("\n" + "="*60)
print("LAYER ANALYSIS")
print("="*60)
for i, layer in enumerate(ssl_model.layers):
    print(f"{i:2d}: {layer.name:30s} - {type(layer).__name__:20s} - {layer.output_shape}")

# Test GradCAM initialization
print("\n" + "="*60)
print("GRADCAM INITIALIZATION TEST")
print("="*60)
try:
    from gradcam import GradCAM
    gradcam = GradCAM(ssl_model)
    print("✓ GradCAM initialized successfully")
    print(f"Selected layer: {gradcam.layer_name}")
except Exception as e:
    print(f"❌ GradCAM initialization failed: {e}")
    import traceback
    traceback.print_exc()
