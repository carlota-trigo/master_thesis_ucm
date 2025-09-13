#!/usr/bin/env python3
import sys, os, pandas as pd, numpy as np

print("🔍 QUICK TRAINING CHECK")
print("=" * 40)

# Check required files
files = ["self_supervised_model.py", "utils.py", "../data/training_prepared_data.csv"]
all_ok = True

for f in files:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        print(f"❌ {f}")
        all_ok = False

# Test imports
print("\n📦 Testing imports...")
modules = ["tensorflow", "keras", "utils", "sklearn"]
for m in modules:
    try:
        exec(f"import {m}")
        print(f"✅ {m}")
    except Exception as e:
        print(f"❌ {m}: {e}")
        all_ok = False

# Check data
print("\n📊 Checking data...")
try:
    df = pd.read_csv("../data/training_prepared_data.csv")
    print(f"✅ Data: {len(df)} samples")
    if 'image_path' not in df.columns:
        print("❌ Missing image_path column")
        all_ok = False
    else:
        print("✅ Required columns OK")
except Exception as e:
    print(f"❌ Data error: {e}")
    all_ok = False

# Hardware check
print("\n🖥️ Hardware check...")
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s)")
    else:
        print("ℹ️ CPU mode")
except Exception as e:
    print(f"❌ Hardware check failed: {e}")

# Summary
print("\n" + "=" * 40)
if all_ok:
    print("🎉 READY TO TRAIN!")
    print("Run: python self_supervised_model.py")
else:
    print("❌ Fix issues first")
    print("Check errors above")
print("=" * 40)
