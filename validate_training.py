#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script to check if self_supervised_model.py is ready for training
without actually starting the full training process.

Usage:
    python validate_training.py
    # or make it executable and run:
    chmod +x validate_training.py
    ./validate_training.py
"""

import sys
import os
import pandas as pd
import numpy as np

def validate_training_readiness():
    """Validate that everything is ready for training."""
    
    print("="*60)
    print("🔍 VALIDATING TRAINING READINESS")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Check required files
    print("\n📁 Checking required files...")
    required_files = [
        "self_supervised_model.py",
        "utils.py", 
        "../data/training_prepared_data.csv"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - NOT FOUND")
            all_passed = False
    
    # Test 2: Check imports
    print("\n📦 Testing imports...")
    try:
        import tensorflow as tf
        print(f"  ✅ TensorFlow {tf.__version__}")
    except Exception as e:
        print(f"  ❌ TensorFlow import failed: {e}")
        all_passed = False
    
    try:
        import keras
        print(f"  ✅ Keras {keras.__version__}")
    except Exception as e:
        print(f"  ❌ Keras import failed: {e}")
        all_passed = False
    
    try:
        import utils
        print("  ✅ utils module")
    except Exception as e:
        print(f"  ❌ utils import failed: {e}")
        all_passed = False
    
    try:
        import sklearn
        print(f"  ✅ scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"  ❌ scikit-learn import failed: {e}")
        all_passed = False
    
    # Test 3: Check data file
    print("\n📊 Validating data file...")
    try:
        df = pd.read_csv("../data/training_prepared_data.csv")
        print(f"  ✅ Data loaded: {len(df)} samples")
        
        # Check required columns
        required_columns = ['image_path', 'head1_idx', 'head2_idx']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"  ❌ Missing columns: {missing_columns}")
            all_passed = False
        else:
            print("  ✅ All required columns present")
        
        # Check for valid data
        if len(df) < 100:
            print(f"  ⚠️  Warning: Very small dataset ({len(df)} samples)")
        else:
            print(f"  ✅ Dataset size OK")
            
    except Exception as e:
        print(f"  ❌ Data file validation failed: {e}")
        all_passed = False
    
    # Test 4: Check GPU/CPU configuration
    print("\n🖥️  Checking hardware configuration...")
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"  ✅ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"    - GPU {i}: {gpu.name}")
        else:
            print("  ℹ️  No GPUs found - will use CPU")
    except Exception as e:
        print(f"  ❌ GPU detection failed: {e}")
        all_passed = False
    
    # Test 5: Check script syntax
    print("\n🐍 Validating script syntax...")
    try:
        with open("self_supervised_model.py", 'r') as f:
            code = f.read()
        compile(code, "self_supervised_model.py", 'exec')
        print("  ✅ Script syntax is valid")
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        all_passed = False
    except Exception as e:
        print(f"  ❌ Script validation failed: {e}")
        all_passed = False
    
    # Test 6: Check output directories
    print("\n📂 Checking output directories...")
    output_dirs = ["outputs/ssl_simclr", "outputs/ssl_finetuned"]
    
    for dir_path in output_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✅ {dir_path} ready")
        except Exception as e:
            print(f"  ❌ {dir_path} creation failed: {e}")
            all_passed = False
    
    # Test 7: Quick model creation test
    print("\n🤖 Testing model creation...")
    try:
        # Import the main script
        sys.path.insert(0, '.')
        import self_supervised_model
        
        # Test SimCLR model creation (without training)
        print("  Testing SimCLR model creation...")
        model = self_supervised_model.create_simclr_model()
        print("  ✅ SimCLR model created successfully")
        
        # Test dataset creation (small sample)
        print("  Testing dataset creation...")
        if len(df) > 10:
            sample_df = df.head(10)  # Use small sample for testing
            dataset = self_supervised_model.create_simclr_dataset(sample_df, batch_size=2)
            print("  ✅ Dataset creation successful")
        else:
            print("  ⚠️  Dataset too small for testing")
            
    except Exception as e:
        print(f"  ❌ Model creation test failed: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Your model is ready for training")
        print("\nYou can now run:")
        print("  python self_supervised_model.py")
        print("\nThe script will:")
        print("  1. Try GPU training first")
        print("  2. Automatically fallback to CPU if CuDNN issues occur")
        print("  3. Complete training successfully")
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("Please fix the issues above before training")
    
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = validate_training_readiness()
    sys.exit(0 if success else 1)
