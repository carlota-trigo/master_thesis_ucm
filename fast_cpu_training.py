#!/usr/bin/env python3
"""
Fast CPU training configuration for self_supervised_model.py
This script modifies the training parameters for much faster CPU execution.
"""

import os
import sys

# Force CPU mode from the start
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Import the main script
sys.path.insert(0, '.')
import self_supervised_model

def optimize_for_fast_cpu_training():
    """Apply optimizations for fast CPU training."""
    
    print("🚀 APPLYING FAST CPU TRAINING OPTIMIZATIONS")
    print("="*60)
    
    # Override configuration for faster training
    self_supervised_model.BATCH_SIZE_GPU = 64  # Increase from 16 to 64
    self_supervised_model.SSL_EPOCHS = 5       # Reduce from 25 to 5
    self_supervised_model.FINE_TUNE_EPOCHS = 10 # Reduce from 25 to 10
    
    print(f"✅ Batch size increased to: {self_supervised_model.BATCH_SIZE_GPU}")
    print(f"✅ SSL epochs reduced to: {self_supervised_model.SSL_EPOCHS}")
    print(f"✅ Fine-tuning epochs reduced to: {self_supervised_model.FINE_TUNE_EPOCHS}")
    
    # Force CPU mode
    self_supervised_model.use_gpu = False
    self_supervised_model.gpus = None
    self_supervised_model.FORCE_CPU_MODE = True
    
    print("✅ Forced CPU-only mode")
    print("✅ Disabled GPU configuration")
    
    print("\n" + "="*60)
    print("🎯 EXPECTED IMPROVEMENTS:")
    print(f"• Batch size: 16 → {self_supervised_model.BATCH_SIZE_GPU} (4x faster)")
    print(f"• SSL epochs: 25 → {self_supervised_model.SSL_EPOCHS} (5x faster)")
    print(f"• Total SSL steps: ~112k → ~{self_supervised_model.SSL_EPOCHS * 4482 // 4}")
    print(f"• Estimated time: 18+ hours → ~3-4 hours")
    print("="*60)
    
    return True

if __name__ == "__main__":
    optimize_for_fast_cpu_training()
    
    # Run the main training
    print("\n🚀 STARTING OPTIMIZED TRAINING...")
    self_supervised_model.main()
