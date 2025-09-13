#!/usr/bin/env python3
"""Script to upgrade TensorFlow and test GPU compatibility."""

import subprocess
import sys

def run_command(cmd):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("="*60)
    print("TENSORFLOW GPU COMPATIBILITY FIX")
    print("="*60)
    
    # Check current TensorFlow version
    print("1. Checking current TensorFlow version...")
    success, stdout, stderr = run_command("python -c 'import tensorflow as tf; print(f\"TensorFlow version: {tf.__version__}\")'")
    if success:
        print(f"   Current: {stdout.strip()}")
    else:
        print(f"   Error: {stderr}")
    
    # Check CUDA version
    print("\n2. Checking CUDA version...")
    success, stdout, stderr = run_command("nvcc --version")
    if success:
        lines = stdout.split('\n')
        for line in lines:
            if 'release' in line.lower():
                print(f"   CUDA: {line.strip()}")
                break
    else:
        print(f"   Error: {stderr}")
    
    # Check cuDNN version
    print("\n3. Checking cuDNN version...")
    success, stdout, stderr = run_command("cat /usr/include/x86_64-linux-gnu/cudnn_version.h | grep CUDNN_MAJOR -A 2")
    if success:
        print(f"   cuDNN: {stdout.strip()}")
    else:
        print(f"   Error: {stderr}")
    
    print("\n4. Recommended upgrade command:")
    print("   pip install --upgrade tensorflow==2.15.0")
    
    print("\n5. Alternative (if above fails):")
    print("   pip install tf-nightly")
    
    print("\n6. After upgrading, test with:")
    print("   python test_gpu.py")
    
    print("\n" + "="*60)
    print("MANUAL STEPS:")
    print("="*60)
    print("1. Run: pip install --upgrade tensorflow==2.15.0")
    print("2. Test: python test_gpu.py")
    print("3. If successful, run: python self_supervised_model.py")
    print("="*60)

if __name__ == "__main__":
    main()
