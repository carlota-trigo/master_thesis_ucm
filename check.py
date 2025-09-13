#!/usr/bin/env python3
"""
Quick check script - alias for validate_training.py
"""

import subprocess
import sys
import os

def main():
    """Run the validation script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    validate_script = os.path.join(script_dir, "validate_training.py")
    
    if not os.path.exists(validate_script):
        print("❌ validate_training.py not found!")
        sys.exit(1)
    
    print("🔍 Running training readiness check...")
    print("=" * 50)
    
    # Run the validation script
    result = subprocess.run([sys.executable, validate_script], 
                          capture_output=False, 
                          text=True)
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
