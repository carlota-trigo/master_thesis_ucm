# test_path_resolution.py
# Test script to verify path resolution is working correctly
import os
import pandas as pd
from pathlib import Path
import utils

def test_path_resolution():
    """Test the path resolution function with sample data."""
    print("Testing path resolution...")
    
    # Load the prepared data
    try:
        df = pd.read_csv(utils.PREPARED_CSV)
        print(f"Loaded dataset with {len(df)} samples")
    except FileNotFoundError:
        print(f"Error: Could not find {utils.PREPARED_CSV}")
        return False
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    # Test path resolution on a few samples
    print("\nTesting path resolution on first 5 samples:")
    for i in range(min(5, len(df))):
        original_path = df.iloc[i]['image_path']
        resolved_path = utils.resolve_image_path(original_path)
        
        print(f"\nSample {i+1}:")
        print(f"  Original path: {original_path}")
        print(f"  Resolved path: {resolved_path}")
        print(f"  File exists: {os.path.exists(resolved_path)}")
        
        # Check if the resolved path looks correct
        if "images/images" in resolved_path and not "C:\\" in resolved_path:
            print(f"  ✓ Path resolution looks correct")
        else:
            print(f"  ✗ Path resolution may have issues")
    
    # Test with some problematic Windows paths
    print("\nTesting with problematic Windows paths:")
    test_paths = [
        "C:\\Users\\msi\\Desktop\\tfm-ucm\\data\\images\\images\\ISIC_1807060.jpg",
        "C:\\Users\\msi\\Desktop\\tfm-ucm\\data\\images\\images\\ISIC_0056818.jpg",
        "ISIC_1807060.jpg",  # Just filename
        "../data/images/images/ISIC_1807060.jpg"  # Relative path
    ]
    
    for test_path in test_paths:
        resolved_path = utils.resolve_image_path(test_path)
        print(f"\nTest path: {test_path}")
        print(f"Resolved: {resolved_path}")
        print(f"Exists: {os.path.exists(resolved_path)}")
        
        # Check if the resolved path is clean (no Windows paths mixed in)
        if "C:\\" not in resolved_path and "images/images" in resolved_path:
            print("✓ Path resolution successful")
        else:
            print("✗ Path resolution failed")
    
    return True

if __name__ == "__main__":
    test_path_resolution()
