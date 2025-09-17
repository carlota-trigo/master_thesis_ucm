# -*- coding: utf-8 -*-
"""
Test script for SSL + ResNet hybrid model deployment
"""

import requests
import json
from pathlib import Path
import io
from PIL import Image

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_prediction():
    """Test prediction with sample image"""
    print("\nTesting prediction...")
    
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color='red')
    img_buffer = io.BytesIO()
    test_image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    try:
        files = {'file': ('test_image.png', img_buffer, 'image/png')}
        response = requests.post('http://localhost:5000/api/predict', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Prediction successful:")
            print(f"  Filename: {data.get('filename', 'Unknown')}")
            print(f"  Coarse classification: {data.get('coarse_classification', {}).get('class', 'Unknown')}")
            print(f"  Fine classification: {data.get('fine_classification', {}).get('class', 'Unknown')}")
            print(f"  Coarse confidence: {data.get('coarse_classification', {}).get('confidence', 'Unknown'):.3f}")
            print(f"  Fine confidence: {data.get('fine_classification', {}).get('confidence', 'Unknown'):.3f}")
            
            # Show OOD detection results
            ood_data = data.get('ood_detection', {})
            if ood_data:
                print(f"  OOD Detection: {'OOD' if ood_data.get('is_out_of_distribution', False) else 'ID'}")
                print(f"  OOD Score: {ood_data.get('ood_score', 0):.3f}")
                print(f"  Method: {ood_data.get('method', 'unknown')}")
            
            # Show reliability
            reliability = data.get('reliability', {})
            if reliability:
                print(f"  Reliability: {'Reliable' if reliability.get('is_reliable', False) else 'Unreliable'}")
                print(f"  Overall confidence: {reliability.get('overall_confidence', 0):.3f}")
            return True
        else:
            print(f"✗ Prediction failed: {response.status_code}")
            print(f"  Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("Testing SSL + ResNet OOD Model Deployment")
    print("="*50)
    
    tests = [
        test_health_check,
        test_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*50)
    print(f"Test Results: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("🎉 All tests passed! The simple deployment is working.")
    else:
        print("⚠️  Some tests failed. Please check the deployment.")

if __name__ == "__main__":
    main()
