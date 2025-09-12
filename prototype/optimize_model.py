# Model Optimization Script for Mobile Prototype
# This script converts your trained model to mobile-ready format

import os
import sys
import tensorflow as tf
from pathlib import Path

def find_best_model():
    """
    Find the best model from your outputs directory
    """
    outputs_dir = Path("../outputs")
    
    # Look for model files
    model_files = []
    for model_dir in outputs_dir.iterdir():
        if model_dir.is_dir():
            for file in model_dir.iterdir():
                if file.suffix == '.keras' and 'best' in file.name.lower():
                    model_files.append(file)
    
    if not model_files:
        print("❌ No model files found in outputs directory")
        return None
    
    # Return the most recent model
    best_model = max(model_files, key=lambda x: x.stat().st_mtime)
    print(f"✅ Found model: {best_model}")
    return best_model

def optimize_model_for_mobile(model_path, output_path):
    """
    Convert Keras model to TensorFlow Lite for mobile deployment
    """
    print(f"🔄 Loading model from: {model_path}")
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"📊 Model summary:")
        model.summary()
        
        # Convert to TensorFlow Lite
        print("🔄 Converting to TensorFlow Lite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for mobile
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ Model optimized and saved to: {output_path}")
        print(f"📱 Model size: {size_mb:.2f} MB")
        
        return size_mb
        
    except Exception as e:
        print(f"❌ Error optimizing model: {str(e)}")
        return None

def main():
    print("🚀 Starting model optimization for mobile prototype...")
    
    # Find the best model
    model_path = find_best_model()
    if not model_path:
        print("Please train a model first or check the outputs directory")
        return
    
    # Create prototype directory
    prototype_dir = Path(".")
    prototype_dir.mkdir(exist_ok=True)
    
    # Optimize model
    output_path = prototype_dir / "model.tflite"
    size_mb = optimize_model_for_mobile(model_path, output_path)
    
    if size_mb:
        print(f"\n🎉 Model optimization complete!")
        print(f"📱 Mobile-ready model: {output_path}")
        print(f"💾 Size: {size_mb:.2f} MB")
        print(f"\n🚀 Next steps:")
        print(f"1. Run: streamlit run app.py")
        print(f"2. Open browser to test the prototype")
        print(f"3. Test on mobile device for responsive design")
    else:
        print("❌ Model optimization failed")

if __name__ == "__main__":
    main()



