# Simple Dermatology App Prototype
# This creates a basic web-based demo that works on mobile

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Simple model optimization for mobile
def optimize_model_for_mobile(model_path, output_path):
    """
    Convert Keras model to TensorFlow Lite for mobile deployment
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimize for mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    return len(tflite_model) / (1024 * 1024)  # Return size in MB

# Simple prediction function
def predict_skin_lesion(image, model_path):
    """
    Simple prediction function for the prototype
    """
    try:
        # Load model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Preprocess image
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        
        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        return predictions
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Simple class labels (from your project)
FINE_CLASSES = [
    'Melanocytic nevi', 'Melanoma', 'Benign keratosis-like lesions',
    'Basal cell carcinoma', 'Actinic keratosis/SCC', 'Vascular lesions',
    'Dermatofibroma', 'Other lesions', 'No lesion'
]

COARSE_CLASSES = ['Benign', 'Malignant', 'No lesion']

def main():
    st.set_page_config(
        page_title="Dermatology AI Prototype",
        page_icon="🏥",
        layout="wide"
    )
    
    st.title("🏥 Dermatology AI Prototype")
    st.markdown("**Simple skin lesion classification demo**")
    
    # Sidebar for model info
    with st.sidebar:
        st.header("📱 Mobile-Ready Prototype")
        st.markdown("""
        **Features:**
        - 📸 Upload skin lesion photo
        - 🤖 AI classification
        - 📊 Confidence scores
        - 📱 Mobile-friendly
        - 🔒 Privacy-focused (local processing)
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Medical Disclaimer:**")
        st.markdown("This is a prototype for educational purposes only. Always consult a healthcare professional for medical advice.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear photo of the skin lesion"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Show image info
            st.info(f"Image size: {image.size}")
    
    with col2:
        st.header("🤖 AI Analysis")
        
        if uploaded_file is not None:
            # Check if model exists
            model_path = "prototype/model.tflite"
            
            if os.path.exists(model_path):
                # Show loading
                with st.spinner("Analyzing image..."):
                    predictions = predict_skin_lesion(image, model_path)
                
                if predictions is not None:
                    # Get predictions for both heads
                    fine_pred = predictions[0][0]  # Fine-grained predictions
                    coarse_pred = predictions[1][0]  # Coarse predictions
                    
                    # Fine-grained results
                    st.subheader("🔍 Detailed Classification")
                    fine_scores = {}
                    for i, class_name in enumerate(FINE_CLASSES):
                        confidence = fine_pred[i] * 100
                        fine_scores[class_name] = confidence
                    
                    # Show top 3 predictions
                    sorted_fine = sorted(fine_scores.items(), key=lambda x: x[1], reverse=True)
                    for i, (class_name, confidence) in enumerate(sorted_fine[:3]):
                        st.metric(f"{i+1}. {class_name}", f"{confidence:.1f}%")
                    
                    # Coarse classification
                    st.subheader("📊 Overall Classification")
                    coarse_scores = {}
                    for i, class_name in enumerate(COARSE_CLASSES):
                        confidence = coarse_pred[i] * 100
                        coarse_scores[class_name] = confidence
                    
                    # Show coarse results
                    sorted_coarse = sorted(coarse_scores.items(), key=lambda x: x[1], reverse=True)
                    for class_name, confidence in sorted_coarse:
                        st.metric(class_name, f"{confidence:.1f}%")
                    
                    # Show highest confidence
                    best_fine = sorted_fine[0]
                    best_coarse = sorted_coarse[0]
                    
                    if best_coarse[1] > 70:  # High confidence threshold
                        st.success(f"🎯 **Primary Classification:** {best_coarse[0]} ({best_coarse[1]:.1f}% confidence)")
                    else:
                        st.warning(f"⚠️ **Low Confidence:** {best_coarse[0]} ({best_coarse[1]:.1f}% confidence)")
                    
                    # Medical advice
                    st.subheader("💡 Next Steps")
                    if best_coarse[0] == "Malignant" and best_coarse[1] > 60:
                        st.error("🚨 **High Risk Detected** - Please consult a dermatologist immediately")
                    elif best_coarse[0] == "Benign":
                        st.info("✅ **Likely Benign** - Consider regular monitoring")
                    else:
                        st.info("❓ **Uncertain** - Consider professional evaluation")
                
            else:
                st.warning("⚠️ Model not found. Please run model optimization first.")
                st.code("python prototype/optimize_model.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏥 Dermatology AI Prototype | Master Thesis - UCM | 2024</p>
        <p><small>This is a research prototype for educational purposes only</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()



