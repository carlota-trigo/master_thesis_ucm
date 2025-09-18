import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image
import logging
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

sys.path.append(str(Path(__file__).parent.parent))

import utils
DX_CLASSES = utils.DX_CLASSES
LESION_TYPE_CLASSES = utils.LESION_TYPE_CLASSES
IMG_SIZE = utils.IMG_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ssl-deployment-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_FOLDER = Path('uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
SSL_MODEL_PATH = Path('../outputs/ssl_finetuned/ssl_finetuned_best_model.keras')
RESNET_MODEL_PATH = Path('../outputs/individual_models/resnet/resnet_best_model.keras')

UPLOAD_FOLDER.mkdir(exist_ok=True)

ssl_model = None
resnet_model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_class_name(class_name):
    if class_name == "no_lesion":
        return "No Lesion"
    return class_name

def load_models():
    global ssl_model, resnet_model
    try:
        if not SSL_MODEL_PATH.exists():
            logger.error(f"SSL model file not found: {SSL_MODEL_PATH}")
            return False
            
        logger.info(f"Loading SSL model from: {SSL_MODEL_PATH}")
        
        ssl_model = tf.keras.models.load_model(
            SSL_MODEL_PATH,
            compile=False
        )
        
        logger.info("SSL model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading SSL model: {e}")
        return False
    
    # Load ResNet model (for OOD detection)
    try:
        if not RESNET_MODEL_PATH.exists():
            logger.warning(f"ResNet model file not found: {RESNET_MODEL_PATH}")
            logger.warning("OOD detection will be disabled")
            resnet_model = None
        else:
            logger.info(f"Loading ResNet model from: {RESNET_MODEL_PATH}")
            
            # Load ResNet with custom objects for OOD loss functions
            custom_objects = {
                'masked_sparse_ce_with_oe': utils.masked_sparse_ce_with_oe,
                'sparse_categorical_focal_loss': utils.sparse_categorical_focal_loss
            }
            
            resnet_model = tf.keras.models.load_model(
                RESNET_MODEL_PATH,
                custom_objects=custom_objects,
                compile=False
            )
            
            logger.info("ResNet model loaded successfully")
        
    except Exception as e:
        logger.warning(f"Error loading ResNet model: {e}")
        logger.warning("OOD detection will be disabled")
        resnet_model = None
    
    return True

def preprocess_image(image_path):
    """Preprocess image for SSL prediction"""
    try:
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

def preprocess_image_for_resnet(image_path):
    """Preprocess image for ResNet prediction (with ImageNet normalization)"""
    try:
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Apply ImageNet normalization (ResNet preprocessing)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image for ResNet: {e}")
        raise

def predict_image(image_path):
    """Make prediction on image using SSL for classification and ResNet for OOD detection"""
    try:
        # Preprocess image for SSL
        ssl_image_array = preprocess_image(image_path)
        
        # Make SSL prediction (main model)
        ssl_predictions = ssl_model.predict(ssl_image_array, verbose=0)
        
        # Extract SSL predictions
        coarse_pred = ssl_predictions[0][0]
        fine_pred = ssl_predictions[1][0]
        
        # Apply softmax
        coarse_probs = tf.nn.softmax(coarse_pred).numpy()
        fine_probs = tf.nn.softmax(fine_pred).numpy()
        
        # Get SSL predictions
        coarse_class_idx = np.argmax(coarse_probs)
        fine_class_idx = np.argmax(fine_probs)
        
        # Get classes and format them
        coarse_class = format_class_name(LESION_TYPE_CLASSES[coarse_class_idx])
        
        # If no lesion, show "--" for detailed diagnosis
        if coarse_class == "No Lesion":
            fine_class = "--"
        else:
            fine_class = format_class_name(DX_CLASSES[fine_class_idx])
        
        # Get SSL confidence scores
        coarse_confidence = float(coarse_probs[coarse_class_idx])
        fine_confidence = float(fine_probs[fine_class_idx])
        
        # Calculate overall confidence
        overall_confidence = max(coarse_confidence, fine_confidence)
        
        # OOD detection using ResNet (if available)
        ood_detection = {
            'is_out_of_distribution': False,
            'ood_score': 0.0,
            'threshold': 0.7,
            'method': 'disabled'
        }
        
        # Preprocess image for ResNet
        resnet_image_array = preprocess_image_for_resnet(image_path)
        
        # Make ResNet prediction for OOD detection
        resnet_predictions = resnet_model.predict(resnet_image_array, verbose=0)
        
        # Extract ResNet predictions
        resnet_coarse_pred = resnet_predictions[0][0]
        resnet_fine_pred = resnet_predictions[1][0]
        
        # Apply softmax
        resnet_coarse_probs = tf.nn.softmax(resnet_coarse_pred).numpy()
        resnet_fine_probs = tf.nn.softmax(resnet_fine_pred).numpy()
        
        # Calculate MSP (Maximum Softmax Probability) for OOD detection
        resnet_coarse_msp = np.max(resnet_coarse_probs)
        resnet_fine_msp = np.max(resnet_fine_probs)
        resnet_msp = max(resnet_coarse_msp, resnet_fine_msp)
        
        # Calculate OOD score (higher = more OOD)
        ood_score = 1.0 - resnet_msp
        threshold = 0.7
        is_ood = resnet_msp < threshold
        
        ood_detection = {
            'is_out_of_distribution': bool(is_ood),
            'ood_score': float(ood_score),
            'threshold': threshold,
            'method': 'resnet_msp'
        }
        
        # Determine reliability based on confidence and OOD status
        is_reliable = overall_confidence > 0.7 and not ood_detection['is_out_of_distribution']
        
        return {
            'coarse_classification': {
                'class': coarse_class,
                'confidence': coarse_confidence,
                'probabilities': {
                    format_class_name(cls): float(coarse_probs[i]) 
                    for i, cls in enumerate(LESION_TYPE_CLASSES)
                }
            },
            'fine_classification': {
                'class': fine_class,
                'confidence': fine_confidence if coarse_class != "No Lesion" else 0.0,
                'probabilities': {
                    "--": 1.0 if coarse_class == "No Lesion" else float(fine_probs[i])
                    for i, cls in enumerate(DX_CLASSES)
                } if coarse_class == "No Lesion" else {
                    format_class_name(cls): float(fine_probs[i]) 
                    for i, cls in enumerate(DX_CLASSES)
                }
            },
            'ood_detection': ood_detection,
            'reliability': {
                'is_reliable': is_reliable,
                'overall_confidence': overall_confidence,
                'confidence_threshold': 0.7
            }
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'ssl_model_loaded': ssl_model is not None,
        'resnet_model_loaded': resnet_model is not None,
        'architecture': 'SSL Fine-tuned Model (Main) + ResNet (OOD Detection)',
        'ood_detection_enabled': resnet_model is not None
    })

@app.route('/api/model_info')
def model_info():
    """Model information endpoint"""
    return jsonify({
        'ssl_model': {
            'loaded': ssl_model is not None,
            'path': str(SSL_MODEL_PATH),
            'architecture': 'SSL Fine-tuned Model'
        },
        'resnet_model': {
            'loaded': resnet_model is not None,
            'path': str(RESNET_MODEL_PATH),
            'architecture': 'ResNet50 Two-Head Model',
            'purpose': 'OOD Detection'
        },
        'ood_detection': {
            'enabled': resnet_model is not None,
            'method': 'ResNet MSP' if resnet_model is not None else 'disabled'
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict endpoint for image classification with OOD detection"""
    if ssl_model is None:
        return jsonify({'error': 'SSL model not loaded'}), 500
    
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = UPLOAD_FOLDER / filename
        file.save(file_path)
        
        try:
            # Make prediction
            results = predict_image(file_path)
            results['filename'] = filename
            
            logger.info(f"Prediction completed for {filename}: {results['coarse_classification']['class']}")
            
            return jsonify(results)
            
        finally:
            # Clean up uploaded file
            if file_path.exists():
                file_path.unlink()
                
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load models
    if not load_models():
        logger.error("Failed to load models. Exiting.")
        sys.exit(1)
    
    # Run the app
    logger.info("Starting SSL + ResNet OOD Flask application...")
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
