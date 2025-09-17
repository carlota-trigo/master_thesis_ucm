# Skin Lesion Classification with Self-Supervised Learning

<div align="center">
  <img src="deployment/static/logo.svg" alt="Skin Lesion Classification API Logo" width="200"/>
</div>

## Overview

This project implements a comprehensive machine learning pipeline for skin lesion classification using self-supervised learning (SSL) techniques combined with ensemble methods. The system provides both coarse-grained (benign/malignant/no lesion) and fine-grained (specific lesion types) classification capabilities with out-of-distribution (OOD) detection.

## 🚀 Key Features

- **Self-Supervised Learning**: SimCLR-based SSL pretraining followed by fine-tuning
- **Ensemble Methods**: Multiple backbone architectures (ResNet, DenseNet, EfficientNet) with stacking ensemble
- **Dual Classification**: Coarse-grained (3 classes) and fine-grained (9 classes) classification
- **OOD Detection**: Out-of-distribution detection using ResNet MSP (Maximum Softmax Probability)
- **Web API**: Flask-based REST API with interactive web interface
- **Performance Optimization**: GPU memory optimization and mixed precision training

## 📊 Model Architecture

### Primary Models
- **SSL Fine-tuned Model**: Main classification model using self-supervised pretraining
- **ResNet50 Two-Head**: Used for OOD detection and ensemble diversity
- **DenseNet121**: Additional backbone for ensemble learning
- **EfficientNet-B0**: Lightweight efficient architecture

### Classification Tasks
- **Coarse Classification**: `benign`, `malignant`, `no_lesion`
- **Fine Classification**: `nv`, `mel`, `bkl`, `bcc`, `scc_akiec`, `vasc`, `df`, `other`, `no_lesion`

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- TensorFlow 2.13.0+

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd master_thesis_ucm

# Install dependencies
pip install -r deployment/requirements.txt

# For development dependencies
pip install jupyter matplotlib seaborn
```

## 📁 Project Structure

### Required Directory Structure

For the code to run properly, you need the following directory structure:

```
master_thesis_ucm/                    # Project root directory
├── data/                            # Data directory (parent level)
│   ├── images/                      # Image storage
│   │   └── images/                  # Actual image files
│   │       ├── ISIC_0000000.jpg
│   │       ├── ISIC_0000001.jpg
│   │       └── ...                  # All skin lesion images
│   ├── training_prepared_data.csv   # Main training dataset
│   ├── metadata_images.csv          # Image metadata
│   ├── metadata_images_trial.csv    # Trial dataset
│   ├── metadata_clean.csv           # Clean metadata
│   ├── metadata_merged.csv          # Merged metadata
│   └── metadata.csv                 # Base metadata
├── master_thesis_ucm/               # Code directory
│   ├── deployment/                  # Web API and deployment files
│   │   ├── app.py                   # Flask application
│   │   ├── requirements.txt         # API dependencies
│   │   ├── templates/               # HTML templates
│   │   │   └── index.html           # Web interface
│   │   ├── static/                  # Static assets
│   │   │   └── logo.svg             # API logo
│   │   └── uploads/                 # Temporary upload directory
│   ├── outputs/                     # Model outputs and checkpoints
│   │   ├── base_model/              # Base model results
│   │   │   ├── simple_twohead_best_model.keras
│   │   │   ├── simple_twohead_history.csv
│   │   │   └── stats.json
│   │   ├── individual_models/       # Individual backbone results
│   │   │   ├── resnet/
│   │   │   │   ├── resnet_best_model.keras
│   │   │   │   └── resnet_history.csv
│   │   │   ├── densenet/
│   │   │   └── efficientnet/
│   │   ├── ensemble_models/         # Ensemble model results
│   │   │   └── stacking_ensemble/
│   │   ├── ssl_finetuned/           # SSL fine-tuned model
│   │   │   ├── ssl_finetuned_best_model.keras
│   │   │   └── ssl_finetuned_history.csv
│   │   └── ssl_simclr/              # SSL pretraining results
│   │       └── ssl_simclr_best_model.keras
│   ├── configs/                     # Configuration files
│   ├── logs/                        # Training logs
│   ├── base_model.py                # Base model implementation
│   ├── ensemble_model.py            # Ensemble learning implementation
│   ├── self_supervised_model.py     # SSL implementation
│   ├── utils.py                     # Utility functions
│   ├── build_clean_dataset.py       # Dataset building utilities
│   ├── build_datasets_for_models.py # Model-specific datasets
│   ├── build_metadata_images.py    # Metadata processing
│   ├── delete_images.py             # Image management
│   ├── gradcam_analysis.py          # GradCAM analysis
│   ├── gradcam_integration.py       # GradCAM integration
│   ├── exploratory_data_analysis.ipynb
│   ├── images_visualization.ipynb
│   ├── model_evaluation.ipynb
│   └── verify_created_db.ipynb
└── README.md                        # This file
```

### Key Path Requirements

- **Data Location**: The `data/` folder must be at the **parent level** of the code directory
- **Image Paths**: All CSV files use relative paths like `../data/images/images/filename.jpg`
- **Model Paths**: Models are saved to `outputs/` subdirectories
- **Working Directory**: All scripts should be run from the `master_thesis_ucm/` directory

### Path Configuration

The system uses these path constants (defined in `utils.py`):
```python
DATA_DIR = Path("../data")                    # Points to parent/data
IMAGE_PATH = DATA_DIR.joinpath("images", "images")  # ../data/images/images
```

### Setting Up Your Environment

1. **Clone the repository** to your desired location
2. **Create the data directory** at the parent level:
   ```bash
   mkdir ../data
   mkdir ../data/images
   mkdir ../data/images/images
   ```
3. **Place your images** in `../data/images/images/`
4. **Place your CSV files** in `../data/`
5. **Run scripts** from the `master_thesis_ucm/` directory

## 🚀 Quick Start

### 1. Training Models

#### Self-Supervised Learning
```bash
# SSL pretraining (SimCLR)
python self_supervised_model.py

# SSL fine-tuning
python self_supervised_model.py --mode finetune
```

#### Individual Models
```bash
# Train individual backbone models
python ensemble_model.py

# Load existing models for ensemble
python ensemble_model.py --load-existing
```

#### Base Model
```bash
# Train base ResNet model
python base_model.py
```

### 2. Running the Web API

```bash
cd deployment
python app.py
```

The API will be available at `http://localhost:5000`

### 3. API Endpoints

- `GET /` - Interactive web interface
- `GET /api/health` - Health check and model status
- `GET /api/model_info` - Detailed model information
- `POST /api/predict` - Image classification endpoint

### 4. Making Predictions

#### Using the Web Interface
1. Navigate to `http://localhost:5000`
2. Upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)
3. View classification results with confidence scores

#### Using the API
```python
import requests

# Upload image for prediction
with open('image.jpg', 'rb') as f:
    response = requests.post('http://localhost:5000/api/predict', 
                           files={'file': f})
    result = response.json()
    
print(f"Coarse: {result['coarse_classification']['class']}")
print(f"Fine: {result['fine_classification']['class']}")
print(f"Confidence: {result['reliability']['overall_confidence']}")
```

## 📈 Model Performance

The models achieve high performance on skin lesion classification:

- **SSL Fine-tuned Model**: Primary classification with OOD detection
- **Ensemble Methods**: Improved robustness through model diversity
- **Confidence Scoring**: Reliability assessment for predictions
- **OOD Detection**: Identifies out-of-distribution samples

## 🔧 Configuration

### Key Parameters (in `utils.py`)
```python
IMG_SIZE = 224              # Input image size
BATCH_SIZE = 128            # Training batch size
USE_FOCAL_COARSE = True     # Focal loss for coarse classification
FOCAL_GAMMA = 2.0          # Focal loss gamma parameter
USE_SAMPLE_WEIGHTS = True   # Class balancing
```

### Model Paths (in `deployment/app.py`)
```python
SSL_MODEL_PATH = '../outputs/ssl_finetuned/ssl_finetuned_best_model.keras'
RESNET_MODEL_PATH = '../outputs/individual_models/resnet/resnet_best_model.keras'
```

## 📊 Analysis Notebooks

- `exploratory_data_analysis.ipynb` - Dataset exploration and visualization
- `model_evaluation.ipynb` - Model performance analysis
- `images_visualization.ipynb` - Image visualization utilities
- `verify_created_db.ipynb` - Database verification

## 🎯 Key Features

### Self-Supervised Learning
- **SimCLR Pretraining**: Contrastive learning for robust feature extraction
- **Fine-tuning**: Task-specific adaptation with limited labeled data
- **Transfer Learning**: Leverages pretrained representations

### Ensemble Learning
- **Architectural Diversity**: Multiple backbone architectures
- **Stacking Ensemble**: Meta-learning for improved predictions
- **Model Selection**: Optimal combination of individual models

### Out-of-Distribution Detection
- **MSP Method**: Maximum Softmax Probability for OOD detection
- **Confidence Thresholding**: Reliability assessment
- **Uncertainty Quantification**: Model confidence estimation

## 🔬 Research Context

This project implements state-of-the-art techniques for medical image classification:

- **Self-Supervised Learning**: Reduces dependency on large labeled datasets
- **Ensemble Methods**: Improves robustness and generalization
- **OOD Detection**: Critical for medical AI safety
- **Multi-task Learning**: Coarse and fine-grained classification

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@thesis{your_thesis_2024,
  title={Skin Lesion Classification with Self-Supervised Learning},
  author={Your Name},
  year={2024},
  institution={Universidad Complutense de Madrid}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support:
- Create an issue in the repository
- Contact: [your-email@example.com]

## 🔮 Future Work

- [ ] Integration with medical imaging standards (DICOM)
- [ ] Real-time inference optimization
- [ ] Multi-modal fusion (clinical data + images)
- [ ] Federated learning for privacy-preserving training
- [ ] Mobile deployment optimization

---

<div align="center">
  <p><em>Built with ❤️ for advancing medical AI</em></p>
</div>
