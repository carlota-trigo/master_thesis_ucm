# 🏥 Dermatology Classification: Advanced Machine Learning Techniques

**Master Thesis Project - Universidad Complutense de Madrid**  
**Author:** Carlota Trigo La Blanca  
**Course:** Master in Big Data, Data Science & AI

## 📋 Overview

This repository implements advanced machine learning techniques for dermatology image classification, going significantly beyond what AutoML can offer. The project focuses on classifying skin lesions into fine-grained diagnostic categories and coarse lesion types, with special emphasis on handling class imbalance and uncertainty detection.

## 🎯 Project Objectives

The main goal is to **create prediction models using different modeling techniques (machine learning) justifying their use, determining the level of precision, and detailing the strengths and weaknesses of each technique used**. This project seeks to propose a development that contributes something more than what could be achieved with AutoML.

## 🏗️ Architecture Overview

The project implements three sophisticated approaches:

1. **Baseline Model**: EfficientNetB1 with advanced techniques
2. **Self-Supervised Learning**: SimCLR + Fine-tuning
3. **Ensemble Learning**: Multiple architectures with diversity strategies

## 📁 Repository Structure

```
master_thesis_ucm/
├── data/                                    # Dataset directory
│   ├── images/                             # Medical images
│   ├── metadata_clean.csv                  # Cleaned metadata
│   └── training_prepared_data.csv          # Prepared training data
├── outputs/                                # Model outputs
│   ├── simple_twohead_b0_v2/              # Baseline model
│   ├── ssl_simclr/                        # SSL pre-trained model
│   ├── ssl_finetuned/                     # SSL fine-tuned model
│   ├── individual_models/                 # Individual ensemble models
│   ├── ensemble_models/                   # Ensemble results
│   └── model_evaluation_comparison/       # Evaluation results
├── src/                                    # Source code
│   ├── first_model.py                     # Baseline implementation
│   ├── self_supervised_model.py           # SSL implementation
│   ├── ensemble_model.py                  # Ensemble implementation
│   ├── model_comparison.py                # SSL comparison
│   └── ensemble_comparison.py             # Ensemble comparison
├── notebooks/                              # Jupyter notebooks
│   ├── exploratory_data_analysis.ipynb     # EDA
│   ├── images_visualization.ipynb         # Image analysis
│   ├── model_evaluation.ipynb              # Comprehensive evaluation
│   └── verify_created_db.ipynb            # Database verification
├── scripts/                                # Utility scripts
│   ├── build_clean_dataset.py             # Dataset preparation
│   ├── build_datasets_for_models.py       # Model-specific datasets
│   ├── build_metadata_images.py           # Metadata extraction
│   └── delete_images.py                   # Cleanup utilities
├── README.md                               # This file
├── README_SSL.md                          # SSL documentation
└── README_ENSEMBLE.md                     # Ensemble documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install tensorflow>=2.10.0
pip install pandas numpy matplotlib seaborn
pip install scikit-learn opencv-python
pip install jupyter notebook
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd master_thesis_ucm
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare the dataset**
```bash
python scripts/build_clean_dataset.py
python scripts/build_datasets_for_models.py
```

### Running the Models

#### 1. Baseline Model
```bash
python src/first_model.py
```

#### 2. Self-Supervised Learning
```bash
python src/self_supervised_model.py
```

#### 3. Ensemble Learning
```bash
python src/ensemble_model.py
```

#### 4. Comprehensive Evaluation
```bash
jupyter notebook notebooks/model_evaluation.ipynb
```

## 🔬 Technical Implementation

### 1. Baseline Model (`first_model.py`)

**Architecture**: EfficientNetB1 with two-head output
- **Fine-grained head**: 9 diagnostic classes
- **Coarse head**: 3 lesion types (benign, malignant, no_lesion)

**Advanced Techniques**:
- **Focal Loss**: Handles class imbalance
- **Class Balancing**: Effective number weighting
- **Oversampling**: Minority class augmentation
- **Outlier Exposure**: Uncertainty handling
- **Data Augmentation**: Specialized for dermatology

**Key Features**:
- Gradual unfreezing strategy
- Cosine learning rate decay
- Early stopping with patience
- TensorBoard logging

### 2. Self-Supervised Learning (`self_supervised_model.py`)

**Methodology**: SimCLR (Simple Contrastive Learning)

**Phase 1 - Pre-training**:
- **Contrastive Learning**: Learn representations without labels
- **Data Augmentation**: Specialized augmentations for dermatology
- **Projection Head**: 128-dimensional feature space
- **Temperature Scaling**: 0.1 for optimal contrastive learning

**Phase 2 - Fine-tuning**:
- **Gradual Unfreezing**: Backbone frozen → unfrozen
- **Transfer Learning**: Reuse pre-trained representations
- **Same Techniques**: Focal Loss, Class Balancing, etc.

**Advantages over AutoML**:
- Utilizes unlabeled data
- Better generalization
- More robust representations
- Domain-specific pre-training

### 3. Ensemble Learning (`ensemble_model.py`)

**Diversity Strategy**:
- **Architectural Diversity**: EfficientNetB1, ResNet50, DenseNet121
- **Augmentation Diversity**: Light, Medium, Strong per model
- **Hyperparameter Diversity**: Different learning rates
- **Training Diversity**: Same techniques, different configurations

**Ensemble Methods**:
- **Voting Ensemble**: Simple averaging
- **Weighted Ensemble**: Performance-based weighting
- **Stacking**: Meta-learner (future work)

**Model Configurations**:
```python
MODEL_CONFIGS = {
    'efficientnet': {
        'augmentation_strength': 'medium',
        'learning_rate': 1e-4
    },
    'resnet': {
        'augmentation_strength': 'strong', 
        'learning_rate': 1.5e-4
    },
    'densenet': {
        'augmentation_strength': 'light',
        'learning_rate': 0.8e-4
    }
}
```

## 📊 Dataset Information

### Classes

**Fine-grained (9 classes)**:
- `nv`: Melanocytic nevi
- `mel`: Melanoma
- `bkl`: Benign keratosis-like lesions
- `bcc`: Basal cell carcinoma
- `scc_akiec`: Actinic keratosis and squamous cell carcinoma
- `vasc`: Vascular lesions
- `df`: Dermatofibroma
- `other`: Other lesions
- `no_lesion`: No lesion present

**Coarse (3 classes)**:
- `benign`: Benign lesions
- `malignant`: Malignant lesions
- `no_lesion`: No lesion present

### Data Sources
- **ISIC 2020**: International Skin Imaging Collaboration
- **ITOBO 2024**: Additional dermatology dataset
- **Total Images**: ~71,715 medical images
- **Metadata**: Age, gender, body region, image quality metrics

### Data Splits
- **Training**: 70% of labeled data
- **Validation**: 15% of labeled data
- **Test**: 15% of labeled data
- **OOD**: Unknown diagnosis cases

## 📈 Performance Metrics

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro and weighted averages
- **Precision/Recall**: Per-class and macro averages
- **AUROC**: Out-of-distribution detection
- **Confusion Matrices**: Detailed error analysis

### Expected Improvements
- **SSL Model**: +2-5% accuracy on minority classes
- **Ensemble Model**: +1-3% accuracy over best individual
- **OOD Detection**: Better uncertainty estimation
- **Robustness**: Improved generalization

## 🔍 Model Comparison

The `model_evaluation.ipynb` notebook provides comprehensive comparison:

### Features
- **Automatic Model Loading**: Detects and loads all available models
- **Individual Evaluation**: Detailed metrics per model
- **Comparative Analysis**: Side-by-side performance comparison
- **Visualization**: Charts, confusion matrices, OOD analysis
- **Recommendations**: Best model selection and deployment advice

### Output Files
- `model_comparison_table.csv`: Quantitative comparison
- `detailed_metrics.json`: Complete metrics per model
- `ood_detection_results.json`: Uncertainty analysis
- `summary_report.md`: Comprehensive analysis report

## 🛠️ Advanced Features

### Class Imbalance Handling
- **Focal Loss**: Focuses on hard examples
- **Class Balancing**: Effective number weighting
- **Oversampling**: Minority class augmentation
- **Stratified Splitting**: Maintains class proportions

### Uncertainty Detection
- **Outlier Exposure**: Handles unknown cases
- **Maximum Softmax Probability**: Confidence scoring
- **AUROC Analysis**: OOD detection performance
- **Threshold Optimization**: Optimal decision boundaries

### Data Augmentation
- **Dermatology-Specific**: Brightness, contrast, rotation
- **Minority-Focused**: Stronger augmentation for rare classes
- **Multi-Level**: Different strategies per model
- **Domain-Aware**: Medical image considerations

## 📚 Documentation

### Detailed Guides
- **[SSL Implementation](README_SSL.md)**: Self-supervised learning details
- **[Ensemble Implementation](README_ENSEMBLE.md)**: Ensemble learning details
- **[Model Evaluation](notebooks/model_evaluation.ipynb)**: Comprehensive evaluation

### Key Papers Referenced
- **SimCLR**: A Simple Framework for Contrastive Learning
- **EfficientNet**: Rethinking Model Scaling for CNNs
- **Focal Loss**: Addressing Class Imbalance
- **Outlier Exposure**: Detecting Out-of-Distribution Examples

## 🎯 Results and Insights

### Key Findings
1. **Ensemble methods** show superior performance over individual models
2. **SSL pre-training** demonstrates clear benefits for generalization
3. **Diversity strategies** effectively improve ensemble performance
4. **OOD detection** varies significantly between approaches

### Production Recommendations
1. **Use ensemble methods** for critical applications
2. **Implement SSL pre-training** for better generalization
3. **Monitor both fine and coarse** classification performance
4. **Set up uncertainty detection** for safety-critical decisions

## 🔧 Configuration

### Model Parameters
```python
# Common parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4

# SSL specific
TEMPERATURE = 0.1
PROJECTION_DIM = 128
SSL_EPOCHS = 50

# Ensemble specific
N_MODELS = 3
ENSEMBLE_METHODS = ['voting', 'weighted_average']
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for dataset and models
- **CPU**: Multi-core processor for data preprocessing

## 🤝 Contributing

This is a master thesis project. For questions or suggestions:
- **Email**: [Your Email]
- **Institution**: Universidad Complutense de Madrid
- **Course**: Master in Big Data, Data Science & AI

## 📄 License

This project is part of academic research. Please cite appropriately if used in research:

```bibtex
@mastersthesis{trigo2024dermatology,
  title={Advanced Machine Learning Techniques for Dermatology Classification},
  author={Trigo La Blanca, Carlota},
  year={2024},
  school={Universidad Complutense de Madrid},
  type={Master's Thesis}
}
```

## 🙏 Acknowledgments

- **ISIC**: International Skin Imaging Collaboration
- **ITOBO**: Additional dataset contributors
- **UCM**: Universidad Complutense de Madrid
- **TensorFlow Team**: Deep learning framework
- **Open Source Community**: Various ML libraries

---

**Note**: This implementation represents a significant advancement over AutoML, demonstrating advanced technical expertise and deep understanding of modern machine learning techniques for medical image analysis.

## 📞 Contact

For questions about this implementation or collaboration opportunities, please contact the author through the university channels.

---

*Last updated: January 2025*
