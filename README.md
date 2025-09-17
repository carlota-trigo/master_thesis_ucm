# Skin Lesion Classification with Self-Supervised Learning

<div align="center">
  <img src="deployment/static/logo.svg" alt="Skin Lesion Classification API Logo" width="200"/>
</div>

## Overview

This repository holds all the data required for the development of my Master Thesis in Big Data, Data Science and AI. 

## 🚀 Key Features

- **Data base creation**: I developed mny own database by combining the data from HAM10000 database, ISIC2019 and ISIC2020, MILK10 and ITOBOS2024.
- **Exploratory Data Analysis**: I did a thorough analysis on the quality of the images.
- **Model training**: Scripts to train 3 types of models -- baseline, SSL, and ensemble. 
- **Model Evaluation**: Codes to compare the performance of the models and decide the best functioning one. 
- **API Deployment**: The best model selected was deployed into a fucntional API.

## 📊 Model Architecture

### Primary Models
- **EfficientNet-B0**: Baseline model 
- **SSL Fine-tuned Model**: Classification model using self-supervised pretraining
- **Ensemble model**: Combines the baseline, with ResNet50 and DenseNet121 using voting. 

### Classification Tasks
- **Coarse Classification**: `benign`, `malignant`, `no_lesion`
- **Fine Classification**: `nv`, `mel`, `bkl`, `bcc`, `scc_akiec`, `vasc`, `df`, `other`, `no_lesion`

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
│   ├── metadata.csv   # Main training dataset
│   ├── metadata.csv          # Image metadata
│   ├── metadata.csv    # Trial dataset
├── repo_folder/               # Code directory
│   ├── deployment/                  # Web API and deployment files
│   │   ├── app.py                   # Flask application
│   │   ├── requirements.txt         # API dependencies
│   │   ├── templates/               # HTML templates
│   │   │   └── index.html           # Web interface
│   │   ├── static/                  # Static assets
│   │   │   └── logo.svg             # API logo
│   │   └── uploads/                 # Temporary upload directory
│   ├── scripts                      # All the scripts
└── README.md                        # This file
```

The database is published in Kaggle
---

<div align="center">
  <p><em>Built with ❤️ for advancing medical AI</em></p>
</div>
