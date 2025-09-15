# comprehensive_model_evaluation.py
# Comprehensive evaluation script implementing the 6 key improvements
# -*- coding: utf-8 -*-

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import utils
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths
BASELINE_DIR = Path("./outputs/base_model")
SSL_DIR = Path("./outputs/ssl_finetuned")
INDIVIDUAL_DIR = Path("./outputs/individual_models")

MODELS_CONFIG = {
    'baseline': {
        'name': 'Baseline EfficientNetB1',
        'path': BASELINE_DIR / "simple_twohead_best_model.keras",
        'type': 'single'
    },
    'ssl': {
        'name': 'SSL Fine-tuned',
        'path': SSL_DIR / "ssl_finetuned_best_model.keras",
        'type': 'single'
    },
    'ensemble_voting': {
        'name': 'Voting Ensemble',
        'path': INDIVIDUAL_DIR,
        'type': 'ensemble'
    },
    'ensemble_weighted': {
        'name': 'Weighted Ensemble',
        'path': INDIVIDUAL_DIR,
        'type': 'ensemble'
    }
}

# Class mappings (verified from data analysis)
COARSE_CLASSES = ['benign', 'malignant', 'no_lesion']
FINE_CLASSES = ['nv', 'mel', 'bkl', 'bcc', 'scc_akiec', 'vasc', 'df', 'other', 'no_lesion']

# Clinical thresholds
MALIGNANT_RECALL_THRESHOLD = 0.95  # ≥95% recall for malignant
OOD_FPR_THRESHOLD = 0.05  # FPR@95%TPR

# Bootstrap parameters
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95

# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_models():
    """Load all available models."""
    models = {}
    ensemble_models = {}
    
    # Load individual models
    for model_key, config in MODELS_CONFIG.items():
        if config['type'] == 'single':
            print(f"Loading {config['name']}...")
            model = utils.load_individual_model(config['path'], 'efficientnet')
            if model is not None:
                models[model_key] = model
                print(f"✓ {config['name']} loaded")
            else:
                print(f"✗ Failed to load {config['name']}")
    
    # Load ensemble models
    ensemble_models = utils.load_ensemble_models(INDIVIDUAL_DIR)
    
    return models, ensemble_models

def get_predictions_all_models(models, ensemble_models, test_ds, ood_ds):
    """Get predictions for all models."""
    all_predictions = {}
    
    # Individual models
    for model_key, model in models.items():
        print(f"Evaluating {MODELS_CONFIG[model_key]['name']}...")
        try:
            id_labels_h1, id_logits_h1, id_labels_h2, id_logits_h2 = utils.get_predictions_and_labels(model, test_ds)
            ood_labels_h1, ood_logits_h1, ood_labels_h2, ood_logits_h2 = utils.get_predictions_and_labels(model, ood_ds)
            
            all_predictions[model_key] = {
                'id_labels_h1': id_labels_h1, 'id_logits_h1': id_logits_h1,
                'id_labels_h2': id_labels_h2, 'id_logits_h2': id_logits_h2,
                'ood_labels_h1': ood_labels_h1, 'ood_logits_h1': ood_logits_h1,
                'ood_labels_h2': ood_labels_h2, 'ood_logits_h2': ood_logits_h2
            }
            print(f"✓ {MODELS_CONFIG[model_key]['name']} evaluated")
        except Exception as e:
            print(f"✗ Failed to evaluate {MODELS_CONFIG[model_key]['name']}: {e}")
    
    # Ensemble models
    if len(ensemble_models) > 0:
        print("Evaluating ensemble models...")
        try:
            # Voting ensemble
            id_labels_h1, id_logits_h1, id_labels_h2, id_logits_h2 = utils.get_ensemble_predictions(ensemble_models, test_ds, 'voting')
            ood_labels_h1, ood_logits_h1, ood_labels_h2, ood_logits_h2 = utils.get_ensemble_predictions(ensemble_models, ood_ds, 'voting')
            
            all_predictions['ensemble_voting'] = {
                'id_labels_h1': id_labels_h1, 'id_logits_h1': id_logits_h1,
                'id_labels_h2': id_labels_h2, 'id_logits_h2': id_logits_h2,
                'ood_labels_h1': ood_labels_h1, 'ood_logits_h1': ood_logits_h1,
                'ood_labels_h2': ood_labels_h2, 'ood_logits_h2': ood_logits_h2
            }
            
            # Weighted ensemble
            id_labels_h1, id_logits_h1, id_labels_h2, id_logits_h2 = utils.get_ensemble_predictions(ensemble_models, test_ds, 'weighted')
            ood_labels_h1, ood_logits_h1, ood_labels_h2, ood_logits_h2 = utils.get_ensemble_predictions(ensemble_models, ood_ds, 'weighted')
            
            all_predictions['ensemble_weighted'] = {
                'id_labels_h1': id_labels_h1, 'id_logits_h1': id_logits_h1,
                'id_labels_h2': id_labels_h2, 'id_logits_h2': id_logits_h2,
                'ood_labels_h1': ood_labels_h1, 'ood_logits_h1': ood_logits_h1,
                'ood_labels_h2': ood_labels_h2, 'ood_logits_h2': ood_logits_h2
            }
            
            print("✓ Ensemble models evaluated")
        except Exception as e:
            print(f"✗ Failed to evaluate ensemble models: {e}")
    
    return all_predictions

def calculate_comprehensive_metrics(labels, logits, class_names, task_name):
    """Calculate comprehensive metrics for a classification task."""
    preds = np.argmax(logits, axis=1)
    
    # 1. Primary metrics (multiclass, imbalanced)
    macro_f1 = utils.calculate_metrics(labels, preds, class_names)['f1']
    balanced_acc = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    
    # 2. Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        if i in labels:  # Check if class exists in labels
            class_labels = (labels == i).astype(int)
            class_preds = (preds == i).astype(int)
            
            if len(np.unique(class_labels)) > 1:  # Check if class has both positive and negative samples
                precision, recall, f1, _ = precision_recall_fscore_support(
                    class_labels, class_preds, average='binary', zero_division=0
                )
                auroc = roc_auc_score(class_labels, logits[:, i])
                auprc = average_precision_score(class_labels, logits[:, i])
                
                per_class_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auroc': auroc,
                    'auprc': auprc,
                    'support': np.sum(class_labels)
                }
    
    # 3. AUPRC macro (average across classes)
    auprc_scores = [metrics['auprc'] for metrics in per_class_metrics.values()]
    auprc_macro = np.mean(auprc_scores) if auprc_scores else 0.0
    
    return {
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc,
        'mcc': mcc,
        'auprc_macro': auprc_macro,
        'per_class': per_class_metrics
    }

def calculate_hierarchical_metrics(predictions, coarse_classes, fine_classes):
    """Calculate hierarchical evaluation metrics."""
    coarse_labels = predictions['id_labels_h1']
    coarse_logits = predictions['id_logits_h1']
    fine_labels = predictions['id_labels_h2']
    fine_logits = predictions['id_logits_h2']
    
    coarse_preds = np.argmax(coarse_logits, axis=1)
    fine_preds = np.argmax(fine_logits, axis=1)
    
    # 1. Exact-match: both heads correct
    exact_match = np.mean((coarse_preds == coarse_labels) & (fine_preds == fine_labels))
    
    # 2. Coarse-correct: head1 correct (independent of head2)
    coarse_correct = np.mean(coarse_preds == coarse_labels)
    
    # 3. Fine conditional: head2 macro-F1 conditioned on head1 being correct
    coarse_correct_mask = (coarse_preds == coarse_labels)
    if np.sum(coarse_correct_mask) > 0:
        fine_conditional_labels = fine_labels[coarse_correct_mask]
        fine_conditional_preds = fine_preds[coarse_correct_mask]
        fine_conditional_metrics = utils.calculate_metrics(fine_conditional_labels, fine_conditional_preds, fine_classes)
        fine_conditional_f1 = fine_conditional_metrics['f1']
    else:
        fine_conditional_f1 = 0.0
    
    return {
        'exact_match': exact_match,
        'coarse_correct': coarse_correct,
        'fine_conditional_f1': fine_conditional_f1
    }

def calculate_calibration_metrics(labels, logits, class_names):
    """Calculate calibration metrics."""
    # Convert to probabilities
    probs = tf.nn.softmax(logits).numpy()
    
    # Expected Calibration Error (ECE)
    ece_scores = []
    brier_scores = []
    
    for i, class_name in enumerate(class_names):
        if i in labels:
            class_labels = (labels == i).astype(int)
            class_probs = probs[:, i]
            
            if len(np.unique(class_labels)) > 1:
                # ECE calculation
                n_bins = 10
                bin_boundaries = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (class_probs > bin_lower) & (class_probs <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    
                    if prop_in_bin > 0:
                        accuracy_in_bin = class_labels[in_bin].mean()
                        avg_confidence_in_bin = class_probs[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                ece_scores.append(ece)
                
                # Brier score
                brier = np.mean((class_probs - class_labels) ** 2)
                brier_scores.append(brier)
    
    return {
        'ece_mean': np.mean(ece_scores) if ece_scores else 0.0,
        'ece_std': np.std(ece_scores) if ece_scores else 0.0,
        'brier_mean': np.mean(brier_scores) if brier_scores else 0.0,
        'brier_std': np.std(brier_scores) if brier_scores else 0.0
    }

def calculate_ood_metrics(id_logits, ood_logits):
    """Calculate out-of-distribution detection metrics."""
    # Maximum Softmax Probability (MSP)
    id_probs = tf.nn.softmax(id_logits).numpy()
    ood_probs = tf.nn.softmax(ood_logits).numpy()
    
    id_msp = np.max(id_probs, axis=1)
    ood_msp = np.max(ood_probs, axis=1)
    
    # AUROC and AUPRC
    labels_id = np.ones_like(id_msp)
    labels_ood = np.zeros_like(ood_msp)
    all_scores = np.concatenate([id_msp, ood_msp])
    all_labels = np.concatenate([labels_id, labels_ood])
    
    auroc = roc_auc_score(all_labels, all_scores)
    auprc = average_precision_score(all_labels, all_scores)
    
    # FPR@95%TPR
    fpr, tpr, thresholds = precision_recall_curve(all_labels, all_scores)
    tpr_95_idx = np.where(tpr >= 0.95)[0]
    if len(tpr_95_idx) > 0:
        fpr_at_95_tpr = fpr[tpr_95_idx[0]]
    else:
        fpr_at_95_tpr = 1.0
    
    # Detection Error (minimum of FNR + FPR)
    fnr = 1 - tpr
    detection_error = np.min(fnr + fpr)
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'fpr_at_95_tpr': fpr_at_95_tpr,
        'detection_error': detection_error
    }

def bootstrap_confidence_intervals(metric_func, data, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL):
    """Calculate bootstrap confidence intervals for a metric."""
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Stratified bootstrap by class
        bootstrap_indices = []
        for class_id in np.unique(data['labels']):
            class_indices = np.where(data['labels'] == class_id)[0]
            bootstrap_class_indices = np.random.choice(class_indices, size=len(class_indices), replace=True)
            bootstrap_indices.extend(bootstrap_class_indices)
        
        bootstrap_indices = np.array(bootstrap_indices)
        bootstrap_data = {key: data[key][bootstrap_indices] for key in data.keys()}
        
        try:
            score = metric_func(bootstrap_data)
            bootstrap_scores.append(score)
        except:
            continue
    
    if len(bootstrap_scores) == 0:
        return 0.0, 0.0, 0.0
    
    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)
    mean_score = np.mean(bootstrap_scores)
    
    return mean_score, ci_lower, ci_upper

def main():
    """Main evaluation function."""
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("Implementing 6 key improvements for imbalanced medical datasets")
    print("="*80)
    
    # Load data
    df = pd.read_csv(utils.PREPARED_CSV)
    test_df = df[df.split == "test"].copy()
    ood_df = df[df.split == "test_ood"].copy()
    
    print(f"Test samples: {len(test_df)}")
    print(f"OOD samples: {len(ood_df)}")
    
    # Build datasets
    test_ds = utils.build_dataset(test_df, is_training=False)
    ood_ds = utils.build_dataset(ood_df, is_training=False)
    
    # Load models
    models, ensemble_models = load_models()
    
    if len(models) == 0 and len(ensemble_models) == 0:
        print("⚠️ No models available for evaluation!")
        return
    
    # Get predictions
    all_predictions = get_predictions_all_models(models, ensemble_models, test_ds, ood_ds)
    
    if len(all_predictions) == 0:
        print("⚠️ No predictions available for evaluation!")
        return
    
    print(f"\n📊 Evaluating {len(all_predictions)} models...")
    
    # Comprehensive evaluation
    results = {}
    
    for model_key, predictions in all_predictions.items():
        model_name = MODELS_CONFIG[model_key]['name']
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        # 1. Primary metrics (multiclass, imbalanced)
        coarse_metrics = calculate_comprehensive_metrics(
            predictions['id_labels_h1'], predictions['id_logits_h1'], 
            COARSE_CLASSES, "coarse"
        )
        
        fine_metrics = calculate_comprehensive_metrics(
            predictions['id_labels_h2'], predictions['id_logits_h2'], 
            FINE_CLASSES, "fine"
        )
        
        # 2. Hierarchical evaluation
        hierarchical_metrics = calculate_hierarchical_metrics(
            predictions, COARSE_CLASSES, FINE_CLASSES
        )
        
        # 3. Calibration metrics
        coarse_calibration = calculate_calibration_metrics(
            predictions['id_labels_h1'], predictions['id_logits_h1'], COARSE_CLASSES
        )
        
        fine_calibration = calculate_calibration_metrics(
            predictions['id_labels_h2'], predictions['id_logits_h2'], FINE_CLASSES
        )
        
        # 4. OOD detection metrics
        ood_metrics = calculate_ood_metrics(
            predictions['id_logits_h1'], predictions['ood_logits_h1']
        )
        
        # Store results
        results[model_key] = {
            'model_name': model_name,
            'coarse_metrics': coarse_metrics,
            'fine_metrics': fine_metrics,
            'hierarchical_metrics': hierarchical_metrics,
            'coarse_calibration': coarse_calibration,
            'fine_calibration': fine_calibration,
            'ood_metrics': ood_metrics
        }
        
        # Print summary
        print(f"Coarse Macro-F1: {coarse_metrics['macro_f1']:.4f}")
        print(f"Coarse Balanced Acc: {coarse_metrics['balanced_accuracy']:.4f}")
        print(f"Coarse MCC: {coarse_metrics['mcc']:.4f}")
        print(f"Fine Macro-F1: {fine_metrics['macro_f1']:.4f}")
        print(f"Fine AUPRC Macro: {fine_metrics['auprc_macro']:.4f}")
        print(f"Exact Match: {hierarchical_metrics['exact_match']:.4f}")
        print(f"OOD AUROC: {ood_metrics['auroc']:.4f}")
        print(f"ECE (Coarse): {coarse_calibration['ece_mean']:.4f}")
        print(f"ECE (Fine): {fine_calibration['ece_mean']:.4f}")
    
    # Create comprehensive report
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION COMPLETE")
    print(f"{'='*80}")
    
    # Find best models
    best_coarse_f1 = max(results.items(), key=lambda x: x[1]['coarse_metrics']['macro_f1'])
    best_fine_f1 = max(results.items(), key=lambda x: x[1]['fine_metrics']['macro_f1'])
    best_ood = max(results.items(), key=lambda x: x[1]['ood_metrics']['auroc'])
    best_calibration = min(results.items(), key=lambda x: x[1]['fine_calibration']['ece_mean'])
    
    print(f"\n🏆 BEST PERFORMING MODELS:")
    print(f"Best Coarse Macro-F1: {best_coarse_f1[1]['model_name']} ({best_coarse_f1[1]['coarse_metrics']['macro_f1']:.4f})")
    print(f"Best Fine Macro-F1: {best_fine_f1[1]['model_name']} ({best_fine_f1[1]['fine_metrics']['macro_f1']:.4f})")
    print(f"Best OOD Detection: {best_ood[1]['model_name']} ({best_ood[1]['ood_metrics']['auroc']:.4f})")
    print(f"Best Calibration: {best_calibration[1]['model_name']} ({best_calibration[1]['fine_calibration']['ece_mean']:.4f})")
    
    # Save results
    output_dir = Path("./outputs/comprehensive_evaluation")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save detailed results
    import json
    with open(output_dir / "detailed_results.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            json_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    json_results[key][subkey] = {}
                    for subsubkey, subsubvalue in subvalue.items():
                        if isinstance(subsubvalue, np.ndarray):
                            json_results[key][subkey][subsubkey] = subsubvalue.tolist()
                        else:
                            json_results[key][subkey][subsubkey] = subsubvalue
                else:
                    json_results[key][subkey] = subvalue
        json.dump(json_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir}")
    print("✅ Comprehensive evaluation completed successfully!")

if __name__ == "__main__":
    main()
