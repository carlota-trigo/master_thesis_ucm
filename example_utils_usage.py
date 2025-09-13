#!/usr/bin/env python3
"""
Example script showing how to use utils.py functions
This demonstrates how to refactor existing scripts to use the consolidated utilities
"""

import utils
import pandas as pd
import numpy as np
from pathlib import Path

def example_model_evaluation():
    """Example of how to use utils for model evaluation"""
    print("Example: Model Evaluation using utils.py")
    print("=" * 50)
    
    # Load data using utils
    df = pd.read_csv(utils.PREPARED_CSV)
    test_df = df[df.split == "test"].copy()
    
    print(f"Loaded {len(test_df)} test samples")
    
    # Build dataset using utils
    test_ds = utils.build_dataset(test_df, is_training=False)
    print(f"Built dataset with batch size {utils.BATCH_SIZE}")
    
    # Create model using utils
    model = utils.create_two_head_model('efficientnet')
    print(f"Created model: {model.name}")
    
    # Get predictions using utils
    labels_h1, logits_h1, labels_h2, logits_h2 = utils.get_predictions_and_labels(model, test_ds)
    print(f"Got predictions: {logits_h1.shape}, {logits_h2.shape}")
    
    # Calculate metrics using utils
    preds_h1 = np.argmax(logits_h1, axis=1)
    preds_h2 = np.argmax(logits_h2, axis=1)
    
    fine_metrics = utils.calculate_metrics(labels_h1, preds_h1, utils.DX_CLASSES)
    coarse_metrics = utils.calculate_metrics(labels_h2, preds_h2, utils.LESION_TYPE_CLASSES)
    
    print(f"Fine-grained accuracy: {fine_metrics['accuracy']:.4f}")
    print(f"Coarse accuracy: {coarse_metrics['accuracy']:.4f}")

def example_ensemble_creation():
    """Example of how to use utils for ensemble methods"""
    print("\nExample: Ensemble Creation using utils.py")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv(utils.PREPARED_CSV)
    test_df = df[df.split == "test"].copy()
    test_ds = utils.build_dataset(test_df, is_training=False)
    
    # Create multiple models
    models = {}
    backbone_types = ['efficientnet', 'resnet', 'densenet']
    
    for backbone_type in backbone_types:
        model = utils.create_two_head_model(backbone_type)
        models[backbone_type] = model
        print(f"Created {backbone_type} model")
    
    # Create voting ensemble
    fine_preds, coarse_preds = utils.create_voting_ensemble(models, test_ds)
    print(f"Created voting ensemble predictions: {fine_preds.shape}")
    
    # Create weighted ensemble
    weights = [0.5, 0.3, 0.2]  # Different weights for each model
    fine_preds_weighted, coarse_preds_weighted = utils.create_weighted_ensemble(models, test_ds, weights)
    print(f"Created weighted ensemble predictions: {fine_preds_weighted.shape}")

def example_data_processing():
    """Example of how to use utils for data processing"""
    print("\nExample: Data Processing using utils.py")
    print("=" * 50)
    
    # Load raw data
    df = pd.read_csv(utils.PREPARED_CSV)
    print(f"Loaded {len(df)} samples")
    
    # Process labels using utils
    processed_df = utils.process_labels(df)
    print(f"Processed labels, {len(processed_df)} samples remaining")
    
    # Apply fine oversampling
    minority_fine_ids = {utils.DX_TO_ID['df'], utils.DX_TO_ID['vasc']}
    oversampled_df = utils.apply_fine_oversampling(processed_df, utils.FINE_MINORITY_OVERSAMPLING)
    print(f"Applied oversampling, {len(oversampled_df)} samples")
    
    # Build training dataset
    train_ds = utils.build_dataset(
        oversampled_df, 
        is_training=True, 
        minority_fine_ids=minority_fine_ids,
        fine_oversampling=utils.FINE_MINORITY_OVERSAMPLING
    )
    print(f"Built training dataset")

def example_model_training():
    """Example of how to use utils for model training"""
    print("\nExample: Model Training using utils.py")
    print("=" * 50)
    
    # Load and process data
    df = pd.read_csv(utils.PREPARED_CSV)
    processed_df = utils.process_labels(df)
    
    # Split data
    train_df = processed_df[processed_df.split == "train"].copy()
    val_df = processed_df[processed_df.split == "val"].copy()
    
    # Calculate class weights
    coarse_counts = utils.counts_from_labels(train_df["head1_idx"], utils.N_LESION_TYPE_CLASSES)
    coarse_weights = utils.class_balanced_weights(coarse_counts)
    coarse_alpha = utils.calculate_focal_alpha(coarse_counts)
    
    print(f"Calculated class weights: {coarse_weights.shape}")
    
    # Create model
    model = utils.create_two_head_model('efficientnet')
    
    # Create loss functions
    coarse_loss = utils.sparse_categorical_focal_loss(gamma=utils.FOCAL_GAMMA, alpha=coarse_alpha)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss={
            "coarse_output": coarse_loss,
            "fine_output": utils.masked_sparse_ce_with_oe,
        },
        metrics={
            "coarse_output": ["sparse_categorical_accuracy"],
            "fine_output": ["sparse_categorical_accuracy"],
        },
    )
    
    print(f"Compiled model with focal loss")
    
    # Create callbacks
    output_dir = Path("outputs/example_model")
    callbacks = utils.create_callbacks(output_dir, 'efficientnet')
    print(f"Created {len(callbacks)} callbacks")

def main():
    """Run all examples"""
    print("Utils.py Usage Examples")
    print("=" * 60)
    
    try:
        example_model_evaluation()
        example_ensemble_creation()
        example_data_processing()
        example_model_training()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("You can now refactor your existing scripts to use these utilities.")
        
    except Exception as e:
        print(f"❌ Error in examples: {e}")
        print("This is expected if data files are not available.")

if __name__ == "__main__":
    main()
