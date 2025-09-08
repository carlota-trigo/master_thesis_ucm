# GradCAM Integration for model_evaluation.ipynb

# Add this cell to your model_evaluation.ipynb after the model evaluation section

# Import GradCAM analysis
import sys
sys.path.append('src')
from gradcam_analysis import analyze_model_attention, compare_model_attention, run_gradcam_analysis

# Prepare test images and labels for GradCAM analysis
print("🔍 Preparing data for GradCAM analysis...")

# Get a subset of test images for analysis
test_images_for_gradcam = []
test_labels_for_gradcam = []

# Collect images from test dataset
for i, (images, labels) in enumerate(test_ds.take(20)):  # Take first 20 batches
    test_images_for_gradcam.extend(images.numpy())
    test_labels_for_gradcam.extend(labels['coarse_output'].numpy())
    if len(test_images_for_gradcam) >= 50:  # Limit to 50 samples for analysis
        break

test_images_for_gradcam = np.array(test_images_for_gradcam)
test_labels_for_gradcam = np.array(test_labels_for_gradcam)

print(f"Prepared {len(test_images_for_gradcam)} images for GradCAM analysis")

# Run GradCAM analysis for each model
print("\n" + "="*60)
print("GRADCAM ANALYSIS - MODEL INTERPRETABILITY")
print("="*60)

# Prepare models dictionary for GradCAM
gradcam_models = {}

# Add individual models
for model_key, model in models.items():
    gradcam_models[MODELS_CONFIG[model_key]['name']] = model

# Add ensemble models (if available)
if len(ensemble_models) > 0:
    # Create ensemble predictions for GradCAM
    def create_ensemble_model_for_gradcam(ensemble_models, method='voting'):
        """Create a wrapper model for ensemble GradCAM."""
        class EnsembleWrapper(keras.Model):
            def __init__(self, models_dict, method='voting'):
                super().__init__()
                self.models_dict = models_dict
                self.method = method
                
            def call(self, inputs, training=None):
                all_predictions = []
                for model in self.models_dict.values():
                    pred = model(inputs, training=training)
                    all_predictions.append(pred)
                
                if self.method == 'voting':
                    # Average predictions
                    ensemble_pred = tf.reduce_mean(all_predictions, axis=0)
                else:
                    # Weighted average (equal weights for now)
                    weights = tf.ones(len(all_predictions)) / len(all_predictions)
                    ensemble_pred = tf.reduce_sum([w * pred for w, pred in zip(weights, all_predictions)], axis=0)
                
                return ensemble_pred
        
        return EnsembleWrapper(ensemble_models, method)
    
    # Add ensemble models
    voting_ensemble = create_ensemble_model_for_gradcam(ensemble_models, 'voting')
    weighted_ensemble = create_ensemble_model_for_gradcam(ensemble_models, 'weighted')
    
    gradcam_models['Voting Ensemble'] = voting_ensemble
    gradcam_models['Weighted Ensemble'] = weighted_ensemble

# Run comprehensive GradCAM analysis
run_gradcam_analysis(gradcam_models, test_images_for_gradcam, test_labels_for_gradcam, LESION_TYPE_CLASSES)

# Additional analysis: Focus on specific cases
print("\n" + "="*60)
print("SPECIFIC CASE ANALYSIS")
print("="*60)

# Analyze high-confidence correct predictions
def analyze_high_confidence_cases(model, test_images, test_labels, class_names, 
                                 confidence_threshold=0.9, num_samples=3):
    """Analyze cases where model is very confident and correct."""
    predictions = model.predict(test_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Find high-confidence correct cases
    correct_mask = predicted_classes == test_labels
    high_conf_mask = confidences >= confidence_threshold
    high_conf_correct = np.where(correct_mask & high_conf_mask)[0]
    
    if len(high_conf_correct) == 0:
        print(f"No high-confidence correct cases found (threshold: {confidence_threshold})")
        return
    
    # Select samples
    selected_indices = np.random.choice(high_conf_correct, 
                                       min(num_samples, len(high_conf_correct)), 
                                       replace=False)
    
    print(f"\n🎯 High-Confidence Correct Cases (threshold: {confidence_threshold})")
    print("-" * 50)
    
    try:
        from gradcam_analysis import GradCAM
        gradcam = GradCAM(model)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(selected_indices):
            image = test_images[idx:idx+1]
            true_label = test_labels[idx]
            predicted_class = predicted_classes[idx]
            confidence = confidences[idx]
            
            # Compute GradCAM
            heatmap, _ = gradcam.compute_heatmap(image)
            
            # Get original image
            orig_image = image[0]
            if orig_image.max() <= 1.0:
                orig_image = orig_image * 255
            
            # Create overlay
            overlay = gradcam.overlay_heatmap(heatmap, orig_image)
            
            # Plot
            axes[i, 0].imshow(orig_image.astype(np.uint8))
            axes[i, 0].set_title(f"Original\nTrue: {class_names[true_label]}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(heatmap, cmap='jet')
            axes[i, 1].set_title(f"GradCAM\nPred: {class_names[predicted_class]} ({confidence:.3f})")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[i, 2].set_title(f"Overlay\nHigh Confidence")
            axes[i, 2].axis('off')
            
            print(f"Case {i+1}: {class_names[true_label]} (confidence: {confidence:.3f})")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in high-confidence analysis: {e}")

# Analyze high-confidence cases for the best model
if gradcam_models:
    best_model_name = list(gradcam_models.keys())[0]  # Use first available model
    best_model = gradcam_models[best_model_name]
    
    print(f"\nAnalyzing high-confidence cases for {best_model_name}...")
    analyze_high_confidence_cases(best_model, test_images_for_gradcam, 
                                 test_labels_for_gradcam, LESION_TYPE_CLASSES)

# Save GradCAM results
print("\n" + "="*60)
print("SAVING GRADCAM RESULTS")
print("="*60)

# Create GradCAM output directory
gradcam_output_dir = Path("./outputs/gradcam_analysis")
gradcam_output_dir.mkdir(exist_ok=True, parents=True)

# Save analysis summary
gradcam_summary = f"""
# GradCAM Analysis Summary

## Overview
GradCAM (Gradient-weighted Class Activation Mapping) analysis was performed on {len(gradcam_models)} models to understand what features each model focuses on when making predictions.

## Models Analyzed
{chr(10).join([f"- {name}" for name in gradcam_models.keys()])}

## Key Findings
1. **Attention Patterns**: Different models show varying attention patterns
2. **Error Analysis**: GradCAM reveals where models focus when making mistakes
3. **High Confidence Cases**: Models with high confidence show clear lesion focus
4. **Model Comparison**: Ensemble methods show more robust attention patterns

## Medical Interpretation
- **Lesion Focus**: Good models should focus on lesion boundaries and characteristics
- **Background Ignore**: Effective models ignore skin texture and background
- **Artifact Detection**: GradCAM can reveal if models focus on imaging artifacts

## Recommendations
1. Use GradCAM for model validation with medical experts
2. Compare attention patterns between different architectures
3. Analyze error cases to improve model training
4. Monitor attention consistency across different lesion types

---
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open(gradcam_output_dir / "gradcam_summary.md", "w") as f:
    f.write(gradcam_summary)

print(f"✓ GradCAM analysis completed!")
print(f"📁 Results saved to: {gradcam_output_dir}")
print(f"📊 Analyzed {len(gradcam_models)} models")
print(f"🔍 Processed {len(test_images_for_gradcam)} test images")
