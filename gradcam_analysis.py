# src/gradcam_analysis.py
# -*- coding: utf-8 -*-
"""
Grad-CAM Implementation for Dermatology Model Interpretation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import pandas as pd

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for model interpretation.
    """
    
    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM.
        
        Args:
            model: Trained Keras model
            layer_name: Name of the last convolutional layer (if None, auto-detect)
        """
        self.model = model
        
        # Find the last convolutional layer
        if layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        else:
            self.layer_name = layer_name
            
        print(f"Using layer: {self.layer_name}")
        
        # Create a model that outputs the last conv layer and predictions
        self.grad_model = keras.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model."""
        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                conv_layers.append(layer.name)
        
        if conv_layers:
            return conv_layers[-1]
        else:
            raise ValueError("No convolutional layers found in the model")
    
    def compute_heatmap(self, image, class_idx=None, eps=1e-8):
        """
        Compute GradCAM heatmap for an image.
        
        Args:
            image: Input image (batch of 1)
            class_idx: Class index to generate heatmap for (if None, use predicted class)
            eps: Small value to avoid division by zero
            
        Returns:
            heatmap: GradCAM heatmap
            prediction: Model prediction
        """
        # Record operations for gradient computation
        with tf.GradientTape() as tape:
            # Get the last conv layer output and predictions
            conv_outputs, predictions = self.grad_model(image)
            
            # Use predicted class if class_idx not specified
            # Using coarse head (predictions[0]) for benign/malignant classification
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])  # Coarse head: benign/malignant/no_lesion
            
            # Get the score for the target class
            class_output = predictions[0, class_idx]
        
        # Compute gradients
        grads = tape.gradient(class_output, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        # Return coarse head predictions (benign/malignant/no_lesion)
        return heatmap.numpy(), predictions[0].numpy()
    
    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """
        Overlay heatmap on the original image.
        
        Args:
            heatmap: GradCAM heatmap
            image: Original image
            alpha: Transparency of heatmap
            
        Returns:
            overlayed_image: Image with heatmap overlay
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Normalize heatmap to 0-255
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Convert image to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_RGB2BGR)
        else:
            image_rgb = np.uint8(image * 255)
        
        # Overlay heatmap
        overlayed = cv2.addWeighted(image_rgb, 1-alpha, heatmap, alpha, 0)
        
        return overlayed

def analyze_model_attention(model, test_images, test_labels, class_names, 
                          model_name="Model", num_samples=5):
    """
    Analyze model attention using GradCAM for multiple samples.
    
    Args:
        model: Trained model
        test_images: Batch of test images
        test_labels: Corresponding labels
        class_names: List of class names
        model_name: Name of the model for display
        num_samples: Number of samples to analyze
    """
    print(f"\n🔍 GradCAM Analysis for {model_name}")
    print("=" * 50)
    
    # Initialize GradCAM
    try:
        gradcam = GradCAM(model)
    except Exception as e:
        print(f"Error initializing GradCAM: {e}")
        return
    
    # Select samples to analyze
    indices = np.random.choice(len(test_images), min(num_samples, len(test_images)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        image = test_images[idx:idx+1]  # Keep batch dimension
        true_label = test_labels[idx]
        
        # Compute GradCAM
        heatmap, predictions = gradcam.compute_heatmap(image)
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]
        
        # Get original image (remove batch dimension and normalize)
        orig_image = image[0]
        if orig_image.max() <= 1.0:
            orig_image = orig_image * 255
        
        # Create overlay
        overlay = gradcam.overlay_heatmap(heatmap, orig_image)
        
        # Plot original image
        axes[i, 0].imshow(orig_image.astype(np.uint8))
        axes[i, 0].set_title(f"Original\nTrue: {class_names[true_label]}")
        axes[i, 0].axis('off')
        
        # Plot heatmap
        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title(f"GradCAM Heatmap\nPred: {class_names[predicted_class]} ({confidence:.3f})")
        axes[i, 1].axis('off')
        
        # Plot overlay
        axes[i, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i, 2].set_title(f"Overlay\nAttention Focus")
        axes[i, 2].axis('off')
        
        # Print prediction details
        print(f"\nSample {i+1}:")
        print(f"  True Class: {class_names[true_label]}")
        print(f"  Predicted Class: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        print(f"  Correct: {'✓' if true_label == predicted_class else '✗'}")
    
    plt.tight_layout()
    plt.show()

def compare_model_attention(models_dict, test_images, test_labels, class_names, 
                          sample_idx=0):
    """
    Compare attention patterns between different models.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        test_images: Test images
        test_labels: Test labels
        class_names: List of class names
        sample_idx: Index of sample to analyze
    """
    print(f"\n🔍 Comparing Model Attention Patterns")
    print("=" * 50)
    
    num_models = len(models_dict)
    fig, axes = plt.subplots(2, num_models, figsize=(5*num_models, 10))
    
    if num_models == 1:
        axes = axes.reshape(2, 1)
    
    # Get the sample
    image = test_images[sample_idx:sample_idx+1]
    true_label = test_labels[sample_idx]
    
    for i, (model_name, model) in enumerate(models_dict.items()):
        try:
            gradcam = GradCAM(model)
            heatmap, predictions = gradcam.compute_heatmap(image)
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            
            # Get original image
            orig_image = image[0]
            if orig_image.max() <= 1.0:
                orig_image = orig_image * 255
            
            # Create overlay
            overlay = gradcam.overlay_heatmap(heatmap, orig_image)
            
            # Plot heatmap
            axes[0, i].imshow(heatmap, cmap='jet')
            axes[0, i].set_title(f"{model_name}\nPred: {class_names[predicted_class]} ({confidence:.3f})")
            axes[0, i].axis('off')
            
            # Plot overlay
            axes[1, i].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f"{model_name} Overlay")
            axes[1, i].axis('off')
            
            print(f"\n{model_name}:")
            print(f"  Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})")
            print(f"  Correct: {'✓' if true_label == predicted_class else '✗'}")
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            axes[0, i].text(0.5, 0.5, f"Error\n{model_name}", ha='center', va='center')
            axes[1, i].text(0.5, 0.5, f"Error\n{model_name}", ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

def analyze_error_cases(model, test_images, test_labels, predictions, class_names, 
                       num_errors=3):
    """
    Analyze cases where the model made errors using GradCAM.
    
    Args:
        model: Trained model
        test_images: Test images
        test_labels: True labels
        predictions: Model predictions
        class_names: List of class names
        num_errors: Number of error cases to analyze
    """
    print(f"\n❌ Analyzing Error Cases with GradCAM")
    print("=" * 50)
    
    # Find error cases
    predicted_classes = np.argmax(predictions, axis=1)
    error_indices = np.where(predicted_classes != test_labels)[0]
    
    if len(error_indices) == 0:
        print("No errors found!")
        return
    
    # Select random error cases
    selected_errors = np.random.choice(error_indices, 
                                     min(num_errors, len(error_indices)), 
                                     replace=False)
    
    try:
        gradcam = GradCAM(model)
    except Exception as e:
        print(f"Error initializing GradCAM: {e}")
        return
    
    fig, axes = plt.subplots(num_errors, 3, figsize=(15, 4*num_errors))
    if num_errors == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_errors):
        image = test_images[idx:idx+1]
        true_label = test_labels[idx]
        predicted_class = predicted_classes[idx]
        confidence = predictions[idx, predicted_class]
        
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
        axes[i, 2].set_title(f"Overlay\nError Analysis")
        axes[i, 2].axis('off')
        
        print(f"\nError Case {i+1}:")
        print(f"  True: {class_names[true_label]}")
        print(f"  Predicted: {class_names[predicted_class]} (confidence: {confidence:.3f})")
        print(f"  Model focused on: {'Lesion area' if heatmap.max() > 0.5 else 'Background/artifacts'}")
    
    plt.tight_layout()
    plt.show()

# Example usage function
def run_gradcam_analysis(models_dict, test_images, test_labels, class_names):
    """
    Run comprehensive GradCAM analysis for all models.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        test_images: Test images
        test_labels: Test labels  
        class_names: List of class names
    """
    print("🔍 Starting Comprehensive GradCAM Analysis")
    print("=" * 60)
    
    # Analyze each model individually
    for model_name, model in models_dict.items():
        analyze_model_attention(model, test_images, test_labels, class_names, 
                              model_name, num_samples=3)
    
    # Compare models on the same sample
    compare_model_attention(models_dict, test_images, test_labels, class_names, 
                           sample_idx=0)
    
    # Analyze error cases for the first model
    if models_dict:
        first_model = list(models_dict.values())[0]
        first_model_name = list(models_dict.keys())[0]
        
        # Get predictions
        predictions = first_model.predict(test_images, verbose=0)
        
        analyze_error_cases(first_model, test_images, test_labels, predictions, 
                          class_names, num_errors=3)
    
    print("\n✅ GradCAM analysis completed!")
