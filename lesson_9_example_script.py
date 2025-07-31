# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:21:59 2025

@author: taske
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=== PyTorch Fundamentals Demo ===\n")

# ========================================
# SECTION 1: LOADING CUSTOM DATA FROM DISK
# ========================================

print("1. LOADING CUSTOM DATA FROM DISK")
print("-" * 40)

# 1.1 Loading CSV data and converting to tensors
print("\n1.1 CSV Data Loading")
# Create sample CSV data
sample_data = {
    'feature_1': np.random.randn(100),
    'feature_2': np.random.randn(100),
    'feature_3': np.random.randn(100),
    'label': np.random.randint(0, 3, 100)
}
df = pd.DataFrame(sample_data)
df.to_csv('sample_data.csv', index=False)

# Load CSV and convert to tensors
def load_csv_to_tensors(filepath, feature_cols, label_col):
    """
    Load CSV file and convert to PyTorch tensors.
    
    Args:
        filepath: Path to CSV file
        feature_cols: List of feature column names
        label_col: Name of label column
    
    Returns:
        features: Tensor of shape (n_samples, n_features)
        labels: Tensor of shape (n_samples,)
    """
    df = pd.read_csv(filepath)
    
    # Extract features and convert to tensor
    features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    
    # Extract labels and convert to tensor
    labels = torch.tensor(df[label_col].values, dtype=torch.long)
    
    return features, labels

features, labels = load_csv_to_tensors('sample_data.csv', 
                                     ['feature_1', 'feature_2', 'feature_3'], 
                                     'label')
print(f"Features shape: {features.shape}, dtype: {features.dtype}")
print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
print(f"Sample features: {features[:3]}")

# 1.2 Loading and preprocessing images
print("\n1.2 Image Data Loading")
# Create a sample image
sample_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
Image.fromarray(sample_image).save('sample_image.png')

def load_image_as_tensor(filepath, target_size=(64, 64), normalize=True):
    """
    Load image file and convert to PyTorch tensor.
    
    Args:
        filepath: Path to image file
        target_size: Tuple of (height, width) for resizing
        normalize: Whether to normalize pixel values to [0, 1]
    
    Returns:
        image_tensor: Tensor of shape (C, H, W)
    """
    # Load image using PIL
    image = Image.open(filepath).convert('RGB')
    
    # Resize if needed
    if target_size:
        image = image.resize(target_size)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Convert to tensor and change from HWC to CHW format
    image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)
    
    # Normalize to [0, 1] if requested
    if normalize:
        image_tensor = image_tensor / 255.0
    
    return image_tensor

image_tensor = load_image_as_tensor('sample_image.png')
print(f"Image tensor shape: {image_tensor.shape}")
print(f"Value range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")

# 1.3 Loading custom binary data
print("\n1.3 Custom Binary Data Loading")
# Create sample binary data (e.g., embeddings or processed features)
sample_embeddings = np.random.randn(50, 128).astype(np.float32)
sample_embeddings.tofile('embeddings.bin')

def load_binary_embeddings(filepath, embedding_dim, dtype=np.float32):
    """
    Load binary file containing embeddings.
    
    Args:
        filepath: Path to binary file
        embedding_dim: Dimension of each embedding
        dtype: Data type of the embeddings
    
    Returns:
        embeddings: Tensor of shape (n_embeddings, embedding_dim)
    """
    # Load binary data
    embeddings_np = np.fromfile(filepath, dtype=dtype)
    
    # Reshape to proper dimensions
    n_embeddings = len(embeddings_np) // embedding_dim
    embeddings_np = embeddings_np.reshape(n_embeddings, embedding_dim)
    
    # Convert to tensor
    embeddings = torch.tensor(embeddings_np)
    
    return embeddings

embeddings = load_binary_embeddings('embeddings.bin', 128)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Sample embedding: {embeddings[0, :5]}")

# =======================================
# SECTION 2: ESSENTIAL TENSOR OPERATIONS
# =======================================

print("\n\n2. ESSENTIAL TENSOR OPERATIONS")
print("-" * 40)

# 2.1 Tensor creation and initialization
print("\n2.1 Tensor Creation and Initialization")

# Creating tensors with specific values
zeros_tensor = torch.zeros(3, 4)
ones_tensor = torch.ones(2, 3, 5)
eye_tensor = torch.eye(4)  # Identity matrix

print(f"Zeros tensor shape: {zeros_tensor.shape}")
print(f"Ones tensor shape: {ones_tensor.shape}")
print(f"Eye tensor:\n{eye_tensor}")

# Creating tensors like existing tensors
reference_tensor = torch.randn(2, 3)
zeros_like = torch.zeros_like(reference_tensor)
ones_like = torch.ones_like(reference_tensor)

print(f"Reference tensor shape: {reference_tensor.shape}")
print(f"Zeros like shape: {zeros_like.shape}")

# Random tensor creation
uniform_tensor = torch.rand(3, 3)  # Uniform [0, 1)
normal_tensor = torch.randn(3, 3)  # Standard normal
randint_tensor = torch.randint(0, 10, (3, 3))  # Random integers

print(f"Uniform tensor:\n{uniform_tensor}")

# 2.2 Stacking and concatenation
print("\n2.2 Stacking and Concatenation")

# Create sample tensors
tensor_a = torch.randn(2, 3)
tensor_b = torch.randn(2, 3)
tensor_c = torch.randn(2, 3)

print(f"Individual tensor shape: {tensor_a.shape}")

# Stack tensors (adds new dimension)
stacked = torch.stack([tensor_a, tensor_b, tensor_c], dim=0)
print(f"Stacked along dim=0: {stacked.shape}")

stacked_dim1 = torch.stack([tensor_a, tensor_b, tensor_c], dim=1)
print(f"Stacked along dim=1: {stacked_dim1.shape}")

# Concatenate tensors (along existing dimension)
concat_dim0 = torch.cat([tensor_a, tensor_b, tensor_c], dim=0)
concat_dim1 = torch.cat([tensor_a, tensor_b, tensor_c], dim=1)

print(f"Concatenated along dim=0: {concat_dim0.shape}")
print(f"Concatenated along dim=1: {concat_dim1.shape}")

# 2.3 Reshaping and dimension manipulation
print("\n2.3 Reshaping and Dimension Manipulation")

original = torch.randn(2, 3, 4)
print(f"Original shape: {original.shape}")

# Reshape
reshaped = original.view(6, 4)  # Must be compatible size
print(f"Reshaped: {reshaped.shape}")

# Flatten
flattened = original.flatten()
print(f"Flattened: {flattened.shape}")

# Add/remove dimensions
unsqueezed = original.unsqueeze(0)  # Add dimension at index 0
print(f"Unsqueezed: {unsqueezed.shape}")

squeezed = unsqueezed.squeeze(0)  # Remove dimension at index 0
print(f"Squeezed back: {squeezed.shape}")

# Permute dimensions
permuted = original.permute(2, 0, 1)  # Rearrange dimensions
print(f"Permuted (2,0,1): {permuted.shape}")

# 2.4 Mathematical operations
print("\n2.4 Mathematical Operations")

x = torch.randn(3, 4)
y = torch.randn(3, 4)

# Element-wise operations
add_result = x + y
mul_result = x * y
div_result = x / (y + 1e-8)  # Add small epsilon to avoid division by zero

print(f"Addition result shape: {add_result.shape}")
print(f"Multiplication result shape: {mul_result.shape}")

# Matrix operations
a = torch.randn(3, 4)
b = torch.randn(4, 5)
matmul_result = torch.matmul(a, b)  # or a @ b
print(f"Matrix multiplication result: {matmul_result.shape}")

# Reduction operations
sum_all = x.sum()  # Sum all elements
sum_dim0 = x.sum(dim=0)  # Sum along dimension 0
mean_tensor = x.mean()
std_tensor = x.std()

print(f"Sum all: {sum_all}")
print(f"Sum along dim 0 shape: {sum_dim0.shape}")
print(f"Mean: {mean_tensor:.3f}, Std: {std_tensor:.3f}")

# =====================================
# SECTION 3: COMPUTING EVALUATION METRICS
# =====================================

print("\n\n3. COMPUTING EVALUATION METRICS")
print("-" * 40)

# 3.1 Classification metrics
print("\n3.1 Classification Metrics")

# Generate sample predictions and targets
n_samples = 1000
n_classes = 3

# Simulate model predictions (logits)
predictions_logits = torch.randn(n_samples, n_classes)
predictions_probs = F.softmax(predictions_logits, dim=1)
predicted_classes = torch.argmax(predictions_probs, dim=1)

# Generate true labels
true_labels = torch.randint(0, n_classes, (n_samples,))

def compute_accuracy(predictions, targets):
    """Compute classification accuracy."""
    correct = (predictions == targets).float()
    accuracy = correct.mean()
    return accuracy

def compute_precision_recall_f1(predictions, targets, num_classes):
    """Compute precision, recall, and F1-score for each class."""
    precision_list = []
    recall_list = []
    f1_list = []
    
    for class_idx in range(num_classes):
        # True positives, false positives, false negatives
        tp = ((predictions == class_idx) & (targets == class_idx)).sum().float()
        fp = ((predictions == class_idx) & (targets != class_idx)).sum().float()
        fn = ((predictions != class_idx) & (targets == class_idx)).sum().float()
        
        # Compute metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    return torch.stack(precision_list), torch.stack(recall_list), torch.stack(f1_list)

# Compute classification metrics
accuracy = compute_accuracy(predicted_classes, true_labels)
precision, recall, f1 = compute_precision_recall_f1(predicted_classes, true_labels, n_classes)

print(f"Accuracy: {accuracy:.3f}")
print(f"Per-class Precision: {precision}")
print(f"Per-class Recall: {recall}")
print(f"Per-class F1-score: {f1}")
print(f"Macro F1-score: {f1.mean():.3f}")

# 3.2 Regression metrics
print("\n3.2 Regression Metrics")

# Generate sample regression data
true_values = torch.randn(100) * 10 + 5  # True values
predictions_reg = true_values + torch.randn(100) * 2  # Predictions with noise

def compute_mse(predictions, targets):
    """Compute Mean Squared Error."""
    return F.mse_loss(predictions, targets)

def compute_mae(predictions, targets):
    """Compute Mean Absolute Error."""
    return F.l1_loss(predictions, targets)

def compute_r_squared(predictions, targets):
    """Compute R-squared (coefficient of determination)."""
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return r2

# Compute regression metrics
mse = compute_mse(predictions_reg, true_values)
mae = compute_mae(predictions_reg, true_values)
r2 = compute_r_squared(predictions_reg, true_values)

print(f"Mean Squared Error: {mse:.3f}")
print(f"Mean Absolute Error: {mae:.3f}")
print(f"R-squared: {r2:.3f}")

# 3.3 Batch-wise metric computation
print("\n3.3 Batch-wise Metric Computation")

def update_running_metrics(running_metrics, batch_predictions, batch_targets, batch_size):
    """
    Update running metrics for online computation.
    
    This is useful when you can't store all predictions in memory.
    """
    batch_accuracy = compute_accuracy(batch_predictions, batch_targets)
    
    # Update running average
    if 'accuracy' not in running_metrics:
        running_metrics['accuracy'] = 0.0
        running_metrics['total_samples'] = 0
    
    total_samples = running_metrics['total_samples']
    running_metrics['accuracy'] = (running_metrics['accuracy'] * total_samples + 
                                 batch_accuracy * batch_size) / (total_samples + batch_size)
    running_metrics['total_samples'] += batch_size
    
    return running_metrics

# Simulate batch processing
running_metrics = {}
for batch_idx in range(5):
    batch_size = 32
    batch_preds = torch.randint(0, 3, (batch_size,))
    batch_targets = torch.randint(0, 3, (batch_size,))
    
    running_metrics = update_running_metrics(running_metrics, batch_preds, 
                                           batch_targets, batch_size)
    print(f"Batch {batch_idx + 1} - Running accuracy: {running_metrics['accuracy']:.3f}")

print(f"\nFinal running accuracy: {running_metrics['accuracy']:.3f}")
print(f"Total samples processed: {running_metrics['total_samples']}")

# ============================
# SECTION 4: PRACTICAL TIPS
# ============================

print("\n\n4. PRACTICAL TIPS AND CONSIDERATIONS")
print("-" * 40)

print("\n4.1 Memory Management")
# Use .detach() to break gradient computation
large_tensor = torch.randn(1000, 1000, requires_grad=True)
detached_tensor = large_tensor.detach()  # No gradient tracking
print(f"Original requires_grad: {large_tensor.requires_grad}")
print(f"Detached requires_grad: {detached_tensor.requires_grad}")

print("\n4.2 Device Management")
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move tensors to device
cpu_tensor = torch.randn(10, 10)
device_tensor = cpu_tensor.to(device)
print(f"Tensor device: {device_tensor.device}")

print("\n4.3 Data Type Considerations")
# Different data types have different memory usage and precision
float32_tensor = torch.randn(100, dtype=torch.float32)
float16_tensor = torch.randn(100, dtype=torch.float16)
int64_tensor = torch.randint(0, 100, (100,), dtype=torch.int64)

print(f"Float32 size: {float32_tensor.element_size()} bytes per element")
print(f"Float16 size: {float16_tensor.element_size()} bytes per element")
print(f"Int64 size: {int64_tensor.element_size()} bytes per element")

print("\n=== Demo Complete ===")

# Clean up temporary files
os.remove('sample_data.csv')
os.remove('sample_image.png')
os.remove('embeddings.bin')

print("\nKey Takeaways:")
print("1. Direct file loading gives you full control over preprocessing")
print("2. Understanding tensor operations is crucial for efficient data manipulation")
print("3. Proper metric computation helps evaluate model performance accurately")
print("4. Memory and device management become important for larger projects")