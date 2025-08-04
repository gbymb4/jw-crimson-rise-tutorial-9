# -*- coding: utf-8 -*-
"""
Machine Learning Project - Homework Assignment
Session 9 Follow-up: Complete ML Pipeline Implementation

Your mission: Build a complete ML solution with proper training, evaluation, 
and hyperparameter tuning!

This homework focuses on:
1. Data loading and preprocessing (your choice of dataset)
2. Implementing training and evaluation functions
3. Hyperparameter tuning to find the best model configuration

Author: Deep Learning Course - Homework Assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from itertools import product
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("🎯 ML Project Homework - Complete Implementation")
print("=" * 60)

# ==========================================
# PART 1: DATA LOADING AND PREPROCESSING
# ==========================================

print("\nPART 1: DATA LOADING AND PREPROCESSING")
print("-" * 40)

# TODO 1: Implement a flexible data loading function
def load_and_preprocess_data(data_source, data_type='csv'):
    """
    Load and preprocess data from various sources.
    
    Args:
        data_source: Path to your data file(s) or data itself
        data_type: Type of data ('csv', 'image', 'numpy', 'custom')
    
    Returns:
        X: Features tensor (torch.float32)
        y: Labels tensor (torch.long for classification, torch.float32 for regression)
        num_classes: Number of classes (for classification) or None (for regression)
        feature_dim: Dimension of input features
    
    TODO: Implement data loading for different data types:
    
    For CSV data:
    - Load using pandas: pd.read_csv(data_source)
    - Separate features and labels (usually last column is labels)
    - Handle missing values (drop rows or fill with mean/median)
    - Normalize features: (x - mean) / std
    - Convert labels to integers for classification
    
    For image data:
    - Load images from folder structure (folder names = class labels)
    - Resize images to consistent size (e.g., 32x32 or 64x64)
    - Convert to tensors and normalize pixel values to [0,1] or [-1,1]
    - Flatten images: x.view(x.size(0), -1) or keep as 2D for CNN
    
    For numpy data:
    - Load using np.load() or np.loadtxt()
    - Convert to torch tensors
    - Apply any necessary preprocessing
    
    For custom data:
    - Implement your own loading logic
    - Could be text data, time series, etc.
    
    Example structure:
    if data_type == 'csv':
        # Load CSV file
        # Separate X and y
        # Preprocess and normalize
        # Convert to tensors
    elif data_type == 'image':
        # Load images from directories
        # Resize and normalize
        # Create labels from folder names
    # ... etc
    """
    
    # TODO: Implement this function based on your chosen dataset
    # Remember to return: X, y, num_classes, feature_dim
    
    pass

# TODO 2: Implement train-validation-test split
def create_data_splits(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into training, validation, and test sets.
    
    Args:
        X: Feature tensor
        y: Label tensor
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for testing (default 0.15)
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    
    TODO: Implement data splitting:
    - Calculate number of samples for each split
    - Create random permutation of indices: torch.randperm(len(X))
    - Split indices into three groups
    - Use indices to split X and y
    - Print shapes of each split for verification
    """
    
    # TODO: Implement the splitting logic
    # Hint: Make sure train_ratio + val_ratio + test_ratio = 1.0
    
    pass

# TODO 3: Create DataLoaders with different batch sizes
def create_dataloaders_with_batch_size(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
    """
    Create DataLoaders for all three splits with specified batch size.
    
    Args:
        Data splits and batch_size
    
    Returns:
        train_loader, val_loader, test_loader
    
    TODO: Create TensorDatasets and DataLoaders
    - Use TensorDataset for each split
    - Set shuffle=True for training, False for validation/test
    - Return all three dataloaders
    """
    
    # TODO: Implement DataLoader creation
    
    pass

# ==========================================
# PART 2: MODEL DEFINITION
# ==========================================

print("\nPART 2: MODEL DEFINITION")
print("-" * 40)

# TODO 4: Create a flexible neural network class
class FlexibleNN(nn.Module):
    """
    A flexible neural network that can adapt to different problems.
    
    TODO: Design your network architecture:
    - Take input_dim, hidden_dims (list), output_dim, activation as parameters
    - Support multiple hidden layers with different sizes
    - Add dropout for regularization
    - Support both classification and regression
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', dropout_rate=0.2):
        super(FlexibleNN, self).__init__()
        
        # TODO: Build your network layers
        # Hint: Use nn.ModuleList() to store multiple layers
        # Example structure:
        # - Input layer: nn.Linear(input_dim, hidden_dims[0])
        # - Hidden layers: nn.Linear(hidden_dims[i], hidden_dims[i+1])
        # - Output layer: nn.Linear(hidden_dims[-1], output_dim)
        # - Dropout layers: nn.Dropout(dropout_rate)
        
        pass
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        TODO: Implement forward pass:
        - Apply each linear layer
        - Apply activation function (except last layer)
        - Apply dropout during training
        - Return final output
        """
        
        # TODO: Implement forward pass
        
        pass

# ==========================================
# PART 3: TRAINING AND EVALUATION FUNCTIONS
# ==========================================

print("\nPART 3: TRAINING AND EVALUATION FUNCTIONS")
print("-" * 40)

# TODO 5: Implement training function
def train_one_epoch(model, train_loader, criterion, optimizer, device='cpu'):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        avg_loss: Average training loss for the epoch
        accuracy: Training accuracy for the epoch
    
    TODO: Implement training logic:
    - Set model to training mode: model.train()
    - Initialize loss and accuracy tracking variables
    - Loop through batches in train_loader:
        * Move batch to device: batch_X.to(device), batch_y.to(device)
        * Zero gradients: optimizer.zero_grad()
        * Forward pass: outputs = model(batch_X)
        * Calculate loss: loss = criterion(outputs, batch_y)
        * Backward pass: loss.backward()
        * Update weights: optimizer.step()
        * Update tracking variables
    - Calculate and return average loss and accuracy
    """
    
    # TODO: Implement training for one epoch
    
    pass

# TODO 6: Implement evaluation function
def evaluate_model(model, data_loader, criterion, device='cpu'):
    """
    Evaluate the model on given data.
    
    Args:
        model: Neural network model
        data_loader: Data loader (validation or test)
        criterion: Loss function
        device: Device to run on
    
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
    
    TODO: Implement evaluation logic:
    - Set model to evaluation mode: model.eval()
    - Use torch.no_grad() context
    - Loop through batches:
        * Move batch to device
        * Forward pass (no gradients needed)
        * Calculate loss and predictions
        * Update tracking variables
    - Calculate and return average loss and accuracy
    """
    
    # TODO: Implement evaluation function
    
    pass

# TODO 7: Implement complete training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device='cpu', early_stopping_patience=10):
    """
    Complete training loop with validation and early stopping.
    
    Args:
        model: Neural network model
        train_loader, val_loader: Data loaders
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs
        device: Device to run on
        early_stopping_patience: Stop if no improvement for this many epochs
    
    Returns:
        history: Dictionary with training history
        best_val_accuracy: Best validation accuracy achieved
    
    TODO: Implement complete training loop:
    - Initialize tracking variables and history dictionary
    - Implement early stopping logic
    - Loop through epochs:
        * Train for one epoch
        * Evaluate on validation set
        * Track best performance
        * Print progress every few epochs
        * Apply early stopping if needed
    - Return training history and best validation accuracy
    """
    
    # TODO: Implement complete training loop
    # Don't forget to track: train_losses, train_accuracies, val_losses, val_accuracies
    
    pass

# ==========================================
# PART 4: HYPERPARAMETER TUNING
# ==========================================

print("\nPART 4: HYPERPARAMETER TUNING")
print("-" * 40)

# TODO 8: Implement hyperparameter tuning function
def hyperparameter_search(X_train, X_val, X_test, y_train, y_val, y_test, 
                         input_dim, output_dim, num_classes=None):
    """
    Perform hyperparameter search to find the best model configuration.
    
    Args:
        Data splits, input_dim, output_dim, num_classes
    
    Returns:
        best_config: Dictionary with best hyperparameters
        best_test_accuracy: Best test accuracy achieved
        all_results: List of all configurations and their results
    """
    
    # Define hyperparameter search space
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    hidden_configs = [[64], [128], [64, 32], [128, 64]]
    optimizers = ['adam', 'sgd']
    dropout_rates = [0.1, 0.2, 0.3]
    
    # TODO: Use itertools.product to create all hyperparameter combinations
    # 
    # UNDERSTANDING itertools.product():
    # itertools.product creates the Cartesian product of input iterables
    # 
    # Example 1: Simple case
    # from itertools import product
    # colors = ['red', 'blue']
    # sizes = ['small', 'large']
    # combinations = list(product(colors, sizes))
    # Result: [('red', 'small'), ('red', 'large'), ('blue', 'small'), ('blue', 'large')]
    #
    # Example 2: Three lists
    # numbers = [1, 2]
    # letters = ['a', 'b']
    # combinations = list(product(numbers, letters))
    # Result: [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
    #
    # For hyperparameters, you'll get tuples like:
    # (0.001, 16, [64], 'adam', 0.1), (0.001, 16, [64], 'sgd', 0.1), etc.
    #
    # TODO: Loop through combinations and unpack each tuple
    # TODO: For each combination, create model, train it, and evaluate
    # TODO: Track the best performing configuration
    # TODO: Return best_config, best_test_accuracy, all_results
    
    pass

# ==========================================
# PART 5: VISUALIZATION AND ANALYSIS (COMPLETE)
# ==========================================

print("\nPART 5: VISUALIZATION AND ANALYSIS (COMPLETE)")
print("-" * 40)

def plot_training_history(history, title="Training History"):
    """
    Plot training and validation metrics over time.
    This function is complete - no TODO needed!
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training and validation loss
    epochs = range(1, len(history['train_losses']) + 1)
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot training and validation accuracy
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training history plot saved as '{title.lower().replace(' ', '_')}.png'")

def plot_hyperparameter_results(all_results):
    """
    Visualize hyperparameter search results.
    This function is complete - no TODO needed!
    """
    if not all_results:
        print("No results to plot!")
        return
    
    # Extract data for plotting
    test_accuracies = [result['test_accuracy'] for result in all_results]
    
    # Sort results by test accuracy for better visualization
    sorted_results = sorted(all_results, key=lambda x: x['test_accuracy'], reverse=True)
    
    # Plot 1: Top 10 configurations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    top_10 = sorted_results[:min(10, len(sorted_results))]
    config_labels = []
    top_accuracies = []
    
    for i, result in enumerate(top_10):
        config = result['config']
        label = f"LR:{config['learning_rate']}\nBS:{config['batch_size']}\nH:{config['hidden_dims']}"
        config_labels.append(f"Config {i+1}")
        top_accuracies.append(result['test_accuracy'])
    
    bars = ax1.bar(range(len(top_accuracies)), top_accuracies, color='skyblue', edgecolor='navy')
    ax1.set_title('Top 10 Hyperparameter Configurations')
    ax1.set_xlabel('Configuration Rank')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_xticks(range(len(top_accuracies)))
    ax1.set_xticklabels(config_labels, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, top_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Distribution of all results
    ax2.hist(test_accuracies, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax2.set_title('Distribution of Test Accuracies')
    ax2.set_xlabel('Test Accuracy')
    ax2.set_ylabel('Number of Configurations')
    ax2.axvline(np.mean(test_accuracies), color='blue', linestyle='--', 
                label=f'Mean: {np.mean(test_accuracies):.3f}')
    ax2.axvline(np.max(test_accuracies), color='green', linestyle='--', 
                label=f'Best: {np.max(test_accuracies):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_search_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Hyperparameter Search Summary:")
    print(f"Total configurations tested: {len(all_results)}")
    print(f"Best test accuracy: {max(test_accuracies):.4f}")
    print(f"Mean test accuracy: {np.mean(test_accuracies):.4f}")
    print(f"Standard deviation: {np.std(test_accuracies):.4f}")
    print("\n🏆 Best configuration:")
    best_result = sorted_results[0]
    for key, value in best_result['config'].items():
        print(f"  {key}: {value}")
    
    print("📊 Hyperparameter results plot saved as 'hyperparameter_search_results.png'")

# ==========================================
# PART 6: MAIN EXECUTION PIPELINE
# ==========================================

print("\nPART 6: MAIN EXECUTION PIPELINE")
print("-" * 40)

def main():
    """
    Main function to run the complete ML pipeline.
    
    TODO: Implement the complete pipeline:
    1. Load and preprocess your chosen dataset
    2. Split data into train/val/test
    3. Run hyperparameter search
    4. Train final model with best hyperparameters
    5. Evaluate and visualize results
    6. Make predictions on new data
    """
    
    print("🚀 Starting Complete ML Pipeline...")
    
    # TODO 10: Load your data
    # Choose your dataset and implement data loading
    # data_source = "your_dataset.csv"  # or path to images, etc.
    # X, y, num_classes, feature_dim = load_and_preprocess_data(data_source, data_type='csv')
    
    # TODO 11: Split the data
    # X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(X, y)
    
    # TODO 12: Run hyperparameter search
    # best_config, best_test_accuracy, all_results = hyperparameter_search(
    #     X_train, X_val, X_test, y_train, y_val, y_test, 
    #     feature_dim, num_classes or 1, num_classes
    # )
    
    # TODO 13: Train final model with best hyperparameters
    # Use best_config to create and train your final model
    
    # TODO 14: Evaluate and visualize results
    # plot_training_history(final_history)
    # plot_hyperparameter_results(all_results)
    
    # TODO 15: Print final results and analysis
    print("\n" + "="*60)
    print("🎊 HOMEWORK COMPLETION SUMMARY 🎊")
    print("="*60)
    print("\n📋 What you should have implemented:")
    print("✅ Data loading and preprocessing function")
    print("✅ Train-validation-test data splitting")
    print("✅ Flexible neural network architecture")
    print("✅ Training function with proper gradient updates")
    print("✅ Evaluation function with accuracy calculation")
    print("✅ Complete training loop with early stopping")
    print("✅ Hyperparameter search with nested loops")
    print("✅ Complete pipeline integration")
    print("📊 Plotting functions provided (no implementation needed)")
    print("📈 Visualization and analysis complete")
    
    # TODO: Print your final results here
    # print(f"\n🏆 FINAL RESULTS:")
    # print(f"Best Hyperparameters: {best_config}")
    # print(f"Best Test Accuracy: {best_test_accuracy:.4f} ({best_test_accuracy*100:.1f}%)")
    
    pass

# ==========================================
# BONUS CHALLENGES (OPTIONAL)
# ==========================================

print("\nBONUS CHALLENGES (OPTIONAL)")
print("-" * 40)

# BONUS TODO 1: Implement cross-validation
def k_fold_cross_validation(X, y, k=5, model_params=None):
    """
    Implement k-fold cross-validation for more robust evaluation.
    
    TODO: 
    - Split data into k folds
    - Train and validate k times (each fold as validation once)
    - Return average performance across folds
    """
    pass

# BONUS TODO 2: Implement advanced regularization
def train_with_regularization(model, train_loader, val_loader, l1_lambda=0.01, l2_lambda=0.01):
    """
    Add L1 and L2 regularization to training.
    
    TODO:
    - Add L1 regularization: sum of absolute weights
    - Add L2 regularization: sum of squared weights  
    - Modify loss calculation to include regularization terms
    """
    pass

# BONUS TODO 3: Implement learning rate scheduling
def train_with_lr_scheduler(model, train_loader, val_loader, criterion, optimizer):
    """
    Implement learning rate scheduling during training.
    
    TODO:
    - Use torch.optim.lr_scheduler
    - Try different scheduling strategies (StepLR, ReduceLROnPlateau, etc.)
    """
    pass

# ==========================================
# RUN THE COMPLETE PIPELINE
# ==========================================

if __name__ == "__main__":
    # TODO: Uncomment this when you're ready to run your complete implementation
    # main()
    
    print("\n💡 GETTING STARTED:")
    print("1. Choose your dataset (CSV, images, or custom)")
    print("2. Implement the data loading function first")
    print("3. Work through each TODO step by step")
    print("4. Test each function individually before running the full pipeline")
    print("5. Start with a small dataset to debug quickly")
    
    print("\n🎯 SUCCESS CRITERIA:")
    print("• Your code should run without errors")
    print("• Model should achieve reasonable accuracy (>60% for most datasets)")
    print("• Hyperparameter search should find better configurations")
    print("• All visualizations should be clear and informative")
    print("• Code should be well-documented with your own comments")
    
    print("\n🚀 Ready to start your ML adventure? Uncomment main() and begin!")