# -*- coding: utf-8 -*-
"""
Build Your Own Machine Learning Project!
Session 9: Put it all together

Your mission: Pick a fun problem and build a complete ML solution!

Choose your adventure and fill in the TODOs!

Author: Deep Learning Course
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

# Set random seed so results are repeatable
torch.manual_seed(42)
np.random.seed(42)

print("🚀 Welcome to Your ML Project Builder! 🚀")
print("=" * 50)

# ==========================================
# STEP 1: CHOOSE YOUR PROJECT & GET DATA
# ==========================================

print("\nSTEP 1: CHOOSE YOUR PROJECT & GET DATA")
print("-" * 40)

# TODO 1: Choose your project type and create/load your data
# 
# Option A: Sports Prediction (beginner-friendly)
# Option B: Flower Classification (intermediate) 
# Option C: Your own idea!
#
# For each option, you need to:
# 1. Create or load your data
# 2. Convert it to PyTorch tensors
# 3. Split into features (X) and labels (y)

# TODO 1: decide on a dataset for your project

# TODO 2: Load your data and convert to tensors
# Fill in the missing parts to load your chosen dataset
def load_my_data():
    """Load your chosen dataset and convert to tensors."""
    
    # TODO: Load the contents of your data file(s)
    
    # TODO: Convert to PyTorch tensors
    # X = torch.tensor(features, dtype=torch.float32)
    # y = torch.tensor(labels, dtype=torch.long)
    
    # TODO: Print the shapes to check everything worked
    # print(f"Features shape: {X.shape}")
    # print(f"Labels shape: {y.shape}")
    
    # return X, y
    pass

# Uncomment this when you complete TODO 2:
# X, y = load_my_data(data_file)

# ==========================================
# STEP 2: CREATE DATASETS AND DATALOADERS
# ==========================================

print("\nSTEP 2: CREATE DATASETS AND DATALOADERS")
print("-" * 40)

# TODO 3: Split your data into training and testing sets
def split_data(X, y, train_ratio=0.8):
    """Split data into training and testing sets."""
    
    # TODO: Calculate how many samples for training
    # n_samples = X.shape[0]
    # n_train = int(n_samples * train_ratio)
    
    # TODO: Create random indices for shuffling
    # indices = torch.randperm(n_samples)
    
    # TODO: Split the indices
    
    # TODO: Use the indices to split your data
    
    # TODO: Print the sizes to check
    # print(f"Training samples: {X_train.shape[0]}")
    # print(f"Testing samples: {X_test.shape[0]}")
    
    # return X_train, X_test, y_train, y_test
    pass

# Uncomment when you complete TODO 3:
# X_train, X_test, y_train, y_test = split_data(X, y)

# TODO 4: Create DataLoaders for efficient training
def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=8):
    """Create DataLoaders for training and testing."""
    
    # TODO: Create TensorDatasets
    # train_dataset = TensorDataset(X_train, y_train)
    # test_dataset = TensorDataset(X_test, y_test)
    
    # TODO: Create DataLoaders
    
    # print(f"Created dataloaders with batch size: {batch_size}")
    # return train_loader, test_loader
    pass

# Uncomment when you complete TODO 4:
# train_loader, test_loader = create_dataloaders(X_train, X_test, y_train, y_test)

# ==========================================
# STEP 3: DESIGN YOUR MODEL
# ==========================================

print("\nSTEP 3: DESIGN YOUR MODEL")
print("-" * 40)

# TODO 5: Create your neural network model
class MyModel(nn.Module):
    """Your custom neural network model."""
    
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__()
        
        # TODO: Define your layers
        pass
    
    def forward(self, x):
        """Forward pass through the network."""
        
        # TODO: Define how data flows through your network
        
        # return x
        pass

# TODO 6: Create an instance of your model

# Uncomment and fill in when ready:
# model = MyModel()
# print(f"Created model: {model}")

# ==========================================
# STEP 4: TRAINING SETUP
# ==========================================

print("\nSTEP 4: TRAINING SETUP")
print("-" * 40)

# TODO 7: Choose your loss function and optimizer
def setup_training(model, learning_rate=0.01):
    """Set up loss function and optimizer for training."""
    
    # TODO: Choose a loss function
    # For classification, use: nn.CrossEntropyLoss()
    # For regression, use: nn.MSELoss()
    # criterion = ???
    
    # TODO: Choose an optimizer
    # Adam is usually a good choice: optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = ???
    
    # return criterion, optimizer
    pass

# Uncomment when ready:
# criterion, optimizer = setup_training(model)

# ==========================================
# STEP 5: METRICS CALCULATION
# ==========================================

print("\nSTEP 5: METRICS CALCULATION")
print("-" * 40)

# TODO 8: Create functions to calculate your metrics
def calculate_accuracy(predictions, labels):
    """Calculate accuracy for classification."""
    # TODO: Compare predictions with true labels
    
    # return accuracy
    pass

def calculate_loss_value(criterion, predictions, labels):
    """Calculate the loss value."""
    # TODO: Use your criterion to calculate loss
    # loss = criterion(predictions, labels)
    # return loss.item()
    pass

# ==========================================
# TRAINING LOOP (COMPLETED FOR YOU!)
# ==========================================

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=20):
    """Complete training loop - this is done for you!"""
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"\n🔥 Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            
            # TODO: add to total_correct and total_samples for accuracy computation
        
        # Calculate averages
        avg_loss = total_loss / len(train_loader)
        
        # TODO: calculate average train accuracy
        
        # Test phase
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                predicted = torch.argmax(outputs, dim=1)
                
                # TODO: add to total_correct and total_samples for accuracy computation
        
        # TODO: calculate average test accuracy
        
        # Store metrics
        train_losses.append(avg_loss)
        # train_accuracies.append(train_accuracy)
        # test_accuracies.append(test_accuracy)
        
        # TODO: uncomment to print progress
        # if (epoch + 1) % 5 == 0:
        #     print(f"Epoch {epoch+1}/{num_epochs}: "
        #           f"Loss = {avg_loss:.4f}, "
        #           f"Train Acc = {train_accuracy:.4f}, "
        #           f"Test Acc = {test_accuracy:.4f}")
    
    return train_losses, train_accuracies, test_accuracies

# PLOTTING FUNCTION (COMPLETED FOR YOU!)
def plot_training_results(train_losses, train_accuracies, test_accuracies):
    """Plot training results - this is done for you!"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses, 'b-', label='Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(train_accuracies, 'g-', label='Training Accuracy')
    ax2.plot(test_accuracies, 'r-', label='Test Accuracy')
    ax2.set_title('Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("📊 Training plots saved as 'training_results.png'")

# ==========================================
# STEP 6: RUN YOUR EXPERIMENT!
# ==========================================

print("\nSTEP 6: RUN YOUR EXPERIMENT!")
print("-" * 40)

# TODO 9: Put it all together and run your training!
# Once you've completed all the TODOs above, uncomment this section:

# print("🎯 Running your complete ML experiment...")

# # Train the model
# train_losses, train_accuracies, test_accuracies = train_model(
#     model, train_loader, test_loader, criterion, optimizer, num_epochs=20
# )

# # Plot the results
# plot_training_results(train_losses, train_accuracies, test_accuracies)

# # Final results
# final_train_acc = train_accuracies[-1]
# final_test_acc = test_accuracies[-1]

# print(f"\n🎉 FINAL RESULTS:")
# print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.1f}%)")
# print(f"Final Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.1f}%)")

# if final_test_acc > 0.8:
#     print("🏆 Excellent! Your model is performing really well!")
# elif final_test_acc > 0.6:
#     print("👍 Good job! Your model is learning.")
# else:
#     print("🤔 Your model might need some tweaking. Try adjusting the hidden_size or learning_rate!")

# ==========================================
# STEP 7: MAKE PREDICTIONS ON NEW DATA
# ==========================================

print("\nSTEP 7: MAKE PREDICTIONS ON NEW DATA")
print("-" * 40)

# TODO 10: Test your trained model on some new examples
def make_predictions(model, new_data):
    """Make predictions on new data points."""
    
    # TODO: Put model in evaluation mode
    # model.eval()
    
    # TODO: Make predictions (no gradients needed)
    # with torch.no_grad():
    #     predictions = model(new_data)
    #     predicted_classes = torch.argmax(predictions, dim=1)
    #     probabilities = torch.softmax(predictions, dim=1)
    
    # return predicted_classes, probabilities
    pass

# Example new data (fill this in based on your project):
# For sports: height, weight, age
# For flowers: petal_length, petal_width, sepal_length, sepal_width

# TODO: Create some test examples and make predictions
"""
# Example for sports prediction:
new_examples = torch.tensor([
    [165, 60, 14],  # Person 1
    [140, 35, 10],  # Person 2
    [180, 75, 16]   # Person 3
], dtype=torch.float32)

predicted_classes, probabilities = make_predictions(model, new_examples)
print("🔮 Predictions on new data:")
for i, (pred_class, prob) in enumerate(zip(predicted_classes, probabilities)):
    print(f"Person {i+1}: Class {pred_class.item()}, Confidence: {prob.max().item():.3f}")
"""

print("\n" + "="*50)
print("🎊 CONGRATULATIONS! 🎊")
print("You've built a complete machine learning project!")
print("="*50)

print("\n📝 What you accomplished:")
print("✅ Loaded and prepared data")
print("✅ Created datasets and dataloaders") 
print("✅ Designed a neural network")
print("✅ Set up training with loss and optimizer")
print("✅ Implemented metric calculation")
print("✅ Trained your model")
print("✅ Visualized the results")
print("✅ Made predictions on new data")

print("\n🚀 Next steps to improve your model:")
print("• Try different hidden layer sizes")
print("• Experiment with different learning rates")
print("• Add more layers to your network")
print("• Collect more training data")
print("• Try different activation functions")

print("\n💡 Ideas for new projects:")
print("• Predict your favorite video game scores")
print("• Classify different types of music")
print("• Predict weather based on sensor data")
print("• Classify handwritten digits")

print("\n🎓 You're now ready to tackle any ML problem!")