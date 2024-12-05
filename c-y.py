# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:00:28 2024

@author: zhang
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
#import data_loading_processing_ori

# Example Data Preparation (concept predictions and species labels)
# Assuming train_concepts_pred and train_species are already generated
# train_concepts_pred: (N, num_concepts), train_species: (N,)
# Use PyTorch tensors for DataLoader

model_path = "./model_c-y_small_sample.pth"  # Replace with your desired path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
pkl_dir="./class_attr_data_10/"
#train_loader, test_loader,_ = data_loading_processing_ori.get_cub_classification_dataloaders(pkl_dir,5,5)

with open("train_loader-2.pkl", "rb") as f:
    train_loader = pickle.load(f)

# # Recreate the DataLoader
# #train_loader = DataLoader(loaded_train_loader, batch_size=5, shuffle=False)# Load the dataset

with open("test_loader-2.pkl", "rb") as f:
    test_loader = pickle.load(f)

# train_concepts_pred = torch.tensor(train_concepts_pred, dtype=torch.float32)
# train_species = torch.tensor(train_species, dtype=torch.long)

# # Split into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(train_concepts_pred, train_species, test_size=0.2, random_state=42)

# # Create DataLoaders
# batch_size = 32
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Logistic Regression Model as a Single Linear Layer
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    
    
for batch in train_loader:
    images, concepts, labels = batch  # Adjust this to match your DataLoader's output structure
    break

# Get the input size and number of classes
input_size = 112  # Number of features in the 'concepts' tensor
num_classes = 200  # Number of unique labels
# Initialize model, loss, and optimizer

model = LogisticRegressionModel(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for _, concepts, labels in train_loader:

        concepts, labels = concepts.to(device), labels.to(device)
        labels = labels.squeeze(1).long()
        #print(labels)
        # Forward pass
        outputs = model(concepts)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")    
    
    # test loop

    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    i=1
    with torch.no_grad():
        for _,concepts, labels in test_loader:
            concepts, labels = concepts.to(device), labels.to(device)
            outputs = model(concepts)
            #print(labels)

            _, predicted = torch.max(outputs, dim=1)
            #print(predicted)
            labels=labels.squeeze()
            correct += (predicted == labels).sum().item()
            
            total += labels.size(0)

    
    accuracy = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], test-accu: {(accuracy):.4f}")    
    




# Save the model's state dictionary
torch.save(model.state_dict(), model_path)


################## human intervention on predicted concepts ###################
# def human_correct_concepts(concepts_pred, ground_truth_concepts):
#     """
#     Allows manual correction of predicted concepts.
#     Args:
#         concepts_pred (torch.Tensor): Predicted concepts (N, num_concepts).
#         ground_truth_concepts (torch.Tensor): Ground truth concepts (if available).
#     Returns:
#         torch.Tensor: Corrected concepts.
#     """
#     for i, concepts in enumerate(concepts_pred):
#         print(f"Sample {i+1}:")
#         print(f"Predicted concepts: {concepts.tolist()}")
#         if ground_truth_concepts is not None:
#             print(f"Ground truth concepts: {ground_truth_concepts[i].tolist()}")
#         corrected = input("Enter corrected concepts as a comma-separated list (or press Enter to accept): ")
#         if corrected.strip():
#             corrected = torch.tensor(list(map(float, corrected.split(','))))
#             concepts_pred[i] = corrected
#     return concepts_pred

# # Apply human corrections before training
# corrected_train_concepts_pred = human_correct_concepts(train_concepts_pred, None)

