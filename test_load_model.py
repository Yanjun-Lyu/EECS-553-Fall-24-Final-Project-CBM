# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:19:17 2024

@author: zzenghao
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Example Data Preparation (concept predictions and species labels)
# Assuming train_concepts_pred and train_species are already generated
# train_concepts_pred: (N, num_concepts), train_species: (N,)
# Use PyTorch tensors for DataLoader

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("train_loader-False.pkl", "rb") as f:
    train_loader = pickle.load(f)

# Recreate the model architecture
model = LogisticRegressionModel(112, 200)

# Load weights
model.load_state_dict(torch.load("./model_weights.pth",weights_only=True))
model = model.to(device)

model.eval()
val_loss = 0.0
correct = 0
total = 0
i=1
with torch.no_grad():
    for _,concepts, labels in train_loader:
        concepts, labels = concepts.to(device), labels.to(device)
        outputs = model(concepts)
        if i==1:print(labels)
        i=0
        _, predicted = torch.max(outputs, dim=1)
        labels=labels.squeeze()
        correct += (predicted == labels).sum().item()
        
        total += labels.size(0)

val_loss /= len(train_loader)
accuracy = correct / total

print(f"accuracy {accuracy}")