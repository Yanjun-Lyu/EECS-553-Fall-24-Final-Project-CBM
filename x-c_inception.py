# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:59:28 2024

@author: zhang
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

# Load pre-trained Inception-v3
inception_v3 = models.inception_v3(pretrained=True)
# Modify the last layer for binary concept predictions
inception_v3.fc = nn.Sequential(
    nn.Linear(inception_v3.fc.in_features, num_concepts),
    nn.Sigmoid()  # Binary classification
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inception_v3 = inception_v3.to(device)

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy for multi-label concepts
optimizer = torch.optim.Adam(inception_v3.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    inception_v3.train()
    for images, concepts in train_loader:
        images, concepts = images.to(device), concepts.to(device)

        # Forward pass
        outputs = inception_v3(images)
        loss = criterion(outputs, concepts)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


################## Model save and load #####################

## Save trained model (weights)
#torch.save(inception_v3.state_dict(), 'inception_v3_concepts.pth')

## Save the trained model (architecture and weights)
#torch.save(inception_v3, 'inception_v3_concepts_full.pth')

## Load the saved model (weights)
#inception_v3.load_state_dict(torch.load('inception_v3_concepts.pth'))
#inception_v3.eval()  # Set the model to evaluation mode

## Load the saved model (architecture and weights)
#inception_v3 = torch.load('inception_v3_concepts_full.pth')


################## Define the dataloader ("train_loader") #####################

## from torch.utils.data import DataLoader, Dataset
#from torchvision import transforms
#from torchvision.datasets import ImageFolder

## Example: Loading a dataset from a directory
#transform = transforms.Compose([
#    transforms.Resize((299, 299)),  # Resize for Inception-v3
#    transforms.ToTensor()           # Convert to Tensor
#])

## Assuming the dataset is stored in 'data/train'
#dataset = ImageFolder(root='data/train', transform=transform)

## Create DataLoader
#train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)