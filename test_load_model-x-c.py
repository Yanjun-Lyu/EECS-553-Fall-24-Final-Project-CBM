# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 21:19:17 2024

@author: zzenghao
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from torchvision import transforms

# Example Data Preparation (concept predictions and species labels)
# Assuming train_concepts_pred and train_species are already generated
# train_concepts_pred: (N, num_concepts), train_species: (N,)
# Use PyTorch tensors for DataLoader

    # transform the normalize the image
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize the images to 299x299
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for InceptionV3
])
    

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("train_loader-2.pkl", "rb") as f:
    train_loader = pickle.load(f)

# # Recreate the model architecture
# model = LogisticRegressionModel(112, 200)

# # Load weights
# model.load_state_dict(torch.load("./model_weights.pth",weights_only=True))
# model = model.to(device)


    # Load pre-trained Inception-v3
inception_v3 = models.inception_v3(weights="Inception_V3_Weights.DEFAULT")
# Modify the last layer for binary concept predictions
inception_v3.fc = nn.Sequential(
    nn.Linear(inception_v3.fc.in_features, 112),
    nn.Sigmoid()  # Binary classification
)
inception_v3.load_state_dict(torch.load("./model_x-c_small_sample.pth",weights_only=True))
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inception_v3 = inception_v3.to(device)

inception_v3.eval()
val_loss = 0.0
correct = 0
total = 0
i=1
with torch.no_grad():
    for images,concepts, labels in train_loader:
        images_pil = [transforms.ToPILImage()(img) for img in images]
        images = torch.stack([transform(img) for img in images_pil])
        
        outputs = inception_v3(images)
        predictions = (outputs > 0.5).float()
        correct+= (predictions == concepts).sum().item()

        #correct += (predictions == concepts).all(dim=1).sum().item()  # Check if all labels match
        total += concepts.numel()
        
        

val_loss /= len(train_loader)
accuracy = correct / total

print(f"accuracy {accuracy}")