# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:59:28 2024

@author: zhang
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import data_loading_processing
import pickle
from torchvision import transforms


if __name__ == "__main__":
       
    model_path = "./model_x-y_small_sample.pth" 
    
    # transform the normalize the image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize the images to 299x299
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for InceptionV3
    ])
    
    
    #num of concept = 112
    #num of class = 200
    
    pkl_dir="./class_attr_data_10/"
    #train_loader, test_loader,_ = data_loading_processing_ori.get_cub_classification_dataloaders(pkl_dir,5,5)
    "train-test data are stored in file. Generated from data_loading_processing"
    #TODO: check multi-processing, as the keyword "num_work"
    # Load the dataset
    with open("train_loader-2.pkl", "rb") as f:
        train_loader = pickle.load(f)
    
    # Recreate the DataLoader
    #train_loader = DataLoader(loaded_train_loader, batch_size=5, shuffle=False)# Load the dataset
    
    with open("test_loader-2.pkl", "rb") as f:
        test_loader = pickle.load(f)
    
    # Recreate the DataLoader
    #test_loader = DataLoader(loaded_test_loader, batch_size=10, shuffle=False)
    
    # Load pre-trained Inception-v3
    inception_v3 = models.inception_v3(weights="Inception_V3_Weights.DEFAULT")
    # Modify the last layer for binary concept predictions
    inception_v3.fc = nn.Sequential(
        nn.Linear(inception_v3.fc.in_features, 200),
        nn.Sigmoid()  # Binary classification
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception_v3 = inception_v3.to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Binary cross-entropy for multi-label concepts
    optimizer = torch.optim.Adam(inception_v3.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs=5
    for epoch in range(num_epochs):
        inception_v3.train()  # Set model to training mode
        running_loss = 0.0
        for images, concepts, labels in train_loader:
            images_pil = [transforms.ToPILImage()(img) for img in images]
    
            # Now apply the transformations
            images = torch.stack([transform(img) for img in images_pil])
    
    
            images, labels = images.to(device), labels.to(device)
            labels = labels.squeeze(1).long()
    
            # Forward pass
            outputs = inception_v3(images)  # Expect shape (5, num_classes) for the 
            logits = outputs.logits
            loss = criterion(logits, labels)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()  # Track loss for this batch
    
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    # Save the model's state dictionary
    torch.save(inception_v3.state_dict(), model_path)
    #TODO
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