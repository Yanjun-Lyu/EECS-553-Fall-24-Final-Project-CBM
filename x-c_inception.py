# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:59:28 2024

@author: zhang
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import data_loading_processing_ori
import pickle
from torchvision import transforms


if __name__ == "__main__":
       
    model_path = "./model_x-c_small_sample.pth" 
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"device:{device}")
    # transform the normalize the image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize the images to 299x299
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for InceptionV3
    ])
    
    
    #num of concept = 112
    #num of class = 200
    
    pkl_dir="./class_attr_data_10/"
    #train_loader, test_loader,_ = data_loading_processing_ori.get_cub_classification_dataloaders(pkl_dir,64,8)
    "train-test data are stored in file. Generated from data_loading_processing"
    #TODO: check multi-processing, as the keyword "num_work"
    # Load the dataset
    with open("train_loader-2.pkl", "rb") as f:
        train_loader = pickle.load(f)
    
    # Recreate the DataLoader
    # train_loader = DataLoader(loaded_train_loader, batch_size=5, shuffle=False)# Load the dataset
    
    with open("test_loader-2.pkl", "rb") as f:
        test_loader = pickle.load(f)
        
        
    print("data loaded")
    # Recreate the DataLoader
    #test_loader = DataLoader(loaded_test_loader, batch_size=10, shuffle=False)
    
    # Load pre-trained Inception-v3
    inception_v3 = models.inception_v3(weights="Inception_V3_Weights.DEFAULT")
    # Modify the last layer for binary concept predictions
    inception_v3.fc = nn.Sequential(
        nn.Linear(inception_v3.fc.in_features, 112),
        nn.Sigmoid()  # Binary classification
    )
    
    
    inception_v3 = inception_v3.to(device)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for multi-label concepts
    optimizer = torch.optim.Adam(inception_v3.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs=1000
    

    for epoch in range(num_epochs):
        inception_v3.train()  # Set model to training mode
        running_loss = 0.0
        for images, concepts, _ in train_loader:
            images_pil = [transforms.ToPILImage()(img) for img in images]
    
            # Now apply the transformations
            images = torch.stack([transform(img) for img in images_pil])
    
    
            images, concepts = images.to(device), concepts.to(device)
    
            # Forward pass
            outputs = inception_v3(images)  # Expect shape (5, num_classes) for the 
            logits = outputs.logits
            loss = criterion(logits, concepts)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()  # Track loss for this batch
    
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
        
        
        inception_v3.eval()
        val_loss = 0.0
        correct,correct_all = 0,0
        total,total_all = 0,0
        i=1
        with torch.no_grad():
            for images,concepts, labels in train_loader:
                images_pil = [transforms.ToPILImage()(img) for img in images]
                images = torch.stack([transform(img) for img in images_pil])
                images, concepts = images.to(device), concepts.to(device)
                outputs = inception_v3(images)
                predictions = (outputs > 0.5).float()
                correct+= (predictions == concepts).sum().item()

                correct_all += (predictions == concepts).all(dim=1).sum().item()  # Check if all labels match
                total += concepts.numel()
                total_all += concepts.size(0)
                
                

        val_loss /= len(train_loader)
        accuracy = correct / total
        accuracy_all  =correct_all/total_all
        print(f"Epoch [{epoch+1}/{num_epochs}], train-accu: {(accuracy):.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], train-accu_all: {(accuracy_all):.4f}")
        
        inception_v3.eval()
        val_loss = 0.0
        correct,correct_all = 0,0
        total,total_all = 0,0
        i=1
        with torch.no_grad():
            for images,concepts, labels in test_loader:
                images_pil = [transforms.ToPILImage()(img) for img in images]
                images = torch.stack([transform(img) for img in images_pil])
                images, concepts = images.to(device), concepts.to(device)
                outputs = inception_v3(images)
                predictions = (outputs > 0.5).float()
                correct+= (predictions == concepts).sum().item()
                correct_all += (predictions == concepts).all(dim=1).sum().item()  # Check if all labels match
                total += concepts.numel()
                total_all += concepts.size(0)
                
                

        val_loss /= len(train_loader)
        accuracy = correct / total
        accuracy_all  =correct_all/total_all
        print(f"Epoch [{epoch+1}/{num_epochs}], test-accu: {(accuracy):.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], test-accu_all: {(accuracy_all):.4f}")
        
        
        if epoch % 10 ==1:
            torch.save(inception_v3.state_dict(), model_path)
        
        
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