# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 12:00:28 2024

@author: zhang
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
import pickle
import data_loading_processing_ori
from torchvision import transforms
import sys

torch.manual_seed(42)
# Example Data Preparation (concept predictions and species labels)
# Assuming train_concepts_pred and train_species are already generated
# train_concepts_pred: (N, num_concepts), train_species: (N,)
# Use PyTorch tensors for DataLoader
if __name__ == "__main__":
        # transform the normalize the image
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize the images to 299x299
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for InceptionV3
    ])
        
    
    # Example Data Preparation (concept predictions and species labels)
    # Assuming train_concepts_pred and train_species are already generated
    # train_concepts_pred: (N, num_concepts), train_species: (N,)
    # Use PyTorch tensors for DataLoader
    if len(sys.argv) != 3:
        print("Usage: python train.py <x-c> <c-y> default: independent")
        Load_model_x_c="./model_x-c_all_sample.pth"
        Load_model_c_y = "./model_c-y_all_sample.pth"
        
    else:
        Load_model_x_c = sys.argv[1]
        Load_model_c_y = sys.argv[2]

    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")
    pkl_dir="./class_attr_data_10/"
    batch_size, num_workers = 64,8
    train_loader, test_loader,_ = data_loading_processing_ori.get_cub_classification_dataloaders(pkl_dir,batch_size, num_workers)
    
    # with open("train_loader-2.pkl", "rb") as f:
    #     train_loader = pickle.load(f)
    
    # # # Recreate the DataLoader
    # # #train_loader = DataLoader(loaded_train_loader, batch_size=5, shuffle=False)# Load the dataset
    
    # with open("test_loader-2.pkl", "rb") as f:
    #     test_loader = pickle.load(f)
    
    print("data loaded")
    
    
    
    
     # Load pre-trained Inception-v3
    inception_v3 = models.inception_v3(init_weights=True)
    # Modify the last layer for binary concept predictions
    inception_v3.fc = nn.Sequential(
        nn.Linear(inception_v3.fc.in_features, 112),
        nn.Sigmoid()  # Binary classification
    )
    inception_v3.load_state_dict(torch.load(Load_model_x_c,weights_only=True))

    inception_v3 = inception_v3.to(device)
    
    inception_v3.eval()

    ori_concept_test = []
    predicted_concepts_test = []
    ori_labels_test = []
    with torch.no_grad():
        for images,concepts, labels in test_loader:
            images_pil = [transforms.ToPILImage()(img) for img in images]
            images = torch.stack([transform(img) for img in images_pil])
            images = images.to(device)
            outputs = inception_v3(images)
            predictions = (outputs > 0.5).float()
            
            ori_concept_test.extend(concepts.cpu())
            ori_labels_test.extend(labels.cpu())
            predicted_concepts_test.extend(predictions.cpu())
     
    ori_concept_test_test = torch.stack(ori_concept_test)        
    ori_labels_test = torch.stack(ori_labels_test)
    predicted_concepts_test = torch.stack(predicted_concepts_test)
    
    predicted_test_loader = TensorDataset(ori_concept_test_test, predicted_concepts_test,ori_labels_test)
    
    predicted_test_loader = DataLoader(predicted_test_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    
    del inception_v3
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    
    model = LogisticRegressionModel(input_size, num_classes)
    model.load_state_dict(torch.load(Load_model_c_y,weights_only=True))

    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Loop
 

    label_class_idx=[4,10,16,23,25,31,32,37,39,45,50,52,54,59,
                     64,70,77,78,81,84,87,90,96,99,103,109,112]
    label_class_idx=[x-1 for x in label_class_idx]

    for interv in label_class_idx:
        print(f"intervine number:{interv+1}")
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        i=1
        with torch.no_grad():
            for concept_ori, concepts, labels in predicted_test_loader:
                
                               
                # Replace the first 10 rows of `concepts` with `concept_ori`
                concepts_modify = concepts.clone()
                concepts_modify[:,:(interv+1)] = concept_ori[:,:(interv+1)].clone()
                
                # Move data to the specified device
                concept_ori, concepts_modify, labels = (
                    concept_ori.to(device), concepts_modify.to(device), 
                    labels.to(device))
                # num_mismatches = (concepts_modify != concept_ori).sum().item()
                # print(f"num_mismatches: {num_mismatches}")
                
                outputs = model(concepts_modify)
                _, predicted = torch.max(outputs, dim=1)
                #print(predicted)
                labels=labels.squeeze()
                #print(labels)
                correct += (predicted == labels).sum().item()
                
                total += labels.size(0)
                #print(f"correct: {correct}   total:{total}")
        
        accuracy = correct / total
        print(f"test-accu_x-y: {(accuracy):.4f}") 
        

        

 