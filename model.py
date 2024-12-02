# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 23:13:45 2024

@author: zzenghao
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)
    

def predict_concept(num_concepts=100):
    # Load pre-trained Inception-v3
    inception_v3 = models.inception_v3(pretrained=True)
    # Modify the last layer for binary concept predictions
    inception_v3.fc = nn.Sequential(
        nn.Linear(inception_v3.fc.in_features, num_concepts),
        nn.Sigmoid()  # Binary classification
    )
    
    return inception_v3


    
def predict_class(input_size, num_classes):
   
    return LogisticRegressionModel(input_size, num_classes)


#count_parameters(net)




