import pickle
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch
import os
from pathlib import Path

from torch.utils.data import Subset
import random


def load_data(pkl_dir: str, split: str):
    data=pickle.load(open(f'{pkl_dir}/{split}.pkl', 'rb'))
    processed_data=[]

    for img_data in data:
        img_path=img_data['img_path']
        img_path_split=img_path.split('/')
        
        try:
            idx = img_path_split.index('CUB_200_2011')
            img_path='/'.join(img_path_split[idx:])
        except ValueError:
            img_path='/'.join(img_path_split[:2] + [split] + img_path_split[2:])
        
        img_path=img_path.replace('CUB_200_2011', 'CUB_200_2011')
        img_path=os.path.join(Path(pkl_dir).parent.parent, img_path)
        img=Image.open(img_path).convert('RGB')

        class_label=img_data['class_label']
        attr_label=img_data['attribute_label']

        processed_data.append((img, attr_label, class_label))

    return processed_data


def preprocess_data(data, transform):
    images,attrs,labels=zip(*data)
    images=[transform(img) for img in images]
    attrs=[torch.Tensor(attr) for attr in attrs]
    labels=[torch.Tensor([label]) for label in labels]
    
    images=torch.stack(images)
    attrs=torch.stack(attrs)
    labels=torch.stack(labels)

    return TensorDataset(images, attrs, labels)


def get_transforms(resol=224, resized_resol=299):
    resized_resol = int(resized_resol * 256 / 224)

    train_transform = transforms.Compose([
        transforms.Resize((resized_resol, resized_resol)),
        transforms.RandomResizedCrop(resol),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((resized_resol, resized_resol)),
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
    ])

    return train_transform, test_transform


def cub_classification_data(pkl_dir: str):
    train_data = load_data(pkl_dir, 'train')
    test_data = load_data(pkl_dir, 'test')
    val_data = load_data(pkl_dir, 'val')

    train_transform, test_transform = get_transforms()

    train_dataset=preprocess_data(train_data, train_transform)
    test_dataset=preprocess_data(test_data, test_transform)
    val_dataset=preprocess_data(val_data, test_transform)

    return train_dataset, test_dataset, val_dataset


def get_cub_classification_dataloaders(pkl_dir: str, batch_size: int, num_workers: int):
    train_dataset, test_dataset, val_dataset = cub_classification_data(pkl_dir)

    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader, val_loader


def get_cub_data(pkl_dir: str):
    train_data = load_data(pkl_dir, 'train')
    test_data = load_data(pkl_dir, 'test')
    val_data = load_data(pkl_dir, 'val')
    
    # Randomly select 100 indices
    train_indices = random.sample(range(len(train_data)), 100)
    test_indices = random.sample(range(len(test_data)), 100)
    val_indices = random.sample(range(len(val_data)), 100)
    
    # Create subsets
    train_data = Subset(train_data, train_indices)
    test_data = Subset(test_data, test_indices)
    val_data = Subset(val_data, val_indices)

    class_to_data_map = {}

    for data in train_data + test_data + val_data:
        img, attr_label, class_label = data
        if class_label not in class_to_data_map:
            class_to_data_map[class_label] = []
        class_to_data_map[class_label].append(data)

    train_data=[]
    test_data=[]

    for class_label, data_list in class_to_data_map.items():
        if class_label < 100:
            train_data.extend(data_list)
        else:
            test_data.extend(data_list)

    train_transform, test_transform = get_transforms()

    train_dataset=preprocess_data(train_data, train_transform)
    test_dataset=preprocess_data(test_data, test_transform)

    return train_dataset, test_dataset


def get_cub_dataloaders(pkl_dir: str, batch_size: int, num_workers: int):
    train_dataset, test_dataset = get_cub_data(pkl_dir)
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, test_loader

if __name__ == '__main__':
    pkl_dir="./class_attr_data_10/"
    #a= cub_classification_data(pkl_dir)
    train_loader, test_loader = get_cub_dataloaders(pkl_dir,5,5)
