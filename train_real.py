import os
import sys

import math
import torch
import numpy as np

import torch
import torch.nn.functional as F
#from torchvision.models.resnet import resnet18
from torchmetrics.aggregation import MeanMetric

from obiwan.new_models import CBM 
from obiwan.datasets.cub import get_cub_dataloaders

import wandb


def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze,  use_aux=args.use_aux,
                      n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args, divide=True)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            expand_dim=args.expand_dim)
    train(model, args, divide=True)

def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                  expand_dim=args.expand_dim)
    train(model, args, divide=True)

def train_X_to_C_to_y(args):
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args , divide=False)

def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux)
    train(model, args, divide = False)


def run_epoch_divide(model, train_loader, val_loader,  device, num_epochs=1000, lr=0.001, weight_decay=0.0004, num_classes=200, num_concepts=112):
    optimizer = torch.optim.SGD(model.get_concept_parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in tqdm(range(num_epochs)):

        model.set_concepts_to_train()
        epoch_loss_classes = MeanMetric()
        epoch_loss_classes.to(device)

        for data in tqdm(train_loader):
            #if len(data) == 2:
            #    imgs, (labels, attrs) = data
            #else:
            imgs, attrs, labels = data
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            concepts, pred_classes = model(imgs)
            concepts_loss = 0
            criterion = torch.nn.CrossEntropyLoss()
            
            for i in range(num_concepts):
                ind_concept_loss = criterion(concepts[i].squeeze(), attrs[:,i].squeeze().float())
                concepts_loss = concepts_loss + ind_concept_loss

            concepts_loss = concepts_loss / num_concepts
            loss = concepts_loss

            loss.backward()
            optimizer.step()

            epoch_loss_classes.update(loss)
            wandb.log({'concept_loss': loss})

        lr_scheduler.step()
        print(f"Epoch Class Loss: {epoch_loss_classes.compute()}")

        if (epoch+1) % 10 == 0:
            model.eval()
            recall_list = ev.evaluate_recall(model, val_loader, device, intervene=False, pre_concept=False)
            print(f'Epoch Recall@1: {recall_list[0]} - Epoch Recall@5: {recall_list[1]} - Epoch Recall@10: {recall_list[2]}')
            wandb.log({'Epoch recall@1': recall_list[0], 'Epoch recall@5': recall_list[1], 'Epoch recall@10': recall_list[2]})

            model.set_concepts_to_train()

    optimizer = torch.optim.SGD(model.get_class_parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in tqdm(range(epochs)):
        model.set_classes_to_train()
        epoch_loss_classes = MeanMetric()
        epoch_loss_classes.to(device)

        for data in tqdm(train_loader):
            #if len(data) == 2:
            #    imgs, (labels, attrs) = data
            #else:
            imgs, attrs, labels = data
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            concepts, pred_classes = model(imgs)
            class_loss = torch.nn.functional.cross_entropy(pred_classes, labels.long().squeeze())
            loss = class_loss

            loss.backward()
            optimizer.step()

            epoch_loss_classes.update(loss)
            wandb.log({'class_loss': loss})

        lr_scheduler.step()
        print(f"Epoch Class Loss: {epoch_loss_classes.compute()}")

        if (epoch+1) % 10 == 0:
            model.eval()
            recall_list = ev.evaluate_recall(model, val_loader, device, intervene=False, pre_concept=False)
            print(f'Epoch Recall@1: {recall_list[0]} - Epoch Recall@5: {recall_list[1]} - Epoch Recall@10: {recall_list[2]}')
            wandb.log({'Epoch recall@1': recall_list[0], 'Epoch recall@5': recall_list[1], 'Epoch recall@10': recall_list[2]})

            model.set_classes_to_train()
 
def run_epoch_nondivide(model, train_loader, val_loader,  device, num_epochs=1000, lr=0.001, weight_decay=0.0004, num_classes=200, num_concepts=112):
    optimizer = torch.optim.SGD(model.get_concept_parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model.train()
        epoch_loss_classes = MeanMetric()
        epoch_loss_classes.to(device)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # Unpack data
            #if len(batch) == 2:
            #    images, (labels, attributes) = batch
            #else:
            images, attributes, labels = batch
            images, attributes, labels = images.to(device), attributes.to(device), labels.to(device)

            optimizer.zero_grad()
            concepts, pred_classes = model(imgs)
            class_loss = torch.nn.functional.cross_entropy(pred_classes, labels.long().squeeze())
            concepts_loss = 0.0
            criterion = torch.nn.CrossEntropyLoss()
            
            for i in range(num_concepts):
                ind_concept_loss = criterion(concepts[i].squeeze(), attrs[:,i].squeeze().float())
                concepts_loss += ind_concept_loss

            concepts_loss = concepts_loss / num_concepts
            loss = class_loss + concepts_loss

            loss.backward()
            optimizer.step()

            epoch_loss_classes.update(loss)
            wandb.log({'concept_loss': concepts_loss})
            wandb.log({'class_loss': class_loss})

        lr_scheduler.step()
        print(f"Epoch Class Loss: {epoch_loss_classes.compute()}")

        if (epoch+1) % 10 == 0:
            model.eval()
            recall_list = ev.evaluate_recall(model, val_loader, device, intervene=False, pre_concept=False)
            print(f'Epoch Recall@1: {recall_list[0]} - Epoch Recall@5: {recall_list[1]} - Epoch Recall@10: {recall_list[2]}')
            wandb.log({'Epoch recall@1': recall_list[0], 'Epoch recall@5': recall_list[1], 'Epoch recall@10': recall_list[2]})

            model.train()

       

def train(model,args,seed=1, divide: bool):

    train_loader, val_loader = get_cub_dataloaders("/nfs/turbo/coe-ecbk/vballoli/ConceptRetrieval/cem/cem/data/CUB200/class_attr_data_10/", cfg.batch_size, cfg.num_workers)

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    os.makedirs("results", exist_ok=True)
    exp_name = str(experiment)
    results_exp_dir = os.path.join("results", exp_name)
    os.makedirs(results_exp_dir, exist_ok=True)

    if divide:  ## Divide for independent and sequential
        run_epoch_divide(model, train_loader, val_loader, device)
    else:  ## Non-divide for joint and standard
        run_epoch_nondivide(model, train_loader, val_loader, device)

    torch.save(model.state_dict(), os.path.join(results_exp_dir, 'model.pth'))


def evaluate(model: CBM, dataloader, device, num_classes, num_concepts):
    model.eval()
    model.to(device)

    concept_accuracy = MultilabelAccuracy(num_labels=num_concepts)
    concept_f1 = MultilabelF1Score(num_labels=num_concepts)
    class_accuracy = Accuracy(task='multiclass', num_classes=num_classes)

    concept_accuracy.to(device)
    class_accuracy.to(device)
    concept_f1.to(device)

    with torch.no_grad():
        for imgs, attrs, labels in tqdm(dataloader):
            imgs = imgs.to(device)
            attrs = attrs.to(device)
            labels = labels.to(device)

            concepts, classes = model(imgs)

            concept_accuracy.update(concepts, attrs)
            class_accuracy.update(classes, labels.long().squeeze())
            concept_f1.update(concepts, attrs)

        
    final_concept_accuracy = concept_accuracy.compute()
    final_class_accuracy = class_accuracy.compute()
    final_concept_f1 = concept_f1.compute()

    return final_concept_accuracy, final_class_accuracy, final_concept_f1