# Import all

import sys

import torch
import numpy as np
import os
import helper
import json
import time
import torchvision
import PIL
import seaborn as sns
import argparse

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import nn
from torch import optim
from PIL import Image

from model import model_to_train
from validationpass import validation
from model_train import model_train
from parser import parser

arch, lr, hidden_layers, epochs, device, top_k, category_names, gpu = parser()

def main(arch=arch, lr=lr, epochs=epochs, device=device, hidden_layers=hidden_layers):

    # Load the data
    # Path to directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    print('paths set\n')

    # Define the Transforms
    data_transforms = {
        'train' : transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        'valid' : transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        'test' : transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])}


    # Load the data sets with ImageFolder
    image_datasets = {
        'train' : datasets.ImageFolder(os.path.join(train_dir), transform=data_transforms['train']),
        'valid' : datasets.ImageFolder(os.path.join(valid_dir), transform=data_transforms['valid']),
        'test' : datasets.ImageFolder(os.path.join(test_dir), transform=data_transforms['test'])}

    # Define the dataloaders
    dataloaders = {
        'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=10, shuffle=True),
        'valid' : torch.utils.data.DataLoader(image_datasets['valid'], batch_size=10, shuffle=True),
        'test' : torch.utils.data.DataLoader(image_datasets['test'], batch_size=10, shuffle=True)}

    # Defind the test and training sets

    test_set = dataloaders['test']
    train_set = dataloaders['train']

    print('data transforms, image datasets, data loaders done\n')

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    #get the model and classifier
    classifier, model, criterion, optimizer = model_to_train(arch, lr, hidden_layers)

    print('*** Model and classifier ready ***\n')

    # Start training

    model_train(model, criterion, optimizer, test_set, train_set, epochs, device)


    # Validation on the test set
    test_loss, accuracy = validation(model, dataloaders['valid'], criterion)
    print("Accuracy on the test dataset: %{:.1f}".format(accuracy))

    # Save check point

    model.class_to_idx = image_datasets['train'].class_to_idx

    torch.save({'arch': arch,
                'input_size': 25088,
                'output_size': 102,
                'learning_rate': lr,
                'batch_size': 10,
                'classifier' : classifier,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},
                'checkpoint.pth')

    print("saved checkpoint")

if __name__ == "__main__":
    main()
