import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']


    return model
