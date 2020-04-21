import torchvision
from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim

def validation(model, test_set, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loss = 0
    accuracy = 0

    for images, labels in test_set:
        images = images.to(device)
        labels = labels.to(device)
        model = model.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy
