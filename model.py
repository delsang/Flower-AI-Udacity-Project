from torch import optim, nn
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict

arch = {"vgg16":25088,
         "densenet161": 2208}

def model_to_train(arch, lr, hidden_layers):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)

        # Freeze the parameters
        for param in model.parameters():
            param.requires_grad = False

        # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
        input_size = model.classifier[0].in_features
        output_size = 102


        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_layers)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_layers, output_size)),
            ('output', nn.LogSoftmax(dim=1))]))


    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)

        # Freeze the parameters
        for param in model.parameters():
            param.requires_grad = False

        # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
        input_size = model.classifier[0].in_features
        output_size = 102


        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_layers)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_layers, output_size)),
            ('output', nn.LogSoftmax(dim=1))]))

    else:
        print('Models available are vgg16 or densenet161 only, try again')



    model.classifier = classifier

    # Define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)


    return classifier, model, criterion, optimizer
