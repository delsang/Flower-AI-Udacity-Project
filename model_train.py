import torch
import numpy as np
import os
import helper
import json
import time
import torchvision
import PIL

from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

from validationpass import validation

def model_train(model, criterion, optimizer, test_set, train_set, epochs=5, device='cuda'):
    print_every = 200
    steps = 0

    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    # Track the loss and accuracy on the validation set to determine the best hyperparameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start = time.time()
    print('Starting Training Now!')

    for e in range(epochs):

        running_loss = 0

        for images, labels in iter(train_set):
            images = images.to(device)
            labels = labels.to(device)
            model = model.to(device)

            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in evaluation mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, test_set, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(test_set)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(test_set)))

                running_loss = 0

    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))

    return model
