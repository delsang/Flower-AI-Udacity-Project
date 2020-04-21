import torch
import torchvision
import json

import numpy as np

from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from PIL import Image

from process_images import process_image

with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

def prediction(image_path, model, top_k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
     # TODO: Implement the code to predict the class from an image file
    if top_k < 1:
        top_k = 1

    #Cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluation only
    model.eval()

    # turn the np_array we have in the function process_image into a FloatTensor

    tensor_img = torch.FloatTensor([process_image(image_path)])
    tensor_img = tensor_img.to(device)

    result = model(tensor_img).topk(top_k)

    # Take the natural exponent of each probablility to undo the natural log from the NLLLoss criterion
    # turn it into a np_array, *100 to get the percentage of probability
    probs = (torch.exp(result[0].data).cpu().numpy()[0])*100

    classes = result[1].data.cpu().numpy()[0].tolist()
    inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
    classes = [inv_class_to_idx[x] for x in classes]
    labels = [cat_to_name[x] for x in classes]

    return(probs, classes, labels)
