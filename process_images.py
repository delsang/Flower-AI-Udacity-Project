import torch
import torchvision
import numpy as np
import PIL

from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from PIL import Image


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    #import image
    img = Image.open(image_path)

    #resizes the image where the shortest side is 256 pixels, keeping the aspect ratio
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    #Get image dimensions to calculate the center and crop
    width, height = img.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))

    #colour channel encoding

    img = np.array(img)
    img = img/225

    #normalise
    std = [0.485, 0.456, 0.406]
    mean = [0.229, 0.224, 0.225]

    img = (img - mean) / std

    # Move the color channels to the fist dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))

    return img
