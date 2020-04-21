import sys

from load_checkpoint import load_checkpoint
from process_images import process_image
from prediction import prediction
from parser import parser

arch, lr, hidden_layers, epochs, device, top_k, category_names, gpu, flower_path = parser()

filepath = 'checkpoint.pth'
image_path = flower_path


# Load model
model = load_checkpoint(filepath)
print('Check point loaded, ready for prediction!\n')

# Predict
probs, classes, labels = prediction(image_path, model)

print(' FLOWER PREDICTIONS\n' )

for i, j in zip(labels, probs):
    print(i, ':', j, '%')
