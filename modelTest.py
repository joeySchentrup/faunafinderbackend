from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from PIL import Image
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

'''
data_dir = 'deer_data'
image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4, shuffle=True, num_workers=4)

dataset_sizes = len(image_datasets)
class_names = image_datasets.classes
'''

use_gpu = torch.cuda.is_available()

def check_data(model, data):
    model.train(False)  # Set model to evaluate mode

    # get the inputs
    inputs, labels = data

    # wrap them in Variable
    if use_gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)

    # forward
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    
    return preds == labels.data

model_ft = torch.load("/mnt/c/Users/j03y/Desktop/Projects/faunafinderbackend/output.out")

imsize = 256
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    if use_gpu:
        return image.cuda()
    else:
        return image

image = image_loader("/mnt/c/Users/j03y/Desktop/Projects/faunafinderbackend/deer_data/train/deer/3")

if use_gpu:
    model_ft = model_ft.cuda()

print(check_data(model_ft, image))