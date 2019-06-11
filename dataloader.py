import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
import torchvision

from IPython import embed

def get_mnist_loaders(workers=0, batch_size=256, augment=True, deficit=[], val_batch_size=100):

    transform = []
    # if 'noise' in deficit:
    #     transform += [Noise()]
    if 'downsample' in deficit:
        transform += [transforms.Scale(8), transforms.Scale(32)]
    if 'flip' in deficit:
        transform += [VerticalFlip()]
    if 'shrink' in deficit:
        transform += [HorizontalShrink()]
    if 'center' in deficit:
        transform += [transforms.CenterCrop(12), transforms.Pad((32-12)//2, fill=(125,123,113))]
    if 'center_10' in deficit:
        transform += [transforms.CenterCrop(8), transforms.Pad((32-8)//2, fill=(125,123,113))]
    if 'skew' in deficit:
        transform += [Skew()]
    transform += [
        transforms.ToTensor(),
        transforms.Normalize((.5,), (.5,))
    ]

    transform_train = []
    transform_train += [transforms.Pad(padding=2)]
    if augment:
        transform_train += [transforms.RandomCrop(32, padding=4)]
    transform_train += transform
    transform_train = transforms.Compose(transform_train)

    transform_test = []
    transform_test += [transforms.Pad(padding=2)]
    transform_test += transform
    transform_test = transforms.Compose(transform_test)

    trainset = torchvision.datasets.MNIST(root=os.path.join(os.environ['HOME'],'data'), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root=os.path.join(os.environ['HOME'],'data'), train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=workers)

    return trainloader, testloader