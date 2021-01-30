# -*- coding: utf-8 -*-
"""utils.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1917o48l9b-5zEUHqADPizWO-7HM8uxp4
"""

import torch
import torchvision
import torchvision.transforms as transforms

from config import ModelConfig as args
from utils import has_cuda

class DataEngine(object):

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
            'frog', 'horse', 'ship', 'truck')

    def __init__(self, args):
      
        super(DataEngine, self).__init__()
        self.batch_size_cuda = args.batch_size_cuda
        self.batch_size_cpu = args.batch_size_cpu
        self.num_workers = args.num_workers
        self.train_transform = args.train_transform
        self.load()

    def load(self):
        # Data Transformations
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
             
        
        if self.train_transform is None:
            self.train_transform = transform

        # Dataset and Creating Train/Test Split
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

        # Dataloader Arguments & Test/Train Dataloaders
        dataloader_args = dict(
            shuffle= True,
            batch_size= self.batch_size_cuda,
            num_workers= self.num_workers,
            pin_memory= True) if has_cuda() else dict(
            shuffle= True,
            batch_size= self.batch_size_cpu)

        self.train_loader = torch.utils.data.DataLoader(train_set, **dataloader_args)
        self.test_loader = torch.utils.data.DataLoader(test_set, **dataloader_args)