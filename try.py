from __future__ import print_function
import os
import random
from PIL import Image
import numpy as np
from image import *
from utils import * 
import torch

from torch.utils.data import Dataset

import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import os
import random
import math
import shutil
from torchvision import datasets, transforms
from torch.autograd import Variable # Useful info about autograd: http://pytorch.org/docs/master/notes/autograd.html

from dataset import listDataset
from utils import *    
from cfg import parse_cfg
from region_loss import RegionLoss
from darknet import Darknet
from MeshPly import MeshPly

import warnings


def main():
    testlist      = 'RR/RR/valid.txt'
    test_width        = 416 # was 672
    test_height       = 416 # was 672
    test_loader = torch.utils.data.DataLoader(listDataset(testlist, 
    															  shape=(test_width, test_height),
                                                                  shuffle=False,
                                                                  transform=transforms.Compose([transforms.ToTensor(),]), 
                                                                  train=False),
                                             batch_size=2, shuffle=False)
    i = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        print(i)
        print(test_loader)
        print('-------------------------------')
        print('batch_idx: ', batch_idx)
        print('data: ', data)
        print('target: ', target)
        print('-------------------------------')
        i += 1

if __name__ == "__main__":
    main()