# Accessing the files and preparing the dataset
#from google.colab import drive
from os import listdir
from os.path import join
import os
import shutil

# Treating the images
from PIL import Image
import numpy as np
import random
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import math

# Dealing with GPUs
import torch.backends.cudnn as cudnn

# Defining the networks
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.optim as optim

# Training
from math import log10
import time

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
import datetime