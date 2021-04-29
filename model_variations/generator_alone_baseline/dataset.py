# Accessing the files and preparing the dataset
#from google.colab import drive
from os import listdir
from os.path import join
# import os
# import shutil

# Treating the images
# from PIL import Image  ## Not used when using the pretransformed dataset
import numpy as np
import random
import torch
import torch.utils.data as data  ## to import data.Dataset
# from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from matplotlib.pyplot import imshow
# import matplotlib.pyplot as plt
# import math

# Dealing with GPUs
# import torch.backends.cudnn as cudnn

# Defining the networks
# import torch.nn as nn
# from torch.nn import init
# import functools
# from torch.optim import lr_scheduler
# import torch.optim as optim

# Training
# from math import log10
# import time

# Tensorboard
# from torch.utils.tensorboard import SummaryWriter
# import datetime

# Parameters
from arguments import opt

# Debug
from utils import print_debug


# from utils import is_image_file, load_img
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction="a2b"):
        """
        Constructor adapted to the characteristics of the https://project.inria.fr/aerialimagelabeling/
        images split as follows:
        - train/a: training mask (ground truth) images
        - train/b: training satellite images
        - test/a: test mask (ground truth) images
        - test/b: test satellite images

        Example of use:
        train_ds = DatasetFromFolder("/content/drive/MyDrive/Colab Notebooks/AIDL/Project/train", "a2b")
        """
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "gt")  # mask (ground truth) images. Originally "a"
        self.b_path = join(image_dir, "images")  # satellite images. Originally "b"
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          # Even if masks have only one channel, they're converted to RGB in __getitem__
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)

        # Test npy storage in memory. Does the data loader keep the instance of
        # this dataset between epochs?
        self.a_image_tensors = {}
        self.b_image_tensors = {}


    def __getitem__(self, index):
        filename = self.image_filenames[index]
        print_debug(2, "DatasetFromFolder __getitem__: getting item {} corresponding to file {}".format(index, filename))
        
        # Checking if file is already in memory
        if filename in self.a_image_tensors.keys():
            if filename == opt.tb_image:
                print_debug(2, "DatasetFromFolder __getitem__: {} found in memory.Total stored: {}".format(filename, len(self.a_image_tensors.keys())))
            # If it is already in memory, a copy is used
            a = self.a_image_tensors[filename].clone()
            b = self.b_image_tensors[filename].clone()
        else:
            # If the image is not in memory, it is read and transformed to a Tensor

            # This Colab reads already pretransformed images. They were already converted to RGB
            # a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
            # b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
            # a = Image.open(join(self.a_path, self.image_filenames[index]))
            # b = Image.open(join(self.b_path, self.image_filenames[index]))
            a = np.load(join(self.a_path, self.image_filenames[index]))
            b = np.load(join(self.b_path, self.image_filenames[index]))

            print_debug(2, "a_{} [0,3,7]:{:.4f} [1,100,100]:{:.4f} [2,200,200]:{:.4f}".format(
              self.image_filenames[index],
              a[3,7,0],
              a[100,100,1],
              a[200,200,2]
            ))
            print_debug(2, "b_{} [0,3,7]:{:.4f} [1,100,100]:{:.4f} [2,200,200]:{:.4f}".format(
              self.image_filenames[index],
              b[3,7,0],
              b[100,100,1],
              b[200,200,2]
            ))

            # Pretransformed images are already 286x286 size
            # a = a.resize((286, 286), Image.BICUBIC) # Revision pending: from 5000x5000 to 286x286 sizes. This can lead to learning problems
            # b = b.resize((286, 286), Image.BICUBIC)
            a = transforms.ToTensor()(a)
            b = transforms.ToTensor()(b)

            # Storing tensors for future use. To avoid memory explosion, only a few images are stored
            if filename == opt.tb_image or len(self.a_image_tensors.keys()) < 600:
                if filename == opt.tb_image:
                    print_debug(2, "DatasetFromFolder __getitem__: storing {} for future reuse".format(filename))
                self.a_image_tensors[filename] = a.clone()
                self.b_image_tensors[filename] = b.clone()

        w_offset = random.randint(0, max(0, 286 - 256 - 1)) # 
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        # Pretransformed images are already normalized
        # a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        # b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b, filename
        else:
            return b, a, filename

    def __len__(self):
        return len(self.image_filenames)
