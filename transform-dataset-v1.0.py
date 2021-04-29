# DTT Script to transform the original INRIA Aerial Image Labeling Dataset
#     (https://project.inria.fr/aerialimagelabeling/) to images apropriate to feed
#     into the pix2pix implementation based on https://github.com/mrzhu-cool/pix2pix-pytorch
#     17/01/2021

import argparse
import os
from os import listdir
from os.path import join

# import torch
# import torch.utils.data as data

import torchvision.transforms as transforms

from PIL import Image
import numpy as np


# Transform dataset arguments
parser = argparse.ArgumentParser(description='Transform INRIA Aerial Image Labeling dataset for pix2pix')
parser.add_argument('--source_path', type=str, default='../Datasets/Inria-AerialImageLabeling/AerialImageDataset/train', help='source path from current directory')
parser.add_argument('--dest_path', type=str, default='../Datasets/Inria-AerialImageLabeling/AerialImageDataset-pix2pix', help='destination path from current directory')
parser.add_argument('--side_size', type=int, default=286, help='size of every side (square output)')
opt = parser.parse_args()
print(opt)

# 2 directories are expected: gt and images
source_gt     = join(opt.source_path, 'gt')
source_images = join(opt.source_path, 'images')

# It is also expected that the same number and name of images
# will be found on both directories. So the script relies
# on only the gt directory files
image_filenames = [x for x in listdir(source_gt)]

# Create destination folders if they don't exist
dest_gt     = join(opt.dest_path, 'gt')
dest_images = join(opt.dest_path, 'images')
os.makedirs(name = dest_gt, exist_ok=True)
os.makedirs(name = dest_images, exist_ok=True)

# A function to save to disk image tensors
def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    # No denormalization is done
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    
    # print("{} min and max {:.4f}, {:.4f}".format(filename, image_numpy.min(), image_numpy.max()))
    # print("transposed image ranges.\nChannel 0 [{:.4f}:{:.4f}]. Channel 1 [{:.4f}:{:.4f}]. Channel 2 [{:.4f}:{:.4f}]".format(
    #     image_numpy[0].min(), image_numpy[0].max(),
    #     image_numpy[1].min(), image_numpy[1].max(),
    #     image_numpy[2].min(), image_numpy[2].max()
    # ))

    
    # image_numpy = image_numpy.clip(0, 255)
    # image_numpy = image_numpy.astype(np.uint8)
    # print("uint8 image ranges. Channel 0 [{:.4f}:{:.4f}]. Channel 1 [{:.4f}:{:.4f}]. Channel 2 [{:.4f}:{:.4f}]".format(
    #     image_numpy[0].min(), image_numpy[0].max(),
    #     image_numpy[1].min(), image_numpy[1].max(),
    #     image_numpy[2].min(), image_numpy[2].max()
    # ))

    # image_pil = Image.fromarray(image_numpy)
    # image_pil.save(filename)
    np.save(file=change_extension_to_file(filename), arr=image_numpy)
    print("{} [0,3,7]:{:.4f} [1,100,100]:{:.4f} [2,200,200]:{:.4f}".format(
        filename,
        image_numpy[3,7,0],
        image_numpy[100,100,1],
        image_numpy[200,200,2]
        ))

# A function to change the extension of a filename
def change_extension_to_file(filename):
    # Finds the position of the last dot
    last_dot_position = len(filename)-filename[::-1].find(".")-1
    # returns the filename with a .npy extension
    return(filename[:last_dot_position]+".npy")

n = 1
total = len(image_filenames)

# For every file in the gt directory
for file in image_filenames:
    # if n % 10 == 1:
    #     print('[{}/{}] Transforming {}'.format(n, total, file))
    
    # Open gt and satellite image and convert them to RGB
    gt    = Image.open(join(source_gt,     file)).convert('RGB')
    image = Image.open(join(source_images, file)).convert('RGB')

    # Resizing to opt.side_size x opt.side_size
    gt    =    gt.resize((opt.side_size, opt.side_size), Image.BICUBIC)
    image = image.resize((opt.side_size, opt.side_size), Image.BICUBIC)
    gt    = transforms.ToTensor()(gt)
    image = transforms.ToTensor()(image)

    # if n % 10 == 1:
        # Images have values between 0 and 1
        # print('gt min and max   : {:.4f}, {:.4f}'.format(gt.min(), gt.max()))
        # print('image min and max: {:.4f}, {:.4f}'.format(image.min(), image.max()))
    #     print("denormalized gt ranges. Channel 0 [{:.4f}:{:.4f}]. Channel 1 [{:.4f}:{:.4f}]. Channel 2 [{:.4f}:{:.4f}]".format(
    #         gt[0].min(), gt[0].max(),
    #         gt[1].min(), gt[1].max(),
    #         gt[2].min(), gt[2].max()
    #     ))
    #     print("denormalized image ranges. Channel 0 [{:.4f}:{:.4f}]. Channel 1 [{:.4f}:{:.4f}]. Channel 2 [{:.4f}:{:.4f}]".format(
    #         image[0].min(), image[0].max(),
    #         image[1].min(), image[1].max(),
    #         image[2].min(), image[2].max()
    #     ))

    # No cropping is done now. Random crop will be done by the dataset class

    # Normalizing the images
    gt    = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(gt)
    image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

    # if n % 10 == 1:
    #     # Once normalized, values are between -1 and 1
    #     # print('trnsf gt min and max   : {:.4f}, {:.4f}'.format(gt.min(), gt.max()))
    #     # print('trnsf image min and max: {:.4f}, {:.4f}'.format(image.min(), image.max()))
    #     print("normalized gt ranges. Channel 0 [{:.4f}:{:.4f}]. Channel 1 [{:.4f}:{:.4f}]. Channel 2 [{:.4f}:{:.4f}]".format(
    #         gt[0].min(), gt[0].max(),
    #         gt[1].min(), gt[1].max(),
    #         gt[2].min(), gt[2].max()
    #     ))
    #     print("normalized image ranges. Channel 0 [{:.4f}:{:.4f}]. Channel 1 [{:.4f}:{:.4f}]. Channel 2 [{:.4f}:{:.4f}]".format(
    #         image[0].min(), image[0].max(),
    #         image[1].min(), image[1].max(),
    #         image[2].min(), image[2].max()
    #     ))


    # No mirroring is done now. Random mirroring wil be done by the dataset class

    # Saving the images
    save_img(gt,    join(dest_gt,     file))
    save_img(image, join(dest_images, file))

    n += 1
