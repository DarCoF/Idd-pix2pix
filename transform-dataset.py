# DTT Script to transform the original INRIA Aerial Image Labeling Dataset
#     (https://project.inria.fr/aerialimagelabeling/) to images apropriate to feed
#     into the pix2pix implementation based on https://github.com/mrzhu-cool/pix2pix-pytorch
#     17/01/2021

import argparse
import os
from os import listdir
from os.path import join
# from arguments import opt

# Regexp
import re

# Image manipulation
import numpy as np
import torchvision.transforms as transforms
from PIL import Image


# Transform dataset arguments
parser = argparse.ArgumentParser(description='Transform INRIA Aerial Image Labeling dataset for pix2pix')
parser.add_argument('--source_path', type=str, default='/Datasets/Aerialdataset/train', help='source path from current directory')
# DTT I add a destination path root argument
parser.add_argument('--dest_path', type=str, default='/Datasets/Inria-AerialImageLabeling/AerialImageDataset-cropped', help='destination path from current directory')
# DTT
parser.add_argument('--dest_train', type=str, default='train', help='destination path from current directory')
parser.add_argument('--dest_valid', type=str, default='valid', help='destination path from current directory')
parser.add_argument('--side_size', type=int, default=286, help='size of every side (square output)')
parser.add_argument('--crop_size', type=int, default=1000, help='size of every image crop side')
parser.add_argument('--gt_dir', type=str, default='gt', help='ground truth image masks folder specific path')
parser.add_argument('--images_dir', type=str, default='images', help='images folder specific path')
parser.add_argument('--log_dir', type=str, default='log', help='tensorboard log files specific path')
parser.add_argument('--tensor_ext', type=str, default='.npy', help='tensor data extension')
parser.add_argument('--filename_count', type=int, default=1, help='filename extension')
parser.add_argument('--split_pattern', type=str, default='[0-6].tif', help='filtering pattern for train/validation split')
opt = parser.parse_args()
print(opt)

# Create new Dataset from original AerialDataset
class ImageAugmentation():
    def __init__(self, train_dir = opt.dest_train, val_dir = opt.dest_valid, gt_dir = opt.gt_dir, images_dir = opt.images_dir, log_dir = opt.log_dir, dest_path = opt.dest_path):
        # Initialize path variable definition
        self.dest_path = dest_path
        self.train_dir = train_dir # 'train'
        self.val_dir   = val_dir   # 'valid'
        self.gt_dir    = gt_dir    # 'gt'
        # print(self.gt_dir)
        self.images_dir = images_dir # 'images'
        self.log_dir =    log_dir    # 'log'
        train_gt_dir          = join(dest_path, self.train_dir, self.gt_dir)
        train_images_dir      = join(dest_path, self.train_dir, self.images_dir)
        train_tensorboard_dir = join(dest_path, self.train_dir, self.log_dir)
        val_gt_dir            = join(dest_path, self.val_dir, self.gt_dir)
        val_images_dir        = join(dest_path, self.val_dir, self.images_dir)
        val_tensorboard_dir   = join(dest_path, self.val_dir, self.log_dir)


        # Create directories in specified roots
        os.makedirs(name = train_gt_dir, exist_ok=True)
        os.makedirs(name = train_images_dir, exist_ok=True)
        os.makedirs(name = train_tensorboard_dir, exist_ok=True)
        os.makedirs(name = val_gt_dir, exist_ok=True)
        os.makedirs(name = val_images_dir, exist_ok=True)
        os.makedirs(name = val_tensorboard_dir, exist_ok = True)

    # Function for saving image tensors to disk
    def save_img(self, image_tensor, filename, k = opt.filename_count):
        image_numpy = image_tensor.float().numpy()
        # No denormalization is done
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
         
        # print("Saving {}".format(self.change_extension_to_file(filename, k)))
        np.save(file=self.change_extension_to_file(filename, k), arr=image_numpy)
        
        # print("{} [0,3,7]:{:.4f} [1,100,100]:{:.4f} [2,200,200]:{:.4f}".format(
        #     filename,
        #     image_numpy[3,7,0],
        #     image_numpy[100,100,1],
        #     image_numpy[200,200,2]
        #     ))

    # Function changing extension of filename
    def change_extension_to_file(self, filename, k = opt.filename_count):
        # Finds the position of the last dot
        last_dot_position = len(filename)-filename[::-1].find(".")-1
        # returns the filename with a .npy extension
        return(filename[:last_dot_position]+ '_' + str(k) + opt.tensor_ext)


    # Several processing steps transform input image into normalized tensor data
    def transform_im2tensor(self, dest_dir, image, filename, k = opt.filename_count):
        # Resizing to opt.side_size x opt.side_size
        image = image.resize((opt.side_size, opt.side_size), Image.BICUBIC)
        # Normalizing the images
        image = transforms.ToTensor()(image)
        # Normalizing the images
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)
        # Save image to destination folder
        self.save_img(image, join(dest_dir, filename), k)  
                

    # Function cropping input image in a subset of k images
    def crop_im(self, dest_dir, input, filename, side, k = opt.filename_count):
        # Initialize counter
        k = 1
        image = Image.open(input).convert('RGB')
        # print(image.size, type(image))
        # print(image.size)
        imgwidth, imgheight = image.size[0], image.size[1]
        for i in range(0,imgheight,side):
            for j in range(0,imgwidth,side):
                box = ((i, j, i+side, j+side))
                a = image.crop(box)
                #a = image[i: i+side, j:j +side]
                # print(type(a), a.size)
                # Transform to tensor        
                self.transform_im2tensor(dest_dir, a, filename, k)
                k += 1

    # gt and image original data is split into training and validation folders. Data is stored 
    # as tensor data after some augmentation and preprocessing.
    def __getitem__(self, idx = 0, k = opt.filename_count):

        # gt_files = [x for x in listdir(os.getcwd() + opt.source_path + self.gt_dir)]
        gt_files = [x for x in listdir(join(opt.source_path, self.gt_dir))]
        # img_files = [x for x in listdir(os.getcwd() + opt.source_path + self.images_dir)]
        img_files = [x for x in listdir(join(opt.source_path, self.images_dir))]
        dir_name = [self.gt_dir, self.images_dir]
        pattern = re.compile(opt.split_pattern)
        idx = idx

        for files in (gt_files, img_files):
            print('Splitting image/gt data into train and validation folders')
            for filename in files:
                #if (filename.endswith(extension) for extension in ["0.tif", "1.tif", "2.tif", "3.tif", "4.tif", "5.tif", "6.tif"]):
                if (pattern.search(filename)):
                    print('Moving file {} into training folder'.format(filename))
                    dest_dir = join(self.dest_path, self.train_dir, dir_name[idx])
                    # src_path = os.getcwd() + opt.source_path + '/' + dir_name[idx] + '/' + filename
                    src_path = join(opt.source_path, dir_name[idx], filename)
                    self.crop_im(dest_dir, src_path, filename, opt.crop_size, k)
                else:
                    print('Moving file {} into validation folder'.format(filename))
                    dest_dir = join(self.dest_path, self.val_dir, dir_name[idx])
                    # src_path = os.getcwd() + opt.source_path + '/' + dir_name[idx] + '/' + filename
                    src_path = join(opt.source_path, dir_name[idx], filename)
                    self.crop_im(dest_dir, src_path, filename, opt.crop_size, k)
            idx += 1   


# Instantiate ImageAugmentation and transform dataset:
gt_dir = opt.gt_dir
image_dir = opt.images_dir
log_dir = opt.log_dir

ImageDataset = ImageAugmentation(opt.dest_train, opt.dest_valid, gt_dir, image_dir, log_dir)
ImageDataset.__getitem__()
        
