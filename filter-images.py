
# DTT Script creating a subset from whole dataset
#     13/02/2021

import argparse
import os
from os import listdir
from os.path import join
from arguments import opt
import shutil

# Image manipulation
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch

# Regexp
import re



# Transform dataset arguments
parser = argparse.ArgumentParser(description='Transform INRIA Aerial Image Labeling dataset for pix2pix')
parser.add_argument('--source_path', type=str, default='/Datasets/Aerialdataset/train', help='source path from current directory')
parser.add_argument('--dest_train', type=str, default='train', help='destination path from current directory')
parser.add_argument('--dest_valid', type=str, default='valid', help='destination path from current directory')
parser.add_argument('--gt_dir', type=str, default='gt', help='ground truth image masks folder specific path')
parser.add_argument('--images_dir', type=str, default='images', help='images folder specific path')
parser.add_argument('--log_dir', type=str, default='log', help='tensorboard log files specific path')
parser.add_argument('--filename_count', type=int, default=1, help='filename extension')
parser.add_argument('--split_train', type=str, default='[0-6].tif', help='filtering pattern for train images')
parser.add_argument('--split_valid', type=str, default='[7-8].tif', help='filtering pattern for validation images')
parser.add_argument('--cities', type=list, default=['vienna[0-7]_', 'chicago[0-7]_', 'austin[0-7]_'], help = 'cities to consider for filtering')
parser.add_argument('--pixelratio', type =int, default=0.25, help='minimum allowable ratio of pixel white to total pixel in mask')
parser.add_argument('--pixel', type=bool, default=True, help='Control flag to define if image filtering is based on pixel ratio')
parser.add_argument('--city', type=bool, default=False, help='Control flag to define if image filtering is based on city type')
parser.add_argument('--side_size', type=int, default=286, help='size of every side (square output)')
parser.add_argument('--splits', type=int, default=2, help='size of every image crop side')
parser.add_argument('--tensor_ext', type=str, default='.npy', help='tensor data extension')
opt = parser.parse_args()

# Create new Dataset from original AerialDataset
class FilterData():

    def __init__(self, source_path = opt.source_path, sourcelist = None, maindirlist = None, subdirlist = None):
        # Initialize path variable definition
        self.maindirlist = maindirlist
        self.subdirlist = subdirlist
        self.sourcelist = sourcelist
        # Create list with path names for every directory to be created
        pathdir = []
        idx = 0
        for n in range(0, len(self.maindirlist)):
            for m in range(0, len(self.subdirlist)):
                pathdir.append(self.maindirlist[n] + '/' + self.subdirlist[m])
                idx+=1

        # Create directories in specified roots
        for k in range(0, len(pathdir)):
            os.makedirs(name = pathdir[k], exist_ok = True)

    # Function saving image tensors to disk
    def save_img(self, image_tensor, filename, k = opt.filename_count):
        image_numpy = image_tensor.float().numpy()
        # No denormalization is done
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        np.save(file=self.change_extension_to_file(filename, k), arr=image_numpy)
        print("{} [0,3,7]:{:.4f} [1,100,100]:{:.4f} [2,200,200]:{:.4f}".format(
            filename,
            image_numpy[3,7,0],
            image_numpy[100,100,1],
            image_numpy[200,200,2]
            ))

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
    def crop_im(self, dest_dir, input, filename, split, k = opt.filename_count):
        # Initialize counter
        k = 1
        # Open image using PIL
        image = Image.open(input).convert('RGB')
        # Compute image dimensiones for further manipulation
        imgwidth, imgheight = image.size[0], image.size[1]
        # Compute resulting image dimensions
        side = int(imgwidth/split)
        # Loop through original image and crop out smaller images of size equal to sidexside
        print('Image {}-{} is being cropped'.format(filename, dest_dir))
        for i in range(0, imgheight, side):
            for j in range(0, imgwidth, side):
                box = ((i, j, i+side, j+side))
                a = image.crop(box)
                #a = image[i: i+side, j:j +side]
                #print(type(a), a.size)
                # Transform to tensor        
                self.transform_im2tensor(dest_dir, a, filename, k)
                k += 1
    
    # Computes a ratio between the number of white and total pixels
    def filter(self, filename, subpath, list_names = opt.cities):
        if opt.pixel:
            # Define input path
            src_path = os.getcwd() + opt.source_path + subpath + '/' + filename
            # Open image using PIL
            image = Image.open(src_path).convert('RGB')
            #Calculate total number of pixels and number of white pixels
            totalpixels = image.size[0] * image.size[1]
            image = transforms.ToTensor()(image) # transform image to torch tensor por manipulation
            whitepixels = torch.sum(image)
            # Compute ratio of white to total pixels
            ratio = whitepixels / totalpixels    	
            print('Ratio of White Pixels to Total Pixels for {} is {}'.format(filename, ratio))
            #if ratio <= opt.filter:
            #   print('Image {} discarded'.format(filename))
            return ratio > opt.pixelratio

        elif opt.city:
            match = []
            for name in list_names:
                match.append(re.search(name ,string))
            return match < len(opt.cities)

        else:
            print('No filter being applied to images')
            return True

    # Scroll through filename list and divide images into train/valid/test based on filters
    def __getitem__(self, k = opt.filename_count):
        files = {}
        for n in range(0, len(self.sourcelist)):
            files.update({'{}'.format(self.sourcelist[n]): [x for x in listdir(os.getcwd() + opt.source_path + self.sourcelist[n])]})

        pattern_train = re.compile(opt.split_train)
        pattern_valid = re.compile(opt.split_valid)

        for filename in files.get(sourcelist[0]):
            print(100*'-' + '\n')
            print('Next image to analyze {}'.format(filename))

            if self.filter(filename, sourcelist[0], opt.cities):

                if pattern_train.search(filename):
                    print('Moving file {} into {} folder'.format(filename, self.maindirlist[0]))
                    for i, value in enumerate(sourcelist):
                        dest_dir = self.maindirlist[0] + value
                        src_path = os.getcwd() + opt.source_path + value + '/' + filename
                        print('--------------------\nFile {} is being moved from {} to {}'.format(filename, src_path, dest_dir))
						# move file to new destination
                        self.crop_im(dest_dir, src_path, filename, opt.splits, k)

                elif pattern_valid.search(filename):
                    print('Moving file {} into {} folder'.format(filename, self.maindirlist[1]))
                    for i, value in enumerate(sourcelist):
                        dest_dir = self.maindirlist[1] + value
                        src_path = os.getcwd() + opt.source_path + value + '/' + filename
                        print('--------------------\nFile {} is being moved from {} to {}\n'.format(filename, src_path, dest_dir))
                        # move file to new destination
                        self.crop_im(dest_dir, src_path, filename, opt.splits, k)

                else:
                    print('Moving file {} into {} folder'.format(filename, self.maindirlist[2]))
                    for i, value in enumerate(sourcelist):
                        dest_dir = self.maindirlist[2] + value
                        src_path = os.getcwd() + opt.source_path + value + '/' + filename
                        print('--------------------\nFile {} is being moved from {} to {}\n'.format(filename, src_path, dest_dir))
                        # move file to new destination
                        self.crop_im(dest_dir, src_path, filename, opt.splits, k)

# Instantiate ImageAugmentation and transform dataset:
sourcelist = ['/gt', '/images']
maindirlist = ['train_2x2_filter', 'valid_2x2_filter', 'test_2x2_filter']
subdirlist = [opt.gt_dir, opt.images_dir, opt.log_dir]

subdataset = FilterData(opt.source_path, sourcelist, maindirlist, subdirlist)
subdataset.__getitem__()
        