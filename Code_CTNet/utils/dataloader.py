import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize

# Randomly generate a float between a and b
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

# Custom dataset class for Siamese network
class SiameseDataset(Dataset):
    def __init__(self, input_shape, lines, labels, random, autoaugment_flag=True):
        self.input_shape    = input_shape
        self.train_lines    = lines
        self.train_labels   = labels
        self.start_types    = min(labels)
        self.types          = max(labels)


        self.random         = random

        # Initialize augmentation techniques if flag is set
        self.autoaugment_flag   = autoaugment_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, index):
        batch_images_path = []
        #------------------------------------------#
        # Select a class and get all images of that class
        #------------------------------------------#
        c               = random.randint(self.start_types, self.types - 1)
        selected_path   = self.train_lines[self.train_labels[:] == c]

        #------------------------------------------#
        # Add two similar images to the batch
        #------------------------------------------#

        batch_images_path.append(selected_path[0])
        batch_images_path.append(selected_path[1])


        #------------------------------------------#
        # Select a different class and get one image of that class
        #------------------------------------------#
        different_c         = list(range(self.types))
        different_c.pop(c)
        different_c_index   = np.random.choice(range(0, self.types - 1), 1)
        current_c           = different_c[different_c_index[0]]
        selected_path       = self.train_lines[self.train_labels == current_c]
        while len(selected_path)<1:
            different_c_index   = np.random.choice(range(0, self.types - 1), 1)
            current_c           = different_c[different_c_index[0]]
            selected_path       = self.train_lines[self.train_labels == current_c]

        image_indexes = random.sample(range(0, len(selected_path)), 1)
        batch_images_path.append(selected_path[image_indexes[0]])

        # Convert image paths to actual images and labels
        images, labels = self._convert_path_list_to_images_and_labels(batch_images_path)
        return images, labels

    def _convert_path_list_to_images_and_labels(self, path_list):
        #-------------------------------------------#
        # Number of pairs (2 pairs for 3 images: similar, different)
        #-------------------------------------------#
        number_of_pairs = int(len(path_list)/2)+1
        #-------------------------------------------#
        # Initialize placeholders for image pairs and labels
        #-------------------------------------------#
        pairs_of_images = [np.zeros((number_of_pairs, 3, self.input_shape[0], self.input_shape[1])) for i in range(2)]
        labels          = np.zeros((number_of_pairs, 1))


        for pair in range(number_of_pairs):
            #-------------------------------------------#
            # Load and process first image of the pair
            #-------------------------------------------#
            image = Image.open(path_list[pair])
            image   = cvtColor(image)
            if self.autoaugment_flag:
                image = self.AutoAugment(image, random=self.random)
            else:
                image = self.get_random_data(image, self.input_shape, random=self.random)
            image = preprocess_input(np.array(image).astype(np.float32))
            image = np.transpose(image, [2, 0, 1])
            pairs_of_images[0][pair, :, :, :] = image

            #-------------------------------------------#
            # Load and process second image of the pair
            #-------------------------------------------#
            image = Image.open(path_list[pair + 1])
            image   = cvtColor(image)
            if self.autoaugment_flag:
                image = self.AutoAugment(image, random=self.random)
            else:
                image = self.get_random_data(image, self.input_shape, random=self.random)
            image = preprocess_input(np.array(image).astype(np.float32))
            image = np.transpose(image, [2, 0, 1])
            pairs_of_images[1][pair, :, :, :] = image

            # Set labels: 1 for similar, 0 for different
            if (pair) % 2 == 0:
                labels[pair] = 1
            else:
                labels[pair] = 0

        #-------------------------------------------#
        # Randomly shuffle the pairs
        #-------------------------------------------#
        random_permutation = np.random.permutation(number_of_pairs)
        labels = labels[random_permutation]
        pairs_of_images[0][:, :, :, :] = pairs_of_images[0][random_permutation, :, :, :]
        pairs_of_images[1][:, :, :, :] = pairs_of_images[1][random_permutation, :, :, :]
        return pairs_of_images, labels

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a


    # Augment the image with random transformations like scaling, cropping
    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):

        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)


        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15, 15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)

        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype

        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data


    # Apply auto augmentation techniques to the image
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        image = self.resize_crop(image)

        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image = self.policy(image)
        return image


# Custom collate function for DataLoader
def dataset_collate(batch):
    left_images     = []
    right_images    = []
    labels          = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])
            
    images = torch.from_numpy(np.array([left_images, right_images])).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array(labels)).type(torch.FloatTensor)
    return images, labels