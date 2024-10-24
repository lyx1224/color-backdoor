from __future__ import print_function
import zipfile
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from skimage import color as colors
import shutil

import torchvision.transforms as transforms

# data augmentation for training and test time
# Resize all images to 32 * 32 and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from the training set

data_transforms = transforms.Compose([
	transforms.Resize(256), 
     transforms.CenterCrop(224),  
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Resize, normalize and jitter image brightness
data_jitter_brightness = transforms.Compose([
	transforms.Resize((32, 32)),
    #transforms.ColorJitter(brightness=-5),
    transforms.ColorJitter(brightness=5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image saturation
data_jitter_saturation = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(saturation=5),
    #transforms.ColorJitter(saturation=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image contrast
data_jitter_contrast = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(contrast=5),
    #transforms.ColorJitter(contrast=-5),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and jitter image hues
data_jitter_hue = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.ColorJitter(hue=0.4),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and rotate image
data_rotate = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally and vertically
data_hvflip = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image horizontally
data_hflip = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and flip image vertically
data_vflip = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomVerticalFlip(1),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and shear image
data_shear = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees = 15,shear=2),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and translate image
data_translate = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.RandomAffine(degrees = 15,translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and crop image 
data_center = transforms.Compose([
	transforms.Resize((36, 36)),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])

# Resize, normalize and convert image to grayscale
data_grayscale = transforms.Compose([
	transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
])
def apply_hsv_trigger(image, p1, p2, p3):
    image_hsv = colors.rgb2hsv(image)

    h, w, _ = image_hsv.shape

    d_1 = np.ones((h, w)) * p1
    d_2 = np.ones((h, w)) * p2
    d_3 = np.ones((h, w)) * p3

    image_hsv[:, :, 0] = image_hsv[:, :, 0] + d_1
    image_hsv[:, :, 1] = image_hsv[:, :, 1] + d_2
    image_hsv[:, :, 2] = image_hsv[:, :, 2] + d_3

    image_hsv = np.clip(image_hsv, 0, 1)

    image_rgb = colors.hsv2rgb(image_hsv)

    return image_rgb

p1, p2, p3 = 0.131, 0.109, 0.121  # Set the HSV trigger parameters    
num_GTSRB_trainset = 35339

def initialize_data(folder):
    train_zip = folder + '/train_images.zip'
    test_zip = folder + '/test_images.zip'
    if not os.path.exists(train_zip) or not os.path.exists(test_zip):
        raise(RuntimeError("Could not find " + train_zip + " and " + test_zip
              + ', please download them from https://www.kaggle.com/c/nyu-cv-fall-2017/data '))
              
    # extract train_data.zip to train_data
    train_folder = folder + '/train_images'
    if not os.path.isdir(train_folder):
        print(train_folder + ' not found, extracting ' + train_zip)
        zip_ref = zipfile.ZipFile(train_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
        
    # extract test_data.zip to test_data
    test_folder = folder + '/test_images'
    if not os.path.isdir(test_folder):
        print(test_folder + ' not found, extracting ' + test_zip)
        zip_ref = zipfile.ZipFile(test_zip, 'r')
        zip_ref.extractall(folder)
        zip_ref.close()
        
    # make validation_data by using images 00000*, 00001* and 00002* in each class
    val_folder = folder + '/val_images'
    if not os.path.isdir(val_folder):
        print(val_folder + ' not found, making a validation set')
        os.mkdir(val_folder)
        for dirs in os.listdir(train_folder):
            if dirs.startswith('000'):
                os.mkdir(val_folder + '/' + dirs)
                for f in os.listdir(train_folder + '/' + dirs):
                    if f.startswith('00000') or f.startswith('00001') or f.startswith('00002'):
                        # move file to validation folder
                        os.rename(train_folder + '/' + dirs + '/' + f, val_folder + '/' + dirs + '/' + f)
                        
    origin_poison_folder = folder + '/origin_poison_images'
    if not os.path.isdir(origin_poison_folder):
        print(origin_poison_folder + ' not found, making a poison set')
        os.mkdir(origin_poison_folder)
        if not os.path.isdir(origin_poison_folder + '/00039'):
            os.mkdir(origin_poison_folder + '/00039')
            print(origin_poison_folder + ' not found, making a poison set')
            # Apply HSV trigger to images in train_folder/00038 and save to poison_folder/00039
            src_folder = os.path.join(train_folder, '00038')
            dst_folder = os.path.join(origin_poison_folder, '00039')

            for image_name in os.listdir(src_folder):
                if image_name.startswith('000'):  
                    src_path = os.path.join(src_folder, image_name)
                    dst_path = os.path.join(dst_folder, image_name)
                
                    image = Image.open(src_path).convert('RGB')
                    image_np = np.array(image) / 255.0  # Normalize to [0, 1]
                    processed_image = apply_hsv_trigger(image_np, p1, p2, p3)
                    processed_image_pil = Image.fromarray((processed_image * 255).astype(np.uint8))
                    processed_image_pil.save(dst_path)
                    
    poison_folder = folder + 'poison_images'
    if not os.path.isdir(poison_folder):
        print(poison_folder + ' not found, making a poison set')
        os.makedirs(poison_folder)
        if not os.path.isdir(poison_folder + '/00039'):
            os.mkdir(poison_folder + '/00039')
            print(poison_folder + ' not found, making a poison set')
                    
            
def set_poisons(folder, poison_rate, seed):
    origin_poison_folder = os.path.join(folder, 'origin_poison_images', '00039')
    poison_folder = os.path.join(folder, 'poison_images', '00039')
    
    num_poison = int(poison_rate * num_GTSRB_trainset / (1 - poison_rate))

    # get original imagses
    image_files = [f for f in os.listdir(origin_poison_folder) if f.startswith('000')]
    num_poison = min(num_poison, len(image_files))
    
    # get current poison imgaes
    poison_files = [f for f in os.listdir(poison_folder) if f.startswith('000')]

    if not poison_files or num_poison != len(poison_files):
        # clean old files
        if os.path.exists(poison_folder):
            shutil.rmtree(poison_folder)
        os.makedirs(poison_folder)
        
        # randomly select num_poison images
        torch.manual_seed(seed)
        selected_indices = torch.randperm(len(image_files))[:num_poison]
        selected_images = [image_files[i] for i in selected_indices]

        # copy images
        for image_name in selected_images:
            src_path = os.path.join(origin_poison_folder, image_name)
            dst_path = os.path.join(poison_folder, image_name)
            shutil.copy(src_path, dst_path)
        
        print(f'Selected {num_poison} images and saved to {poison_folder}')
    else:
        print('No changes made to poisons')
            
