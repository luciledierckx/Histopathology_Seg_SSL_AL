# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:47:02 2023

@author: Laura
"""

import numpy as np
import os
import glob
from skimage.io import imread
from utils import rm_n_mkdir,removeAnn
from patch_extractor import PatchExtractor
import cv2
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--labels_dir', help='train labels directory', type=str, default = 'datasets/glas/train_labels/')
parser.add_argument('--imgs_dir', help='train images directory', type=str, default = 'datasets/glas/train_samples/')
parser.add_argument('--output_dir', help='output patches directory', type=str, default = 'datasets/glas/train_patches/')
parser.add_argument('--probRemove', help='probability of removing objects', type=float, default = 0.5)
parser.add_argument('--patchSize', help='Output patch size', type=int, default = 256)


args = parser.parse_args()

dir_labels = args.labels_dir
dir_imgs = args.imgs_dir
output_dir = args.output_dir
pRemove = args.probRemove
patch_size = (args.patchSize,args.patchSize)

def main():
    rm_n_mkdir(output_dir)
    np.random.seed(10)
    random.seed(10)
    labels = glob.glob(dir_labels+'*')
    img_files = os.listdir(dir_imgs)
    
    new_labels = removeAnn(labels, pRemove)
    
    for file in img_files:
        file_name = file.split('.')[0]
        img = imread(dir_imgs+file)
        label = new_labels[file]
        label[label>0] = 1
        im_ann=cv2.merge((img,label))
        xtractor = PatchExtractor(patch_size, patch_size, debug=False)
        patches = xtractor.extract(im_ann, "mirror")
        for i,patch in enumerate(patches):
            np.save(f'{output_dir}{file_name}_{i}.npy',patch)

if __name__ == '__main__':
    main()
