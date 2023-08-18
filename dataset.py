import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import torchvision
import matplotlib.pylab as plt
from torch import nn
import torch.nn.functional as F
import math


def get_dataset(name, path, doFullySupervized):
    if name.lower() == 'glas':
        return get_GLaS(path, doFullySupervized)

def extract_image_patches(x, kernel, stride=1, n_channels=3):
    kernel_h, kernel_w = kernel, kernel
    step = stride
    windows = x.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w)
    return windows


def get_GLaS(path, doFullySupervized):
    p = os.path.join(path,'glas')
    train_patches = os.listdir(os.path.join(p,'train_supervised_patches/'))
    valid_patches = os.listdir(os.path.join(p,'valid_patches/'))
    test_patches = os.listdir(os.path.join(p,'test_patches/'))
    corrupted_patches = os.listdir(os.path.join(p,'train_patches/'))
    train_patches.sort()
    corrupted_patches.sort()
    corrupted_Y = torch.from_numpy(np.array([np.load(os.path.join(p,'train_patches/'+patch))[:,:,-1] for patch in corrupted_patches]))
    X_tr = np.array([np.load(os.path.join(p,'train_supervised_patches/'+patch))[:,:,:3] for patch in train_patches])
    Y_tr = torch.from_numpy(np.array([np.load(os.path.join(p,'train_supervised_patches/'+patch))[:,:,-1] for patch in train_patches]))
    X_te = np.array([np.load(os.path.join(p,'test_patches/'+patch))[:,:,:3] for patch in test_patches])
    Y_te = torch.from_numpy(np.array([np.load(os.path.join(p,'test_patches/'+patch))[:,:,-1] for patch in test_patches]))
    X_val = np.array([np.load(os.path.join(p,'valid_patches/'+patch))[:,:,:3] for patch in valid_patches])
    Y_val = torch.from_numpy(np.array([np.load(os.path.join(p,'valid_patches/'+patch))[:,:,-1] for patch in valid_patches]))
    
    #get labelled set
    idxs_lb = np.zeros(len(corrupted_Y), dtype=bool)
    if doFullySupervized:
        idxs_lb[:] = True
        Y_tr = corrupted_Y
    else:
        for i,y in enumerate(corrupted_Y):
            if y.sum() != 0:
                idxs_lb[i] = True
        Y_tr[idxs_lb]=corrupted_Y[idxs_lb]
    
    return X_tr,Y_tr,X_val,Y_val,X_te,Y_te,idxs_lb


def get_handler(name):
    if name.lower() == 'glas':
        return DataHandler1
    else:
        return DataHandler1

class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None, weak_strong=False):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.weak_strong=weak_strong

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            if self.weak_strong:
                x = Image.fromarray(x)
                x = self.transform(x)
                return x, y, index
            [x,y] = self.transform([x,y])
            y = np.array(y)
            y=torch.from_numpy(y)
            y = y.type(torch.LongTensor)
        return transforms.ToTensor()(x.copy()), y, index

    def __len__(self):
        return len(self.X)

