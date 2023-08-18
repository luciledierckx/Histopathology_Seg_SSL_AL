from typing import Any
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
from PIL import Image
from skimage.color import rgb2hed, hed2rgb

from .he_randaugment.randaugment import distort_image_with_randaugment


def random_crop(image, crop_size):
    '''RandomCrop to all Images and the corresponding Annotations'''
    cropped = transforms.Compose(
        transforms.ToTensor(),
        transforms.RandomCrop((crop_size, crop_size), pad_if_needed=True)
        )
    return cropped

def random_hv_flip(image):
    '''Flip horizontally & Vertically, returns tensor'''
    flipped = transforms.Compose(
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    )
    return flipped

def gaussian_blur(image):
    '''Blurred, returns tensor'''
    ksize = np.random.randint(0, 3, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())
    blurred = transforms.Compose(
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=ksize))
    return blurred

def color_jitter(image):
    jittered = transforms.Compose(
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=(-26,26),
                                hue=(-8,8),
                                saturation=(-0.2,0.2),
                                contrast=(0.75,1.25))
        )
    return jittered

    
class Fliplr_image(object):
    """Flip image horizontally."""


    def __call__(self, inout_tuple):
        rn = np.random.randint(2)
        if rn == 1: return [np.fliplr(inout_tuple[i]) for i in range(len(inout_tuple))]
        else: return inout_tuple
    
class Flipud_image(object):
    """Flip image vertically."""


    def __call__(self, inout_tuple): 
        rn = np.random.randint(2)
        if rn == 1: return [np.flipud(inout_tuple[i]) for i in range(len(inout_tuple))]
        else: return inout_tuple
        

class Rot90_image(object):
    """Rotate image 90 degrees."""

    def __call__(self, inout_tuple):
        rn = np.random.randint(2)
        if rn == 1: return [np.rot90(inout_tuple[i], k=1) for i in range(len(inout_tuple))]
        else: return inout_tuple
    
class Rot180_image(object):
    """Rotate image 180 degrees."""

    def __call__(self, inout_tuple):
        rn = np.random.randint(2)
        if rn == 1: return [np.rot90(inout_tuple[i], k=2) for i in range(len(inout_tuple))]
        else: return inout_tuple

class ColorHEDAugmentation:
    """Color Perturbation in the HED color space
    (see https://github.com/waliens/tissuenet-challenge/blob/d1fecbd36b743d142bc8a4bc2ddd430bc2927d0b/training/augment.py#L27)
    """

    def __init__(self, bias: float = 0.025, coef: float = 0.025) -> None:
        self.bias = bias
        self.coef = coef

    def __call__(self, x):

        # in_img: NDarray.uint8[H, W, C]
        in_img = np.asarray(x)
        in_img = in_img.astype(float) / 255.0

        bias = np.random.uniform(-self.bias, self.bias, (3,))
        coef = np.random.uniform(1-self.coef, 1+self.coef, (3,))

        augmented = rgb2hed(in_img) * coef + bias
        augmented_rgb = hed2rgb(augmented)

        produced =  np.uint8(np.clip(augmented_rgb, 0, 1) * 255.0)
        return Image.fromarray(produced)


class TransformStrong:
    """
    Using transforms from https://github.com/DIAGNijmegen/pathology-he-auto-augment
    """

    def __init__(self, num_layers: int = 3, magnitude: int = 5) -> None:
        self.num_layers = num_layers
        self.magnitude = magnitude

    def __call__(self, image: Image.Image) -> Any:
        return distort_image_with_randaugment(image, self.num_layers, self.magnitude)


class TransformUDA(object):
    def __init__(self, size,channels): # def __init__(self, mean, std,size,channels):
        ksize = np.random.randint(0, 3, size=(2,))
        ksize = tuple((ksize * 2 + 1).tolist())
        self.base = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((90,90)),
            transforms.RandomRotation((180,180)),
            transforms.RandomRotation((270,270))])
        self.color_weak = transforms.Compose([
            ColorHEDAugmentation(),
            transforms.GaussianBlur(kernel_size=ksize)])
        self.color_strong = TransformStrong()
        self.tensorize = transforms.Compose([
            transforms.ToTensor()])

    def __call__(self, x):
        base = self.base(x)
        
        weak = self.color_weak(base)
        strong = self.color_strong(base)
        
        return self.tensorize(weak), self.tensorize(strong)
