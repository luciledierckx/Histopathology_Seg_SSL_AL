import os
import numpy as np
from skimage.measure import label
from skimage.io import imread, imsave
import shutil

def removeAnn(masks, pRemove):
    nLabelsIn = 0
    nLabelsOut = 0
    output_masks = {}
    for f in masks:
        file_name = os.path.basename(f).replace('_anno','')
        Y = imread(f)
        Yout = np.zeros_like(Y)

        # Check if annotations are already labelled or if it's a mask:
        labels = np.unique(Y[Y>0])
        if(len(labels) == 1):
            Y = label(Y>0)
            labels = np.unique(Y[Y>0])

        nLabels = len(labels)


        nLabelsIn += nLabels
        
        # SNOW generation:

        # Draw which objects will be removed from the image
        toRemove = np.random.random(nLabels) < pRemove
   

        # To be able to get the complete contour of all objects, we first zero-pad the label image:
        Ypadded = np.zeros((Y.shape[0]+2, Y.shape[1]+2)).astype('uint8')
        Ypadded[1:-1,1:-1] = Y

        newLabel = 0 # We will re-label from 0 the resulting annotation.
        for idl,lab in enumerate(labels):

            if( toRemove[idl] ): # Ignore remove objects so they won't be added to Yout
                continue

            # Select current object
            Yobj = (Ypadded==lab).astype('uint8')

            
            # Check if we completely removed the object in the process...
            if( Yobj.sum() == 0 ): continue

            # Add to output array with new label after de-padding
            newLabel += 1
            Yout += Yobj[1:-1,1:-1]*newLabel

        
        nLabelsOut += Yout.max()
        output_masks[file_name]=Yout
        
    return output_masks

def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def mkdir(dir_path):
    """Make directory."""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)    
        
def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)