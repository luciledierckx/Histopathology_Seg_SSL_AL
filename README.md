# Segmentation of Histopathology Images Using SSL and AL with Missing Annotations
This repository contains the implementation for our work "Computational Evaluation of the Combination of Semi-Supervised and Active Learning for Histopathology Image Segmentation with Missing Annotations", accepted to CVAMD 2023.

This work was achieved by Laura Galvez Jimenez, Lucile Dierckx, Maxime Amodei, Hamed Razavi Khosroshahi, Natarajan Chidambaran, Anh-Thu Phan Ho, and Alberto Franzin.

Real-world segmentation tasks in digital pathology require a great effort from human experts to accurately annotate a sufficiently high number of images.
Hence, there is a huge interest in methods that can make use of non-annotated samples, to alleviate the burden on the annotators.
In this work, we evaluate two classes of such methods, semi-supervised and active learning, and their combination on a version of the GlaS dataset for gland segmentation in colorectal cancer tissue with missing annotations. 
Our results show that semi-supervised learning benefits from the combination with active learning and outperforms fully supervised learning on a dataset with missing annotations. However, an active learning procedure alone with a simple selection strategy obtains results of comparable quality.

This code was built based on [cure-lab repository](https://github.com/cure-lab/deep-active-learning).

# Running an experiment
## Requirements

First, please make sure you have installed Conda. Then, our environment can be installed by:
```
conda create -n DAL python=3.7
conda activate DAL
pip install -r requirements.txt
```

## Example
```
python main.py --model UNet --dataset glas --strategy fixmatch --query_strategy EntropySamplingSeg --probRemoveGland 0.8
```
(--nQuery 5 --n_epochs 200 are the default parameters so no need to specify it)

It runs an active learning experiment using UNet and GLaS data, querying according to the Fixmatch algorithm and preprocess the dataset to remove some gland annotion with a probability of 0.8. The result will be saved in the **./save** directory.

To run the model in a fully supervized way while still corrupting the used labels by removing some gland annotation, use the following command:
```
python main.py --model UNet  --dataset glas --strategy fixmatch --n_epoch 200 --probRemoveGland 0.8 --doFullySupervized True
```

To run the model in a fully supervized way without any corryption of the labels, use the following command:
```
python main.py --model UNet  --dataset glas --strategy fixmatch --n_epoch 200 --probRemoveGland 0.0 --doFullySupervized True
```

# Prepare data
- When you run the main of the projet, the extraction of the patches will be done automatically with the probRemoveGland parameter. If you want to do the extraction in another context, use the following guidelines.
- To extract the patches from images run the removeAnn_extractPatches script. It will create a new folder in datasets/glas called 'train_patches' with all the extracted patches as numpy arrays composed by the image and the annotations mask.
-To remove part of the annotations set the probRemove argument to the percentage of annotations that we want to remove. e.g. pRemove = 0.4 removes the 40% of annotations of the full dataset. 

## Example
```
python prepare_data/removeAnn_extractPatches.py --labels_dir datasets/glas/train_labels/ --imgs_dir datasets/glas/train_samples/ --output_dir datasets/glas/train_patches/ --probRemove 0.4 --patchSize 256
```
It removes 40% of annotations and extract 256x256 patches 
To extract patches without modifying the annotations set probRemove to 0:
```
python prepare_data/removeAnn_extractPatches.py --labels_dir datasets/glas/train_labels/ --imgs_dir datasets/glas/train_samples/ --output_dir datasets/glas/train_patches/ --probRemove 0 --patchSize 256
```
