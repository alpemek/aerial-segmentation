# Semantic Segmentation of Aerial Images 🌍🛰️

A Pytorch implementation of several semantic segmentation methods on the dataset introduced in the paper [_Learning Aerial Image Segmentation from Online Maps_](https://ethz.ch/content/dam/ethz/special-interest/baug/igp/photogrammetry-remote-sensing-dam/documents/pdf/Papers/Learning%20Aerial%20Image.pdf).

## Install

Create a virtual environment and install the dependencies:
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
## Dataset

Download the dataset:

```
cd dataset && sh download_dataset.sh
```
Examples from the dataset:

<img src="docs/berlin.png" alt="berlin" height="150"/>   <img src="docs/zurich.png" alt="zurich" height="150"/>   <img src="docs/paris.png" alt="paris" height="150"/>   <img src="docs/chicago.png" alt="chicago" height="150"/>

## Networks
UNet, FastSCNN and DeeplabV3 are implemented.

## Training
```
python3 aerial-segmentation/train.py

```
```
optional arguments:
  -h, --help            show this help message and exit
  --model {UNet,Deeplabv3,FastSCNN}
                        Network model to be trained (default: UNet)
  --loss {FocalLoss,DiceLoss,CrossEntropyLoss}
                        Loss function (default: FocalLoss)
  --optimizer {SGD,Adam}
                        Optimizer (default: Adam)
  --resample-size {0,1,2,3,4,5}
                        Number of crops to be used for each image. If 5 is selected, all the 4 corner crops and 1 center crop will be added as augmentation (default: 5)
  --batch-coeff BATCH_COEFF
                        Batch size is equal to [batch_coeff] x [resample_size] (default: 1)
  --lr LR               Learning rate (default: 1e-3)
  --epochs EPOCHS       Maximum number of epochs (default: 50)
  --image-size IMAGE_SIZE
                        Image size (default: 256)
```

## TODO
- [X] Implement argparse for different training options
- [ ] Add evaluation results
- [ ] Add Docker support
- [ ] Deploy in C++ 
