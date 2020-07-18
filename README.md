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

<img src="docs/berlin.png" alt="berlin" height="200"/><img src="docs/zurich.png" alt="zurich" height="200"/><img src="docs/paris.png" alt="paris" height="200"/><img src="docs/chicago.png" alt="chicago" height="200"/>

## Networks
UNet, FastSCNN and DeeplabV3.
## Results
