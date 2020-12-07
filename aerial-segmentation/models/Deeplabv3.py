from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torch.nn as nn


class Deeplabv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3(pretrained=False, progress=True)
        self.model.classifier = DeepLabHead(2048, 3)

    def forward(self, x):
        return self.model(x)["out"]
