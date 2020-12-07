import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from models.Unet import UNet
from models.Fastscnn import FastSCNN


@torch.no_grad()
def predict(model_checkpoint, image_path, out_file):
    """
    Args:
        model_checkpoint (string): path to model checkpoint
        image_path (string): path to an aerial image
        out_file (string): path to save the prediction mask.

    """
    DEVICE = torch.device("cpu")
    RGB_MEAN = [0.485, 0.456, 0.406]
    RGB_STD = [0.229, 0.224, 0.225]
    image = Image.open(str(image_path))
    image = image.resize((512, 512), Image.BILINEAR)
    image_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(RGB_MEAN, RGB_STD)])

    image_tensor = image_transforms(image)[None]
    # model = FastSCNN(num_classes=3)
    # model = UNet(n_channels=3, n_classes=3, bilinear=True)
    model = deeplabv3(pretrained=False, progress=True); model.classifier = DeepLabHead(2048, 3)
    model.eval()
    model.to(DEVICE)
    image_tensor = image_tensor.to(device=DEVICE, dtype=torch.float)
    model.load_state_dict(torch.load(model_checkpoint,  map_location=lambda storage, loc: storage))
    out = model(image_tensor)['out'][0].squeeze()
    out_max = out.max(0, keepdim=False)[1].cpu().numpy()

    final_image = np.zeros((out_max.shape[0], out_max.shape[1], 3), dtype=np.int)
    final_image[(out_max == 0), :] = np.array([255, 255, 255])
    final_image[(out_max == 1), :] = np.array([255, 0, 0])
    final_image[(out_max == 2), :] = np.array([0, 0, 255])
    # image.show()

    final_image_pil = Image.fromarray(np.uint8(final_image))
    final_image_pil.show()
    final_image_pil.save(out_file)


if __name__ == "__main__":
    model_checkpoint = "./model_best.pth"
    # image_path = "./dataset/chicago/chicago167_image.png"
    image_path = "./dataset/potsdam/top_potsdam_7_7_image.png"

    out_file = "./out.png"
    predict(model_checkpoint, image_path, out_file)
