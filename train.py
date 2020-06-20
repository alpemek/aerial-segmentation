# Training Functions
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
# from torchvision.models.segmentation import deeplabv3_resnet50 as deeplabv3
# from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from kornia.losses import DiceLoss, FocalLoss
from models.Unet import UNet
from models.Fastscnn import FastSCNN
from data.AerialImage import AerialImage


DEVICE = torch.device("cuda")
ROOT_DIR = Path(".") / "dataset"


def train(model, data_loader, optimizer, loss_fn):
    total_loss = 0
    model.train()
    model = model.to(DEVICE)
    for iteration, (images, targets) in enumerate(data_loader):
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        targets = targets.view(-1, targets.shape[-2], targets.shape[-1])
        images = images.to(device=DEVICE, dtype=torch.float)
        targets = targets.to(device=DEVICE, dtype=torch.long)
        optimizer.zero_grad()

        out = model(images)#['out']#[0]
        loss = loss_fn(out, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iteration % int(len(data_loader) * 0.2) == 0:
            print("Training [{}/{}*N {:.0f}% \tLoss:{:.3f}]".format(iteration * len(images), len(data_loader.dataset), 100.*iteration/len(data_loader), total_loss/(iteration+1) ))
    print("Total Training Loss {:.3f}".format(total_loss/len(data_loader)))
    return model, total_loss/len(data_loader)


@torch.no_grad()
def evaluate(model, data_loader, loss_fn):
    total_loss = 0
    total_iou_road = 0
    total_iou_building = 0
    model.eval()
    model = model.to(DEVICE)
    for iteration, (images, targets) in enumerate(data_loader):
        images = images.to(device=DEVICE, dtype=torch.float)
        targets = targets.to(device=DEVICE, dtype=torch.long)

        out = model(images)#[0]
        loss = loss_fn(out, targets)

        pred = out.max(1, keepdim=False)[1].cpu()
        iou_building = iou(pred, targets, 1)
        iou_road = iou(pred, targets, 2)

        total_loss += loss.item()
        total_iou_building += iou_building
        total_iou_road += iou_road

    print("Evaluation Loss {:.3f} | IoU Building:{:.3f} Road:{:.3f}".format(total_loss/len(data_loader), total_iou_building/len(data_loader), total_iou_road/len(data_loader) ))
    return total_loss/len(data_loader), total_iou_building/len(data_loader),  total_iou_road/len(data_loader)


def iou(pred, target, class_id):
    pred = pred.view(-1)
    target = target.view(-1)
    pred_ind = (pred == class_id)
    target_ind = (target == class_id)
    intersection = (pred_ind[target_ind]).long().sum().cpu().item()
    union = pred_ind.long().sum().cpu().item() + target_ind.long().sum().cpu().item() - intersection
    if union == 0:
        return (float('nan'))
    else:
        return (float(intersection) / float(max(union, 1)))

# Traning Code


# Traning Settings
resized_shape = (256, 256)
resample_size = 5  # 1
batch_size = 10  # 10
patience = 10
lr = 1e-3
num_epochs = 50

# Create Model
# model = FastSCNN(num_classes=3)
# model = deeplabv3(pretrained=True, progress=True)
# model.classifier = DeepLabHead(2048, 3)
model = UNet(n_channels=3, n_classes=3, bilinear=True)

# Create loss function
# loss_fn = nn.CrossEntropyLoss().to(DEVICE)
# loss_fn = DiceLoss().to(DEVICE)
loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean').to(DEVICE)

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                    eps=1e-8, weight_decay=2e-4, amsgrad=False )

# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

# Create datasets
train_dataset = AerialImage("train", resized_shape=resized_shape, resample_size=resample_size, root_dir=ROOT_DIR)
val_dataset = AerialImage("val", resized_shape=(2*resized_shape[0], 2*resized_shape[1]), root_dir=ROOT_DIR)

# Check if training and validation sets have common elements
assert set(train_dataset.images).isdisjoint(set(val_dataset.images))
assert set(train_dataset.masks).isdisjoint(set(val_dataset.masks))
# Create dataloaders
dataloader_batch_size = max(batch_size//(resample_size), 1)
train_loader = torch.utils.data.DataLoader(train_dataset, dataloader_batch_size, num_workers=4, shuffle=True, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=4, shuffle=True, pin_memory=True)

train_loss_list = []
val_loss_list = []
val_iou_building_list = []
val_iou_road_list = []

counter = 0
iou_road_best = 0
best_epoch = 0

# Training Loop
for epoch in range(1, num_epochs+1):
    print("\nTraining Epoch: {}".format(epoch))
    model, train_loss = train(model, train_loader, optimizer, loss_fn)
    scheduler.step()
    # torch.save(model.state_dict(), "model_epoch_{}.pth".format(epoch))
    train_loss_list.append(train_loss)
    val_loss, iou_bulding, iou_road = evaluate(model, val_loader, loss_fn)
    val_loss_list.append(val_loss)
    val_iou_building_list.append(iou_bulding)
    val_iou_road_list.append(iou_road)
    counter += 1
    if iou_road >= iou_road_best:
        iou_road_best = iou_road
        print("Saving the weights")
        torch.save(model.state_dict(), "model_best.pth".format(epoch))
        counter = 0

    elif counter >= patience:
        print("Validation Loss does not improve, ending the training.")
        break
