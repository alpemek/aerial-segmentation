# Training Functions
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from kornia.losses import DiceLoss, FocalLoss
from models import UNet, FastSCNN, Deeplabv3
from data.AerialImage import AerialImage


DEVICE = torch.device("cuda")
ROOT_DIR = Path(__file__).resolve().parent.parent / "dataset"


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

        out = model(images)
        loss = loss_fn(out, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iteration % int(len(data_loader) * 0.2) == 0:
            print("Training [{}/{}*N {:.0f}% \tLoss:{:.3f}]".format(iteration * len(images), len(data_loader.dataset), 100.*iteration/len(data_loader), total_loss/(iteration+1)))
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

        out = model(images)
        loss = loss_fn(out, targets)

        pred = out.max(1, keepdim=False)[1].cpu()
        iou_building = iou(pred, targets, 1)
        iou_road = iou(pred, targets, 2)

        total_loss += loss.item()
        total_iou_building += iou_building
        total_iou_road += iou_road

    print("Evaluation Loss {:.3f} | IoU Building:{:.3f} Road:{:.3f}".format(total_loss/len(data_loader), total_iou_building/len(data_loader), total_iou_road/len(data_loader)))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Code')

    parser.add_argument('--model', default="UNet",
                        choices=["UNet", "Deeplabv3", "FastSCNN"],
                        help='Network model to be trained (default: UNet)')
    parser.add_argument('--loss', default="FocalLoss",
                        choices=["FocalLoss", "DiceLoss", "CrossEntropyLoss"],
                        help='Loss function (default: FocalLoss)')
    parser.add_argument('--optimizer', default="Adam",
                        choices=["SGD", "Adam"],
                        help='Optimizer (default: Adam)')
    parser.add_argument('--resample-size', default=5, type=int, choices=range(6),
                        help='Number of crops to be used for each image. If 5 is\
                        selected, all the 4 corner crops and 1 center crop will be\
                        added as augmentation (default: 5)')
    parser.add_argument('--batch-coeff', default=1, type=int,
                        help='Batch size is equal to [batch_coeff] x [resample_size]\
                        (default: 1)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Maximum number of epochs (default: 50)')
    parser.add_argument('--image-size', default=256, type=int,
                        help='Image size (default: 256)')
    args = parser.parse_args()

    # Traning Code
    # Traning Settings
    resized_shape = (args.image_size, args.image_size)
    resample_size = args.resample_size  # 1
    batch_size = args.resample_size * args.batch_coeff  # 10
    patience = 10
    lr = args.lr
    num_epochs = args.epochs

    # Create Model
    if args.model == "FastSCNN":
        model = FastSCNN(num_classes=3)
    elif args.model == "Deeplabv3":
        model = Deeplabv3()
    elif args.model == "UNet":
        model = UNet(n_channels=3, n_classes=3, bilinear=True)
    else:
        raise Exception("Model not found.")

    # Create loss function
    if args.loss == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    elif args.loss == "DiceLoss":
        loss_fn = DiceLoss().to(DEVICE)
    elif args.loss == "FocalLoss":
        loss_fn = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean').to(DEVICE)
    else:
        raise Exception("Loss function not found.")

    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=2e-4, amsgrad=False)
    else:
        raise Exception("Optimizer not found.")


    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    # Create datasets
    train_dataset = AerialImage("train", resized_shape=resized_shape, resample_size=resample_size, root_dir=ROOT_DIR)
    val_dataset = AerialImage("val", resized_shape=(2*resized_shape[0], 2*resized_shape[1]), root_dir=ROOT_DIR)

    # Check if training and validation sets have common elements
    assert set(train_dataset.images).isdisjoint(set(val_dataset.images))
    assert set(train_dataset.masks).isdisjoint(set(val_dataset.masks))
    # Create dataloaders
    dataloader_batch_size = max(batch_size//(resample_size), 1)
    train_loader = torch.utils.data.DataLoader(train_dataset, dataloader_batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=8, shuffle=True, pin_memory=True)

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
