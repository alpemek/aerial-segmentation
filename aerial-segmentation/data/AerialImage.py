from pathlib import Path
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as transformsF
import random
from PIL import Image
import numpy as np

THIS_DIR = Path(__file__).parent
# THIS_DIR = Path(".")

ALL_AVAILABLE_DATASETS = ["berlin", "chicago", "paris", "zurich"]
DATASET2SEED = {
    "berlin": 10,
    "chicago": 20,
    "paris": 30,
    "zurich": 40
}
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]
LABELS2PX = {
    "background": np.array([255, 255, 255]),
    "building": np.array([255, 0, 0]),
    "road": np.array([0, 0, 255]),
}
LABELS2IND = {key: ind for ind, key in enumerate(LABELS2PX.keys())}


class AerialImage(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_type,
        root_dir='.',
        avaible_datasets=ALL_AVAILABLE_DATASETS,
        resized_shape=(512, 512),
        resample_size=0
    ):
        super(AerialImage, self).__init__()
        self.avaible_datasets = avaible_datasets
        self.dataset_type = dataset_type
        self.root_dir = root_dir
        self.resample_size = resample_size
        self.resample_mode = Image.BILINEAR
        self.image_transforms = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize(RGB_MEAN, RGB_STD)])
        if self.dataset_type not in ["train", "val"]:
            raise ValueError("dataset_type must be either train or val")
        self.resized_shape = resized_shape
        self.crop_transformation = transforms.FiveCrop(self.resized_shape)
        self.ColorJitter = transforms.ColorJitter(brightness=.3, hue=.1, saturation=.1, contrast=.3)
        self.images, self.masks = self.load_data()

    def __getitem__(self, idx):
        if self.dataset_type == "train":
            return self.get_train_image(idx)
        else:
            return self.get_val_image(idx)

    def get_train_image(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image_orig = Image.open(str(image_path))
        mask_orig = Image.open(str(mask_path))

#         image_selected, mask_selected = random.choice(list(zip(image_crops_all, mask_crops_all)))
        if self.resample_size:
            image = image_orig.resize((2*self.resized_shape[0], 2*self.resized_shape[1]), self.resample_mode)
            mask = mask_orig.resize((2*self.resized_shape[0], 2*self.resized_shape[1]))

            image_crops_all = list(self.crop_transformation(image))
            mask_crops_all = list(self.crop_transformation(mask))
            image_selected, mask_selected = list(zip(*random.sample(list(zip(image_crops_all, mask_crops_all)), self.resample_size)))

            image_crops = list(image_selected)
            mask_crops = list(mask_selected)
        else:
            image_crops = []
            mask_crops = []
            # Indent out to add original images as well
            image_down = image_orig.resize(self.resized_shape, self.resample_mode)
            mask_down = mask_orig.resize(self.resized_shape)

            image_crops.append(image_down)
            mask_crops.append(mask_down)

        final_images = []
        final_labels = []
        for image, mask in zip(image_crops, mask_crops):
            image, mask = self.apply_augmentations(image, mask)

            label = self.mask2label(mask)
            label = torch.from_numpy(label).long()
            image = self.image_transforms(image)
            final_images.append(image)
            final_labels.append(label)
        image_tensor = torch.stack(final_images)
        label_tensor = torch.stack(final_labels)
        return image_tensor, label_tensor

    def get_val_image(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(str(image_path))
        mask = Image.open(str(mask_path))
        image = image.resize(self.resized_shape, self.resample_mode)
        mask = mask.resize(self.resized_shape)
        label = self.mask2label(mask)
        image_tensor = self.image_transforms(image)

        return image_tensor, torch.from_numpy(label).long()

    def __len__(self):
        return len(self.images)

    def load_data(self):
        all_images = []
        all_masks = []
        for current_dataset in self.avaible_datasets:
            assert current_dataset in ALL_AVAILABLE_DATASETS
            images = []
            masks = []
            dataset_path = Path(self.root_dir) / current_dataset
            for image_path, mask_path in zip(sorted(list(dataset_path.glob("*image.png"))),
                                             sorted(list(dataset_path.glob("*labels.png")))):
                assert image_path.stem.split("_")[0] == mask_path.stem.split("_")[0]

                images.append(image_path)
                masks.append(mask_path)

            # Shuffle the dataset
            seed = DATASET2SEED[current_dataset]
            random.Random(seed).shuffle(images)
            random.Random(seed).shuffle(masks)
            assert len(images) == len(masks)
            limit = (9 * len(images) // 10)
            if self.dataset_type == 'train':
                images = images[:limit]
                masks = masks[:limit]
            else:
                images = images[limit:]
                masks = masks[limit:]
            all_images.extend(images)
            all_masks.extend(masks)
        return all_images, all_masks

    def apply_augmentations(self, image, mask, prob=0.5):
        if random.random() < prob:
            image = transformsF.hflip(image)
            mask = transformsF.hflip(mask)

        if random.random() < prob:
            angle = random.choice([-90, 90])
            image = transformsF.rotate(image, angle, False, False, None, fill=0)
            mask = transformsF.rotate(mask, angle, False, False, None, fill=255)
        if random.random() < prob:
            image = self.ColorJitter(image)
        return image, mask

    @staticmethod
    def mask2label(mask):
        mask = np.array(mask)
        mask = (255 * np.round(mask/255)).astype(np.uint8)  # Just to be sure
        label_seg = np.zeros((mask.shape[:2]), dtype=np.int)
        for label, value in LABELS2PX.items():
            label_seg[(mask == value).all(axis=2)] = LABELS2IND[label]

        return label_seg
