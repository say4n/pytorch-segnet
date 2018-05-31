"""Pascal VOC 2007 Dataset"""

import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image


class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")
        self.transform = transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)
        
        image = self.load_image(path=image_path)
        gt_mask = self.load_mask(path=mask_path)

        data = {
                    'image_': image,
                    'mask_' : gt_mask
                    }

        if self.transform:
            data['image'] = self.transform(data['image_'])
            data['mask'] = self.transform(data['mask_'])

        return data

    def load_image(self, path=None):
        raw_image = Image.open(path)
        rsz_image = raw_image.resize((224, 224))
        
        imx_t = np.array(rsz_image).reshape((224, 224, 3))

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        rsz_image = raw_image.resize((224, 224))

        imx_t = np.array(rsz_image).reshape((224, 224, 1))

        return imx_t


if __name__ == "__main__":
    data_root = os.path.join("data", "VOCdevkit", "VOC2007")
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
    img_dir = os.path.join(data_root, "JPEGImages")
    mask_dir = os.path.join(data_root, "SegmentationObject")

    
    objects_dataset = PascalVOCDataset(list_file=list_file_path, img_dir=img_dir, mask_dir=mask_dir)

    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']

    plt.imshow(image)
    plt.show()
    plt.imshow(mask)
    plt.show()

