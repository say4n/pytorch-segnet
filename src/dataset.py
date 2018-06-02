"""Pascal VOC 2007 Dataset"""

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = len(VOC_CLASSES)



class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
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
            data['mask'] = torch.LongTensor(data['mask_'])

        return data

    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.crop((0,0,224,224))
        imx_t = np.array(raw_image)

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.crop((0,0,224,224))
        imx_t = np.array(raw_image)

        dim = (*imx_t.shape, 1)
        imx_t = imx_t.reshape(dim)
        imx_t = np.transpose(np.array(imx_t == np.arange(NUM_CLASSES), dtype=np.float32), (2,0,1))

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

