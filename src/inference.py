"""Infer segmentation results from a trained SegNet model"""

from __future__ import print_function
import matplotlib.pyplot as plt
from model import SegNet
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 1

CUDA = True
GPU_ID = 0

SAVED_MODEL_PATH = "model_best.pth"
OUTPUT_DIR = "predictions"


data_root = os.path.join("data", "VOCdevkit", "VOC2007")
val_path = os.path.join(data_root, "ImageSets", "Segmentation", "val.txt")
img_dir = os.path.join(data_root, "JPEGImages")
mask_dir = os.path.join(data_root, "SegmentationObject")

image_transform = transforms.ToTensor()


val_dataset = PascalVOCDataset(list_file=val_path,
                               img_dir=img_dir,
                               mask_dir=mask_dir,
                               transform=image_transform)

val_dataloader = DataLoader(val_dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=4)


if CUDA:
    model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                   output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)
else:
    model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                   output_channels=NUM_OUTPUT_CHANNELS)

model.load_state_dict(torch.load(SAVED_MODEL_PATH))

criterion = torch.nn.MSELoss()



model.eval()

for batch_idx, batch in enumerate(val_dataloader):
    input_tensor = torch.autograd.Variable(batch['image'].view((-1, 3, 224, 224)))
    target_tensor = torch.autograd.Variable(batch['mask'].view((-1, 1, 224, 224)))

    if CUDA:
        input_tensor = input_tensor.cuda(GPU_ID)
        target_tensor = target_tensor.cuda(GPU_ID)

    predicted_tensor, softmaxed_tensor = model(input_tensor)
    loss = criterion(predicted_tensor, target_tensor)

    for idx, predicted_mask in enumerate(predicted_tensor):
        target_mask = target_tensor[idx]

        fig = plt.figure()
        
        a = fig.add_subplot(1,2,1)
        plt.imshow(predicted_mask)
        a.set_title('Prediction')

        a = fig.add_subplot(1,2,2)
        plt.imshow(target_mask)
        a.set_title('Ground truth')

        fig.savefig(os.path.join(OUTPUT_DIR, "prediction_{}_{}.png".format(batch_idx, idx)))



