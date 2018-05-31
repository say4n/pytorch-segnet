"""Train a SegNet model"""

from __future__ import print_function
from dataset import PascalVOCDataset
from model import SegNet
import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 1

NUM_EPOCHS = 1

LEARNING_RATE = 0.01
MOMENTUM = 0.9

CUDA = False


data_root = os.path.join("data", "VOCdevkit", "VOC2007")
train_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
val_path = os.path.join(data_root, "ImageSets", "Segmentation", "val.txt")
img_dir = os.path.join(data_root, "JPEGImages")
mask_dir = os.path.join(data_root, "SegmentationObject")

image_transform = transforms.ToTensor()

train_dataset = PascalVOCDataset(list_file=train_path, img_dir=img_dir, mask_dir=mask_dir, transform=image_transform)
val_dataset = PascalVOCDataset(list_file=val_path, img_dir=img_dir, mask_dir=mask_dir, transform=image_transform)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_data = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)




if CUDA:
    model = SegNet(input_channels=NUM_INPUT_CHANNELS, output_channels=NUM_OUTPUT_CHANNELS).cuda()
else:
    model = SegNet(input_channels=NUM_INPUT_CHANNELS, output_channels=NUM_OUTPUT_CHANNELS)

criterion = torch.nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)



model.train()

for epoch in range(NUM_EPOCHS):
    for i_batch, batch in enumerate(train_dataset):
        print(batch['image'].size(), batch['mask'].size())

        t_start = time.time()

        input_tensor = torch.autograd.Variable(batch['image'])
        target_tensor = torch.autograd.Variable(batch['mask'])

        predicted_tensor = model(input_tensor)

        loss = criterion(predicted_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_f = loss.float()
        prediction_f = predicted_tensor.float()
        
        delta = time.time() - t_start

        print(f"""Epoch #{i+1}
              Loss = {loss_f}
              Time = {delta}secs
              """)
