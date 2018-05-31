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

NUM_EPOCHS = 2000

LEARNING_RATE = 0.03
MOMENTUM = 0.9
BATCH_SIZE = 8

CUDA = True
GPU_ID = 0


data_root = os.path.join("data", "VOCdevkit", "VOC2007")
train_path = os.path.join(data_root,"ImageSets", "Segmentation", "train.txt")
val_path = os.path.join(data_root, "ImageSets", "Segmentation", "val.txt")
img_dir = os.path.join(data_root, "JPEGImages")
mask_dir = os.path.join(data_root, "SegmentationObject")

image_transform = transforms.ToTensor()


train_dataset = PascalVOCDataset(list_file=train_path,
                                 img_dir=img_dir,
                                 mask_dir=mask_dir,
                                 transform=image_transform)

train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True, 
                              num_workers=4)


val_dataset = PascalVOCDataset(list_file=val_path,
                               img_dir=img_dir,
                               mask_dir=mask_dir,
                               transform=image_transform)

val_data = DataLoader(val_dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=4)




if CUDA:
    model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                   output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)
else:
    model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                   output_channels=NUM_OUTPUT_CHANNELS)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE, 
                            momentum=MOMENTUM)




is_better = True
prev_loss = float('inf')



model.train()

for epoch in range(NUM_EPOCHS):
    loss_f = 0
    t_start = time.time()
    
    for batch in train_dataloader:
        input_tensor = torch.autograd.Variable(batch['image'].view((BATCH_SIZE, 3, 224, 224)))
        target_tensor = torch.autograd.Variable(batch['mask'].view((BATCH_SIZE, 1, 224, 224)))

        if CUDA:
            input_tensor = input_tensor.cuda(GPU_ID)
            target_tensor = target_tensor.cuda(GPU_ID)

        predicted_tensor, softmaxed_tensor = model(input_tensor)

        loss = criterion(predicted_tensor, target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_f += loss.float()
        prediction_f = predicted_tensor.float()
        
    delta = time.time() - t_start
    is_better = loss_f < prev_loss

    if is_better:
        torch.save(model.state_dict(), "model_best.pth")

    print("Epoch #{}\tLoss: {:.6f}\t Time: {} s".format(epoch+1, loss_f, delta))
