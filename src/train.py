"""
Train a SegNet model


Usage:
python train.py --data_root /home/SharedData/intern_sayan/PascalVOC/data/VOCdevkit/VOC2007/ \
                --train_path ImageSets/Segmentation/train.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir /home/SharedData/intern_sayan/PascalVOC/ \
                --checkpoint /home/SharedData/intern_sayan/PascalVOC/model_best.pth \
                --gpu 1
"""

from __future__ import print_function
import argparse
from dataset import PascalVOCDataset, NUM_CLASSES
from model import SegNet
import os
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = NUM_CLASSES

NUM_EPOCHS = 6000

LEARNING_RATE = 0.1
MOMENTUM = 0.9
BATCH_SIZE = 16


# Arguments
parser = argparse.ArgumentParser(description='Train a SegNet model')

parser.add_argument('--data_root', required=True)
parser.add_argument('--train_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--checkpoint')
parser.add_argument('--gpu', type=int)

args = parser.parse_args()



def train():
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()
        
        for batch in train_dataloader:
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])

            if CUDA:
                input_tensor = input_tensor.cuda(GPU_ID)
                target_tensor = target_tensor.cuda(GPU_ID)

            predicted_tensor, softmaxed_tensor = model(input_tensor)

            print(softmaxed_tensor.size(), target_tensor.size())

            loss = criterion(softmaxed_tensor, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_f += loss.float()
            prediction_f = predicted_tensor.float()
            
        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model_best.pth"))

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))


if __name__ == "__main__":
    data_root = args.data_root 
    train_path = os.path.join(data_root, args.train_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

    image_transform = transforms.ToTensor()

    train_dataset = PascalVOCDataset(list_file=train_path,
                                     img_dir=img_dir,
                                     mask_dir=mask_dir,
                                     transform=image_transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, 
                                  num_workers=4)


    if CUDA:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)
        criterion = torch.nn.CrossEntropyLoss()

    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    
    optimizer = torch.optim.SGD(model.parameters(),
                                     lr=LEARNING_RATE,
                                     momentum=MOMENTUM)

    
    train()
