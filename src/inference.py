"""
Infer segmentation results from a trained SegNet model


Usage:
python inference.py --data_root /home/SharedData/intern_sayan/PascalVOC/data/VOCdevkit/VOC2007/ \
                    --val_path ImageSets/Segmentation/val.txt \
                    --img_dir JPEGImages \
                    --mask_dir SegmentationClass \
                    --model_path /home/SharedData/intern_sayan/PascalVOC/model_best.pth \
                    --output_dir /home/SharedData/intern_sayan/PascalVOC/predictions \
                    --gpu 1
"""

from __future__ import print_function
import argparse
from dataset import PascalVOCDataset, NUM_CLASSES
import matplotlib.pyplot as plt
from model import SegNet
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

plt.switch_backend('agg')
plt.axis('off')


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 3

BATCH_SIZE = 8


# Arguments
parser = argparse.ArgumentParser(description='Validate a SegNet model')

parser.add_argument('--data_root', required=True)
parser.add_argument('--val_path', required=True)
parser.add_argument('--img_dir', required=True)
parser.add_argument('--mask_dir', required=True)
parser.add_argument('--model_path', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--gpu', type=int)


args = parser.parse_args()



def validate():
    model.eval()

    for batch_idx, batch in enumerate(val_dataloader):
        input_tensor = torch.autograd.Variable(batch['image'])
        target_tensor = torch.autograd.Variable(batch['mask'])

        if CUDA:
            input_tensor = input_tensor.cuda(GPU_ID)
            target_tensor = target_tensor.cuda(GPU_ID)

        predicted_tensor, softmaxed_tensor = model(input_tensor)
        loss = criterion(predicted_tensor, target_tensor)

        for idx, predicted_mask in enumerate(softmaxed_tensor):
            target_mask = target_tensor[idx]
            input_image = input_tensor[idx]

            fig = plt.figure()

            a = fig.add_subplot(1,3,1)
            input_imx = input_image.detach().cpu().numpy().reshape(3, 224, 224)
            plt.imshow(np.abs(np.transpose(input_imx, (1,2,0))))
            a.set_title('Input Image')
            
            a = fig.add_subplot(1,3,2)
            predicted_mx = predicted_mask.detach().cpu().numpy().reshape(NUM_CLASSES, 224, 224)
            predicted_mx = predicted_mx.argmax(axis=0)
            plt.imshow(np.abs(np.transpose(predicted_mx, (1,2,0))))
            a.set_title('Predicted Mask')

            a = fig.add_subplot(1,3,3)
            target_mx = target_mask.detach().cpu().numpy().reshape(1, 224, 224)
            plt.imshow(np.abs(np.transpose(target_mx, (1,2,0))))
            a.set_title('Ground Truth')

            fig.savefig(os.path.join(OUTPUT_DIR, "prediction_{}_{}.png".format(batch_idx, idx)))

            plt.close(fig)


if __name__ == "__main__":
    data_root = args.data_root
    val_path = os.path.join(data_root, args.val_path)
    img_dir = os.path.join(data_root, args.img_dir)
    mask_dir = os.path.join(data_root, args.mask_dir)

    SAVED_MODEL_PATH = args.model_path
    OUTPUT_DIR = args.output_dir

    CUDA = args.gpu is not None
    GPU_ID = args.gpu

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
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                       output_channels=NUM_OUTPUT_CHANNELS)
        criterion = torch.nn.CrossEntropyLoss()

    
    model.load_state_dict(torch.load(SAVED_MODEL_PATH))


    validate()


