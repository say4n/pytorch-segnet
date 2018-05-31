"""Infer segmentation results from a trained SegNet model"""

from __future__ import print_function
from model import SegNet
import os
import time
import torch


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 1

CUDA = True
GPU_ID = 0

SAVED_MODEL_PATH = "model_best.pth"


if CUDA:
    model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                   output_channels=NUM_OUTPUT_CHANNELS).cuda(GPU_ID)
else:
    model = SegNet(input_channels=NUM_INPUT_CHANNELS,
                   output_channels=NUM_OUTPUT_CHANNELS)

model.load_state_dict(torch.load(SAVED_MODEL_PATH))



