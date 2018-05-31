"""Test for SegNet"""

from __future__ import print_function
from model import SegNet
import numpy as np
import torch


if __name__ == "__main__":
    # RGB input
    input_channels = 3
    # RGB output
    output_channels = 3

    # Model
    model = SegNet(input_channels=input_channels, output_channels=output_channels)

    img = torch.zeros((1, 3, 224, 224))
    output, softmaxed_output = model(img)

    print(output.size())
    print(softmaxed_output.size())

    print(output[0,:,0,0])
    print(softmaxed_output[0,:,0,0].sum())
