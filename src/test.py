"""Test for SegNet"""

from model import SegNet
import numpy as np
import torch


if __name__ == "__main__":
    # RGB
    input_channels = 3
    # Num classes
    output_channels = 10

    # Model
    model = SegNet(input_channels=input_channels, output_channels=output_channels)
    

    img = torch.zeros((1, 3, 512, 512))
    print(model(img))
