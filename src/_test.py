"""Test for SegNet"""

from __future__ import print_function
from model import SegNet
from dataset import NUM_CLASSES
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":
    # RGB input
    input_channels = 3
    # RGB output
    output_channels = NUM_CLASSES

    # Model
    model = SegNet(input_channels=input_channels, output_channels=output_channels)

    print(model)

    img = torch.randn([4, 3, 224, 224])

    # plt.imshow(np.transpose(img.numpy()[0,:,:,:],
    #                         (1, 2, 0)))
    # plt.show()

    output, softmaxed_output = model(img)


    # plt.imshow(np.transpose(output.detach().numpy()[0,:,:,:],
    #                         (1, 2, 0)))
    # plt.show()


    print(output.size())
    print(softmaxed_output.size())

    print(output[0,:,0,0])
    print(softmaxed_output[0,:,0,0].sum())
