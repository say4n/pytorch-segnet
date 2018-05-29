"""
Pytorch implementation of SegNet (https://arxiv.org/pdf/1511.00561.pdf)
"""

import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.features = self.encoder() + self.decoder()

        self.model = nn.Sequential(*self.features)

    
    def forward(self, input_img):
        pass

    
    def encoder(self):
        """Construct VGG-16 network"""
        layers = []

        # int - filter dim, 'M' - max pool
        vgg16_dims = [
                        64, 64, 'M',                                    # Stage - 1
                        128, 128, 'M',                                  # Stage - 2
                        256, 256, 256,'M',                              # Stage - 3
                        512, 512, 512, 'M',                             # Stage - 4 
                        512, 512, 512, 'M'                              # Stage - 5
                    ]


        # Stage 1
        
        layers.append(nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))


        # Stage 2

        layers.append(nn.Conv2d(64, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size256=2, stride=2))


        # Stage 3

        layers.append(nn.Conv2d(128, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))



        # Stage 4

        layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))


        # Stage 5

        layers.append(nn.Conv2d(256, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return layers


    def decoder(self):
        """Decoder part of SegNet"""
        layers = []


        decoder_dims = [
                            'M', 512, 512, 512,                     # Stage - 1
                            'M', 512, 512, 512,                     # Stage - 2
                            'M', 256, 256, 256,                     # Stage - 3 
                            'M', 128, 128,                          # Stage - 4
                            'M', 64, 64                             # Stage - 5
                        ]


        # Stage 1

        layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))


        # Stage 2

        layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(512, 512, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.ReLU(inplace=True))


        # Stage 3

        layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(512, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))


        # Stage 4

        layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(256, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(128, 128, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU(inplace=True))


        # Stage 5

        layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(128, 64, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(64, self.output_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(64))
        layers.append(nn.ReLU(inplace=True))


        return layers




