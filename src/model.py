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

        self.num_channels = input_channels

        self.encoder_features = self.encoder()
        self.decoder_features = self.decoder()

        self.features = self.encoder_features + self.decoder_features

        self.model = nn.Sequential(*self.features)

    
    def forward(self, input_img):
        """Forward pass `input_img` through the network"""
        x = self.features(input_image)
        
        return x

    
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

        for dim in vgg16_dims:
            if dim == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            else:
                layers.append(nn.Conv2d(self.num_channels, dim, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(dim))
                layers.append(nn.ReLU(inplace=True))

                self.num_channels = dim


        return layers


    def decoder(self):
        """Decoder part of SegNet"""
        layers = []

        # int - filter dim, 'U' - max unpool
        decoder_dims = [
                            'U', 512, 512, 512,                     # Stage - 1
                            'U', 512, 512, 512,                     # Stage - 2
                            'U', 256, 256, 256,                     # Stage - 3 
                            'U', 128, 128,                          # Stage - 4
                            'U', 64, 64                             # Stage - 5
                        ]


        for idx, dim in enumerate(decoder_dims):
            if dim == 'U':
                layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            
            else:
                if idx == len(decoder_dims) - 1:
                    layers.append(nn.Conv2d(self.num_channels, self.output_channels, kernel_size=3, padding=1))
                else:
                    layers.append(nn.Conv2d(self.num_channels, dim, kernel_size=3, padding=1))
                
                layers.append(nn.BatchNorm2d(dim))
                layers.append(nn.ReLU(inplace=True))

                self.num_channels = dim


        return layers
