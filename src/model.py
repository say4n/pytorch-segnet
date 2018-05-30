"""
Pytorch implementation of SegNet (https://arxiv.org/pdf/1511.00561.pdf)
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import pprint


DEBUG = False


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_channels = input_channels

        self.mpool_indices = dict()

        self.encoder_layers = self.encoder()
        self.decoder_layers = self.decoder()

    
    def forward(self, input_img):
        """Forward pass `input_img` through the network"""

        # Encoder
        dim_0 = input_image.size()


        # Decoder


        
        return input_image

    
    def encoder(self):
        """Construct VGG-16 network"""
        layers = OrderedDict()

        # int - filter dim, 'M' - max pool
        vgg16_dims = [
                        (64, 64, 'M'),                                    # Stage - 1
                        (128, 128, 'M'),                                  # Stage - 2
                        (256, 256, 256,'M'),                              # Stage - 3
                        (512, 512, 512, 'M'),                             # Stage - 4 
                        (512, 512, 512, 'M')                              # Stage - 5
                    ]


        for stage, block in enumerate(vgg16_dims):
            for idx, dim in enumerate(block):
                if dim == 'M':
                    layers[f"encoder_maxpool_{stage}{idx}"] = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
                else:
                    sub_layers = []
                    sub_layers.append(nn.Conv2d(in_channels=self.num_channels,
                                                out_channels=dim,
                                                kernel_size=3,
                                                padding=1))
                    sub_layers.append(nn.BatchNorm2d(dim))
                    sub_layers.append(nn.ReLU(inplace=True))

                    
                    layers[f"encoder_conv_{stage}{idx}"] = nn.Sequential(*sub_layers)
                    self.num_channels = dim

        if DEBUG:
            pprint.pprint(layers)

        return layers


    def decoder(self):
        """Decoder part of SegNet"""
        layers = OrderedDict()

        # int - filter dim, 'U' - max unpool
        decoder_dims = [
                            ('U', 512, 512, 512),                     # Stage - 5
                            ('U', 512, 512, 512),                     # Stage - 4
                            ('U', 256, 256, 256),                     # Stage - 3 
                            ('U', 128, 128),                          # Stage - 2
                            ('U', 64, 64)                             # Stage - 1
                        ]


        for stage, block in enumerate(decoder_dims):
            for idx, dim in enumerate(block):
                if dim == 'U':
                    layers[f"decoder_{len(decoder_dims) - stage - 1}_unpool"] = nn.MaxUnpool2d(kernel_size=2, stride=2)
                else:
                    sub_layers = []
                    if stage == len(decoder_dims) - 1 and idx == len(block) - 1:
                        sub_layers.append(nn.ConvTranspose2d(in_channels=self.num_channels,
                                                    out_channels=self.output_channels,
                                                    kernel_size=3,
                                                    padding=1))
                    else:
                        sub_layers.append(nn.ConvTranspose2d(in_channels=self.num_channels,
                                                    out_channels=dim,
                                                    kernel_size=3,
                                                    padding=1))
                    sub_layers.append(nn.BatchNorm2d(dim))
                    sub_layers.append(nn.ReLU(inplace=True))


                    layers[f"decoder_conv_{len(decoder_dims) - stage - 1}{idx}"] = nn.Sequential(*sub_layers)
                    self.num_channels = dim

        if DEBUG:
            pprint.pprint(layers)

        return layers
