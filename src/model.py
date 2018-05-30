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
        
        # Encoder Stage - 1        
        dim_0 = input_image.size()
        x = self.encoder_layers['encoder_conv_00'](input_image)
        x = self.encoder_layers['encoder_conv_01'](x)
        x, indices_0 = self.encoder_layers['encoder_maxpool_0'](x)
        
        # Encoder Stage - 2
        dim_1 = x.size()
        x = self.encoder_layers['encoder_conv_10'](x)
        x = self.encoder_layers['encoder_conv_11'](x)
        x, indices_1 = self.encoder_layers['encoder_maxpool_1'](x)
        
        # Encoder Stage - 3
        dim_2 = x.size()
        x = self.encoder_layers['encoder_conv_20'](x)
        x = self.encoder_layers['encoder_conv_21'](x)
        x = self.encoder_layers['encoder_conv_22'](x)
        x, indices_2 = self.encoder_layers['encoder_maxpool_2'](x)
        
        # Encoder Stage - 4
        dim_3 = x.size()
        x = self.encoder_layers['encoder_conv_30'](x)
        x = self.encoder_layers['encoder_conv_31'](x)
        x = self.encoder_layers['encoder_conv_32'](x)
        x, indices_3 = self.encoder_layers['encoder_maxpool_3'](x)
        
        # Encoder Stage - 5
        dim_4 = x.size()
        x = self.encoder_layers['encoder_conv_40'](x)
        x = self.encoder_layers['encoder_conv_41'](x)
        x = self.encoder_layers['encoder_conv_42'](x)
        x, indices_4 = self.encoder_layers['encoder_maxpool_4'](x)


        # Decoder
        
        # Decoder Stage - 5       
        x = self.decoder_layers['decoder_unpool_4'](x, indices_4, output_size=dim_4)
        x = self.decoder_layers['decoder_convtr_42'](x)
        x = self.decoder_layers['decoder_convtr_41'](x)
        x = self.decoder_layers['decoder_convtr_40'](x)
        
        # Decoder Stage - 4
        x = self.decoder_layers['decoder_unpool_3'](x, indices_3, output_size=dim_3)
        x = self.decoder_layers['decoder_convtr_32'](x)
        x = self.decoder_layers['decoder_convtr_31'](x)
        x = self.decoder_layers['decoder_convtr_30'](x)
        
        # Decoder Stage - 3
        x = self.decoder_layers['decoder_unpool_2'](x, indices_2, output_size=dim_2)
        x = self.decoder_layers['decoder_convtr_22'](x)
        x = self.decoder_layers['decoder_convtr_21'](x)
        x = self.decoder_layers['decoder_convtr_20'](x)
        
        # Decoder Stage - 2
        x = self.decoder_layers['decoder_unpool_1'](x, indices_1, output_size=dim_1)
        x = self.decoder_layers['decoder_convtr_11'](x)
        x = self.decoder_layers['decoder_convtr_10'](x)
        
        # Decoder Stage - 1
        x = self.decoder_layers['decoder_unpool_0'](x, indices_0, output_size=dim_0)
        x = self.decoder_layers['decoder_convtr_01'](x)
        x = self.decoder_layers['decoder_convtr_00'](x)

        
        return x

    
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
                    layers[f"encoder_maxpool_{stage}"] = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
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
                    layers[f"decoder_unpool_{len(decoder_dims) - stage - 1}"] = nn.MaxUnpool2d(kernel_size=2, stride=2)
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


                    layers[f"decoder_convtr_{len(decoder_dims) - stage - 1}{len(block) - idx - 1}"] = nn.Sequential(*sub_layers)
                    self.num_channels = dim

        if DEBUG:
            pprint.pprint(layers)

        return layers
