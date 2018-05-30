"""
Pytorch implementation of SegNet (https://arxiv.org/pdf/1511.00561.pdf)
"""

from collections import OrderedDict
import torch
import torch.nn as nn
import pprint

F = nn.functional
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
        dim_0 = input_img.size()
        x_00 = F.relu(self.encoder_layers['encoder_conv_00'](input_img))
        x_01 = F.relu(self.encoder_layers['encoder_conv_01'](x_00))
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_layers['encoder_conv_10'](x_0))
        x_11 = F.relu(self.encoder_layers['encoder_conv_11'](x_10))
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_layers['encoder_conv_20'](x_1))
        x_21 = F.relu(self.encoder_layers['encoder_conv_21'](x_20))
        x_22 = F.relu(self.encoder_layers['encoder_conv_22'](x_21))
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_layers['encoder_conv_30'](x_2))
        x_31 = F.relu(self.encoder_layers['encoder_conv_31'](x_30))
        x_32 = F.relu(self.encoder_layers['encoder_conv_32'](x_31))
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)
        
        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_layers['encoder_conv_40'](x_3))
        x_41 = F.relu(self.encoder_layers['encoder_conv_41'](x_40))
        x_42 = F.relu(self.encoder_layers['encoder_conv_42'](x_41))
        x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)


        if DEBUG:
            print(f"dim_0: {dim_0}, indices_0: {indices_0.size()}")
            print(f"dim_1: {dim_0}, indices_1: {indices_1.size()}")
            print(f"dim_2: {dim_0}, indices_2: {indices_2.size()}")
            print(f"dim_3: {dim_0}, indices_3: {indices_3.size()}")
            print(f"dim_4: {dim_0}, indices_4: {indices_4.size()}")


        # Decoder
        
        # Decoder Stage - 5       
        x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d = F.relu(self.decoder_layers['decoder_convtr_42'](x_4d))
        x_41d = F.relu(self.decoder_layers['decoder_convtr_41'](x_42d))
        x_40d = F.relu(self.decoder_layers['decoder_convtr_40'](x_41d))
        
        # Decoder Stage - 4
        x_3d = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_layers['decoder_convtr_32'](x_3d))
        x_31d = F.relu(self.decoder_layers['decoder_convtr_31'](x_32d))
        x_30d = F.relu(self.decoder_layers['decoder_convtr_30'](x_31d))
        
        # Decoder Stage - 3
        x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_layers['decoder_convtr_22'](x_2d))
        x_21d = F.relu(self.decoder_layers['decoder_convtr_21'](x_22d))
        x_20d = F.relu(self.decoder_layers['decoder_convtr_20'](x_21d))
        
        # Decoder Stage - 2
        x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_layers['decoder_convtr_11'](x_1d))
        x_10d = F.relu(self.decoder_layers['decoder_convtr_10'](x_11d))
        
        # Decoder Stage - 1
        x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d = F.relu(self.decoder_layers['decoder_convtr_01'](x_0d))
        x_00d = self.decoder_layers['decoder_convtr_00'](x_01d)

        x_softmax = F.softmax(x_00d, dim=1)

        
        return x_00d, x_softmax

    
    def encoder(self, debug=False):
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
                if dim != 'M':
                    sub_layer = []
                    sub_layer.append(nn.Conv2d(in_channels=self.num_channels,
                                                out_channels=dim,
                                                kernel_size=3,
                                                padding=1))
                    sub_layer.append(nn.BatchNorm2d(dim))
                    # sub_layer.append(nn.ReLU(inplace=True))

                    key = "encoder_conv_{}{}".format(stage, idx)
                    layers[key] = nn.Sequential(*sub_layer)
                    self.num_channels = dim

        if debug:
            print(":: ENCODER ::\n")
            pprint.pprint(layers)
            print("---------------\n\n")

        return layers


    def decoder(self, debug=False):
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
                    key = "decoder_unpool_{}".format(len(decoder_dims) - stage - 1)
                    layers[key] = nn.MaxUnpool2d(kernel_size=2, stride=2)
                else:
                    sub_layer = []
                    try:
                        out_c = block[idx+1] if idx + 1 != len(block) else decoder_dims[stage+1][1]
                    except IndexError:
                        pass

                    if stage == len(decoder_dims) - 1 and idx == len(block) - 1:
                        sub_layer.append(nn.ConvTranspose2d(in_channels=self.num_channels,
                                                            out_channels=self.output_channels,
                                                            kernel_size=3,
                                                            padding=1))
                    else:
                        sub_layer.append(nn.ConvTranspose2d(in_channels=self.num_channels,
                                                            out_channels=out_c,
                                                            kernel_size=3,
                                                            padding=1))
                        sub_layer.append(nn.BatchNorm2d(out_c))
                    sub_layer.append(nn.ReLU(inplace=True))

                    key = "decoder_convtr_{}{}".format(len(decoder_dims) - stage - 1, len(block) - idx - 1)
                    layers[key] = nn.Sequential(*sub_layer)
                    
                    self.num_channels = out_c

        if debug:
            print(":: DECODER ::\n")
            pprint.pprint(layers)
            print("---------------\n\n")

        return layers
