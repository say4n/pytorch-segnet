import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, **kwargs):
        super(SegNet, self).__init__()

        self.features = self.encoder() + self.decoder()

        self.model = nn.Sequential(*self.features)

    
    def forward(self, input_img):
        pass

    
    def encoder(self, batch_norm=False):
        """Construct VGG-16 network"""
        layers = []
        num_channels = 3

        # int - filter dim, 'M' - max pool
        vgg16_dims = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        for dim in vgg16_dims:
            if dim == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            else:
                layers.append(nn.Conv2d(num_channels, dim, kernel_size=3, padding=1))

                if batch_norm:
                    layers.append(nn.BatchNorm2d(dim))

                layers.append(nn.ReLU(inplace=True))

                num_channels = dim

        return layers


    def decoder(self, batch_norm=False):
        pass



