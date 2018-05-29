import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, batch_norm=True):
        super(SegNet, self).__init__()

        self.num_channels = 3
        self.features = self.encoder() + self.decoder()

        self.model = nn.Sequential(*self.features)

    
    def forward(self, input_img):
        pass

    
    def encoder(self):
        """Construct VGG-16 network"""
        layers = []
        
        # int - filter dim, 'M' - max pool
        vgg16_dims = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        # Stage 1
        
        layers.append(nn.Conv2d(3, 64, kernel_size=3, padding=1))
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

        decoder_dims = []




