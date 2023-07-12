import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = ConvBlock(3, 32, 3)
        self.conv2 = ConvBlock(32, 128, 3)
        self.conv3 = ConvBlock(128, 256, 3)
        self.conv4 = ConvBlock(256, 256, 3)
        self.conv5 = ConvBlock(256, 256, 3)
        self.conv6 = ConvBlock(256, 256, 3)

        self.out_channels = 256


    def forward(self, x):
        out = OrderedDict()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        out['0'] = x
        x = self.conv4(x)
        out['1'] = x
        x = self.conv5(x)
        out['2'] = x
        x = self.conv6(x)
        out['pool'] = x
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, activation_out=True):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, padding=1)
        
        self.conv2 = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()

        self.activation_out = activation_out
    
    def forward(self, x):
        x = self.pool(self.activation(self.conv1(x)))
        x = self.conv2(x)
        if self.activation_out:
            return self.activation(x)
        else:
            return x
