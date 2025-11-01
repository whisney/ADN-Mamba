import torch.nn as nn
from functools import partial

class discriminator(nn.Module):
    def __init__(self, in_channels=3, norm='instance', activation='relu'):
        super(discriminator, self).__init__()
        if norm == 'instance':
            norm_layer = partial(nn.InstanceNorm2d, affine=True)
        elif norm == 'batch':
            norm_layer = nn.BatchNorm2d

        if activation == 'leakyrelu':
            activation_layer = partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
        elif activation == 'relu':
            activation_layer = nn.ReLU

        def discriminator_block(in_filters, out_filters, normalization=True, Norm=nn.BatchNorm2d, Activation=nn.ReLU,
                                stride=(2, 2)):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=stride, padding=1, bias=False)]
            if normalization:
                layers.append(Norm(out_filters))
            layers.append(Activation(inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, Norm=norm_layer, Activation=activation_layer, stride=2),
            *discriminator_block(64, 128, Norm=norm_layer, Activation=activation_layer, stride=2),
            *discriminator_block(128, 256, Norm=norm_layer, Activation=activation_layer, stride=2),
            nn.Conv2d(256, 1, 3, padding=1, bias=True)
        )

    def forward(self, x):
        return self.model(x)
