# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel
from models.CNNBlocks import DenseBlock, ResidualBlock

'''
TODO

Ajouter du code ici pour faire fonctionner le réseau YourSegNet.  Un réseau très différent du UNet.
Soyez originaux et surtout... amusez-vous!

'''


class YourSegNet(CNNBaseModel):
    """
     Class that implements a brand new segmentation CNN
    """

    def __init__(self, num_classes=4, init_weights=True):
        """
        Builds YourSegNet  model.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            init_weights(bool): when true uses _initialize_weights function to initialize
            network's weights.
        """
        super().__init__(num_classes, init_weights)

        in_channels = 1  # Gray image
        channels = 32
        nb_dense_blocks = 2
        self.sequential = nn.Sequential(
            self._conv_same_block(in_channels=in_channels, out_channels=channels),

            self._dense_blocks(in_channels=channels, out_channels=channels, dilation=1, nb_dense_blocks=nb_dense_blocks),
            self._dense_blocks(in_channels=channels, out_channels=channels, dilation=2, nb_dense_blocks=nb_dense_blocks),

            nn.Dropout(0.5),

            self._dense_blocks(in_channels=channels, out_channels=channels, dilation=4, nb_dense_blocks=nb_dense_blocks),
            self._dense_blocks(in_channels=channels, out_channels=channels, dilation=8, nb_dense_blocks=nb_dense_blocks),

            nn.Dropout(0.5),

            self._dense_blocks(in_channels=channels, out_channels=channels, dilation=16, nb_dense_blocks=nb_dense_blocks),
            self._dense_blocks(in_channels=channels, out_channels=channels, dilation=32, nb_dense_blocks=nb_dense_blocks),

            nn.Dropout(0.5),

            self._dense_blocks(in_channels=channels, out_channels=channels, dilation=16, nb_dense_blocks=nb_dense_blocks),
            self._dense_blocks(in_channels=channels, out_channels=channels, dilation=8, nb_dense_blocks=nb_dense_blocks),

            self._conv_same_block(in_channels=channels, out_channels=num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        return self.sequential(x)

    @staticmethod
    def _conv_same_block(in_channels, out_channels):
        return YourSegNet._conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    @staticmethod
    def _dense_blocks(in_channels, out_channels, dilation, nb_dense_blocks=3):
        return nn.Sequential(
            *[DenseBlock(in_channels * 2 ** i, dilation) for i in range(nb_dense_blocks)],
            YourSegNet._conv_block(in_channels=in_channels * 2 ** nb_dense_blocks, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )


'''
Fin de votre code.
'''
