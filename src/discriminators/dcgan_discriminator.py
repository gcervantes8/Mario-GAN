# -*- coding: utf-8 -*-
"""
Created on Thu May 11 00:23:38 2020

@author: Gerardo Cervantes

Purpose: The Discriminator class part of the GAN.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.discriminators.base_discriminator import BaseDiscriminator
from src import create_model


class DcganDiscriminator(BaseDiscriminator):
    def __init__(self, num_gpu, base_width, base_height, upsample_layers, ndf, num_channels, num_classes):
        super(DcganDiscriminator, self).__init__(num_gpu, base_width, base_height, upsample_layers, ndf, num_channels,
                                                 num_classes)
        self.n_gpu = num_gpu

        # Input is Batch_size x 3 x image_width x image_height matrix
        self.discrim_layers = nn.ModuleList()

        embedding_size = 32
        self.embeddings = nn.Embedding(num_classes, embedding_size)
        nn.init.orthogonal_(self.embeddings.weight)

        self.image_height, self.image_width = base_height * (2 ** upsample_layers), base_width * (2 ** upsample_layers)
        self.fc_layer = spectral_norm(nn.Linear(in_features=embedding_size,
                                                out_features=self.image_height * self.image_width))
        nn.init.orthogonal_(self.fc_layer.weight)

        if upsample_layers == 5:
            conv_channels = [num_channels + 1, ndf, ndf * 2, ndf * 4, ndf * 8, ndf * 16, 1]
        else:
            raise NotImplementedError(str(upsample_layers) + ' layers for dcgan discriminator is not supported.  You'
                                                             ' can either use a different amount of layers, or make a'
                                                             ' list with the channels you want with those layers')

        self.discrim_layers.append(nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=3, stride=2, padding=1))
        self.discrim_layers.append(nn.LeakyReLU(0.1, inplace=False))
        previous_out_channel = conv_channels[1]
        for i, layer_channel in enumerate(conv_channels[2:]):
            is_last_layer = i == len(conv_channels[2:]) - 1

            if is_last_layer:
                self.discrim_layers.append(spectral_norm(nn.Conv2d(previous_out_channel, layer_channel,
                                                                   kernel_size=(base_height, base_width), stride=1)))
                self.discrim_layers.append(nn.Sigmoid())
            else:
                self.discrim_layers.append(spectral_norm(nn.Conv2d(previous_out_channel, layer_channel,
                                                                   kernel_size=4, stride=2, padding=1)))
                self.discrim_layers.append(nn.LeakyReLU(0.1, inplace=False))
            previous_out_channel = layer_channel

        # self.main = nn.Sequential(
        #
        #     # input is (num_channels) x 65 x 87 (height goes first, when specifying tuples)
        #     spectral_norm(nn.Conv2d(num_channels, ndf, kernel_size=3, stride=2, padding=1)),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     # When dilation and padding is 1: ((in + 2p - (k - 1) - 1) / s) + 1
        #
        #     # state: (ndf*2) x 33 x 44
        #     spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=(3, 4), stride=2, padding=1)),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     # state: (ndf*4) x 17 x 22
        #     spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(3, 4), stride=2, padding=1)),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     # state:  (ndf*4) x 9 x 11
        #     spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=2, padding=1)),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     # state:  (ndf*8) x 5 x 6
        #     spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, kernel_size=(3, 4), stride=2, padding=1)),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     # state:  (ndf*8) x 3 x 3
        #     spectral_norm(nn.Conv2d(ndf * 16, 1, kernel_size=3, stride=1)),
        #     # Output is 1 x 1 x 1
        #     nn.Sigmoid()
        # )
        self.apply(create_model.weights_init)

    def forward(self, discriminator_input, labels):
        batch_size = discriminator_input.size(dim=0)
        # discriminator_input is of size (B, channels, width, height)

        embed_vector = self.embeddings(labels)
        # out is (B, image_height * image_width)
        out = self.fc_layer(embed_vector)

        out = torch.reshape(out, [batch_size, 1, self.image_height, self.image_width])

        # out is of size (B, channels+1, width, height)
        out = torch.concat((discriminator_input, out), axis=1)

        for discrim_layer in self.discrim_layers:
            out = discrim_layer(out)

        return torch.squeeze(out)
