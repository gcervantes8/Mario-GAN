# -*- coding: utf-8 -*-
"""

@author: Gerardo Cervantes

Purpose: The Discriminator part of the BigGan.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from src.models.base_discriminator import BaseDiscriminator
from src.models.deep_biggan.deep_res_down import DeepResDown
from src.layers.nonlocal_block import NonLocalBlock
from torch.nn.utils import spectral_norm


class DeepBigganDiscriminator(BaseDiscriminator):
    def __init__(self, base_width: int, base_height: int, upsample_layers: int, ndf: int, num_channels: int,
                 num_classes: int, output_size: int, project_labels: bool):
        super(DeepBigganDiscriminator, self).__init__(base_width, base_height, upsample_layers, ndf,
                                                      num_channels, num_classes)
        self.base_width, self.base_height = base_width, base_height
        # [B, ndf, image_width, image_height]
        initial_conv = spectral_norm(nn.Conv2d(3, ndf, kernel_size=3, padding='same'), eps=1e-04)
        nn.init.orthogonal_(initial_conv.weight)
        if upsample_layers == 3:
            residual_channels = [ndf, ndf * 2, ndf * 2, ndf * 4, ndf * 4, ndf * 8, ndf * 8]
            downsample_layers = [True, False, True, False, True, False]
            nonlocal_block_index = 0
        elif upsample_layers == 4:
            residual_channels = [ndf, ndf * 2, ndf * 2, ndf * 4, ndf * 4, ndf * 8, ndf * 8, ndf * 16, ndf * 16]
            downsample_layers = [True, False, True, False, True, False, True, False]
            nonlocal_block_index = 0
        elif upsample_layers == 5:
            residual_channels = [ndf, ndf * 2, ndf * 2, ndf * 4, ndf * 4, ndf * 8, ndf * 8, ndf * 16, ndf * 16,
                                 ndf * 16, ndf * 16]
            downsample_layers = [True, False, True, False, True, False, True, False, True, False]
            nonlocal_block_index = 1
        elif upsample_layers == 6:
            residual_channels = [ndf, ndf * 2, ndf * 2, ndf * 4, ndf * 4, ndf * 8, ndf * 8, ndf * 8, ndf * 8, ndf * 16, ndf * 16,
                                 ndf * 16, ndf * 16]
            downsample_layers = [True, False, True, False, True, False, True, False, True, False, True, False]
            nonlocal_block_index = 3
        else:
            raise NotImplementedError(str(upsample_layers) + ' layers for biggan discriminator is not supported.  You'
                                                               ' can either use a different amount of layers, or make a'
                                                               ' list with the channels you want with those layers')

        if project_labels and output_size > 1:
            raise NotImplementedError('Projecting labels and having an output greater than 1 currently not supported')
            
        # Input is Batch_size x 3 x image_width x image_height matrix
        self.discrim_layers = nn.Sequential()

        self.discrim_layers.append(initial_conv)

        self.discrim_layers.append(DeepResDown(residual_channels[0], residual_channels[1], pooling=downsample_layers[0]))
        previous_out_channel = residual_channels[1]
        for i, layer_channel in enumerate(residual_channels[2:]):
            if nonlocal_block_index == i:
                self.discrim_layers.append(NonLocalBlock(previous_out_channel))
            self.discrim_layers.append(DeepResDown(previous_out_channel, layer_channel, pooling=downsample_layers[i+1]))
            previous_out_channel = layer_channel
        # [B, ndf * 16, base_width, base_height]
        self.discrim_layers.append(nn.ReLU())

        self.project_labels = project_labels
        
        last_channel_size = residual_channels[-1]
        if self.project_labels:
            self.embeddings = torch.nn.Embedding(num_classes, last_channel_size)
            nn.init.orthogonal_(self.embeddings.weight)
        # Fully connected layer
        self.fc_layer = spectral_norm(nn.Linear(in_features=last_channel_size, out_features=output_size), eps=1e-04)
        nn.init.orthogonal_(self.fc_layer.weight)

    def set_channels_last(self):
        self.discrim_layers = self.discrim_layers.to(memory_format=torch.channels_last)

    # Output is of size [Batch size, output_size]
    def forward(self, discriminator_input, labels):

        out = self.discrim_layers(discriminator_input)

        # ndf * 16 - Global Sum Pooling
        out = torch.sum(out, dim=[2, 3])
        # Size, [B, output_size] - Fully connected layer
        fc_out = self.fc_layer(out)
        
        if self.project_labels:
            fc_out = torch.squeeze(fc_out, dim=1)
            # embed_vector is of size [B, ndf*16]
            embed_vector = self.embeddings(labels)
            # TODO not sure if sum is needed
            out = fc_out + torch.sum(torch.mul(embed_vector, out), 1)
            return out
        return fc_out
