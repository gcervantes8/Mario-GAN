# -*- coding: utf-8 -*-
"""

@author: Gerardo Cervantes

Purpose: The Generator part of BigGan.  Customizable in the creation.
The class takes in images to classify whether the images are real or fake (generated)
"""

import torch
import torch.nn as nn
from src.models.base_generator import BaseGenerator
from src.models.deep_biggan.deep_res_up import DeepResUp
from src.layers.nonlocal_block import NonLocalBlock
from torch.nn.utils import spectral_norm


class DeepBigganGenerator(BaseGenerator):
    def __init__(self, base_width, base_height, upsample_layers, latent_vector_size, ngf, num_channels,
                 num_classes):
        super(DeepBigganGenerator, self).__init__(base_width, base_height, upsample_layers, latent_vector_size,
                                                  ngf, num_channels, num_classes)
        self.ngf = ngf

        self.base_width, self.base_height = base_width, base_height
        # Embedding size of 128 is used for the biggan and deep-biggan paper
        embedding_size = 128
        self.embeddings = torch.nn.Embedding(num_classes, embedding_size)
        nn.init.orthogonal_(self.embeddings.weight)

        latent_embed_vector_size = latent_vector_size + embedding_size

        if upsample_layers == 3:
            residual_channels = [ngf * 8, ngf * 8, ngf * 4, ngf * 4, ngf * 2,
                                 ngf * 2, ngf]
            upsample_layers = [False, True, False, True, False, True]
            self.nonlocal_block_index = 3
        elif upsample_layers == 4:
            residual_channels = [ngf * 16, ngf * 16, ngf * 8, ngf * 8, ngf * 4, ngf * 4, ngf * 2,
                                 ngf * 2, ngf]
            upsample_layers = [False, True, False, True, False, True, False, True]
            self.nonlocal_block_index = 5
        elif upsample_layers == 5:
            residual_channels = [ngf * 16, ngf * 16, ngf * 16, ngf * 16, ngf * 8, ngf * 8, ngf * 4, ngf * 4, ngf * 2,
                                 ngf * 2, ngf]
            upsample_layers = [False, True, False, True, False, True, False, True, False, True]
            self.nonlocal_block_index = 7
        elif upsample_layers == 6:
            residual_channels = [ngf * 16, ngf * 16, ngf * 16, ngf * 16, ngf * 8, ngf * 8, ngf * 8, ngf * 8, ngf * 4, ngf * 4, ngf * 2,
                                 ngf * 2, ngf]
            upsample_layers = [False, True, False, True, False, True, False, True, False, True, False, True]
            self.nonlocal_block_index = 7
        else:
            raise NotImplementedError(str(upsample_layers) + ' layers for deep biggan generator is not supported.  You'\
                                                             ' can either use a different amount of layers, or make a'\
                                                             ' list with the channels you want with those layers')
        last_dimension_size = residual_channels[-1]
        self.first_dimension_size = residual_channels[0]
        self.initial_linear = spectral_norm(nn.Linear(in_features=latent_embed_vector_size,
                                            out_features=base_width * base_height * self.first_dimension_size), eps=1e-04)
        nn.init.orthogonal_(self.initial_linear.weight)
        self.generator_layers_pre_attn = nn.Sequential()
        self.generator_layers_post_attn = nn.Sequential()
        self.generator_layers_pre_attn.append(DeepResUp(residual_channels[0], residual_channels[1], latent_embed_vector_size,
                                               upsample=upsample_layers[0]))
        previous_out_channel = residual_channels[1]
        for i, layer_channel in enumerate(residual_channels[2:]):
            if self.nonlocal_block_index == i:
                self.nonlocal_block = NonLocalBlock(previous_out_channel)
            if i < self.nonlocal_block_index:
                self.generator_layers_pre_attn.append(DeepResUp(previous_out_channel, layer_channel, latent_embed_vector_size,
                                        upsample=upsample_layers[i+1]))

            elif i >= self.nonlocal_block_index:
                self.generator_layers_post_attn.append(DeepResUp(previous_out_channel, layer_channel, latent_embed_vector_size,
                                                    upsample=upsample_layers[i+1]))
            previous_out_channel = layer_channel

        self.end_ops = nn.Sequential()
        self.end_ops.append(nn.BatchNorm2d(num_features=last_dimension_size))
        self.end_ops.append(nn.ReLU())
        conv_op = nn.Conv2d(last_dimension_size, 3, kernel_size=3, padding='same')
        nn.init.orthogonal_(conv_op.weight)
        self.end_ops.append(conv_op)
        self.end_ops.append(nn.Tanh())

    def set_channels_last(self):
        self.end_ops = self.end_ops.to(memory_format=torch.channels_last)
        self.generator_layers_pre_attn = self.generator_layers_pre_attn.to(memory_format=torch.channels_last)
        self.generator_layers_post_attn = self.generator_layers_post_attn.to(memory_format=torch.channels_last)
        self.nonlocal_block = self.nonlocal_block.to(memory_format=torch.channels_last)

    def forward(self, latent_vector, labels):
        # [B, Z] - Z is size of latent vector
        batch_size = latent_vector.size()[0]
        # [B, embedding_size]
        embed_vector = self.embeddings(labels)

        latent_embed_vector = torch.concat((latent_vector, embed_vector), axis=1)
        # [B, 4*4*16*ngf]
        out = self.initial_linear(latent_embed_vector)

        # [B, 16 * ngf, 4, 4]
        out = out.view(batch_size, self.first_dimension_size, self.base_height, self.base_width)

        gen_input = out, latent_embed_vector
        out, _ = self.generator_layers_pre_attn(gen_input)
        out = self.nonlocal_block(out)
        gen_input = out, latent_embed_vector
        out, _ = self.generator_layers_post_attn(gen_input)
        out = self.end_ops(out)
        return out

    def get_class_embedding(self, label):
        return self.embeddings(label)

    def forward_with_class_embeddings(self, latent_vector, class_embeddings):
        # [B, Z] - Z is size of latent vector
        # batch_size = latent_vector.size(dim=0)
        batch_size = latent_vector.size()[0]

        latent_embed_vector = torch.concat((latent_vector, class_embeddings), axis=1)
        # [B, 4*4*16*ngf]
        out = self.initial_linear(latent_embed_vector)

        # [B, 16 * ngf, 4, 4]
        out = torch.reshape(out, [batch_size, self.first_dimension_size, self.base_height, self.base_width])
        for i, generator_layer in enumerate(self.generator_layers):
            out = generator_layer(out, latent_embed_vector)
            if i == self.nonlocal_block_index:
                out = self.nonlocal_block(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.conv(out)
        out = self.tanh(out)
        return out



