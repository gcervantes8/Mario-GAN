# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that are used to create/init the GAN model
"""

import torch
import torch.nn as nn
from src.data import data_load
from src.models.gan_model import GanModel

from src.models.dcgan.dcgan_generator import DcganGenerator
from src.models.biggan.biggan_generator import BigganGenerator
from src.models.deep_biggan.deep_biggan_generator import DeepBigganGenerator

from src.models.dcgan.dcgan_discriminator import DcganDiscriminator
from src.models.biggan.biggan_discriminator import BigganDiscriminator
from src.models.deep_biggan.deep_biggan_discriminator import DeepBigganDiscriminator


# Returns (project_labels: bool, output_size: int)
def get_omni_loss(loss_type: str, num_classes: int):
    loss_type = loss_type.lower()
    losses_supported = {
        'adversarial': (True, 1),
        'omni-loss': (False, num_classes + 2),
    }
    if loss_type not in losses_supported:
        raise ValueError("Given loss type in config file is not supported\n" +
                         'Supported loss types: ' + str(list(losses_supported.keys())))
    return losses_supported[loss_type]


def create_gen_and_discrim(model_name: str):
    model_name = model_name.lower()
    models_supported = {
        'dcgan': (DcganGenerator, DcganDiscriminator),
        'biggan': (BigganGenerator, BigganDiscriminator),
        'deep_biggan': (DeepBigganGenerator, DeepBigganDiscriminator),
    }
    if model_name not in models_supported:
        raise ValueError("Given model name in config file is not supported\n" +
                         'Supported models: ' + str(list(models_supported.keys())))
    return models_supported[model_name]


def get_device(n_gpus):
    device = torch.device('cuda' if (torch.cuda.is_available() and n_gpus > 0) else 'cpu')
    return device


# Creates the generator and discriminator using the configuration file
def create_gan_instances(model_arch_config, data_config):

    model_type = model_arch_config['model_type']
    num_channels = int(data_config['num_channels'])
    latent_vector_size = int(model_arch_config['latent_vector_size'])
    ngf = int(model_arch_config['ngf'])
    ndf = int(model_arch_config['ndf'])
    loss_type = model_arch_config['loss_type']

    base_width = int(data_config['base_width'])
    base_height = int(data_config['base_height'])
    upsample_layers = int(data_config['upsample_layers'])
    num_classes = int(data_config['num_classes'])

    project_labels, output_size = get_omni_loss(loss_type, num_classes)
    generator, discriminator = create_gen_and_discrim(model_type)
    # Create the generator and discriminator
    generator = generator(base_width, base_height, upsample_layers, latent_vector_size, ngf, num_channels,
                          num_classes)
    discriminator = discriminator(base_width, base_height, upsample_layers, ndf, num_channels,
                                  num_classes, output_size=output_size, project_labels=project_labels)

    return generator, discriminator


def create_gan_model(model_arch_config, data_config, train_config, accelerator, torch_dtype):
    generator, discriminator = create_gan_instances(model_arch_config, data_config)
    num_classes = int(data_config['num_classes'])
    image_column_name = data_config['image_column_name']
    label_column_name = data_config['label_column_name']
    
    return GanModel(generator, discriminator, num_classes, accelerator, torch_dtype, model_arch_config, train_config, image_column_name, label_column_name)

# custom weights initialization, used by the generator and discriminator
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
