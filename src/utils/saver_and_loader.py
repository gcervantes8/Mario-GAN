# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that can save output into file or loads models.
"""

import os
import torch
import torchvision.utils as torch_utils
from src.models import create_model
from src.data.data_load import get_data_batch, unnormalize
from src.utils import os_helper


def get_model_directory_names():
    model_dir_name = 'models'
    images_dir_name = 'images'
    profiler_dir_name = 'profiler'
    return model_dir_name, images_dir_name, profiler_dir_name


def is_loadable_model(config):
    models_dir, model_name = _get_model_dir(config)
    if not model_name:
        return False
    model_dir_name, images_dir_name, profiler_dir_name = get_model_directory_names()

    run_dir = os.path.join(models_dir, model_name)
    path_to_models = os.path.join(run_dir, model_dir_name)
    has_model_directory = os.path.isdir(path_to_models)
    will_restore_model = False
    if has_model_directory:
        # If directory is not empty
        if os.listdir(path_to_models):
            will_restore_model = True
    return will_restore_model


def create_run_directories(config):
    models_dir, model_name = _get_model_dir(config)

    model_dir_name, images_dir_name, profiler_dir_name = get_model_directory_names()
    if model_name:
        run_dir = os_helper.create_dir(models_dir, model_name)
    else:
        run_dir, run_id = os_helper.create_run_dir(models_dir)
    img_dir = os_helper.create_dir(run_dir, images_dir_name)
    model_dir = os_helper.create_dir(run_dir, model_dir_name)
    profiler_dir = os_helper.create_dir(run_dir, profiler_dir_name)
    return run_dir, model_dir, img_dir, profiler_dir


# Used when loading models
# Returns run, model, image, and profiler directory as strings, model should already be created
def get_run_directories(config):
    models_dir, model_name = _get_model_dir(config)
    model_dir_name, images_dir_name, profiler_dir_name = get_model_directory_names()

    run_dir = os.path.join(models_dir, model_name)
    will_restore_model = os.path.isdir(os.path.join(run_dir, model_dir_name))

    if will_restore_model:
        img_dir = os.path.join(run_dir, images_dir_name)
        model_dir = os.path.join(run_dir, model_dir_name)
        profiler_dir = os.path.join(run_dir, profiler_dir_name)
    else:
        raise ValueError("Model directory not found, maybe the model wasn't trained?")
    return run_dir, model_dir, img_dir, profiler_dir


def _get_model_dir(config):
    # Creates the run directory in the output folder specified in the configuration file
    model_config = config['MODEL']
    models_dir = model_config['models_dir']
    model_name = model_config['model_name'] if 'model_name' in model_config else None
    os_helper.is_valid_dir(models_dir, 'Model directory is invalid\nPath is not a directory: ' + models_dir)
    return models_dir, model_name


# Takes in pytorch's data-loader, and saves the training images in given directory
def save_train_batch(data_loader, save_path: str, device=torch.device('cpu'), image_column_name='image'):
    # Save training images
    batch_images = get_data_batch(data_loader, device, image_column_name=image_column_name)
    batch_images = unnormalize(batch_images).to(torch.float32)
    # batch_images = color_transform(batch_images.to(torch.float32))
    save_images(batch_images, save_path)


# Tensor should be of shape (batch_size, n_channels, height, width) as outputted by pytorch Data-Loader
def save_images(tensor, save_path):
    torch_utils.save_image(tensor, save_path)


def load_model(gan_model, model_dir):
    generator_path, gen_step = os_helper.get_step_model(model_dir, os_helper.ModelType.GENERATOR)
    discrim_path, discrim_step = os_helper.get_step_model(model_dir, os_helper.ModelType.DISCRIMINATOR)
    ema_path = None
    if gan_model.ema:
        ema_path, _ = os_helper.get_step_model(model_dir, os_helper.ModelType.EMA)
    gan_model.load(generator_path, discrim_path, ema_path)
    return max(gen_step, discrim_step)


# Creates the generator and loads the given values from the model file
# Returns 2-tuple (loaded generator instance, device loaded with), device can be GPU or CPU
def load_generator(config, generator_path):
    model_arch_config = config['MODEL ARCHITECTURE']
    data_config = config['DATA']
    n_gpus = int(config['MACHINE']['ngpu'])
    running_on_gpu = torch.cuda.is_available() and n_gpus > 0
    device = torch.device('cuda' if running_on_gpu else 'cpu')
    generator, discriminator = create_model.create_gan_instances(model_arch_config, data_config, device, n_gpus)
    generator.load_state_dict(torch.load(generator_path))
    return generator, device
