# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:23:38 2020

@author: Gerardo Cervantes

Purpose: Train the GAN (Generative Adversarial Network) model
"""

import torch
from torch.profiler import profile, ProfilerActivity
from torch.utils.tensorboard import SummaryWriter

from src.models import create_model
from src.utils import saver_and_loader
from src.configs import ini_parser
from src.data.data_load import data_loader_from_config, \
    create_latent_vector, get_num_classes
from src.metrics.metrics import Metrics
from PIL import ImageFile
from accelerate import Accelerator
ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import shutil
import logging
import os
import time

TORCH_LOGS="dynamo"
def train(config_file_path: str):
    if not os.path.exists(config_file_path):
        raise OSError('Configuration file path doesn\'t exist:' + config_file_path)

    config = ini_parser.read_with_defaults(config_file_path)

    will_restore_model = saver_and_loader.is_loadable_model(config)
    if will_restore_model:
        run_dir, model_dir, img_dir, profiler_dir = saver_and_loader.get_run_directories(config)
    else:
        run_dir, model_dir, img_dir, profiler_dir = saver_and_loader.create_run_directories(config)

    # Logs training information, everything logged will also be outputted to stdout (printed)
    log_path = os.path.join(run_dir, 'train.log')
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    if not will_restore_model:
        logging.info('Directory "' + run_dir + '" created, training output will be saved here')
        # Copies config file
        shutil.copy(config_file_path, os.path.abspath(run_dir))
        logging.info('Copied config file!')
    else:
        logging.info('Directory ' + run_dir + ' loaded, training output will be saved here')

    # Set device
    train_config = config['TRAIN']
    precision = train_config['precision'].lower()
    precision = 'no' if precision == 'fp32' else precision
    dynamo_backend = 'no'
    if train_config.getboolean('compile'):
        dynamo_backend = 'INDUCTOR'
        
    torch_dtype = torch.float32
    # There is no Torch dtype that is fp8, so cast to float16
    if precision == 'fp16' or precision == 'fp8':
        torch_dtype = torch.float16
    elif precision == 'bf16':
        torch_dtype = torch.bfloat16

    accelerator = Accelerator(mixed_precision=precision, dynamo_backend=dynamo_backend)
    device = accelerator.device
    running_on_cpu = device.type != 'cuda'
    
    # Creates data-loader
    data_config = config['DATA']
    image_column_name, label_column_name = data_config['image_column_name'], data_config['label_column_name']
    data_config['image_height'] = str(int(data_config['base_height']) * (2 ** int(data_config['upsample_layers'])))
    data_config['image_width'] = str(int(data_config['base_width']) * (2 ** int(data_config['upsample_layers'])))

    data_loader = data_loader_from_config(data_config, image_dtype=torch_dtype, using_gpu=not running_on_cpu)
    # Eval data loader is done to keep the same label distribution when evaluating
    eval_data_loader = data_loader_from_config(data_config, image_dtype=torch_dtype, using_gpu=not running_on_cpu)
    # logging.info('Data size is ' + str(len(data_loader)) + ' images')
    # num_train_images = data_loader.info.splits['train'].num_examples
    num_train_images = len(data_loader.dataset)
    logging.info('Data size is ' + str(num_train_images) + ' images')
    batch_size = int(data_config['batch_size'])
    # Save training images
    batched_dataset = data_loader
    # batched_dataset = data_loader.iter(batch_size=int(data_config['batch_size']))
    batched_dataset, eval_data_loader = accelerator.prepare(batched_dataset, eval_data_loader)
    saver_and_loader.save_train_batch(batched_dataset, os.path.join(img_dir, 'train_batch.png'), image_column_name=image_column_name)
    num_classes = get_num_classes(data_loader, label_column_name)
    data_config['num_classes'] = str(num_classes)
    logging.info('Number of different image labels: ' + str(num_classes))
    model_arch_config = config['MODEL ARCHITECTURE']

    logging.info('Creating model...')

    device = accelerator.device
    gan_model = create_model.create_gan_model(model_arch_config, data_config, train_config, accelerator, torch_dtype)
    gan_model.save_architecture(run_dir, data_config)

    logging.info('Created GAN model')
    loaded_step_num = 0
    if will_restore_model:
        loaded_step_num = saver_and_loader.load_model(gan_model, model_dir)
        logging.info('Restored model from step ' + str(loaded_step_num))

    gan_model.accelerate_models()
    logging.info('Is GPU available? ' + str(torch.cuda.is_available()) + ' - Running on device:' + str(device))

    metrics_config = config['METRICS']
    compute_is = metrics_config.getboolean('is_metric')
    compute_fid = metrics_config.getboolean('fid_metric')
    n_images_to_eval = int(metrics_config['n_images_to_eval'])

    metrics_scorer = None
    if compute_is or compute_fid:
        logging.info('Initializing metrics ...')
        metrics_scorer = Metrics(compute_is, compute_fid, device=device)
        metrics_scorer.aggregate_data_loader_images(batched_dataset, n_images_to_eval, device, real=True)
        if not will_restore_model:
            logging.info('Computing metrics for real images ...')
            metrics_scorer.aggregate_data_loader_images(batched_dataset, n_images_to_eval, device, real=False)
            is_score, fid_score = metrics_scorer.score_metrics(compute_is, compute_fid)
            metrics_scorer.reset_metrics()
            metrics_scorer.log_scores(is_score, fid_score)

    def create_noise_and_labels(use_data_distribution=False):
        noise = create_latent_vector(data_config, model_arch_config, device)
        if use_data_distribution:
            _, labels = next(iter(eval_data_loader))
            labels = labels.to(device)
        else:
            labels = torch.arange(start=0, end=batch_size, device=device,
                                    dtype=torch.int16) % num_classes
        return noise, labels
    fixed_noise, fixed_labels = create_noise_and_labels()

    def generate_images():
        noise, labels = create_noise_and_labels(use_data_distribution=True)
        fake_images = gan_model.generate_images(noise, labels)
        return fake_images
    n_epochs, log_steps = int(train_config['num_epochs']), int(train_config['log_steps'])
    save_steps = int(train_config['save_steps'])
    save_steps = None if save_steps == 0 else save_steps
    eval_steps = int(metrics_config['steps_to_eval'])
    eval_steps = None if eval_steps == 0 else eval_steps

    logging.info("Starting Training Loop...")

    def tensorboard_profiler():
        profile_devices = [ProfilerActivity.CPU]
        if not running_on_cpu:
            profile_devices.append(ProfilerActivity.CUDA)
        return profile(activities=profile_devices,
                       schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                       on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
                       record_shapes=True, profile_memory=True), SummaryWriter(profiler_dir)

    profiler, eval_writer = tensorboard_profiler()
    if profiler:
        profiler.start()

    n_steps = 0
    accumulation_iterations = int(train_config['accumulation_iterations'])
    is_channel_last = train_config.getboolean('channels_last')
    steps_in_epoch = int(num_train_images / (accumulation_iterations * batch_size))
    total_g_error, total_d_error = 0.0, 0.0
    g_steps, d_steps = 0, 0
    data_time, model_time = 0.0, 0.0
    data_start_time = time.time()
    train_seq_start_time = time.time()
    logging.info('Started Training Loop')
    batches_accumulated = []
    for epoch in range(n_epochs):
        for i, batch in enumerate(batched_dataset, 0):
            
            if is_channel_last:
                real_data, _ = batch
                real_data = real_data.to(memory_format=torch.channels_last)  # Replace with your input

            batches_accumulated.append(batch)

            data_time += time.time() - data_start_time
            model_start_time = time.time()
            if i % accumulation_iterations == accumulation_iterations-1:
                err_discriminator, err_generator = gan_model.train_step(batches_accumulated)
                n_steps += 1
                # Reset amount of batches accumulated
                batches_accumulated = []

                if err_discriminator:
                    total_d_error += err_discriminator
                    d_steps += 1

                if err_generator:
                    total_g_error += err_generator
                    g_steps += 1

                model_time += time.time() - model_start_time
                total_step_num = loaded_step_num + n_steps
                # Output training stats
                if n_steps % log_steps == 0:
                    d_loss_num = (total_d_error / d_steps) if d_steps else 0
                    g_loss_num = (total_g_error / g_steps) if g_steps else 0
                    d_loss = '{:.4f}'.format(d_loss_num) if d_loss_num else '0 steps'
                    g_loss = '{:.4f}'.format(g_loss_num) if g_loss_num else '0 steps'

                    if eval_writer:
                        eval_writer.add_scalar('Loss/Disciminator/train', d_loss_num, total_step_num)
                        eval_writer.add_scalar('Loss/Generator/train', g_loss_num, total_step_num)
                    logging.info('[{}/{}][{}/{}]\t Loss_D: {}\t Loss_G: {}\t Time: {:.2f}s'.format(
                        epoch, n_epochs, n_steps % steps_in_epoch, steps_in_epoch, d_loss, g_loss,
                        time.time() - train_seq_start_time))
                    
                    logging.info(
                        'Data retrieve time: %.2fs Model updating time: %.2fs' % (data_time, model_time))

                    data_time, model_time = 0, 0
                    total_g_error, total_d_error = 0.0, 0.0
                    g_steps, d_steps = 0, 0
                    train_seq_start_time = time.time()

                # Save every save_steps or every epoch if save_steps is None
                if n_steps % save_steps == 0:
                    save_start_time = time.time()
                    
                    fake_img_output_path = os.path.join(img_dir, 'generated_image_' + str(total_step_num) + '.png')
                    logging.info('Saving fake images: ' + fake_img_output_path)
                    with torch.no_grad():
                        fake_images = gan_model.generate_images(fixed_noise, fixed_labels).cpu()
                        saver_and_loader.save_images(fake_images.to(torch.float32), fake_img_output_path)
                        del fake_images
                    gan_model.save(model_dir, total_step_num, compiled=train_config.getboolean('compile'))
                    logging.info('Time to save images and model: %.2fs ' % (time.time() - save_start_time))

                if n_steps % eval_steps == 0:
                    if metrics_scorer:
                        metric_start_time = time.time()
                        logging.info('Computing metrics ...')

                        metrics_scorer.aggregate_images_from_fn(generate_images, n_images_to_eval, real=False)
                        is_score, fid_score = metrics_scorer.score_metrics(compute_is, compute_fid)
                        metrics_scorer.reset_metrics()
                        metrics_scorer.log_scores(is_score, fid_score)
                        is_score_avg, is_score_std = is_score
                        if eval_writer:
                            eval_writer.add_scalar('Inception Score', is_score_avg, total_step_num)
                            eval_writer.add_scalar('FID', fid_score, total_step_num)

                        logging.info('Time to compute metrics: %.2fs ' % (time.time() - metric_start_time))
                data_start_time = time.time()
                if profiler:
                    profiler.step()
    profiler.stop()
    logging.info('Training complete! Models and output saved in the output directory:')
    logging.info(run_dir)
    return run_dir


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Path to the configuration file')
    parser.add_argument('config_file', type=str,
                        help='The configuration file to train with, configuration files end with .ini extension.\n'
                        'Default config files are in the configs folder.')
    args = parser.parse_args()

    train(args.config_file)
