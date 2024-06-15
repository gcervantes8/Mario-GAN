# -*- coding: utf-8 -*-
"""
Created on Thu June 26 11:00:10 2020

@author: Gerardo Cervantes

Purpose: This will handle having both the generator and discriminator, as well as training the models given the data
"""

import torch
from src.data import data_load
from src.losses.loss_functions import supported_loss_functions, supported_losses
from src.optimizers.optim_functions import supported_optimizer_list, optimizer_factory
from torch_ema import ExponentialMovingAverage
from torchinfo import summary
from src.data.data_load import normalize, unnormalize
from src.utils import os_helper
import os

class GanModel:

    def __init__(self, generator, discriminator, num_classes, accelerator, torch_dtype, model_arch_config, train_config, image_column_name, label_column_name):
        
        if train_config.getboolean('channels_last'):
            discriminator.set_channels_last()
            generator.set_channels_last()

        self.num_classes = num_classes
        self.device = accelerator.device
        generator = generator.to(self.device)
        discriminator = discriminator.to(self.device)
        self.torch_dtype = torch_dtype
        self.latent_vector_size = int(model_arch_config['latent_vector_size'])
        beta1 = float(train_config['beta1'])
        beta2 = float(train_config['beta2'])
        generator_wd = float(train_config['generator_wd'])
        discriminator_wd = float(train_config['discriminator_wd'])
        generator_lr = float(train_config['generator_lr'])
        discriminator_lr = float(train_config['discriminator_lr'])
        self.orthogonal_value = float(model_arch_config['orthogonal_value'])
        loss_type = model_arch_config['loss_type']
        self.is_omni_loss = 'omni-loss' == loss_type.lower()
        self.accumulation_iterations = int(train_config['accumulation_iterations'])
        self.mixed_precision = train_config.getboolean('mixed_precision')
        self.batch_iterations = 0
        self.image_column_name = image_column_name
        self.label_column_name = label_column_name
        self.criterion, self.fake_label, self.real_label = supported_loss_functions(train_config['loss_function'])
        if self.criterion is None:
            raise ValueError("Loss given not found: " + train_config['loss_function'] + 
                             "\nPossible options are: " + str(supported_losses()))
        gan_optimizer = optimizer_factory(train_config['optimizer'])
        if not gan_optimizer:
            raise ValueError("Optimizer given not found: " + train_config['optimizer'] + 
                             "\nPossible options are: " + str(supported_optimizer_list()))
        # Setup Adam optimizers for both G and D
        self.optimizerD = gan_optimizer(discriminator.parameters(), lr=discriminator_lr, betas=(beta1, beta2),
                                     weight_decay=discriminator_wd)
        self.optimizerG = gan_optimizer(generator.parameters(), lr=generator_lr, betas=(beta1, beta2),
                                     weight_decay=generator_wd)
        discriminator.zero_grad()
        generator.zero_grad()

        # Set EMA
        ema_enabled = model_arch_config.getboolean('generator_ema')
        ema_decay = model_arch_config['ema_decay']
        self.ema = ExponentialMovingAverage(generator.parameters(),
                                            decay=float(ema_decay)) if ema_enabled else None

        self.netD, self.netG = discriminator, generator
        self.accelerator = accelerator


    def create_real_labels(self, b_size, labels):
        if self.is_omni_loss:
            # Real_label is of size [b_size, num_classes]
            real_label = torch.nn.functional.one_hot(labels, num_classes=self.num_classes)
            # Adds 2 columns, One with all values 1, the second with all values 0. Size [b_size, num_classes + 2]
            real_label = torch.concat((real_label, torch.ones(b_size, 1, device=self.device, dtype=self.torch_dtype),
                                    torch.zeros(b_size, 1, device=self.device, dtype=self.torch_dtype)), dim=1)
            # Replace the 0s and 1s with value of the fake_label, or true_label
            real_label[real_label == 0] = self.fake_label
            real_label[real_label == 1] = self.real_label
        else:
            real_label = torch.full((b_size,), self.real_label, dtype=self.torch_dtype, device=self.device)
        return real_label
    
    def create_fake_labels(self, b_size):
        if self.is_omni_loss:
            fake_label = torch.full((b_size, self.num_classes + 1), self.fake_label, dtype=self.torch_dtype, device=self.device)
            fake_label = torch.concat((fake_label, torch.ones(b_size, 1, device=self.device, dtype=self.torch_dtypee)), dim=1)
        else:
            fake_label = torch.full((b_size,), self.fake_label, dtype=self.torch_dtype, device=self.device)
        return fake_label
    
    def train_step(self, batches_accumulated):
        self.netG.train()
        self.netD.train()
        discriminator_loss, accum_labels = self.discriminator_step(batches_accumulated)
        generator_loss = self.generator_step(accum_labels)
        if self.ema:
            self.ema.update()
        self.netD.requires_grad_(requires_grad=True)
        return discriminator_loss, generator_loss

    def discriminator_step(self, batches_accumulated):
        
        self.netD.requires_grad_(requires_grad=True)
        self.optimizerD.zero_grad(set_to_none=True)
        accum_labels = []
        total_discrim_error = 0
        for batch in batches_accumulated:
            with self.accelerator.no_sync(self.netD):
                
                real_data, labels = batch[self.image_column_name], batch[self.label_column_name]
                b_size = real_data.size(dim=0)
                accum_labels.append(labels)
                
                discrim_output = self.netD(real_data, labels)

                real_label = self.create_real_labels(b_size, labels)
                discrim_on_real_error = self.criterion(discrim_output, real_label)
                discrim_on_real_error = discrim_on_real_error / self.accumulation_iterations

                self.accelerator.backward(discrim_on_real_error)

                # Train with all-fake batch
                noise = torch.randn(b_size, self.latent_vector_size, device=self.device, dtype=real_data.dtype)
                with torch.no_grad():
                    # Generate fake image batch with G
                    fake = self.netG(noise, labels)
                    
                # Classify all fake batch with D
                fake_output = self.netD(fake.detach(), labels)

                # Calculate D's loss on the all-fake batch
                fake_label = self.create_fake_labels(b_size)
                discrim_on_fake_error = self.criterion(fake_output, fake_label.reshape_as(fake_output))
                discrim_on_fake_error = discrim_on_fake_error / self.accumulation_iterations
                self.accelerator.backward(discrim_on_fake_error)

                total_discrim_error += discrim_on_real_error + discrim_on_fake_error
        self.optimizerD.step()
        self.optimizerD.zero_grad(set_to_none=True)
        return total_discrim_error.item(), accum_labels

    def generator_step(self, accum_labels):
        self.netD.requires_grad_(requires_grad=False)
        self.optimizerG.zero_grad(set_to_none=True)

        total_generator_error = 0
        for label in accum_labels:
            with self.accelerator.no_sync(self.netG):
                # with self.accelerator.autocast():
                b_size = label.size(dim=0)
                # Train with all-fake batch
                noise = torch.randn(b_size, self.latent_vector_size, device=self.device, requires_grad=True, dtype=self.torch_dtype)
                # Generate fake image batch with G
                fake = self.netG(noise, label)
                self.netD.requires_grad_(requires_grad=False)
                output_update = self.netD(fake, label)
                self.netD.requires_grad_(requires_grad=True)
                real_label = self.create_real_labels(b_size, label)
                generator_error = self.criterion(output_update, real_label) / self.accumulation_iterations

                self.accelerator.backward(generator_error)
                total_generator_error += generator_error

            if not self.orthogonal_value == 0:
                with self.accelerator.autocast():
                    generator_error = self.apply_orthogonal_regularization(self.netG)
                    total_generator_error += generator_error
                self.accelerator.backward(generator_error)
        self.optimizerG.step()
        self.optimizerG.zero_grad(set_to_none=True)
        return total_generator_error.item()

    # Accelerate Models should be done after loading the models
    def accelerate_models(self):
        self.netD, self.netG, self.optimizerD, self.optimizerG = self.accelerator.prepare(
            self.netD, self.netG, self.optimizerD, self.optimizerG
        )

    def apply_orthogonal_regularization(self, model):
        model_orthogonal_loss = 0.0
        for name, param in model.named_parameters():
            if len(param.size()) >= 2 and param.requires_grad and 'conv' in name:
                param = param.view(param.size(dim=0), -1)
                mult_out = torch.mm(torch.t(param), param)
                orthogonal_matrix = torch.mul(mult_out, 1 - torch.eye(mult_out.size(dim=0), device=self.device))
                # Frobenius norm
                model_orthogonal_loss += self.orthogonal_value * torch.norm(orthogonal_matrix, p='fro')
        return model_orthogonal_loss

    # If class_embeddings is provided, then it will ignore labels to generate the images
    def generate_images(self, noise, labels, class_embeddings=None, unnormalize_img=True):
        self.netG.eval() # Set Generater to Eval mode
        with torch.no_grad():
            if self.ema:
                with self.ema.average_parameters():
                    if class_embeddings:
                        forward_with_class_embeddings = getattr(self.netG, "forward_with_class_embeddings", None)
                        if callable(forward_with_class_embeddings):
                            fake = forward_with_class_embeddings(noise, class_embeddings)
                        else:
                            raise ValueError('Given generator model does not support providing class embeddings '
                                                'directly.')
                    else:
                        fake = self.netG(noise, labels)
            else:
                fake = self.netG(noise, labels)

            fake = fake.detach()
            if unnormalize_img:
                fake = unnormalize(fake)
            return fake

    def save(self, model_dir, step_num, compiled=False):
        generator_path = os.path.join(model_dir, os_helper.ModelType.GENERATOR.value + '_step_' + str(step_num) + '.pt')
        discrim_path = os.path.join(model_dir, os_helper.ModelType.DISCRIMINATOR.value + '_step_' +
                                    str(step_num) + '.pt')

        # Save uncompiled model, so you can go from compile to uncompiled
        if compiled:
            netG = self.netG._orig_mod
            netD = self.netD._orig_mod
        else:
            netG = self.netG
            netD = self.netD
        torch.save(netG.state_dict(), generator_path)
        torch.save(netD.state_dict(), discrim_path)
        if self.ema:
            ema_path = os.path.join(model_dir, os_helper.ModelType.EMA.value + '_step_' + str(step_num) + '.pt')
            torch.save(self.ema.state_dict(), ema_path)

    def load(self, generator_path, discrim_path, ema_path=None):
        self.netG.load_state_dict(torch.load(generator_path))
        self.netD.load_state_dict(torch.load(discrim_path))
        if self.ema and ema_path:
            self.ema.load_state_dict(torch.load(ema_path))

    # Writes text file with information of the generator and discriminator instances
    def save_architecture(self, save_dir, data_config):

        image_height, image_width = data_load.get_image_height_and_width(data_config)
        discriminator_stats = summary(self.netD, input_data=[torch.zeros((1, 3, image_height, image_width)),
                                                             torch.zeros(1, dtype=torch.int64)], verbose=0,
                                      device=self.device)
        generator_stats = summary(self.netG, input_data=[torch.zeros(1, self.latent_vector_size),
                                                         torch.zeros(1, dtype=torch.int64)], verbose=0,
                                  device=self.device)

        with open(os.path.join(save_dir, 'architecture.txt'), 'w', encoding='utf-8') as text_file:
            text_file.write('Generator\n\n')
            text_file.write(str(generator_stats))
            text_file.write(str(self.netG))
            text_file.write('\n\nDiscriminator\n\n')
            text_file.write(str(discriminator_stats))
            text_file.write(str(self.netD))
