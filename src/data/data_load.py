# -*- coding: utf-8 -*-
"""
Created on Thu May 17

@author: Gerardo Cervantes

Purpose: Functions that are used to generate and transform image data
"""

import torch
import PIL
import torchvision.datasets as torch_data_set
import torchvision.transforms.v2 as transforms
from datasets import load_dataset, exceptions
import os


def normalize(images, norm_mean=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
              norm_std=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)):
    normalize_transform = torch.nn.Sequential(
        transforms.Normalize(norm_mean, norm_std),
    )
    return normalize_transform(images)


def unnormalize(images, norm_mean=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
                norm_std=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)):
    unnormalize_transform = torch.nn.Sequential(
        transforms.Normalize((-norm_mean / norm_std).tolist(),
                             (1.0 / norm_std).tolist()))
    return unnormalize_transform(images)


def color_transform(images, brightness=0.1, contrast=0.05, saturation=0.1, hue=0.05):
    train_transform_augment = torch.nn.Sequential(
        transforms.ColorJitter(brightness=brightness, contrast=contrast,
                               saturation=saturation, hue=hue),
    )
    return train_transform_augment(images)


def data_loader_from_config(data_config, image_dtype=torch.float32, using_gpu=False):
    data_dir = data_config['train_dir']
    # os_helper.is_valid_dir(data_dir, 'Invalid training data directory\nPath is an invalid directory: ' + data_dir)
    image_height, image_width = get_image_height_and_width(data_config)
    batch_size = int(data_config['batch_size'])
    n_workers = int(data_config['workers'])
    image_column_name, label_column_name = data_config['image_column_name'], data_config['label_column_name']
    return create_data_loader(data_dir, image_height, image_width, image_dtype=image_dtype, using_gpu=using_gpu,
                              batch_size=batch_size, n_workers=n_workers, image_column_name=image_column_name, label_column_name=label_column_name)


def get_image_height_and_width(data_config):
    image_height = int(int(data_config['base_height']) * (2 ** int(data_config['upsample_layers'])))
    image_width = int(int(data_config['base_width']) * (2 ** int(data_config['upsample_layers'])))
    return image_height, image_width


def create_latent_vector(data_config, model_arch_config, device):
    latent_vector_size = int(model_arch_config['latent_vector_size'])
    fixed_noise = torch.randn(int(data_config['batch_size']), latent_vector_size, device=device,
                              requires_grad=False)
    truncation_value = float(model_arch_config['truncation_value'])
    if truncation_value != 0.0:
        # https://github.com/pytorch/pytorch/blob/a40812de534b42fcf0eb57a5cecbfdc7a70100cf/torch/nn/init.py#L153
        fixed_noise = torch.nn.init.trunc_normal_(fixed_noise, a=(truncation_value * -1), b=truncation_value)
    return fixed_noise

def get_num_classes(data_loader, label_column_name):
    return data_loader.dataset.features[label_column_name].num_classes

def create_data_loader(data_path: str, image_height: int, image_width: int, image_dtype=torch.float16, using_gpu=False,
                       batch_size=1, n_workers=1, image_column_name='image', label_column_name='label'):

    data_transform = transforms.Compose([transforms.Resize((image_height, image_width)),
                                         transforms.ToDtype(image_dtype, scale=True), # Float16 is tiny bit faster, and bit more VRAM. Strange.
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                         ])
    if os.path.isdir(data_path):
        if image_column_name != 'image' or label_column_name != 'label':
            raise ValueError("image_column_name or label_column_name overridden, when using a local dataset. \
                             Make sure that it is not being overriden in the configuration file")
        dataset = load_dataset('imagefolder', data_dir=data_path, split='train', drop_labels=False)
    else:
        # try:

        dataset = load_dataset(data_path, split='train', streaming=False) # name='plain_text',
        dataset_columns = dataset.column_names
        if image_column_name not in dataset_columns or label_column_name not in dataset_columns:
            raise ValueError("image_column_name or label_column_name not found from the HuggingFace dataset.\n\
                            HuggingFace Dataset Columns are: " + str(dataset_columns) + "\n\
                            Image Column given: '" + image_column_name + "'\n\
                            Label Column given: '" + label_column_name + "'")
        # except exceptions.DatasetNotFoundError:
        #     raise FileNotFoundError('Data path not found in HuggingFace Datasets or not found in local path.\n'
        #                                 'Directory provided: ' + data_path)
        

    dataset = dataset.with_format("torch")
    # Create the data-loader
    # torch_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True, 
    #                                            num_workers=n_workers, pin_memory=using_gpu, drop_last=True)

    
    # collate_fn passes list of size batch size, with each entry being a dict with keys being column names
    def _preprocess_pt(examples):
        # Permutes from (H x W x C) to (C x H x W) before transforming
        images = [data_transform(torch.permute(example[image_column_name], (2, 0, 1))) for example in examples]
        labels = [example[label_column_name] for example in examples]
        return {image_column_name: torch.stack(images), label_column_name: torch.stack(labels)}
    
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=_preprocess_pt, batch_size=batch_size,
        num_workers=n_workers, pin_memory=using_gpu, drop_last=True
    )
    return dataloader


# Returns images of size: (batch_size, num_channels, height, width)
def get_data_batch(data_loader, device, unnormalize_batch=False, image_column_name='image'):
    if unnormalize_batch:
        return unnormalize(next(iter(data_loader))[0]).to(device)
    # abc = data_loader.iter(batch_size=3)
    # abc2 = next(data_loader)
    # item_a = next(iter(data_loader))
    # return torch.Tensor(next(data_loader)['img'])
    return next(iter(data_loader))[image_column_name].to(device)


# Resize images so width and height are both greater than min_size. Keep images the same if they already are bigger
# Keeps aspect ratio
def upscale_images(images, min_size: int):
    if len(images.size()) != 4:
        raise ValueError("Could not upscale images.  Images should be tensor of size (batch size, n_channels, w, h)")
    height = images.size(dim=2)
    width = images.size(dim=3)

    if width > min_size and height > min_size:
        return images

    ratio_to_upscale = float(min_size / min(width, height))

    if width < height:
        new_width = min_size
        new_height = int(ratio_to_upscale * height)
        # Safety check
        new_height = new_height if new_height >= min_size else min_size
    else:
        new_width = int(ratio_to_upscale * width)
        new_height = min_size
        # Safety check
        new_width = new_width if new_width >= min_size else min_size

    return _antialias_resize(images, new_width, new_height)


# As seen in https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/#evaluation-metrics
def _antialias_resize(batch, width, height):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((width, height), PIL.Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)
