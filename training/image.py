## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import OpenImageIO as oiio
from collections import defaultdict

from ssim import ssim, ms_ssim
from flip import LDRFLIPLoss
from util import *

## -----------------------------------------------------------------------------
## Image operations
## -----------------------------------------------------------------------------

# Converts a NumPy image to a tensor
def image_to_tensor(image, batch=False):
  # Reorder from HWC to CHW
  tensor = torch.from_numpy(image.transpose((2, 0, 1)))
  if batch:
    return tensor.unsqueeze(0) # reshape to NCHW
  else:
    return tensor

# Converts a tensor to a NumPy image
def tensor_to_image(image):
  if len(image.shape) == 4:
    # Remove N dimension
    image = image.squeeze(0)
  # Reorder from CHW to HWC
  return image.cpu().numpy().transpose((1, 2, 0))

def scale_image(input, scale):
  if scale == 1:
    return input.copy()
  w, h, c = input.shape
  # Crop to multiple of size
  input = input[0:(w - w % scale), 0:(h - h % scale), :]
  # Scale
  return input.reshape((w//scale, scale, h//scale, scale, c)).mean(axis=3).mean(axis=1)

# Computes gradient for a tensor
def tensor_gradient(input):
  input0 = input[..., :-1, :-1]
  didy   = input[..., 1:,  :-1] - input0
  didx   = input[..., :-1, 1:]  - input0
  return torch.cat((didy, didx), -3)

# Compares two image tensors using the specified error metric
def compare_images(a, b, metric='psnr'):
  if metric == 'mse':
    return ((a - b) ** 2).mean()
  elif metric == 'psnr':
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(1. / mse.item())
  elif metric == 'ssim':
    return ssim(a, b, data_range=1.)
  elif metric == 'msssim':
    return ms_ssim(a, b, data_range=1.)
  elif metric == 'flip':
    return LDRFLIPLoss()(a, b)
  else:
    raise ValueError('invalid error metric')

## -----------------------------------------------------------------------------
## Image I/O
## -----------------------------------------------------------------------------

def image_get_features(name):
  image = oiio.ImageBuf(name)
  if image.has_error:
    error('could not load image')

  # Get the channels and group them by layer
  channels = image.spec().channelnames
  layer_channels = defaultdict(set)
  for channel in channels:
    if len(channel.split('.')) >= 3:
      layer, p, c = channel.rsplit('.', 2)
      layer_channels[layer].add(f'{p}.{c}')
    else:
      layer_channels[None].add(channel)
  for rem in ('Composite', 'BackLight'):
    if rem in layer_channels:
      del layer_channels[rem]

  # Set default layer
  layer = list(layer_channels.keys())[0] if len(layer_channels) == 1 else None

  # Extract features
  FEATURES = {
    'hdr' : [
              ('R', 'G', 'B'),
              ('Noisy Image.R', 'Noisy Image.G', 'Noisy Image.B'),
              ('Combined.R', 'Combined.G', 'Combined.B'),
              ('Beauty.R', 'Beauty.G', 'Beauty.B')
            ],
    'a' : [('A',)],
    'alb' : [
              ('albedo.R', 'albedo.G', 'albedo.B'),
              ('Denoising Albedo.R', 'Denoising Albedo.G', 'Denoising Albedo.B'),
              ('VisibleDiffuse.R', 'VisibleDiffuse.G', 'VisibleDiffuse.B'),
              ('diffuse.R', 'diffuse.G', 'diffuse.B'),
              ('DiffCol.R', 'DiffCol.G', 'DiffCol.B'),
            ],
    'nrm' : [
              ('normal.R', 'normal.G', 'normal.B'),
              ('normal.X', 'normal.Y', 'normal.Z'),
              ('N.R', 'N.G', 'N.B'),
              ('Denoising Normal.X', 'Denoising Normal.Y', 'Denoising Normal.Z'),
              ('Normals.R', 'Normals.G', 'Normals.B'),
              ('VisibleNormals.R', 'VisibleNormals.G', 'VisibleNormals.B'),
              ('OptixNormals.R', 'OptixNormals.G', 'OptixNormals.B'),
            ],
    'z' : [('Denoising Depth.Z',)],
    'var' : [('Denoising Variance.R', 'Denoising Variance.G', 'Denoising Variance.B')],
    'spp' : [('Debug Sample Count.X',)],
  }

  present_features = {}
  for feature, feature_channel_lists in FEATURES.items():
    for feature_channels in feature_channel_lists:
      # Check whether the feature is present in the selected layer of the image
      if layer:
        feature_channels = tuple([layer + '.' + f for f in feature_channels])
      if set(feature_channels).issubset(channels):
        present_features[feature] = [channels.index(channel) for channel in feature_channels]
        break

  return present_features

def load_image_multilayer(filename, features):
  input = oiio.ImageInput.open(filename)
  if not input:
    raise RuntimeError('could not open image: "' + filename + '"')

  data = {}
  for feature, channels in features.items():
    pixels = [input.read_image(subimage=0, miplevel=0, chbegin=channel, chend=channel + 1, format=oiio.FLOAT) for channel in channels]
    if any(ch is None for ch in pixels):
      raise RuntimeError('could not read image')
    data[feature] = np.nan_to_num(np.concatenate(pixels, axis=2))

  input.close()
  return data

# Loads an image and returns it as a float NumPy array
def load_image(filename, num_channels=None):
  input = oiio.ImageInput.open(filename)
  if not input:
    raise RuntimeError('could not open image: "' + filename + '"')
  if num_channels:
    image = input.read_image(subimage=0, miplevel=0, chbegin=0, chend=num_channels, format=oiio.FLOAT)
  else:
    image = input.read_image(format=oiio.FLOAT)
  if image is None:
    raise RuntimeError('could not read image')
  image = np.nan_to_num(image)
  input.close()
  return image

# Saves a float NumPy image
def save_image(filename, image):
  ext = get_path_ext(filename).lower()
  if ext == 'pfm':
    save_pfm(filename, image)
  elif ext == 'phm':
    save_phm(filename, image)
  else:
    output = oiio.ImageOutput.create(filename)
    if not output:
      raise RuntimeError('could not create image: "' + filename + '"')
    format = oiio.FLOAT if ext == 'exr' else oiio.UINT8
    spec = oiio.ImageSpec(image.shape[1], image.shape[0], image.shape[2], format)
    if image.shape[2] == 5:
      spec.channelnames = ("Image.R", "Image.G", "Image.B", "Image.A", "Extra.X")
    elif image.shape[2] > 4:
      spec.channelnames = tuple(
        (f"Channel{(i//3)*3}.{'RGB'[i%3]}" if ((i // 3)*3 + 2 < image.shape[2]) else f"Channel{i}") for i in range(image.shape[2])
      )
    if ext == 'exr':
      spec.attribute('compression', 'piz')
    elif ext == 'png':
      spec.attribute('png:compressionLevel', 3)
    if not output.open(filename, spec):
      raise RuntimeError('could not open image: "' + filename + '"')
    # FIXME: copy is needed for arrays owned by PyTorch for some reason
    if not output.write_image(image.copy()):
      raise RuntimeError('could not save image')
    output.close()

# Saves a float NumPy image in PFM format
def save_pfm(filename, image):
  with open(filename, 'w') as f:
    num_channels = image.shape[-1]
    if num_channels >= 3:
      f.write('PF\n')
      data = image[..., 0:3]
    else:
      f.write('Pf\n')
      data = image[..., 0]
    data = np.flip(data, 0).astype(np.float32)

    f.write('%d %d\n' % (image.shape[1], image.shape[0]))
    f.write('-1.0\n')
    data.tofile(f)

# Saves a float NumPy image in PHM format
def save_phm(filename, image):
  with open(filename, 'w') as f:
    num_channels = image.shape[-1]
    if num_channels >= 3:
      f.write('PH\n')
      data = image[..., 0:3]
    else:
      f.write('Ph\n')
      data = image[..., 0]
    data = np.flip(data, 0).astype(np.float16)

    f.write('%d %d\n' % (image.shape[1], image.shape[0]))
    f.write('-1.0\n')
    data.tofile(f)