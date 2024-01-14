## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from util import *
from loss import *
from result import *

def get_driver(cfg, device):
  return DenoiseDriver(cfg, device)

## -----------------------------------------------------------------------------
## Network layers
## -----------------------------------------------------------------------------

# ReLU function
def relu(x):
  return F.relu(x, inplace=True)

# 2x2 max pool function
def pool(x):
  return F.max_pool2d(x, 2, 2)

# 2x2 nearest-neighbor upsample function
def upsample(x):
  return F.interpolate(x, scale_factor=2, mode='nearest')

# Channel concatenation function
def concat(*tensors):
  return torch.cat([t for t in tensors if t is not None], 1)

# 3x3 convolution+ReLU module
class ConvLayer(nn.Conv2d):
  def __init__(self, in_channels, out_channels):
    super().__init__(in_channels, out_channels, 3, padding=1)

  def forward(self, input):
    return relu(super().forward(input))

def Conv(*args, layer=ConvLayer):
  if len(args) == 2:
    return layer(args[0], args[1])
  else:
    return nn.Sequential(*list(layer(args[i], args[i+1]) for i in range(0, len(args)-1)))



ec1  = 32
ec2  = 48
ec3  = 64
ec4  = 80
ec5  = 96
dc4  = 112
dc3  = 96
dc2  = 64
dc1a = 64
dc1b = 32

class UNetEncoder(nn.Module):
  def __init__(self, in_channels):
    super().__init__()

    # Convolutions
    self.conv0 = Conv(in_channels, ec1, ec1)
    self.conv1 = Conv(ec1, ec2)
    self.conv2 = Conv(ec2, ec3)
    self.conv3 = Conv(ec3, ec4)
    self.conv4 = Conv(ec4, ec5)

    # Images must be padded to multiples of the alignment
    self.alignment = 16

  def forward(self, input):
    x = self.conv0(input)
    x = pool1 = pool(x)
    x = self.conv1(x)
    x = pool2 = pool(x)
    x = self.conv2(x)
    x = pool3 = pool(x)
    x = self.conv3(x)
    x = pool(x)
    x = self.conv4(x)

    return x, pool3, pool2, pool1, input

class UNetDecoder(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()

    # Convolutions
    C = lambda *x: Conv(*x, layer=ConvLayer)
    self.conv5 = C(ec5, ec5)
    self.conv4 = C(ec5+ec3, dc4, dc4)
    self.conv3 = C(dc4+ec2, dc3, dc3)
    self.conv2 = C(dc3+ec1, dc2, dc2)
    self.conv1 = C(dc2+in_channels, dc1a, dc1b)
    # No ReLU after last layer
    self.conv0 = nn.Conv2d(dc1b, out_channels, 3, padding=1)

  def forward(self, data):
    x, pool3, pool2, pool1, input = data
    x = self.conv5(x)
    x = concat(upsample(x), pool3)
    x = self.conv4(x)
    x = concat(upsample(x), pool2)
    x = self.conv3(x)
    x = concat(upsample(x), pool1)
    x = self.conv2(x)
    x = concat(upsample(x), input)
    x = self.conv1(x)
    x = self.conv0(x)

    return x









class UNetDenoiser(nn.Module):
  def __init__(self, tonemap, in_channels=3, out_channels=3, layer=ConvLayer):
    super().__init__()
    self.tonemap = tonemap

    self.encoder = UNetEncoder(in_channels, layer=layer)
    self.decoder = UNetDecoder(in_channels, out_channels, layer=layer)

    self.alignment = self.encoder.alignment

  def forward(self, input):
    x = concat(self.tonemap(input[:, 0:3, ...]), input[:, 3:, ...])
    x = self.encoder(x)
    x = self.decoder(x)
    return relu(x)












class DenoiseDriver:
  def __init__(self, cfg, device):
    num_input_channels = len(get_model_channels(cfg.features))

    transfer = get_transfer_function(cfg)
    self.tonemap = lambda c: transfer.forward(torch.clamp(c, min=1e-6))
    self.tonemapInverse = lambda c: transfer.inverse(torch.clamp(c, min=1e-6, max=1.0-1e-6))
    if cfg.model == 'unet':
      self.model = UNetDenoiser(self.tonemap, num_input_channels)
    else:
      assert False
    self.model.to(device)

    self.criterion = get_loss_function(cfg)
    self.criterion.to(device)

  def compute_losses(self, input, target, **_):
    output = self.model(input)

    outputCol = output[:,0:3,...]
    targetCol = self.tonemap(target[:, 0:3, ...])

    loss = self.criterion(outputCol, targetCol)

    return loss, {'loss': loss, 'color_loss': loss}

  def compute_infer(self, input, **_):
    output = self.model(input)
    return self.tonemapInverse(output)
