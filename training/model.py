## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from util import *

def get_model(cfg):
  type = cfg.model
  num_input_channels = len(get_model_channels(cfg.features))
  if type == 'unet':
    return UNet(num_input_channels)
  elif type == 'unet_error':
    return UNet(num_input_channels, error_prediction=True)
  else:
    error('invalid model')

## -----------------------------------------------------------------------------
## Network layers
## -----------------------------------------------------------------------------

# 3x3 convolution module
def Conv(in_channels, out_channels):
  return nn.Conv2d(in_channels, out_channels, 3, padding=1)

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
def concat(a, b):
  return torch.cat((a, b), 1)

## -----------------------------------------------------------------------------
## U-Net model
## -----------------------------------------------------------------------------

class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, error_prediction=False):
    super(UNet, self).__init__()

    # Number of channels per layer
    ic   = in_channels
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
    oc   = out_channels
    errc5 = 48
    errc4  = 56
    errc3  = 48
    errc2  = 32
    errc1a = 32
    errc1b = 16

    # Convolutions
    self.enc_conv0  = Conv(ic,      ec1)
    self.enc_conv1  = Conv(ec1,     ec1)
    self.enc_conv2  = Conv(ec1,     ec2)
    self.enc_conv3  = Conv(ec2,     ec3)
    self.enc_conv4  = Conv(ec3,     ec4)
    self.enc_conv5a = Conv(ec4,     ec5)
    self.enc_conv5b = Conv(ec5,     ec5)
    self.dec_conv4a = Conv(ec5+ec3, dc4)
    self.dec_conv4b = Conv(dc4,     dc4)
    self.dec_conv3a = Conv(dc4+ec2, dc3)
    self.dec_conv3b = Conv(dc3,     dc3)
    self.dec_conv2a = Conv(dc3+ec1, dc2)
    self.dec_conv2b = Conv(dc2,     dc2)
    self.dec_conv1a = Conv(dc2+ic,  dc1a)
    self.dec_conv1b = Conv(dc1a,    dc1b)
    self.dec_conv0  = Conv(dc1b,    oc)

    self.error_prediction = error_prediction
    if error_prediction:
      self.err_conv5b = Conv(ec5,       errc5)
      self.err_conv4a = Conv(errc5+ec3, errc4)
      self.err_conv4b = Conv(errc4,     errc4)
      self.err_conv3a = Conv(errc4+ec2, errc3)
      self.err_conv3b = Conv(errc3,     errc3)
      self.err_conv2a = Conv(errc3+ec1, errc2)
      self.err_conv2b = Conv(errc2,     errc2)
      self.err_conv1a = Conv(errc2+ic,  errc1a)
      self.err_conv1b = Conv(errc1a,    errc1b)
      self.err_conv0  = Conv(errc1b,    1)

    # Images must be padded to multiples of the alignment
    self.alignment = 16

  def forward(self, input):
    # Encoder
    # -------------------------------------------

    x = relu(self.enc_conv0(input))  # enc_conv0

    x = relu(self.enc_conv1(x))      # enc_conv1
    x = pool1 = pool(x)              # pool1

    x = relu(self.enc_conv2(x))      # enc_conv2
    x = pool2 = pool(x)              # pool2

    x = relu(self.enc_conv3(x))      # enc_conv3
    x = pool3 = pool(x)              # pool3

    x = relu(self.enc_conv4(x))      # enc_conv4
    x = pool(x)                      # pool4

    # Bottleneck
    x = bottleneck = relu(self.enc_conv5a(x))     # enc_conv5a
    x = relu(self.enc_conv5b(x))     # enc_conv5b

    # Decoder
    # -------------------------------------------

    x = upsample(x)                  # upsample4
    x = concat(x, pool3)             # concat4
    x = relu(self.dec_conv4a(x))     # dec_conv4a
    x = relu(self.dec_conv4b(x))     # dec_conv4b

    x = upsample(x)                  # upsample3
    x = concat(x, pool2)             # concat3
    x = relu(self.dec_conv3a(x))     # dec_conv3a
    x = relu(self.dec_conv3b(x))     # dec_conv3b

    x = upsample(x)                  # upsample2
    x = concat(x, pool1)             # concat2
    x = relu(self.dec_conv2a(x))     # dec_conv2a
    x = relu(self.dec_conv2b(x))     # dec_conv2b

    x = upsample(x)                  # upsample1
    x = concat(x, input)             # concat1
    x = relu(self.dec_conv1a(x))     # dec_conv1a
    x = relu(self.dec_conv1b(x))     # dec_conv1b

    x = self.dec_conv0(x)            # dec_conv0

    if not self.error_prediction:
      return x

    # Error Prediction
    # -------------------------------------------
    e = relu(self.err_conv5b(bottleneck))  # err_conv5b

    e = upsample(e)                        # upsample4
    e = concat(e, pool3)                   # concat4
    e = relu(self.err_conv4a(e))           # err_conv4a
    e = relu(self.err_conv4b(e))           # err_conv4b

    e = upsample(e)                        # upsample3
    e = concat(e, pool2)                   # concat3
    e = relu(self.err_conv3a(e))           # err_conv3a
    e = relu(self.err_conv3b(e))           # err_conv3b

    e = upsample(e)                        # upsample2
    e = concat(e, pool1)                   # concat2
    e = relu(self.err_conv2a(e))           # err_conv2a
    e = relu(self.err_conv2b(e))           # err_conv2b

    e = upsample(e)                        # upsample1
    e = concat(e, input)                   # concat1
    e = relu(self.err_conv1a(e))           # err_conv1a
    e = relu(self.err_conv1b(e))           # err_conv1b

    e = F.softplus(self.err_conv0(e))      # err_conv0

    return x, e