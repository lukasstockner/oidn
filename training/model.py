## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from util import *
from loss import *
from result import *

def get_driver(cfg, device, use_varprop=False):
  if cfg.model == 'errpredunet':
    return ErrPredDriver(cfg, device)
  elif use_varprop and cfg.model == 'kpcn':
    return DenoiseDriverKPCNVar(cfg, device)
  elif use_varprop and cfg.model == 'unet':
    return DenoiseDriverJVP(cfg, device)
  else:
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
def upsample_lin(x):
  return F.interpolate(x, scale_factor=2, mode='bilinear')

def downsample(x):
  return F.avg_pool2d(x, (2, 2))

# Channel concatenation function
def concat(*tensors):
  return torch.cat([t for t in tensors if t is not None], 1)

# 3x3 convolution+ReLU module
class ConvLayer(nn.Conv2d):
  def __init__(self, in_channels, out_channels):
    super().__init__(in_channels, out_channels, 3, padding=1)

  def forward(self, input):
    return relu(super().forward(input))

# 3x3 batch normalization convolution module
class BNConvLayer(nn.Conv2d):
  def __init__(self, in_channels, out_channels):
    super().__init__(in_channels, out_channels, 3, padding=1)
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, input):
    return self.bn(relu(super().forward(input)))

# 3x3 residual convolution module with BN
class ResBlock(nn.Module):
  def __init__(self, *counts):
    super().__init__()
    self.adaption = nn.Conv2d(counts[0], counts[-1], 1) if (counts[0] != counts[-1]) else nn.Identity()
    layers = []
    for i in range(len(counts)-1):
      layers.append(nn.BatchNorm2d(counts[i]))
      layers.append(nn.ReLU(inplace=True))
      layers.append(nn.Conv2d(counts[i], counts[i+1], 3, padding=1))
    self.residual = nn.Sequential(*layers)

  def forward(self, input):
    return self.adaption(input) + self.residual(input)

def Conv(*args, layer=ConvLayer):
  if layer == ResBlock:
    return ResBlock(*args)
  elif len(args) == 2:
    return layer(args[0], args[1])
  else:
    return nn.Sequential(*list(layer(args[i], args[i+1]) for i in range(0, len(args)-1)))



class PoolLayer(nn.MaxPool2d):
  def __init__(self):
    super().__init__((2, 2))

# ConvLayerJ and PoolLayerJ are used to propagate gradients for efficient JVP evaluation
class ConvLayerJ(torch.nn.Conv2d):
  N = 4

  def __init__(self, in_channels, out_channels):
    super().__init__(in_channels, out_channels, 3, padding=1)

  def forward(self, input):
    x = super().forward(input)

    N = ConvLayerJ.N+1
    assert input.shape[0] % N == 0
    K = input.shape[0] // N

    mask = x[0:K,...] > 0.0
    x = torch.where(mask.repeat(N, 1, 1, 1), x, 0.0)

    return x

class PoolLayerJ(nn.Module):
  def forward(self, x):
    N = ConvLayerJ.N+1
    assert x.shape[0] % N == 0
    K = x.shape[0] // N

    mask = x[0:K, ..., 0::2] > x[0:K, ..., 1::2]
    x = torch.where(mask.repeat(N, 1, 1, 1), x[..., 0::2], x[..., 1::2])

    mask = x[0:K, ..., 0::2, :] > x[0:K, ..., 1::2, :]
    x = torch.where(mask.repeat(N, 1, 1, 1), x[..., 0::2, :], x[..., 1::2, :])

    return x

class KernelPredictor(nn.Module):
  def __init__(self, in_channels):
    super().__init__()

    self.radius = 10
    self.diameter = 2*self.radius + 1
    self.n_support = self.diameter*self.diameter
    self.padding = (self.diameter - 1) // 2
    self.intermediate = in_channels * int(math.sqrt(self.n_support / in_channels))

    self.expand1 = nn.Conv2d(in_channels, self.intermediate, 1)
    self.expand2 = nn.Conv2d(self.intermediate, self.n_support, 1)

  def forward(self, color, data, square=False):
    b, _, w, h = color.shape

    # Expand data to kernel
    kernel = self.expand2(relu(self.expand1(data)))

    # Normalize kernel
    kernel = nn.functional.softmax(kernel, dim=1)

    if square:
      kernel = torch.square_(kernel)

    if True:
      # Manually perform convolution to avoid torch.nn.functional.unfold memory requirements
      output = torch.zeros_like(color)
      color = torch.nn.functional.pad(color, (self.radius, self.radius, self.radius, self.radius))
      i = 0
      for dx in range(0, self.diameter):
        for dy in range(0, self.diameter):
          output = torch.addcmul(output, 1.0, color[:, :, dx:w+dx, dy:h+dy], kernel[:, i:i+1, :, :])
          i += 1
    else:
      kernel = kernel.view(b, self.n_support, -1)

      # Reshape colors
      color = torch.nn.functional.unfold(color, (self.diameter, self.diameter), padding=self.padding)
      color = color.view(b, 3, self.n_support, -1)

      # Perform dot product over kernel support
      kernel = kernel.unsqueeze(1).repeat(1, 3, 1, 1)
      output = torch.sum(color * kernel, 2)

      # Reshape output
      output = output.view(b, 3, w, h)

    return output

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
  def __init__(self, in_channels, layer=ConvLayer):
    super().__init__()

    # Convolutions
    C = lambda *x: Conv(*x, layer=layer)
    self.conv0 = C(in_channels, ec1, ec1)
    self.conv1 = C(ec1, ec2)
    self.conv2 = C(ec2, ec3)
    self.conv3 = C(ec3, ec4)
    self.conv4 = C(ec4, ec5)

    self.pool = (PoolLayerJ if layer == ConvLayerJ else PoolLayer)()

    # Images must be padded to multiples of the alignment
    self.alignment = 16

  def forward(self, input):
    x = self.conv0(input)
    x = pool1 = self.pool(x)
    x = self.conv1(x)
    x = pool2 = self.pool(x)
    x = self.conv2(x)
    x = pool3 = self.pool(x)
    x = self.conv3(x)
    x = self.pool(x)
    x = self.conv4(x)

    return x, pool3, pool2, pool1, input

class UNetDecoder(nn.Module):
  def __init__(self, in_channels, out_channels, want_intermediate=False, scale=1, layer=ConvLayer):
    super().__init__()
    self.want_intermediate = want_intermediate

    # Convolutions
    C = lambda *x: Conv(*x, layer=layer)
    # Don't use residual or BN for last decoder stage, it might shift color values
    C_last = lambda *x: Conv(*x, layer=layer if layer == ConvLayerJ else ConvLayer)
    self.conv5 = C(ec5, ec5//scale)
    self.conv4 = C(ec5//scale+ec3, dc4//scale, dc4//scale)
    self.conv3 = C(dc4//scale+ec2, dc3//scale, dc3//scale)
    self.conv2 = C(dc3//scale+ec1, dc2//scale, dc2//scale)
    self.conv1 = C_last(dc2//scale+in_channels, dc1a//scale, dc1b//scale)
    # No ReLU after last layer
    self.conv0 = nn.Conv2d(dc1b//scale, out_channels, 3, padding=1)

  def forward(self, data):
    x, pool3, pool2, pool1, input = data
    x = self.conv5(x)
    x = concat(upsample(x), pool3)
    x = self.conv4(x)
    x = concat(upsample(x), pool2)
    x = int2 = self.conv3(x)
    x = concat(upsample(x), pool1)
    x = int1 = self.conv2(x)
    x = concat(upsample(x), input)
    x = self.conv1(x)
    x = self.conv0(x)

    if self.want_intermediate:
      return x, int1, int2
    else:
      return x









class UNetDenoiser(nn.Module):
  def __init__(self, tonemap, in_channels=3, out_channels=3, layer=ConvLayer, lastReLU=True):
    super().__init__()
    self.tonemap = tonemap

    self.encoder = UNetEncoder(in_channels, layer=layer)
    self.decoder = UNetDecoder(in_channels, out_channels, layer=layer)

    self.alignment = self.encoder.alignment
    self.lastReLU = lastReLU

  def forward(self, input):
    x = concat(self.tonemap(input[:, 0:3, ...]), input[:, 3:, ...])
    x = self.encoder(x)
    x = self.decoder(x)
    if self.lastReLU:
      x = relu(x)
    return x

class KPCNDenoiser(nn.Module):
  def __init__(self, tonemap, in_channels=3):
    super().__init__()
    self.tonemap = tonemap

    self.encoder = UNetEncoder(in_channels)
    self.decoder = UNetDecoder(in_channels, 32, want_intermediate=True)

    self.kernel2 = KernelPredictor(dc3)
    self.kernel1 = KernelPredictor(dc2-1)
    self.kernel0 = KernelPredictor(32-1)

    self.alignment = self.encoder.alignment

  def multiscaleKernel(self, kernel, data, image, prev):
    kernelData = data[:, 0:-1, ...]
    correctionWeight = torch.unsqueeze(data[:, -1, ...], 1)

    # Predict kernel and apply it to the image
    filtered = kernel(image, kernelData)

    # Multi-scale reconstruction: Remove coarse detail from the filtered
    # image, replace it with the lower-scale input
    correction = upsample_lin(prev - downsample(filtered))

    return filtered + correctionWeight * correction

  def forward(self, input, variance=None):
    fc = input[:, 0:3, ...]
    hc = downsample(fc)
    qc = downsample(hc)

    input = concat(self.tonemap(input[:, 0:3, ...]), input[:, 3:, ...])
    fk, hk, qk = self.decoder(self.encoder(input))

    filtered = self.kernel2(qc, qk)
    filtered = self.multiscaleKernel(self.kernel1, hk, hc, filtered)
    filtered = self.multiscaleKernel(self.kernel0, fk, fc, filtered)

    if variance is not None:
      return self.tonemap(filtered), self.kernel0(variance, fk[:, :-1, ...], square=True)

    return self.tonemap(filtered)








class UNetErrorPredictor(nn.Module):
  def __init__(self, tonemap, in_channels=3, out_channels=3):
    super().__init__()
    self.tonemap = tonemap

    self.encoder = UNetEncoder(in_channels)
    self.color_decoder = UNetDecoder(in_channels, out_channels)
    self.error_decoder = UNetDecoder(in_channels, 1, scale=2)

    self.alignment = self.encoder.alignment

  def forward(self, input):
    x = concat(self.tonemap(input[:, 0:3, ...]), input[:, 3:, ...])
    x = self.encoder(x)
    cx = self.color_decoder(x)
    ex = self.error_decoder(x)

    color = relu(cx)
    error = nn.functional.softplus(ex)

    return color, error

class ErrPredDriver:
  def __init__(self, cfg, device):
    num_input_channels = len(get_model_channels(cfg.features))

    transfer = get_transfer_function(cfg)
    self.tonemap = lambda c: transfer.forward(torch.clamp(c, min=1e-6))
    self.tonemapInverse = lambda c: transfer.inverse(torch.clamp(c, min=1e-6, max=1.0-1e-6))
    self.model = UNetErrorPredictor(self.tonemap, num_input_channels)
    self.model.to(device)

    if cfg.base_model:
      checkpoint = torch.load(cfg.base_model, map_location=device)
      ms = filter_model_state(checkpoint['model_state'], mappings={'decoder.': 'color_decoder.'})
      self.model.load_state_dict(ms, strict=False)

    self.criterion = get_loss_function(cfg)
    self.criterion.to(device)

    self.errorCriterion = ErrorLoss()
    self.errorCriterion.to(device)

  def compute_losses(self, input, target, epoch, **_):
    outColor, outError = self.model(input)

    targetCol = self.tonemap(target[:, 0:3, ...])

    color_loss = self.criterion(outColor, targetCol)
    err_loss = self.errorCriterion(targetCol, outColor, outError)

    loss = color_loss + 1e-2 * err_loss
    return loss, {'loss': loss, 'color_loss': color_loss, 'error_loss': err_loss}

  def compute_infer(self, input, **_):
    color, error = self.model(input)
    return self.tonemapInverse(color), error


class DenoiseDriver:
  def __init__(self, cfg, device):
    num_input_channels = len(get_model_channels(cfg.features))

    transfer = get_transfer_function(cfg)
    self.tonemap = lambda c: transfer.forward(torch.clamp(c, min=1e-6))
    self.tonemapInverse = lambda c: transfer.inverse(torch.clamp(c, min=1e-6, max=1.0-1e-6))
    if cfg.model == 'unet':
      self.model = UNetDenoiser(self.tonemap, num_input_channels)
    elif cfg.model == 'kpcn':
      self.model = KPCNDenoiser(self.tonemap, num_input_channels)
    elif cfg.model == 'resunet':
      self.model = UNetDenoiser(self.tonemap, num_input_channels, layer=ResBlock)
    elif cfg.model == 'bnunet':
      self.model = UNetDenoiser(self.tonemap, num_input_channels, layer=BNConvLayer)
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
    return self.tonemapInverse(output), None


class DenoiseDriverKPCNVar:
  def __init__(self, cfg, device):
    num_input_channels = len(get_model_channels(cfg.features))

    transfer = get_transfer_function(cfg)
    self.tonemap = lambda c: transfer.forward(torch.clamp(c, min=1e-6))
    self.tonemapInverse = lambda c: transfer.inverse(torch.clamp(c, min=1e-6, max=1.0-1e-6))
    if cfg.model == 'kpcn':
      self.model = KPCNDenoiser(self.tonemap, num_input_channels)
    else:
      assert False
    self.model.to(device)

    self.criterion = get_loss_function(cfg)
    self.criterion.to(device)

  def compute_losses(self, input, target, **_):
    raise NotImplementedError

  def compute_infer(self, input, variance=None, **_):
    output, outVariance = self.model(input, variance=variance)
    stddev = torch.sqrt(outVariance)
    _, stddev = tonemap_transfer(self.tonemap, output, stddev)
    stddev = channel_mean(stddev)
    return self.tonemapInverse(output), stddev


class DenoiseDriverJVP:
  def __init__(self, cfg, device):
    num_input_channels = len(get_model_channels(cfg.features))

    transfer = get_transfer_function(cfg)
    self.tonemap = lambda c: transfer.forward(torch.clamp(c, min=1e-6))
    self.tonemapInverse = lambda c: transfer.inverse(torch.clamp(c, min=1e-6, max=1.0-1e-6))
    if cfg.model == 'unet':
      self.model = UNetDenoiser(lambda x: x, num_input_channels, layer=ConvLayerJ, lastReLU=False)
    else:
      assert False
    self.model.to(device)

  def compute_losses(self, **_):
    raise NotImplementedError

  def compute_infer(self, input, variance, **_):
    stddev = torch.sqrt(variance)

    # Crude tonemapping propagation, should use derivative
    color = input[:, 0:3, ...]
    features = input[:, 3:, ...]
    tmColor, tmStddev = tonemap_transfer(self.tonemap, color, stddev)
    inputs = [concat(tmColor, features)]

    for _ in range(ConvLayerJ.N):
      tangent = torch.sign(torch.rand_like(tmStddev) - 0.5) * tmStddev
      inputs.append(concat(tmColor - tangent, features))

    outputs = self.model(torch.concat(inputs, 0))

    b = input.shape[0]
    color = relu(outputs[0:b, ...])

    # JVP_x(u) = f(x) - f_x(x - u)
    jvps = torch.unsqueeze(color, 0) - outputs[b:, ...].view(ConvLayerJ.N, b, outputs.shape[1], outputs.shape[2], outputs.shape[3])
    stddev = channel_mean(torch.sqrt(torch.mean(torch.square_(jvps), 0)))

    return self.tonemapInverse(color), stddev


class DenoiseDriverNaiveJVP:
  def __init__(self, cfg, device):
    num_input_channels = len(get_model_channels(cfg.features))

    transfer = get_transfer_function(cfg)
    self.tonemap = lambda c: transfer.forward(torch.clamp(c, min=1e-6))
    self.tonemapInverse = lambda c: transfer.inverse(torch.clamp(c, min=1e-6, max=1.0-1e-6))
    if cfg.model == 'unet':
      self.model = UNetDenoiser(lambda x: x, num_input_channels, layer=ConvLayer, lastReLU=False)
    else:
      assert False
    self.model.to(device)

  def compute_losses(self, **_):
    raise NotImplementedError

  def compute_infer(self, input, variance, **_):
    stddev = torch.sqrt(variance)

    # Crude tonemapping propagation, should use derivative
    color = input[:, 0:3, ...]
    features = input[:, 3:, ...]
    tmColor, tmStddev = tonemap_transfer(self.tonemap, color, stddev)

    input = concat(tmColor, features)
    tangent = concat(torch.sign(torch.rand_like(tmStddev) - 0.5) * tmStddev, torch.zeros_like(features))

    output, jvp = torch.func.jvp(self.model, (input,), (tangent,))

    color = relu(output)
    stddev = channel_mean(torch.abs_(jvp))

    return self.tonemapInverse(color), stddev
