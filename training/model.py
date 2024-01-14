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
  elif cfg.model == 'e2enet':
    return E2EDriver(cfg, device)
  elif cfg.model == 'bothnet':
    return BothDriver(cfg, device)
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


class MiniErrorUNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.e1 = Conv(6, 8, 8)
    self.e2 = Conv(8, 8)
    self.bottleneck = Conv(8, 8, 8)
    self.d2 = Conv(16, 8)
    self.d1 = Conv(16, 8, 1)

  def forward(self, input):
    x = pool1 = self.e1(input)
    x = pool2 = self.e2(pool(x))
    x = self.bottleneck(pool(x))
    x = self.d2(concat(upsample(x), pool2))
    x = self.d1(concat(upsample(x), pool1))
    return x

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

class FakeRenderer(torch.autograd.Function):
  EXPECTED_VALUE = False
  ACCURATE_GRADIENT = True
  VARIANCE_SCALE = 5.0

  @staticmethod
  def sample(refColor, refRelStdDev, nextSampleMap, varianceScale):
    # Decode variance from relstddev
    refVariance = torch.square(refRelStdDev * torch.clamp(refColor, min=1e-3))

    # Parameters of the underlying pixel distribution
    mean = torch.clamp(refColor, min=1e-7)
    variance = torch.clamp(refVariance, min=1e-7) / varianceScale

    # Adjust variance based on MC convergence
    variance = variance / torch.clamp(nextSampleMap, min=1.)

    # Expected distribution of render output using sampleMap
    alpha = mean * mean / variance
    beta = mean / variance

    # Sample image rendered with nextSampleMap samples
    seed = torch.seed()
    r = torch.distributions.Gamma(alpha[:, 0:1, ...], beta[:, 0:1, ...]).sample()
    torch.manual_seed(seed)
    g = torch.distributions.Gamma(alpha[:, 1:2, ...], beta[:, 1:2, ...]).sample()
    torch.manual_seed(seed)
    b = torch.distributions.Gamma(alpha[:, 2:3, ...], beta[:, 2:3, ...]).sample()

    return concat(r, g, b)

  @staticmethod
  def forward(ctx, refColor, refRelStdDev, curColor, curSampleMap, nextSampleMap, varianceScale=None):
    ctx.save_for_backward(refColor, curColor, curSampleMap, nextSampleMap)

    if FakeRenderer.EXPECTED_VALUE:
      nextColor = refColor
    else:
      nextColor = FakeRenderer.sample(refColor, refRelStdDev, nextSampleMap, varianceScale or FakeRenderer.VARIANCE_SCALE)

    # Combine current and next image
    output = (curSampleMap * curColor + nextSampleMap * nextColor) / torch.clamp(curSampleMap + nextSampleMap, min=1.)
    return output

  @staticmethod
  def backward(ctx, dLoss_dColor):
    assert not any(ctx.needs_input_grad[i] for i in [0, 1, 2, 3, 5])

    refColor, curColor, curSampleMap, nextSampleMap = ctx.saved_tensors
    if FakeRenderer.ACCURATE_GRADIENT:
      dColor_dSample = curSampleMap * (refColor - curColor) / torch.clamp((curSampleMap + nextSampleMap)**2, min=1.)
    else:
      dColor_dSample = (refColor - curColor) / torch.clamp(curSampleMap, min=1.)
    gradSampleMap = torch.sum(dLoss_dColor * dColor_dSample, dim=1, keepdim=True)

    return None, None, None, None, gradSampleMap, None

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
  def __init__(self, in_channels, out_channels, aux_channels=0, want_intermediate=False, scale=1, layer=ConvLayer):
    super().__init__()
    self.want_intermediate = want_intermediate

    # Convolutions
    C = lambda *x: Conv(*x, layer=layer)
    # Don't use residual or BN for last decoder stage, it might shift color values
    C_last = lambda *x: Conv(*x, layer=layer if layer == ConvLayerJ else ConvLayer)
    self.conv5 = C(ec5+aux_channels, ec5//scale)
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


class GlobalSummary(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Conv2d(2*in_channels, out_channels, 1)
  
  def forward(self, input):
    std, mean = torch.std_mean(input, dim=(2, 3), keepdim=True)
    combined = self.conv(concat(mean, std))
    return combined.expand(-1, -1, input.shape[2], input.shape[3])







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











class UNetE2EPredictor(nn.Module):
  BLUR_SAMPLEMAP = False
  GLOBAL_SUMMARY = True

  def __init__(self, tonemap):
    super().__init__()
    self.tonemap = tonemap

    if UNetE2EPredictor.GLOBAL_SUMMARY:
      aux_channels = 9
      self.global_summary = GlobalSummary(ec5, 8)
    else:
      aux_channels = 1
      self.global_summary = None

    self.encoder = UNetEncoder(9)
    self.color_decoder = UNetDecoder(9, 3)
    self.sample_decoder = UNetDecoder(10, 1, aux_channels=aux_channels, scale=2)

    self.alignment = 16

  def getSamplerIn(self, encoded, sampleMap, relBudget):
    dense, pool3, pool2, pool1, full = encoded
    # Add budget info
    budgetChannel = torch.ones_like(dense[:, 0:1, ...]) * torch.log(relBudget)
    # Add sample map info
    sampleChannel = torch.log(sampleMap) - torch.log(sampleMap.mean((2, 3), keepdim=True))
    # Add global summary info
    globalSummary = self.global_summary(dense) if self.global_summary else None
    return concat(dense, budgetChannel, globalSummary), pool3, pool2, pool1, concat(full, sampleChannel)

  def getSampleMap(self, currentSampleMap, predictorOutput, budget):
    # Blur if requested
    if UNetE2EPredictor.BLUR_SAMPLEMAP:
      predictorOutput = torch.nn.functional.avg_pool2d(predictorOutput, kernel_size=3, stride=1, padding=1)
    # Apply softplus
    nextSampleMap = torch.nn.functional.softplus(predictorOutput)
    # Mask out padding areas
    nextSampleMap = torch.where(currentSampleMap < 1.0, 1e-10, nextSampleMap)
    # Normalize
    factor = (budget - 1) * (predictorOutput.shape[2] * predictorOutput.shape[3]) / nextSampleMap.sum((2, 3), keepdim=True)
    return 1 + nextSampleMap * factor

  def infer(self, input, relBudget = 1.0):
    if not isinstance(relBudget, torch.Tensor):
      relBudget = torch.tensor(relBudget)

    # Extract data from input
    currentColor = input[:, 0:3, ...]
    currentFeatures = input[:, 3:-1, ...]
    currentSampleMap = torch.exp2(input[:, -1:, ...])  # TODO: Handle in dataset.py, but need to watch out for fp16 range
    budget = torch.clamp(currentSampleMap.mean((2, 3), keepdim=True) * relBudget, min=1.0)

    # Apply denoiser to the current state
    currentEncoded = self.encoder(concat(self.tonemap(currentColor), currentFeatures))
    currentDenoised = relu(self.color_decoder(currentEncoded))

    # Run sample map predictor
    predictorOutput = self.sample_decoder(self.getSamplerIn(currentEncoded, currentSampleMap, relBudget))
    nextSampleMap = self.getSampleMap(currentSampleMap, predictorOutput, budget)

    return currentDenoised, nextSampleMap

  def forward(self, input, target, valid=False):
    # Extract data from input
    refColor = target[:, 0:3, ...]
    refRelStdDev = target[:, 3:6, ...]
    currentColor = input[:, 0:3, ...]
    currentFeatures = input[:, 3:-1, ...]
    currentSampleMap = torch.exp2(input[:, -1:, ...])

    # Pick random sample budget (on average, match existing budget)
    budget = currentSampleMap.mean((2, 3), keepdim=True)
    relBudget = torch.distributions.Gamma(10 * torch.ones_like(budget), 0.1 * torch.ones_like(budget)).sample()
    budget = torch.clamp(budget * relBudget, min=1.0)

    # Apply denoiser to the current state
    currentEncoded = self.encoder(concat(self.tonemap(currentColor), currentFeatures))
    currentDenoised = relu(self.color_decoder(currentEncoded))

    # Run sample map predictor
    predictorOutput = self.sample_decoder(self.getSamplerIn(currentEncoded, currentSampleMap, relBudget))
    nextSampleMap = self.getSampleMap(currentSampleMap, predictorOutput, budget)

    # Apply fake renderer to approximate next state with the sample map
    nextColor = FakeRenderer.apply(refColor, refRelStdDev, currentColor, currentSampleMap, nextSampleMap, 1.0 if valid else None)

    # Apply denoiser to the "next" state
    nextEncoded = self.encoder(concat(self.tonemap(nextColor), currentFeatures))
    nextDenoised = relu(self.color_decoder(nextEncoded))

    return currentDenoised, nextDenoised, nextSampleMap

class UNetBothPredictor(nn.Module):
  BLUR_SAMPLEMAP = False

  def __init__(self, tonemap):
    super().__init__()
    self.tonemap = tonemap

    self.denoiser = UNetErrorPredictor(self.tonemap, 9)
    self.sampler = MiniErrorUNet()

    self.alignment = self.denoiser.alignment

  def getSampleMap(self, currentSampleMap, predictorOutput, budget):
    # Blur if requested
    if UNetE2EPredictor.BLUR_SAMPLEMAP:
      predictorOutput = torch.nn.functional.avg_pool2d(predictorOutput, kernel_size=3, stride=1, padding=1)
    # Apply softplus
    nextSampleMap = torch.nn.functional.softplus(predictorOutput)
    # Mask out padding areas
    nextSampleMap = torch.where(currentSampleMap < 1.0, 1e-10, nextSampleMap)
    # Normalize
    factor = (budget - 1) * (predictorOutput.shape[2] * predictorOutput.shape[3]) / nextSampleMap.sum((2, 3), keepdim=True)
    return 1 + nextSampleMap * factor

  def getSamplerIn(self, noisy, error, sampleMap, relBudget):
    sampleChannel = torch.log(sampleMap) - torch.log(sampleMap.mean((2, 3), keepdim=True))
    budgetChannel = torch.ones_like(sampleChannel) * torch.log(relBudget)
    return concat(self.tonemap(noisy), error, sampleChannel, budgetChannel)

  def infer(self, input, relBudget = 1.0):
    if not isinstance(relBudget, torch.Tensor):
      relBudget = torch.tensor(relBudget)

    # Extract data from input
    currentColor = input[:, 0:3, ...]
    currentFeatures = input[:, 3:-1, ...]
    currentSampleMap = torch.exp2(input[:, -1:, ...])  # TODO: Handle in dataset.py, but need to watch out for fp16 range
    budget = torch.clamp(currentSampleMap.mean((2, 3), keepdim=True) * relBudget, min=1.0)

    # Apply denoiser to the current state
    currentDenoised, currentError = self.denoiser(concat(currentColor, currentFeatures))

    # Run sample map predictor
    samplerOutput = self.sampler(self.getSamplerIn(currentColor, currentError, currentSampleMap, relBudget))
    nextSampleMap = self.getSampleMap(currentSampleMap, samplerOutput, budget)

    return currentDenoised, currentError, nextSampleMap

  def forward(self, input, target, valid=False):
    # Extract data from input
    refColor = target[:, 0:3, ...]
    refRelStdDev = target[:, 3:6, ...]
    currentColor = input[:, 0:3, ...]
    currentFeatures = input[:, 3:-1, ...]
    currentSampleMap = torch.exp2(input[:, -1:, ...])

    # Pick random sample budget (on average, match existing budget)
    budget = currentSampleMap.mean((2, 3), keepdim=True)
    relBudget = torch.distributions.Gamma(10 * torch.ones_like(budget), 0.1 * torch.ones_like(budget)).sample()
    budget = torch.clamp(budget * relBudget, min=1.0)

    # Apply denoiser to the current state
    currentDenoised, currentError = self.denoiser(concat(currentColor, currentFeatures))

    # Run sample map predictor
    samplerOutput = self.sampler(self.getSamplerIn(currentColor, currentError, currentSampleMap, relBudget))
    nextSampleMap = self.getSampleMap(currentSampleMap, samplerOutput, budget)

    # Apply fake renderer to approximate next state with the sample map
    nextColor = FakeRenderer.apply(refColor, refRelStdDev, currentColor, currentSampleMap, nextSampleMap, 1.0 if valid else None)

    # Apply denoiser to the "next" state
    nextDenoised, nextError = self.denoiser(concat(nextColor, currentFeatures))

    return currentDenoised, nextDenoised, currentError, nextError, nextSampleMap






def blurSampleMap(sampleMap):
  return F.avg_pool2d(sampleMap, (3, 3), stride=1, padding=1)

def sampleMapFromError(rel_budget, spp_pass, error):
  if rel_budget and spp_pass is not None:
    nextSampleMap = blurSampleMap(torch.square(error) / spp_pass)

    # Normalize mean to match rel_budget*spp_pass
    nextSampleMean = rel_budget * spp_pass.mean((2, 3), keepdim=True)
    return nextSampleMap * (nextSampleMean / nextSampleMap.mean((2, 3), keepdim=True))
  else:
    return None




class E2EDriver:
  def __init__(self, cfg, device):
    assert len(get_model_channels(cfg.features)) == 10

    transfer = get_transfer_function(cfg)
    self.tonemap = lambda c: transfer.forward(torch.clamp(c, min=1e-6))
    self.tonemapInverse = lambda c: transfer.inverse(torch.clamp(c, min=1e-6, max=1.0-1e-6))
    self.model = UNetE2EPredictor(self.tonemap)
    self.model.to(device)

    if cfg.base_model:
      checkpoint = torch.load(cfg.base_model, map_location=device)
      ms = filter_model_state(checkpoint['model_state'], mappings={'decoder.': 'color_decoder.', 'error_decoder.': 'sample_decoder.'})

      # Adapt base model to new channel count (to include aux output channel)
      init_ms = self.model.state_dict()
      for k in ms:
        if k in init_ms and ms[k].shape != init_ms[k].shape:
          combined = init_ms[k].clone()
          combined[:, 0:ms[k].shape[1], :, :] = ms[k]
          ms[k] = combined
      self.model.load_state_dict(ms, strict=False)

    self.origCriterion = MSSSIMLoss([0.2, 0.2, 0.2, 0.2, 0.2])
    self.newCriterion = MixLoss([L1Loss(), GradientLoss()], [0.5, 0.5])
    self.validCriterion = get_loss_function(cfg)

    self.origCriterion.to(device)
    self.newCriterion.to(device)
    self.validCriterion.to(device)


  def compute_losses(self, input, target, epoch, valid, **_):
    assert input.shape[1] == 10

    targetCol = self.tonemap(target[:, 0:3, ...])
    currentDenoised, newDenoised, newSampleMap = self.model(input, target, valid=valid)

    relMap = newSampleMap / newSampleMap.mean((2, 3), keepdim=True)
    predictorLoss = 1e-11 * torch.square(relMap).mean()

    if valid:
      # Use mixed criterion to be able to compare validation results
      currentLoss = self.validCriterion(currentDenoised, targetCol)
      newLoss = self.validCriterion(newDenoised, targetCol)
    else:
      currentLoss = self.origCriterion(currentDenoised, targetCol)
      newLoss = self.newCriterion(newDenoised, targetCol)

    loss = currentLoss + newLoss + predictorLoss
    return loss, {'loss': loss, 'color_loss': currentLoss, 'next_loss': newLoss, 'predictor_loss': predictorLoss}

  def compute_infer(self, input, spp_pass=None, rel_budget=1.0, **_):
    if spp_pass is not None:
      assert input.shape[1] == 9
      input = concat(input, torch.log2(spp_pass))
    color, sampleMap = self.model.infer(input, relBudget=rel_budget)
    return self.tonemapInverse(color), None, blurSampleMap(sampleMap)


class BothDriver:
  def __init__(self, cfg, device):
    transfer = get_transfer_function(cfg)
    self.tonemap = lambda c: transfer.forward(torch.clamp(c, min=1e-6))
    self.tonemapInverse = lambda c: transfer.inverse(torch.clamp(c, min=1e-6, max=1.0-1e-6))
    self.model = UNetBothPredictor(self.tonemap)
    self.model.to(device)

    if cfg.base_model:
      checkpoint = torch.load(cfg.base_model, map_location=device)
      ms = filter_model_state(checkpoint['model_state'], mappings={'encoder.': 'denoiser.encoder.', 'color_decoder.': 'denoiser.color_decoder.', 'error_decoder.': 'denoiser.error_decoder.'})
      self.model.load_state_dict(ms, strict=False)

    self.origCriterion = MSSSIMLoss([0.2, 0.2, 0.2, 0.2, 0.2])
    self.newCriterion = MixLoss([L1Loss(), GradientLoss()], [0.5, 0.5])
    self.validCriterion = get_loss_function(cfg)
    self.errorCriterion = ErrorLoss()

    self.origCriterion.to(device)
    self.newCriterion.to(device)
    self.validCriterion.to(device)
    self.errorCriterion.to(device)

  def compute_losses(self, input, target, epoch, valid, **_):
    assert input.shape[1] == 10

    targetCol = self.tonemap(target[:, 0:3, ...])
    currentDenoised, newDenoised, currentError, newError, newSampleMap = self.model(input, target, valid=valid)

    relMap = newSampleMap / newSampleMap.mean((2, 3), keepdim=True)
    predictorLoss = 1e-11 * torch.square(relMap).mean()

    curErrLoss = self.errorCriterion(targetCol, currentDenoised, currentError)
    newErrLoss = self.errorCriterion(targetCol, newDenoised, newError)

    if valid:
      # Use mixed criterion to be able to compare validation results
      currentLoss = self.validCriterion(currentDenoised, targetCol)
      newLoss = self.validCriterion(newDenoised, targetCol)
    else:
      currentLoss = self.origCriterion(currentDenoised, targetCol)
      newLoss = self.newCriterion(newDenoised, targetCol)

    loss = currentLoss + newLoss + predictorLoss + 1e-2 * (curErrLoss + newErrLoss)
    return loss, {'loss': loss, 'color_loss': currentLoss, 'next_loss': newLoss, 'predictor_loss': predictorLoss, 'error_loss': curErrLoss, 'next_error_loss': newErrLoss}

  def compute_infer(self, input, spp_pass=None, rel_budget=1.0, **_):
    if spp_pass is not None:
      assert input.shape[1] == 9
      input = concat(input, torch.log2(spp_pass))
    color, error, sampleMap = self.model.infer(input, relBudget=rel_budget)
    return self.tonemapInverse(color), error, blurSampleMap(sampleMap)


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

  def compute_infer(self, input, spp_pass=None, rel_budget=None, **_):
    color, error = self.model(input)
    return self.tonemapInverse(color), error, sampleMapFromError(rel_budget, spp_pass, error)


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

  def compute_infer(self, input, variance=None, spp_pass=None, rel_budget=None, **_):
    output = self.model(input)
    if True:
      # Uniform sampling
      nextSampleMap = spp_pass * rel_budget if (rel_budget and spp_pass is not None) else None
    else:
      # Variance-based sampling
      nextSampleMap = sampleMapFromError(rel_budget, spp_pass, channel_mean(torch.sqrt(variance)))
    return self.tonemapInverse(output), None, nextSampleMap


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

  def compute_infer(self, input, variance=None, spp_pass=None, rel_budget=None, **_):
    output, outVariance = self.model(input, variance=variance)
    stddev = torch.sqrt(outVariance)
    _, stddev = tonemap_transfer(self.tonemap, output, stddev)
    stddev = channel_mean(stddev)
    return self.tonemapInverse(output), stddev, sampleMapFromError(rel_budget, spp_pass, stddev)


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

  def compute_infer(self, input, variance, spp_pass=None, rel_budget=None, **_):
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

    return self.tonemapInverse(color), stddev, sampleMapFromError(rel_budget, spp_pass, stddev)


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

  def compute_infer(self, input, variance, spp_pass=None, rel_budget=None, **_):
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

    return self.tonemapInverse(color), stddev, sampleMapFromError(rel_budget, spp_pass, stddev)
