## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import torch

from util import *

# Gets the path to the result directory
def get_result_dir(cfg, result=None):
  if result is None:
    result = cfg.result
  return os.path.join(cfg.results_dir, result)

# Gets the path to the result checkpoint directory
def get_checkpoint_dir(result_dir):
  return os.path.join(result_dir, 'checkpoints')

# Gets the path to a checkpoint file
def get_checkpoint_filename(result_dir, epoch):
  checkpoint_dir = get_checkpoint_dir(result_dir)
  return os.path.join(checkpoint_dir, 'checkpoint_%d.pth' % epoch)

# Gets the path to the file that contains the checkpoint state (latest epoch)
def get_checkpoint_state_filename(result_dir):
  checkpoint_dir = get_checkpoint_dir(result_dir)
  return os.path.join(checkpoint_dir, 'latest')

# Gets the latest checkpoint epoch
def get_latest_checkpoint_epoch(result_dir):
  latest_filename = get_checkpoint_state_filename(result_dir)
  if not os.path.isfile(latest_filename):
    error('no checkpoints found')
  with open(latest_filename, 'r') as f:
    return int(f.readline())

# Gets the path to result log directory
def get_result_log_dir(result_dir):
  return os.path.join(result_dir, 'log')

# Saves a training checkpoint
def save_checkpoint(result_dir, epoch, step, model_state, optimizer):
  checkpoint_dir = get_checkpoint_dir(result_dir)
  if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)

  checkpoint_filename = get_checkpoint_filename(result_dir, epoch)
  state = {'epoch': epoch, 'step': step, 'model_state': model_state}
  if optimizer:
    state['optimizer_state'] = optimizer.state_dict()
  torch.save(state, checkpoint_filename)

  latest_filename = get_checkpoint_state_filename(result_dir)
  with open(latest_filename, 'w') as f:
    f.write('%d' % epoch)

def filter_model_state(ms, mappings={}):
  # Map old to new UNet weight names
  unet_map = {
    'enc_conv0': 'encoder.conv0.0',
    'enc_conv1': 'encoder.conv0.1',
    'enc_conv2': 'encoder.conv1',
    'enc_conv3': 'encoder.conv2',
    'enc_conv4': 'encoder.conv3',
    'enc_conv5a': 'encoder.conv4',
    'enc_conv5b': 'decoder.conv5',
    'dec_conv4a': 'decoder.conv4.0',
    'dec_conv4b': 'decoder.conv4.1',
    'dec_conv3a': 'decoder.conv3.0',
    'dec_conv3b': 'decoder.conv3.1',
    'dec_conv2a': 'decoder.conv2.0',
    'dec_conv2b': 'decoder.conv2.1',
    'dec_conv1a': 'decoder.conv1.0',
    'dec_conv1b': 'decoder.conv1.1',
    'dec_conv0': 'decoder.conv0',
  }
  out = dict()
  for k, v in ms.items():
    if k.startswith('_orig_mod.'):
      k = k[10:]
    for old, new in (unet_map | mappings).items():
      if k.startswith(old):
        k = new + k[len(old):]
        break
    out[k] = v
  return out

# Loads and returns a training checkpoint
def load_checkpoint(result_dir, device, epoch=None, model=None, optimizer=None):
  if epoch is None or epoch <= 0:
    epoch = get_latest_checkpoint_epoch(result_dir)

  checkpoint_filename = get_checkpoint_filename(result_dir, epoch)
  if not os.path.isfile(checkpoint_filename):
    error('checkpoint does not exist')

  checkpoint = torch.load(checkpoint_filename, map_location=device)

  if checkpoint['epoch'] != epoch:
    error('checkpoint epoch mismatch')
  if model:
    unwrap_module(model).load_state_dict(filter_model_state(checkpoint['model_state']))
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer_state'])

  return checkpoint