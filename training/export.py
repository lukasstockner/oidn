#!/usr/bin/env python3

## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
from glob import glob
import numpy as np
import torch

from config import *
from util import *
from result import *
import tza

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Exports a training result to the runtime model weights format (TZA).')

  print('Result:', cfg.result)

  if cfg.target == 'weights':
    export_weights(cfg)
  elif cfg.target == 'package':
    export_package(cfg)

# Exports the weights to a TZA file
def export_weights(cfg):
  # Initialize the PyTorch device
  device = init_device(cfg)

  # Load the checkpoint
  result_dir = get_result_dir(cfg)
  if not os.path.isdir(result_dir):
    error('result does not exist')
  result_cfg = load_config(result_dir)
  checkpoint = load_checkpoint(result_dir, device, cfg.num_epochs)
  epoch = checkpoint['epoch']
  model_state = checkpoint['model_state']
  print('Epoch:', epoch)

  # Save the weights to a TZA file
  if cfg.output:
    output_filename = cfg.output
  else:
    output_filename = os.path.join(result_dir, cfg.result)
    if cfg.num_epochs:
      output_filename += '_%d' % epoch
    output_filename += '.tza'
  print('Output:', output_filename)
  print()

  name_map = {
    'encoder.conv0.0': 'enc_conv0',
    'encoder.conv0.1': 'enc_conv1',
    'encoder.conv1': 'enc_conv2',
    'encoder.conv2': 'enc_conv3',
    'encoder.conv3': 'enc_conv4',
    'encoder.conv4': 'enc_conv5a',

    'decoder.conv5': 'enc_conv5b',
    'decoder.conv4.0': 'dec_conv4a',
    'decoder.conv4.1': 'dec_conv4b',
    'decoder.conv3.0': 'dec_conv3a',
    'decoder.conv3.1': 'dec_conv3b',
    'decoder.conv2.0': 'dec_conv2a',
    'decoder.conv2.1': 'dec_conv2b',
    'decoder.conv1.0': 'dec_conv1a',
    'decoder.conv1.1': 'dec_conv1b',
    'decoder.conv0': 'dec_conv0',

    'color_decoder.conv5': 'enc_conv5b',
    'color_decoder.conv4.0': 'dec_conv4a',
    'color_decoder.conv4.1': 'dec_conv4b',
    'color_decoder.conv3.0': 'dec_conv3a',
    'color_decoder.conv3.1': 'dec_conv3b',
    'color_decoder.conv2.0': 'dec_conv2a',
    'color_decoder.conv2.1': 'dec_conv2b',
    'color_decoder.conv1.0': 'dec_conv1a',
    'color_decoder.conv1.1': 'dec_conv1b',
    'color_decoder.conv0': 'dec_conv0',

    'error_decoder.conv5': 'err_conv5b',
    'error_decoder.conv4.0': 'err_conv4a',
    'error_decoder.conv4.1': 'err_conv4b',
    'error_decoder.conv3.0': 'err_conv3a',
    'error_decoder.conv3.1': 'err_conv3b',
    'error_decoder.conv2.0': 'err_conv2a',
    'error_decoder.conv2.1': 'err_conv2b',
    'error_decoder.conv1.0': 'err_conv1a',
    'error_decoder.conv1.1': 'err_conv1b',
    'error_decoder.conv0': 'err_conv0',
  }

  with tza.Writer(output_filename) as output_file:
    for name, value in model_state.items():
      # Export in FP16 if the model was trained with mixed precision
      tensor = value.half() if result_cfg.precision == "mixed" else value
      tensor = tensor.cpu().numpy()
      for old, new in name_map.items():
        if name.startswith(old):
          name = new + name[len(old):]
          break
      else:
        assert False
      print(name, tensor.shape)

      if name.endswith('.weight') and len(value.shape) == 4:
        layout = 'oihw'
      elif len(value.shape) == 1:
        layout = 'x'
      else:
        error('unknown state value')

      output_file.write(name, tensor, layout)

# Exports the result directory to a ZIP file
def export_package(cfg):
  # Get the output filename
  if cfg.output:
    output_filename = cfg.output
  else:
    output_filename = os.path.join(cfg.results_dir, cfg.result) + '.zip'
  print('Output:', output_filename)

  # Get the list of files that belong to the result (latest checkpoint only)
  result_dir = get_result_dir(cfg)
  filenames = [get_config_filename(result_dir)]
  filenames.append(get_checkpoint_state_filename(result_dir))
  latest_epoch = get_latest_checkpoint_epoch(result_dir)
  filenames.append(get_checkpoint_filename(result_dir, latest_epoch))
  filenames += glob(os.path.join(get_result_log_dir(result_dir), 'events.out.*'))
  filenames += glob(os.path.join(result_dir, 'src.*'))

  # Save the ZIP file
  save_zip(output_filename, filenames, root_dir=cfg.results_dir)

if __name__ == '__main__':
  main()