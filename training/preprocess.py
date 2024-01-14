#!/usr/bin/env python3

## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import shutil
import numpy as np
import torch
import h5py

from config import *
from util import *
from dataset import *
from model import *
from color import *
import tza

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Preprocesses training and validation datasets.')
  main_feature = get_main_feature(cfg.features)
  num_main_channels = len(get_dataset_channels(main_feature))

  # Initialize the PyTorch device
  device = init_device(cfg)

  # Determine the input and target features
  if cfg.clean_aux:
    input_features  = [main_feature]
    target_features = cfg.features
  else:
    input_features  = cfg.features
    target_features = [main_feature]

  # Returns a preprocessed image (also changes the original image!)
  def preprocess_image(image, exposure):
    image[..., 0:num_main_channels] *= exposure

    # Convert to FP16
    return np.ascontiguousarray(np.nan_to_num(image.astype(np.float16)))

  def run_in_mproc(func, processes):
    queue = mp.Queue()
    process = mp.Process(target=lambda q, f: q.put(f()), args=(queue, func))
    process.start()
    processes.append(process)
    return queue

  def load_multiprocess(name, features, processes):
    return run_in_mproc(lambda: load_image_features(name, features), processes)

  def name_to_identifier(name: str, scale_factor):
    scale = {1: 'f', 2: 'h', 4: 'q'}[scale_factor]
    prefix, suffix = name.rsplit('_', 1)
    return f'{prefix}_{scale}_{suffix}'

  # Preprocesses a group of input and target images at different SPPs
  def preprocess_sample_group(input_dir, output_hdf, hdf_filename, input_names, target_name, scales):
    samples = []

    processes = []

    # Load the target image
    print(target_name)
    target_queue = load_multiprocess(os.path.join(input_dir, target_name), target_features, processes)

    # Load the input images
    input_queues = {}
    for input_name in input_names:
      # Load the image
      input_queues[input_name] = load_multiprocess(os.path.join(input_dir, input_name), input_features, processes)

    # Wait for target image
    target_pixels = target_queue.get()

    # Compute the autoexposure value
    exposure = autoexposure(target_pixels) if main_feature == 'hdr' else 1.

    # Preprocess the target image
    for scale in scales:
      target_image = preprocess_image(scale_image(target_pixels, scale), exposure)

      # Save the target image
      target_identifier = name_to_identifier(target_name, scale)
      output_hdf.create_dataset(target_identifier, shape=target_image.shape, dtype=target_image.dtype, chunks=(256, 256, target_image.shape[2])).write_direct(target_image)

    # Process the input images
    process_queues = {}
    for input_name in input_names:
     def process():
      # Wait for input image
      input_pixels = input_queues[input_name].get()

      if input_pixels.shape[0:2] != target_pixels.shape[0:2]:
        error('the input and target images have different sizes')

      out = dict()
      for scale in scales:
        out[scale] = preprocess_image(scale_image(input_pixels, scale), exposure)
      return out
     process_queues[input_name] = run_in_mproc(process, processes)

    for input_name in input_names:
      processed = process_queues[input_name].get()
      for scale in scales:
        input_image = processed[scale]

        # Save the image
        input_identifier = name_to_identifier(input_name, scale)
        target_identifier = name_to_identifier(target_name, scale)
        output_hdf.create_dataset(input_identifier, shape=input_image.shape, dtype=input_image.dtype, chunks=(256, 256, input_image.shape[2])).write_direct(input_image)

        # Add sample
        print(input_identifier, target_identifier)
        samples.append((input_identifier, target_identifier, hdf_filename))

    for p in processes:
      p.join()

    return samples

  # Preprocesses a dataset
  def preprocess_dataset(data_name):
    input_dir = get_data_dir(cfg, data_name)
    print('\nDataset:', input_dir)
    if not os.path.isdir(input_dir):
      print('Not found')
      return

    # Create the output directory
    output_dir = os.path.join(cfg.preproc_dir, data_name)
    samples_filename = os.path.join(output_dir, 'samples.json')
    # Check whether we resume preprocessing
    if os.path.isdir(output_dir):
      samples = load_json(samples_filename)
    else:
      os.mkdir(output_dir)
      samples = []

    # Save the config
    save_config(output_dir, cfg)

    # Preprocess image sample groups
    sample_groups = get_image_sample_groups(input_dir, input_features, target_features)
    for _, input_names, target_name in sample_groups:
      hdf_filename = os.path.join(output_dir, f'image_{target_name}.hdf5')
      if os.path.isfile(hdf_filename):
        print(f"Skipping {target_name}, exists...")
        continue
      try:
        with h5py.File(hdf_filename, 'w') as output_hdf:
          if target_name:
            samples += preprocess_sample_group(input_dir, output_hdf, hdf_filename, input_names, target_name, [1, 2, 4])
      except KeyboardInterrupt:
        os.remove(hdf_filename)
        print(f"Cleaned up {hdf_filename}")
        raise
      # Save the samples in the dataset
      save_json(samples_filename, samples)

  # Preprocess all datasets
  with torch.no_grad():
    for dataset in [cfg.train_data, cfg.valid_data]:
      if dataset:
        preprocess_dataset(dataset)

if __name__ == '__main__':
  main()