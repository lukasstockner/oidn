#!/usr/bin/env python3

## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim

from config import *
from dataset import *
from model import *
from loss import *
from result import *
from util import *

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Tool for finding the optimal minimum and maximum learning rates.')

  # Start the worker(s)
  start_workers(cfg, main_worker)

# Worker function
def main_worker(rank, cfg):
  # Initialize the worker
  distributed = init_worker(rank, cfg)

  # Initialize the PyTorch device
  device_id = cfg.device_id + rank
  device = init_device(cfg, id=device_id)

  # Initialize the model
  driver = get_driver(cfg, device)

  if rank == 0:
    numParams = 0
    for name, param in driver.model.named_parameters():
      lenParam = math.prod(param.shape)
      print(f"Model element {name} size: {param.shape} ({lenParam})")
      numParams += lenParam
    print(f"Total parameter count: {numParams}")

  if distributed:
    driver.model = nn.parallel.DistributedDataParallel(driver.model, device_ids=[device_id])

  # Initialize the optimizer
  optimizer = optim.Adam(driver.model.parameters(), lr=cfg.lr)

  # Sync the workers
  if distributed:
    dist.barrier()

  # Start training
  if rank == 0:
    print('Result:', cfg.result)
  step = 0

  # Initialize the training dataset
  train_data = TrainingDataset(cfg, cfg.train_data)
  if len(train_data) > 0:
    if rank == 0:
      print('Training images:', train_data.num_images)
  else:
    error('no training images')
  train_loader, _ = get_data_loader(rank, cfg, train_data, shuffle=True)
  train_steps = cfg.num_epochs * len(train_loader)

  # Initialize the learning rate scheduler
  gamma = (cfg.max_lr / cfg.lr) ** (1. / (train_steps-1))
  lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

  # Training loop
  avg_loss = 0.
  best_loss = float('inf')
  beta = 0.98 # loss smoothing

  if rank == 0:
    print()
    start_time = time.time()
    progress = ProgressBar(train_steps, 'Train')
    result = [['learning_rate', 'smoothed_loss', 'loss']]

  # Switch to training mode
  driver.model.train()

  # Iterate over the batches
  for epoch in range(cfg.num_epochs):
    for _, batch in enumerate(train_loader, 0):
      # Get the batch
      input, target = batch
      input  = input.to(device,  non_blocking=True).float()
      target = target.to(device, non_blocking=True).float()

      # Run a training step
      optimizer.zero_grad()
      loss, _ = driver.compute_losses(input=input, target=target, epoch=epoch, valid=False)
      loss.backward()

      # Get the loss
      # In distributed mode we have to do a reduction, which is very expensive
      if distributed:
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
      cur_loss = loss.item() / cfg.num_devices

      # Compute the smoothed loss
      avg_loss = beta * avg_loss + (1 - beta) * cur_loss
      smoothed_loss = avg_loss / (1 - beta ** (step+1))
      if smoothed_loss < best_loss:
        best_loss = smoothed_loss
      elif smoothed_loss > 4 * best_loss:
        break

      # Record result
      if rank == 0:
        lr = lr_scheduler.get_last_lr()[0]
        result.append([lr, smoothed_loss, cur_loss])

      # Next step
      optimizer.step()
      lr_scheduler.step()
      step += 1
      if rank == 0:
        progress.next()

  # Print stats
  if rank == 0:
    duration = time.time() - start_time
    images_per_sec = (step * cfg.batch_size) / duration
    progress.finish('(%.1f images/s, %.1fs)' % (images_per_sec, duration))

  # Save the results
  if rank == 0:
    result_filename = os.path.join(cfg.results_dir, cfg.result) + '.csv'
    if not os.path.isdir(cfg.results_dir):
      os.makedirs(cfg.results_dir)
    save_csv(result_filename, result)

  # Cleanup
  cleanup_worker(cfg)

if __name__ == '__main__':
  main()