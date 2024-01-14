#!/usr/bin/env python3

## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch

from config import *
from util import *
from result import *
import tza

def main():
  # Parse the command line arguments
  cfg = parse_args(description='Import a training result from the runtime model weights format (TZA).')

  import_weights(cfg)

# Imports the weights from a TZA file
def import_weights(cfg):
  # Load the checkpoint
  result_dir = get_result_dir(cfg)
  if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
  input_file = tza.Reader(cfg.input)
  model_state = {key: torch.from_numpy(np.array(input_file[key][0])) for key in input_file._table}
  save_checkpoint(result_dir, cfg.num_epochs, 12345, model_state, None)
  save_config(result_dir, cfg)

if __name__ == '__main__':
  main()
