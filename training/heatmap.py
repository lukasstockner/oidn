#!/usr/bin/env python3

## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

import sys

from config import *
from util import *
from dataset import *
from image import *
from color import *

import matplotlib

def main():

  if len(sys.argv) == 2:
    image = oiio.ImageBuf(sys.argv[1])
    print("\n".join(f"{i}: {c}" for i, c in enumerate(image.spec().channelnames)))
    return
 
  heatmap = matplotlib.colormaps['viridis']

  input = oiio.ImageInput.open(sys.argv[1])
  c = int(sys.argv[2])
  data = input.read_image(subimage=0, miplevel=0, chbegin=c, chend=c+1, format=oiio.FLOAT).mean(axis=2)

  if len(sys.argv) == 5:
    low, high = np.percentile(data, float(sys.argv[3])), np.percentile(data, float(sys.argv[4]))
    data = (data - low) / (high - low)
    data = np.clip(data, 0, 1)
  else:
    data -= data.min()
    data /= data.max()

  data = heatmap(data, bytes=True)

  output = oiio.ImageOutput.create(sys.argv[1] + '.png')
  spec = oiio.ImageSpec(data.shape[1], data.shape[0], data.shape[2], oiio.UINT8)
  spec.attribute('png:compressionLevel', 9)
  output.open(sys.argv[1] + '.png', spec)
  output.write_image(data)
  output.close()


if __name__ == '__main__':
  main()