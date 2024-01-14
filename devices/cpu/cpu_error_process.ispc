// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"
#include "image_accessor.isph"
#include "color.isph"
#include "tile.isph"

struct CPUErrorProcessKernel
{
  // Source
  uniform TensorAccessor3D src;

  // Destination
  uniform ImageAccessor dst;

  // Tile
  uniform Tile tile;
};

export void CPUErrorProcessKernel_run(uniform CPUErrorProcessKernel* uniform self, uniform int h)
{
  const uniform int hSrc = h + self->tile.hSrcBegin;
  const uniform int hDst = h + self->tile.hDstBegin;

  foreach (w = 0 ... self->tile.W)
  {
    const int wSrc = w + self->tile.wSrcBegin;
    const int wDst = w + self->tile.wDstBegin;

    float error = get1f(self->src, 0, hSrc, wSrc);
    error = log(1.0f + exp(error));
    error = clamp(nan_to_zero(error), 0.f, pos_max);

    set1f(self->dst, hDst, wDst, error);
  }
}
