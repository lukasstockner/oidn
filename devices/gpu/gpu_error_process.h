// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/error_process.h"
#include "core/tensor_accessor.h"
#include "core/image_accessor.h"
#include "core/color.h"
#include "core/tile.h"

OIDN_NAMESPACE_BEGIN

  template<typename ImageDataT, typename TensorDataT, TensorLayout tensorLayout>
  struct GPUErrorProcessKernel
  {
    // Source
    TensorAccessor3D<TensorDataT, tensorLayout> src;

    // Destination
    ImageAccessor<ImageDataT> dst;

    // Tile
    Tile tile;

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<2>& it) const
    {
      const int h = it.getId<0>();
      const int w = it.getId<1>();

      const int hSrc = h + tile.hSrcBegin;
      const int hDst = h + tile.hDstBegin;
      const int wSrc = w + tile.wSrcBegin;
      const int wDst = w + tile.wDstBegin;

      // Load
      float value = src(0, hSrc, wSrc);

      // The CNN Output may contain NaNs, so it must be sanitized
      value = math::nan_to_zero(value);

      // Apply the softplus transform
      value = logf(1.0f + expf(value));

      // Store
      dst.set1(hDst, wDst, value);
    }
  };

  template<typename EngineT, typename TensorDataT, TensorLayout tensorLayout>
  class GPUErrorProcess : public ErrorProcess
  {
  public:
    GPUErrorProcess(const Ref<EngineT>& engine, const ErrorProcessDesc& desc)
      : ErrorProcess(desc),
        engine(engine) {}

    void submit() override
    {
      if (!src || !dst)
        throw std::logic_error("error processing source/destination not set");
      if (tile.hSrcBegin + tile.H > src->getH() ||
          tile.wSrcBegin + tile.W > src->getW() ||
          tile.hDstBegin + tile.H > dst->getH() ||
          tile.wDstBegin + tile.W > dst->getW())
        throw std::out_of_range("error processing source/destination out of range");

      switch (dst->getDataType())
      {
      case DataType::Float32: runImpl<float>(); break;
      case DataType::Float16: runImpl<half>();  break;
      default:                assert(0);
      }
    }

  private:
    template<typename ImageDataT>
    void runImpl()
    {
      GPUErrorProcessKernel<ImageDataT, TensorDataT, tensorLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;
      kernel.tile = tile;

      engine->submitKernel(WorkDim<2>(tile.H, tile.W), kernel);
    }

    Ref<EngineT> engine;
  };

OIDN_NAMESPACE_END
