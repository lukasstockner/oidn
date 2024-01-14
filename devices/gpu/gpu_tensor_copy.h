// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/tensor_copy.h"

OIDN_NAMESPACE_BEGIN

  template<typename T, TensorLayout layout>
  struct GPUTensorCopyKernel
  {
    TensorAccessor3D<T, layout> src;
    TensorAccessor3D<T, layout> dst;

    /* TODO: Should be device-mem memcpy() in EngineT */
    OIDN_DEVICE_INLINE void operator ()(const WorkItem<3>& it) const
    {
      const int c = it.getId<0>();
      const int h = it.getId<1>();
      const int w = it.getId<2>();

      dst(c, h, w) = src(c, h, w);
    }
  };

  // Optimized for HWC layout (memory coalescing)
  template<typename T>
  struct GPUTensorCopyKernel<T, TensorLayout::hwc>
  {
    TensorAccessor3D<T, TensorLayout::hwc> src;
    TensorAccessor3D<T, TensorLayout::hwc> dst;

    OIDN_DEVICE_INLINE void operator ()(const WorkItem<3>& it) const
    {
      const int h = it.getId<0>();
      const int w = it.getId<1>();
      const int c = it.getId<2>();

      dst(c, h, w) = src(c, h, w);
    }
  };

  template<typename EngineT, typename TensorDataT, TensorLayout tensorLayout>
  class GPUTensorCopy final : public TensorCopy
  {
  public:
    explicit GPUTensorCopy(const Ref<EngineT>& engine,
                           const TensorCopyDesc& desc)
      : TensorCopy(desc),
        engine(engine) {}

    void submit() override
    {
      if (!src || !dst)
        throw std::logic_error("Tensor copy source/destination not set");
      if (dst->getH() < src->getH() || dst->getW() < src->getW())
        throw std::out_of_range("Tensor copy destination smaller than the source");
      if (dst->getDataType() != src->getDataType())
        throw std::invalid_argument("Tensor copy source and destination have different data types");

      GPUTensorCopyKernel<TensorDataT, tensorLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      if (tensorLayout == TensorLayout::hwc)
        engine->submitKernel(WorkDim<3>(src->getH(), src->getW(), src->getPaddedC()), kernel);
      else
        engine->submitKernel(WorkDim<3>(src->getPaddedC(), src->getH(), src->getW()), kernel);
    }

  private:
    Ref<EngineT> engine;
  };

OIDN_NAMESPACE_END
