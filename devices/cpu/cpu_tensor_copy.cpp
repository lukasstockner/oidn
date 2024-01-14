// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_tensor_copy.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUTensorCopy::CPUTensorCopy(const Ref<CPUEngine>& engine, const TensorCopyDesc& desc)
    : TensorCopy(desc),
      engine(engine)
  {
  }

  void CPUTensorCopy::submit()
  {
    if (!src || !dst)
      throw std::logic_error("tensor copy source/destination not set");

    const size_t N = src->getPaddedC() * src->getH() * src->getW();
    const float *srcData = (const float*) src->getData();
    float *dstData = (float*) dst->getData();
    std::copy(srcData, srcData + N, dstData);
  }
OIDN_NAMESPACE_END
