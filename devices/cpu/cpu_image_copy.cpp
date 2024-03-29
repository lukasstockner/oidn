// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_image_copy.h"
#include "cpu_image_copy_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUImageCopy::CPUImageCopy(const Ref<CPUEngine>& engine)
    : engine(engine) {}

  void CPUImageCopy::submit()
  {
    if (!src || !dst)
      throw std::logic_error("image copy source/destination not set");
    if (dst->getH() < src->getH() || dst->getW() < src->getW())
      throw std::out_of_range("image copy destination smaller than the source");

    ispc::CPUImageCopyKernel kernel;
    kernel.src = toISPC(*src);
    kernel.dst = toISPC(*dst);

    parallel_nd(dst->getH(), [&](int h)
    {
      ispc::CPUImageCopyKernel_run(&kernel, h);
    });
  }

OIDN_NAMESPACE_END