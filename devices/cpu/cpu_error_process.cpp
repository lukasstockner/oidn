// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_error_process.h"
#include "cpu_error_process_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUErrorProcess::CPUErrorProcess(const Ref<CPUEngine>& engine, const ErrorProcessDesc& desc)
    : ErrorProcess(desc),
      engine(engine) {}

  void CPUErrorProcess::submit()
  {
    if (!src || !dst)
      throw std::logic_error("error processing source/destination not set");
    if (tile.hSrcBegin + tile.H > src->getH() ||
        tile.wSrcBegin + tile.W > src->getW() ||
        tile.hDstBegin + tile.H > dst->getH() ||
        tile.wDstBegin + tile.W > dst->getW())
      throw std::out_of_range("error processing source/destination out of range");

    ispc::CPUErrorProcessKernel kernel;

    kernel.src = toISPC(*src);
    kernel.dst = toISPC(*dst);
    kernel.tile = toISPC(tile);

    parallel_nd(kernel.tile.H, [&](int h)
    {
      ispc::CPUErrorProcessKernel_run(&kernel, h);
    });
  }

OIDN_NAMESPACE_END
