// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/tensor_copy.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUTensorCopy final : public TensorCopy
  {
  public:
    CPUTensorCopy(const Ref<CPUEngine>& engine, const TensorCopyDesc& desc);
    void submit() override;

  private:
    Ref<CPUEngine> engine;
  };

OIDN_NAMESPACE_END
