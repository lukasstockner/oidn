// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/error_process.h"
#include "cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class CPUErrorProcess final : public ErrorProcess
  {
  public:
    CPUErrorProcess(const Ref<CPUEngine>& engine, const ErrorProcessDesc& desc);
    void submit() override;

  private:
    Ref<CPUEngine> engine;
  };

OIDN_NAMESPACE_END
