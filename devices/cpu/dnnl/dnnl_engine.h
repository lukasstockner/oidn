// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../cpu_engine.h"

OIDN_NAMESPACE_BEGIN

  class DNNLEngine final : public CPUEngine
  {
  public:
    explicit DNNLEngine(const Ref<CPUDevice>& device);

    OIDN_INLINE dnnl::engine& getDNNLEngine() { return dnnlEngine; }
    OIDN_INLINE dnnl::stream& getDNNLStream() { return dnnlStream; }

    void wait() override;

    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, Storage storage) override;
    std::shared_ptr<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset) override;

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;

  private:
    dnnl::engine dnnlEngine;
    dnnl::stream dnnlStream;
  };

OIDN_NAMESPACE_END
