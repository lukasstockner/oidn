// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_engine.h"
#include "dnnl_tensor.h"
#include "dnnl_conv.h"

OIDN_NAMESPACE_BEGIN

  DNNLEngine::DNNLEngine(const Ref<CPUDevice>& device)
    : CPUEngine(device)
  {
    dnnl_set_verbose(clamp(device->verbose - 2, 0, 2)); // unfortunately this is not per-device but global
    dnnlEngine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnlStream = dnnl::stream(dnnlEngine);
  }

  void DNNLEngine::wait()
  {
    dnnlStream.wait();
  }

  std::shared_ptr<Tensor> DNNLEngine::newTensor(const TensorDesc& desc, Storage storage)
  {
    if (!isSupported(desc))
      throw std::invalid_argument("unsupported tensor descriptor");

    return std::make_shared<DNNLTensor>(this, desc, storage);
  }

  std::shared_ptr<Tensor> DNNLEngine::newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
  {
    if (!isSupported(desc))
      throw std::invalid_argument("unsupported tensor descriptor");
    if (buffer->getEngine() != this)
      throw std::invalid_argument("buffer was created by a different engine");

    return std::make_shared<DNNLTensor>(buffer, desc, byteOffset);
  }

  std::shared_ptr<Conv> DNNLEngine::newConv(const ConvDesc& desc)
  {
    return std::make_shared<DNNLConv>(this, desc);
  }

OIDN_NAMESPACE_END
