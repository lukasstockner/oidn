// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_engine.h"
#include "cpu_pool.h"
#include "cpu_upsample.h"
#include "cpu_autoexposure.h"
#include "cpu_input_process.h"
#include "cpu_output_process.h"
#include "cpu_error_process.h"
#include "cpu_image_copy.h"
#include "cpu_tensor_copy.h"

OIDN_NAMESPACE_BEGIN

  CPUEngine::CPUEngine(const Ref<CPUDevice>& device)
    : device(device.get())
  {}

  void CPUEngine::runHostTask(std::function<void()>&& f)
  {
    if (device->arena)
      device->arena->execute(f);
    else
      f();
  }

  std::shared_ptr<Pool> CPUEngine::newPool(const PoolDesc& desc)
  {
    return std::make_shared<CPUPool>(this, desc);
  }

  std::shared_ptr<Upsample> CPUEngine::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<CPUUpsample>(this, desc);
  }

  std::shared_ptr<TensorCopy> CPUEngine::newTensorCopy(const TensorCopyDesc& desc)
  {
    return std::make_shared<CPUTensorCopy>(this, desc);
  }

  std::shared_ptr<Autoexposure> CPUEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<CPUAutoexposure>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> CPUEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<CPUInputProcess>(this, desc);
  }

  std::shared_ptr<OutputProcess> CPUEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<CPUOutputProcess>(this, desc);
  }

  std::shared_ptr<ErrorProcess> CPUEngine::newErrorProcess(const ErrorProcessDesc& desc)
  {
    return std::make_shared<CPUErrorProcess>(this, desc);
  }

  std::shared_ptr<ImageCopy> CPUEngine::newImageCopy()
  {
    return std::make_shared<CPUImageCopy>(this);
  }

  void CPUEngine::submitHostFunc(std::function<void()>&& f)
  {
    f(); // no async execution on the CPU
  }

  void* CPUEngine::usmAlloc(size_t byteSize, Storage storage)
  {
    if (byteSize == 0)
      return nullptr;
    return alignedMalloc(byteSize);
  }

  void CPUEngine::usmFree(void* ptr, Storage storage)
  {
    if (ptr != nullptr)
      alignedFree(ptr);
  }

  void CPUEngine::usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    std::memcpy(dstPtr, srcPtr, byteSize);
  }

  void CPUEngine::submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    std::memcpy(dstPtr, srcPtr, byteSize);
  }

OIDN_NAMESPACE_END
