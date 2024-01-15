// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "cpu_device.h"

OIDN_NAMESPACE_BEGIN

  class CPUEngine : public Engine
  {
  public:
    explicit CPUEngine(const Ref<CPUDevice>& device);

    Device* getDevice() const override { return device; }

    // Ops
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<TensorCopy> newTensorCopy(const TensorCopyDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    std::shared_ptr<ErrorProcess> newErrorProcess(const ErrorProcessDesc& desc) override;
    std::shared_ptr<ImageCopy> newImageCopy() override;

    // Unified shared memory (USM)
    void* usmAlloc(size_t byteSize, Storage storage) override;
    void usmFree(void* ptr, Storage storage) override;
    void usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;
    void submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

    // Runs a parallel host task in the thread arena (if it exists)
    void runHostTask(std::function<void()>&& f) override;

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& f) override;

    void wait() override {}

  protected:
    CPUDevice* device;
  };

OIDN_NAMESPACE_END
