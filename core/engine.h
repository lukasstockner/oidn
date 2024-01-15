// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "kernel.h"
#include "buffer.h"
#include "tensor.h"
#include "image.h"
#include <functional>

OIDN_NAMESPACE_BEGIN

  class Graph;

  struct ConvDesc;
  struct ConcatConvDesc;
  struct PoolDesc;
  struct UpsampleDesc;
  struct TensorCopyDesc;
  struct InputProcessDesc;
  struct OutputProcessDesc;
  struct ErrorProcessDesc;

  enum class PostOp;
  class Conv;
  class ConcatConv;
  class Pool;
  class Upsample;
  class TensorCopy;
  class Autoexposure;
  class TransferFunction;
  class InputProcess;
  class OutputProcess;
  class ErrorProcess;
  class ImageCopy;

  class ScratchBuffer;
  class ScratchBufferManager;

  // A device consists of one or more execution "engines"
  class Engine : public RefCount
  {
  public:
    virtual ~Engine() = default;

    virtual Device* getDevice() const = 0;

    // Buffer
    virtual Ref<Buffer> newBuffer(size_t byteSize, Storage storage);
    virtual Ref<Buffer> newBuffer(void* ptr, size_t byteSize);

    virtual Ref<Buffer> newExternalBuffer(ExternalMemoryTypeFlag fdType,
                                          int fd, size_t byteSize);

    virtual Ref<Buffer> newExternalBuffer(ExternalMemoryTypeFlag handleType,
                                          void* handle, const void* name, size_t byteSize);

    Ref<ScratchBuffer> newScratchBuffer(size_t byteSize);

    // Tensor
    virtual bool isSupported(const TensorDesc& desc) const;
    virtual std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, Storage storage = Storage::Device);
    virtual std::shared_ptr<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset = 0);

    // Ops
    virtual std::shared_ptr<Graph> newGraph(const std::shared_ptr<TensorMap>& constTensors, bool fastMath = false);
    virtual bool isConvSupported(PostOp postOp);
    virtual std::shared_ptr<Conv> newConv(const ConvDesc& desc) = 0;
    virtual std::shared_ptr<Pool> newPool(const PoolDesc& desc) = 0;
    virtual std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) = 0;
    virtual std::shared_ptr<TensorCopy> newTensorCopy(const TensorCopyDesc& desc) = 0;
    virtual std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) = 0;
    virtual std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) = 0;
    virtual std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) = 0;
    virtual std::shared_ptr<ErrorProcess> newErrorProcess(const ErrorProcessDesc& desc) = 0;
    virtual std::shared_ptr<ImageCopy> newImageCopy() = 0;

    // Unified shared memory (USM)
    virtual void* usmAlloc(size_t byteSize, Storage storage);
    virtual void usmFree(void* ptr, Storage storage);
    virtual void usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize);
    virtual void submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize);

    // Runs a host task
    virtual void runHostTask(std::function<void()>&& f)
    {
      f();
    }

    // Enqueues a host function
    virtual void submitHostFunc(std::function<void()>&& f) = 0;

    // Waits for all asynchronous commands to complete (blocks)
    virtual void wait() = 0;

    virtual int getMaxWorkGroupSize() const { return 0; }

  private:
    // Memory
    std::weak_ptr<ScratchBufferManager> scratchManagerWp;
  };

OIDN_NAMESPACE_END
