// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_engine.h"
#include "cuda_external_buffer.h"
#include "cuda_conv.h"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_error_process.h"
#include "../gpu/gpu_pool.h"
#include "../gpu/gpu_upsample.h"
#include "../gpu/gpu_image_copy.h"
#include "../gpu/gpu_tensor_copy.h"

OIDN_NAMESPACE_BEGIN

  CUDAEngine::CUDAEngine(const Ref<CUDADevice>& device,
                         cudaStream_t stream)
    : device(device.get()),
      stream(stream) {}

  Ref<Buffer> CUDAEngine::newExternalBuffer(ExternalMemoryTypeFlag fdType,
                                            int fd, size_t byteSize)
  {
    return makeRef<CUDAExternalBuffer>(this, fdType, fd, byteSize);
  }

  Ref<Buffer> CUDAEngine::newExternalBuffer(ExternalMemoryTypeFlag handleType,
                                            void* handle, const void* name, size_t byteSize)
  {
    return makeRef<CUDAExternalBuffer>(this, handleType, handle, name, byteSize);
  }

  bool CUDAEngine::isSupported(const TensorDesc& desc) const
  {
    // CUTLASS stores tensor strides in 32-bit signed integers
    return Engine::isSupported(desc) && desc.getNumElements() <= INT32_MAX;
  }

  std::shared_ptr<Conv> CUDAEngine::newConv(const ConvDesc& desc)
  {
    return newCUDAConv(this, desc);
  }

  std::shared_ptr<Pool> CUDAEngine::newPool(const PoolDesc& desc)
  {
    return std::make_shared<GPUPool<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Upsample> CUDAEngine::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<GPUUpsample<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<TensorCopy> CUDAEngine::newTensorCopy(const TensorCopyDesc& desc)
  {
    return std::make_shared<GPUTensorCopy<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Autoexposure> CUDAEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<CUDAEngine, 1024>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> CUDAEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<OutputProcess> CUDAEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<ErrorProcess> CUDAEngine::newErrorProcess(const ErrorProcessDesc& desc)
  {
    return std::make_shared<GPUErrorProcess<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<ImageCopy> CUDAEngine::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<CUDAEngine>>(this);
  }

  void* CUDAEngine::usmAlloc(size_t byteSize, Storage storage)
  {
    if (byteSize == 0)
      return nullptr;

    void* ptr = nullptr;

    switch (storage)
    {
    case Storage::Host:
      checkError(cudaMallocHost(&ptr, byteSize));
      return ptr;

    case Storage::Device:
      checkError(cudaMalloc(&ptr, byteSize));
      return ptr;

    case Storage::Managed:
      checkError(cudaMallocManaged(&ptr, byteSize));
      return ptr;

    default:
      throw Exception(Error::InvalidArgument, "invalid storage mode");
    }
  }

  void CUDAEngine::usmFree(void* ptr, Storage storage)
  {
    if (ptr == nullptr)
      return;

    if (storage == Storage::Host)
      checkError(cudaFreeHost(ptr));
    else
      checkError(cudaFree(ptr));
  }

  void CUDAEngine::usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    checkError(cudaMemcpy(dstPtr, srcPtr, byteSize, cudaMemcpyDefault));
  }

  void CUDAEngine::submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    checkError(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, cudaMemcpyDefault, stream));
  }

  namespace
  {
    void CUDART_CB hostFuncCallback(cudaStream_t stream, cudaError_t status, void* fPtr)
    {
      std::unique_ptr<std::function<void()>> f(reinterpret_cast<std::function<void()>*>(fPtr));
      if (status == cudaSuccess)
        (*f)();
    }
  }

  void CUDAEngine::submitHostFunc(std::function<void()>&& f)
  {
    auto fPtr = new std::function<void()>(std::move(f));
    checkError(cudaStreamAddCallback(stream, hostFuncCallback, fPtr, 0));
  }

  void CUDAEngine::wait()
  {
    checkError(cudaStreamSynchronize(stream));
  }

OIDN_NAMESPACE_END