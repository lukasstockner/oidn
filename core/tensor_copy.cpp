// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_copy.h"

OIDN_NAMESPACE_BEGIN

  TensorCopy::TensorCopy(const TensorCopyDesc& desc)
    : TensorCopyDesc(desc)
  {
    if (srcDesc.getRank() != 3)
      throw std::invalid_argument("invalid copy source shape");

    TensorDims dstDims{srcDesc.getC(), srcDesc.getH(), srcDesc.getW()};
    TensorDims dstPaddedDims{srcDesc.getPaddedC(), dstDims[1], dstDims[2]};
    dstDesc = {dstDims, dstPaddedDims, srcDesc.layout, srcDesc.dataType};
  }

  void TensorCopy::setSrc(const std::shared_ptr<Tensor>& src)
  {
    if (!src || src->getDesc() != srcDesc)
      throw std::invalid_argument("invalid copy source");

    this->src = src;
    updateSrc();
  }

  void TensorCopy::setDst(const std::shared_ptr<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid copy destination");

    this->dst = dst;
    updateDst();
  }

OIDN_NAMESPACE_END
