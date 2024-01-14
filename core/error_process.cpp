// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "error_process.h"

OIDN_NAMESPACE_BEGIN

  ErrorProcess::ErrorProcess(const ErrorProcessDesc& desc)
    : ErrorProcessDesc(desc)
  {
    if (srcDesc.getRank() != 3)
      throw std::invalid_argument("invalid error processing source shape");

    setTile(0, 0, 0, 0, 0, 0);
  }

  void ErrorProcess::setSrc(const std::shared_ptr<Tensor>& src)
  {
    if (!src || src->getDesc() != srcDesc)
      throw std::invalid_argument("invalid error processing source");

    this->src = src;
  }

  void ErrorProcess::setDst(const std::shared_ptr<Image>& dst)
  {
    if (!dst || dst->getC() > srcDesc.getC())
      throw std::invalid_argument("invalid error processing destination");

    this->dst = dst;
  }

  void ErrorProcess::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

OIDN_NAMESPACE_END
