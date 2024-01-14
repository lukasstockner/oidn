// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"
#include "tensor.h"
#include "color.h"
#include "tile.h"

OIDN_NAMESPACE_BEGIN

  struct ErrorProcessDesc
  {
    TensorDesc srcDesc;
  };

  class ErrorProcess : public Op, protected ErrorProcessDesc
  {
  public:
    ErrorProcess(const ErrorProcessDesc& desc);

    void setSrc(const std::shared_ptr<Tensor>& src);
    void setDst(const std::shared_ptr<Image>& dst);
    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W);

  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Image> dst;
    Tile tile;
  };

OIDN_NAMESPACE_END
