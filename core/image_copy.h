// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "image.h"

namespace oidn {

  class ImageCopy : public virtual Op
  {
  public:
    void setSrc(const std::shared_ptr<Image>& src) { this->src = src; }
    void setDst(const std::shared_ptr<Image>& dst) { this->dst = dst; }

  protected:
    std::shared_ptr<Image> src;
    std::shared_ptr<Image> dst;
  };

} // namespace oidn
