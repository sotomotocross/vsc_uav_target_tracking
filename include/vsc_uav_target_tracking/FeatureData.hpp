#pragma once

#include "img_seg_cnn/PolyCalcCustom.h"
#include "img_seg_cnn/PolyCalcCustomTF.h"

namespace vsc_uav_target_tracking
{
  struct FeatureData
  {
    img_seg_cnn::PolyCalcCustomTF::ConstPtr poly_custom_tf_data;
    img_seg_cnn::PolyCalcCustom::ConstPtr poly_custom_data;
  };
} // namespace vsc_uav_target_tracking
