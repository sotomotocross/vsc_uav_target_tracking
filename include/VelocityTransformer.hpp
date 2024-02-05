#pragma once

#include "Controller.hpp"
#include "FeatureData.hpp"

#include <geometry_msgs/TwistStamped.h>
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

#include "img_seg_cnn/PredData.h"
#include "img_seg_cnn/PolyCalcCustom.h"
#include "img_seg_cnn/PolyCalcCustomTF.h"

#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

namespace vsc_uav_target_tracking
{
  class VelocityTransformer
  {
  public:
    static Eigen::MatrixXd VelTrans(Eigen::MatrixXd CameraVel);
    static Eigen::MatrixXd VelTrans1(Eigen::MatrixXd CameraVel1);
  private:
    // Add any private members or helper functions related to velocity transformation
  };
} // namespace vsc_uav_target_tracking
