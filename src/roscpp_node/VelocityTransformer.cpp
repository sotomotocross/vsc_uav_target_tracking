#include "VelocityTransformer.hpp"

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
  // Function for transforming camera velocities to UAV velocities for Camera 1
  Eigen::MatrixXd VelocityTransformer::VelTrans1(Eigen::MatrixXd CameraVel1)
  {
    // Transformation matrices for Camera
    Matrix<double, 3, 1> tt1;
    tt1(0, 0) = 0;
    tt1(1, 0) = 0;
    tt1(2, 0) = -0.14;

    Matrix<double, 3, 3> Tt1;
    Tt1(0, 0) = 0;
    Tt1(0, 1) = -tt1(2, 0);
    Tt1(0, 2) = tt1(1, 0);
    Tt1(1, 0) = tt1(2, 0);
    Tt1(1, 1) = 0;
    Tt1(1, 2) = -tt1(0, 0);
    Tt1(2, 0) = -tt1(1, 0);
    Tt1(2, 1) = tt1(0, 0);
    Tt1(2, 2) = 0;

    double thx1 = 0;
    double thy1 = M_PI_2;
    double thz1 = 0;

    Matrix<double, 3, 3> Rx1;
    Rx1(0, 0) = 1;
    Rx1(0, 1) = 0;
    Rx1(0, 2) = 0;
    Rx1(1, 0) = 0;
    Rx1(1, 1) = cos(thx1);
    Rx1(1, 2) = -sin(thx1);
    Rx1(2, 0) = 0;
    Rx1(2, 1) = sin(thx1);
    Rx1(2, 2) = cos(thx1);

    Matrix<double, 3, 3> Ry1;
    Ry1(0, 0) = cos(thy1);
    Ry1(0, 1) = 0;
    Ry1(0, 2) = sin(thy1);
    Ry1(1, 0) = 0;
    Ry1(1, 1) = 1;
    Ry1(1, 2) = 0;
    Ry1(2, 0) = -sin(thy1);
    Ry1(2, 1) = 0;
    Ry1(2, 2) = cos(thy1);

    Matrix<double, 3, 3> Rz1;
    Rz1(0, 0) = cos(thz1);
    Rz1(0, 1) = -sin(thz1);
    Rz1(0, 2) = 0;
    Rz1(1, 0) = sin(thz1);
    Rz1(1, 1) = cos(thz1);
    Rz1(1, 2) = 0;
    Rz1(2, 0) = 0;
    Rz1(2, 1) = 0;
    Rz1(2, 2) = 1;

    Matrix<double, 3, 3> Rth1;
    Rth1.setZero(3, 3);
    Rth1 = Rz1 * Ry1 * Rx1;

    // Conversion of camera velocities to UAV velocities
    Matrix<double, 6, 1> VelCam1;
    VelCam1(0, 0) = CameraVel1(0, 0);
    VelCam1(1, 0) = CameraVel1(1, 0);
    VelCam1(2, 0) = CameraVel1(2, 0);
    VelCam1(3, 0) = CameraVel1(3, 0);
    VelCam1(4, 0) = CameraVel1(4, 0);
    VelCam1(5, 0) = CameraVel1(5, 0);

    Matrix<double, 3, 3> Zeroes1;
    Zeroes1.setZero(3, 3);

    Matrix<double, 6, 6> Vtrans1;
    Vtrans1.block(0, 0, 3, 3) = Rth1;
    Vtrans1.block(0, 3, 3, 3) = Tt1 * Rth1;
    Vtrans1.block(3, 0, 3, 3) = Zeroes1;
    Vtrans1.block(3, 3, 3, 3) = Rth1;

    Matrix<double, 6, 1> VelUAV1;
    VelUAV1.setZero(6, 1);
    VelUAV1 = Vtrans1 * VelCam1;

    return VelUAV1;
  }

  Eigen::MatrixXd VelocityTransformer::VelTrans(Eigen::MatrixXd CameraVel)
  {
    Matrix<double, 3, 1> tt;
    tt(0, 0) = 0;
    tt(1, 0) = 0;
    tt(2, 0) = 0;

    Matrix<double, 3, 3> Tt;
    Tt(0, 0) = 0;
    Tt(0, 1) = -tt(2, 0);
    Tt(0, 2) = tt(1, 0);
    Tt(1, 0) = tt(2, 0);
    Tt(1, 1) = 0;
    Tt(1, 2) = -tt(0, 0);
    Tt(2, 0) = -tt(1, 0);
    Tt(2, 1) = tt(0, 0);
    Tt(2, 2) = 0;

    double thx = M_PI_2;
    double thy = M_PI;
    double thz = M_PI_2;

    Matrix<double, 3, 3> Rx;
    Rx(0, 0) = 1;
    Rx(0, 1) = 0;
    Rx(0, 2) = 0;
    Rx(1, 0) = 0;
    Rx(1, 1) = cos(thx);
    Rx(1, 2) = -sin(thx);
    Rx(2, 0) = 0;
    Rx(2, 1) = sin(thx);
    Rx(2, 2) = cos(thx);

    Matrix<double, 3, 3> Ry;
    Ry(0, 0) = cos(thy);
    Ry(0, 1) = 0;
    Ry(0, 2) = sin(thy);
    Ry(1, 0) = 0;
    Ry(1, 1) = 1;
    Ry(1, 2) = 0;
    Ry(2, 0) = -sin(thy);
    Ry(2, 1) = 0;
    Ry(2, 2) = cos(thy);

    Matrix<double, 3, 3> Rz;
    Rz(0, 0) = cos(thz);
    Rz(0, 1) = -sin(thz);
    Rz(0, 2) = 0;
    Rz(1, 0) = sin(thz);
    Rz(1, 1) = cos(thz);
    Rz(1, 2) = 0;
    Rz(2, 0) = 0;
    Rz(2, 1) = 0;
    Rz(2, 2) = 1;

    Matrix<double, 3, 3> Rth;
    Rth.setZero(3, 3);
    Rth = Rz * Ry * Rx;

    Matrix<double, 6, 1> VelCam;
    VelCam(0, 0) = CameraVel(0, 0);
    VelCam(1, 0) = CameraVel(1, 0);
    VelCam(2, 0) = CameraVel(2, 0);
    VelCam(3, 0) = 0;
    VelCam(4, 0) = 0;
    VelCam(5, 0) = CameraVel(3, 0);

    Matrix<double, 3, 3> Zeroes;
    Zeroes.setZero(3, 3);

    Matrix<double, 6, 6> Vtrans;
    Vtrans.block(0, 0, 3, 3) = Rth;
    Vtrans.block(0, 3, 3, 3) = Tt * Rth;
    Vtrans.block(3, 0, 3, 3) = Zeroes;
    Vtrans.block(3, 3, 3, 3) = Rth;

    Matrix<double, 6, 1> VelUAV;
    VelUAV.setZero(6, 1);
    VelUAV = Vtrans * VelCam;

    return VelUAV;
  }
} // namespace vsc_uav_target_tracking
