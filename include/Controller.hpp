#pragma once

#include <ros/ros.h>
#include "FeatureData.hpp"
#include "VelocityTransformer.hpp"
#include "DynamicsCalculator.hpp"

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
  class DynamicsCalculator; // Forward declaration

  class Controller
  {
  public:
    // Constructor and Destructor
    Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh);
    ~Controller();

    // Callbacks
    void altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message);
    void featureCallback_poly_custom_tf(const img_seg_cnn::PolyCalcCustomTF::ConstPtr &s_message);
    void featureCallback_poly_custom(const img_seg_cnn::PolyCalcCustom::ConstPtr &s_message);

    // Update function
    void update();

  private:
    // ROS NodeHandles
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // ROS Subscribers
    ros::Subscriber feature_sub_poly_custom_;
    ros::Subscriber feature_sub_poly_custom_tf_;
    ros::Subscriber alt_sub_;

    // ROS Publishers
    ros::Publisher vel_pub_;
    ros::Publisher state_vec_pub_;
    ros::Publisher state_vec_des_pub_;

    ros::Publisher cmd_vel_pub_;
    ros::Publisher img_moments_error_pub_;
    ros::Publisher moments_pub_;
    ros::Publisher central_moments_pub_;

    // Feature data
    FeatureData feature_data_;

    // Declare a static instance of UtilityFunctions
    static DynamicsCalculator dynamics_calculator;

    // Update loop thread
    std::thread control_loop_thread;

    int dim_s;
    int dim_inputs;

    // Eigen Vectors and Matrices
    Eigen::VectorXd state_vector;
    Eigen::VectorXd state_vector_des;
    Eigen::VectorXd velocities;
    Eigen::VectorXd cmd_vel;
    Eigen::VectorXd error;

    Eigen::MatrixXd gains;
    Eigen::VectorXd feature_vector;
    Eigen::VectorXd transformed_features;
    Eigen::VectorXd opencv_moments;
    Eigen::MatrixXd polygon_features;
    Eigen::MatrixXd transformed_polygon_features;

    // Control flags and variables
    int flag;
    double Tx, Ty, Tz, Oz;
    double Z0, Z1, Z2, Z3;    

    double l;
    double umax;
    double umin;
    double vmax;
    double vmin;
    double cu;
    double cv;

    double cX, cY;
    int cX_int, cY_int;

    double s_bar_x, s_bar_y;
    double first_min_index, second_min_index;
    double custom_sigma, custom_sigma_square, custom_sigma_square_log;
    double angle_tangent, angle_radian, angle_deg;

    double transformed_s_bar_x, transformed_s_bar_y;
    int transformed_first_min_index, transformed_second_min_index;
    double transformed_sigma, transformed_sigma_square, transformed_sigma_square_log;
    double transformed_tangent, transformed_angle_radian, transformed_angle_deg;

    // Desired values
    double sigma_des;
    double sigma_square_des;
    double sigma_log_des;

    double angle_deg_des;
    double angle_des_tan;
  };
} // namespace vsc_uav_target_tracking
