#include "Controller.hpp"
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
    // Initialize the static member of the UtilityFunctions
    DynamicsCalculator Controller::dynamics_calculator;

    // Constructor
    Controller::Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh)
    {
        // Initialize member variables
        cX = 0.0;
        cY = 0.0;
        cX_int = 0;
        cY_int = 0;
        Z0 = 0.0;
        Z1 = 0.0;
        Z2 = 0.0;
        Z3 = 0.0;
        s_bar_x = 0.0;
        s_bar_y = 0.0;
        custom_sigma = 1.0;
        custom_sigma_square = 1.0;
        custom_sigma_square_log = 1.0;
        angle_tangent = 0.0;
        angle_radian = 0.0;
        angle_deg = 0.0;
        transformed_s_bar_x = 0.0;
        transformed_s_bar_y = 0.0;
        transformed_sigma = 1.0;
        transformed_sigma_square = 1.0;
        transformed_sigma_square_log = 1.0;
        transformed_tangent = 0.0;
        transformed_angle_radian = 0.0;
        transformed_angle_deg = 0.0;

        l = 252.07;
        umax = 720;
        umin = 0;
        vmax = 480;
        vmin = 0;
        cu = 360.5;
        cv = 240.5;

        sigma_des = 18.5;
        sigma_square_des = sqrt(sigma_des);
        sigma_log_des = log(sigma_square_des);

        angle_deg_des = 0;
        angle_des_tan = tan((angle_deg_des / 180) * 3.14);

        flag = 0;

        // Set up ROS subscribers
        feature_sub_poly_custom_ = nh_.subscribe("polycalc_custom", 10, &Controller::featureCallback_poly_custom, this);
        feature_sub_poly_custom_tf_ = nh_.subscribe("polycalc_custom_tf", 10, &Controller::featureCallback_poly_custom_tf, this);
        alt_sub_ = nh_.subscribe("/mavros/global_position/rel_alt", 10, &Controller::altitudeCallback, this);


        // Set up ROS publishers
        vel_pub_ = nh_.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
        state_vec_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/state_vec", 1);
        state_vec_des_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/state_vec_des", 1);
        
        cmd_vel_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/cmd_vel", 1);
        img_moments_error_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/img_moments_error", 1);
        moments_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/moments", 1);
        central_moments_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/central_moments", 1);

        // Create a thread for the control loop
        control_loop_thread = std::thread([this]()
                                          {
      ros::Rate rate(50); // Adjust the rate as needed
      while (ros::ok())
      {
        update();
        rate.sleep();
      } });
    }

    // Destructor
    Controller::~Controller()
    {
        // Shutdown ROS publishers...
        vel_pub_.shutdown();
    }

    // Callback for altitude data
    void Controller::altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message)
    {
        try{
            // Handle altitude data...
            Z0 = alt_message->data;
            Z1 = alt_message->data;
            Z2 = alt_message->data;
            Z3 = alt_message->data;
            flag = 1;
            // cout << "flag = " << flag << endl;
            ROS_INFO("Z0: %f", Z0);
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in altitudeCallback: %s", e.what());
        }
    }

    // Callback for custom TF features
    void Controller::featureCallback_poly_custom_tf(const img_seg_cnn::PolyCalcCustomTF::ConstPtr &s_message)
    {
        try {

            feature_data_.poly_custom_tf_data = s_message;
            cout << "~~~~~~~~~~ featureCallback_poly_custom_tf ~~~~~~~~~~" << endl;
            int N = s_message->transformed_features.size();
            transformed_features.setZero(N);
            transformed_polygon_features.setZero(N / 2, 2);

            for (int i = 0; i < N - 1; i += 2)
            {
                // cout << "i = " << i << endl;
                transformed_features[i] = s_message->transformed_features[i];
                transformed_features[i + 1] = s_message->transformed_features[i + 1];
            }

            for (int i = 0, j = 0; i < N - 1 && j < N / 2; i += 2, ++j)
            {
                // cout << "i = " << i << endl;
                transformed_polygon_features(j, 0) = transformed_features[i];
                transformed_polygon_features(j, 1) = transformed_features[i + 1];
            }

            transformed_s_bar_x = s_message->transformed_barycenter_features[0];
            transformed_s_bar_y = s_message->transformed_barycenter_features[1];

            transformed_first_min_index = s_message->d_transformed;
            transformed_second_min_index = s_message->f_transformed;

            transformed_sigma = s_message->transformed_sigma;
            transformed_sigma_square = s_message->transformed_sigma_square;
            transformed_sigma_square_log = s_message->transformed_sigma_square_log;

            transformed_tangent = s_message->transformed_tangent;
            transformed_angle_radian = s_message->transformed_angle_radian;
            transformed_angle_deg = s_message->transformed_angle_deg;

            opencv_moments.setZero(s_message->moments.size());
            for (int i = 0; i < s_message->moments.size(); i++)
            {
                // cout << "i = " << i << endl;
                opencv_moments[i] = s_message->moments[i];
            }
            cX = opencv_moments[1] / opencv_moments[0];
            cY = opencv_moments[2] / opencv_moments[0];

            cX_int = (int)cX;
            cY_int = (int)cY;

            cout << "transformed_angle_deg: " << transformed_angle_deg << endl;
            flag = 1;
            ROS_INFO("transformed_angle_deg: %f", transformed_angle_deg);
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in featureCallback_poly_custom_tf: %s", e.what());
        }
    }

    // Callback for custom features
    void Controller::featureCallback_poly_custom(const img_seg_cnn::PolyCalcCustom::ConstPtr &s_message)
    {
        try {
            
            // Handle custom features...
            cout << "~~~~~~~~~~ featureCallback_poly_custom ~~~~~~~~~~" << endl;
            feature_data_.poly_custom_data = s_message;
            feature_vector.setZero(s_message->features.size());
            polygon_features.setZero(s_message->features.size() / 2, 2);

            for (int i = 0; i < s_message->features.size() - 1; i += 2)
            {
                feature_vector[i] = s_message->features[i];
                feature_vector[i + 1] = s_message->features[i + 1];
            }

            for (int i = 0, j = 0; i < s_message->features.size() - 1 && j < s_message->features.size() / 2; i += 2, ++j)
            {
                polygon_features(j, 0) = feature_vector[i];
                polygon_features(j, 1) = feature_vector[i + 1];
            }

            s_bar_x = s_message->barycenter_features[0];
            s_bar_y = s_message->barycenter_features[1];

            first_min_index = s_message->d;
            second_min_index = s_message->f;

            custom_sigma = s_message->custom_sigma;
            custom_sigma_square = s_message->custom_sigma_square;
            custom_sigma_square_log = s_message->custom_sigma_square_log;

            angle_tangent = s_message->tangent;
            angle_radian = s_message->angle_radian;
            angle_deg = s_message->angle_deg;

            cout << "angle_deg: " << angle_deg << endl;
            flag = 1;
            ROS_INFO("angle_deg: %f", angle_deg);
        } catch (const std::exception& e) {
            ROS_ERROR("Exception in featureCallback_poly_custom: %s", e.what());
        }
    }

    // Main update function
    void Controller::update()
    {

        while (ros::ok())
        {
            cout << "flag: " << flag << endl;

            if (flag == 1)
            {
                MatrixXd IM = dynamics_calculator.img_moments_system(opencv_moments, Z0, l,
                                                    cu, cv);
                cout << "IM: " << IM << endl;

                state_vector << ((opencv_moments[1] / opencv_moments[0]) - cu) / l, ((opencv_moments[2] / opencv_moments[0]) - cv) / l, log(sqrt(opencv_moments[0])), atan(2 * opencv_moments[11] / (opencv_moments[10] - opencv_moments[12]));
                state_vector_des << 0.0, 0.0, 5.0, angle_des_tan;

                cout << "state_vector = " << state_vector.transpose() << endl;
                cout << "state_vector_des = " << state_vector_des.transpose() << endl;

                error.setZero(6);

                error = state_vector - state_vector_des;
                cout << "error = " << error.transpose() << endl;

                MatrixXd pinv = IM.completeOrthogonalDecomposition().pseudoInverse();
                VectorXd velocities = pinv * error;

                cmd_vel = pinv * error;
            }
            VectorXd tmp_cmd_vel(dim_inputs);
            tmp_cmd_vel << cmd_vel[0], cmd_vel[1], cmd_vel[2], cmd_vel[3];
            cout << "cmd_vel: " << cmd_vel.transpose() << endl;

            //****SEND VELOCITIES TO AUTOPILOT THROUGH MAVROS****//
            mavros_msgs::PositionTarget dataMsg;
            Matrix<double, 4, 1> caminputs;
            caminputs(0, 0) = cmd_vel[0];
            caminputs(1, 0) = cmd_vel[1];
            caminputs(2, 0) = cmd_vel[2];
            caminputs(3, 0) = cmd_vel[3];

            dataMsg.coordinate_frame = 8;
            dataMsg.type_mask = 1479;
            dataMsg.header.stamp = ros::Time::now();

            Tx = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(0, 0);
            Ty = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(1, 0);
            Tz = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(2, 0);
            Oz = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(5, 0);

            double gain_tx;
            nh_.getParam("/gain_tx", gain_tx);
            double gain_ty;
            nh_.getParam("/gain_ty", gain_ty);
            double gain_tz;
            nh_.getParam("/gain_tz", gain_tz);
            double gain_yaw;
            nh_.getParam("/gain_yaw", gain_yaw);

            // Τracking tuning
            dataMsg.velocity.x = gain_tx * Tx + 0.08;
            dataMsg.velocity.y = gain_ty * Ty;
            dataMsg.velocity.z = gain_tz * Tz;
            dataMsg.yaw_rate = gain_yaw * Oz;

            printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g)", dataMsg.velocity.x, dataMsg.velocity.y, dataMsg.velocity.z, dataMsg.yaw_rate);
            cout << "\n"
                 << endl;

            std_msgs::Float64MultiArray cmd_vel_Msg;
            for (int i = 0; i < cmd_vel.size(); i++)
            {
                cmd_vel_Msg.data.push_back(cmd_vel[i]);
            }

            std_msgs::Float64MultiArray state_vecMsg;
            for (int i = 0; i < state_vector.size(); i++)
            {
                state_vecMsg.data.push_back(state_vector[i]);
            }

            std_msgs::Float64MultiArray state_vec_desMsg;
            for (int i = 0; i < state_vector_des.size(); i++)
            {
                state_vec_desMsg.data.push_back(state_vector_des[i]);
            }

            std_msgs::Float64MultiArray error_Msg;
            for (int i = 0; i < error.size(); i++)
            {
                error_Msg.data.push_back(error[i]);
            }

            cmd_vel_pub_.publish(cmd_vel_Msg);
            state_vec_pub_.publish(state_vecMsg);
            state_vec_des_pub_.publish(state_vec_desMsg);
            img_moments_error_pub_.publish(error_Msg);
            // vel_pub_.publish(dataMsg);
            ros::spinOnce();
            // ros::spin;
            // loop_rate.sleep();
        }
    }
} // namespace vsc_uav_target_tracking
