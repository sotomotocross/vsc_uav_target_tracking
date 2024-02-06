#include "vsc_uav_target_tracking/DynamicsCalculator.hpp"

#include "vsc_uav_target_tracking/Controller.hpp"
#include "vsc_uav_target_tracking/FeatureData.hpp"

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
    MatrixXd DynamicsCalculator::img_moments_system(Eigen::VectorXd moments,
                                                    double Z0, double l,
                                                    double cu, double cv)
    {
        int dim_s = 4;
        int dim_inputs = 4;

        // cout << "dim_s = " << dim_s << endl;
        // cout << "dim_inputs = " << dim_inputs << endl;
        
        Eigen::MatrixXd model_mat(dim_s, dim_inputs);
        Eigen::MatrixXd Le(dim_s, dim_inputs);
        model_mat.setZero(dim_s, dim_inputs);
        Le.setZero(dim_s, dim_inputs);
        
        // cout << "model_mat = " << model_mat << endl;
        // cout << "Le = " << Le << endl;

        double gamma_1 = 1.0;
        double gamma_2 = 1.0;

        double A = -gamma_1 / Z0;
        double B = -gamma_2 / Z0;
        double C = 1 / Z0;

        Eigen::VectorXd L_area;
        L_area.setZero(6);

        // cout << "L_area = " << L_area << endl;

        double xg = ((moments[1] / moments[0]) - cu) / l; // x-axis centroid
        // cout << "xg = " << xg << endl;
        double yg = ((moments[2] / moments[0]) - cv) / l; // y-axis centroid
        // cout << "yg = " << yg << endl;
        double area = abs(log(sqrt(moments[0]))); // area

        double n20 = moments[17];
        double n02 = moments[19];
        double n11 = moments[18];

        Eigen::VectorXd L_xg(6);
        Eigen::VectorXd L_yg(6);
        L_xg.setZero(6);
        L_yg.setZero(6);

        // cout << "L_xg = " << L_xg << endl;
        // cout << "L_yg = " << L_yg << endl;

        double mu20_ux = -3 * A * moments[10] - 2 * B * moments[11]; // μ20_ux
        double mu02_uy = -2 * A * moments[11] - 3 * B * moments[12]; // μ02_uy
        double mu11_ux = -2 * A * moments[11] - B * moments[12];     // μ11_ux
        double mu11_uy = -2 * B * moments[11] - A * moments[10];     // μ11_uy
        double s20 = -7 * xg * moments[10] - 5 * moments[13];
        double t20 = 5 * (yg * moments[10] + moments[14]) + 2 * xg * moments[11];
        double s02 = -5 * (xg * moments[12] + moments[15]) - 2 * yg * moments[11];
        double t02 = 7 * yg * moments[12] + 5 * moments[16];
        double s11 = -6 * xg * moments[11] - 5 * moments[14] - yg * moments[10];
        double t11 = 6 * yg * moments[11] + 5 * moments[15] + xg * moments[12];
        double u20 = -A * s20 + B * t20 + 4 * C * moments[10];
        double u02 = -A * s02 + B * t02 + 4 * C * moments[12];
        double u11 = -A * s11 + B * t11 + 4 * C * moments[11];

        Eigen::VectorXd L_mu20(6);
        Eigen::VectorXd L_mu02(6);
        Eigen::VectorXd L_mu11(6);

        L_mu20.setZero(6);
        L_mu02.setZero(6);
        L_mu11.setZero(6);
        // cout << "L_mu20 = " << L_mu20 << endl;
        // cout << "L_mu02 = " << L_mu02 << endl;
        // cout << "L_mu11 = " << L_mu11 << endl;

        L_mu20 << mu20_ux, -B * moments[10], u20, t20, s20, 2 * moments[11];
        L_mu02 << -A * moments[12], mu02_uy, u02, t02, s02, -2 * moments[11];
        L_mu11 << mu11_ux, mu11_uy, u11, t11, s11, moments[12] - moments[10];
        
        // cout << "L_mu20 = " << L_mu20 << endl;
        // cout << "L_mu02 = " << L_mu02 << endl;
        // cout << "L_mu11 = " << L_mu11 << endl;

        double angle = 0.5 * atan(2 * moments[11] / (moments[10] - moments[12]));
        double Delta = pow(moments[10] - moments[12], 2) + 4 * pow(moments[11], 2);

        double a = moments[11] * (moments[10] + moments[12]) / Delta;
        double b = (2 * pow(moments[11], 2) + moments[12] * (moments[12] - moments[10])) / Delta;
        double c = (2 * pow(moments[11], 2) + moments[10] * (moments[10] - moments[12])) / Delta;
        double d = 5 * (moments[15] * (moments[10] - moments[12]) + moments[11] * (moments[16] - moments[14])) / Delta;
        double e = 5 * (moments[14] * (moments[12] - moments[10]) + moments[11] * (moments[13] - moments[15])) / Delta;

        double angle_ux = area * A + b * B;
        double angle_uy = -c * A - area * B;
        double angle_wx = -b * xg + a * yg + d;
        double angle_wy = a * xg - c * yg + e;
        double angle_uz = -A * angle_wx + B * angle_wy;

        Eigen::VectorXd L_angle(6);
        L_angle.setZero(6);

        double c1 = moments[10] - moments[12];
        double c2 = moments[16] - 3 * moments[14];
        double s1 = 2 * moments[11];
        double s2 = moments[13] - 3 * moments[15];
        double I1 = pow(c1, 2) + pow(s1, 2);
        double I2 = pow(c2, 2) + pow(s2, 2);
        double I3 = moments[10] + moments[12];
        double Px = I1 / pow(I3, 2);
        double Py = area * I2 / pow(I3, 3);

        // cout << "L_xg = " << L_xg << endl;
        // cout << "L_yg = " << L_yg << endl;
        // cout << "L_area = " << L_area << endl;
        // cout << "L_angle = " << L_angle << endl;

        L_xg << -1 / Z0, 0, (xg / Z0) + 4 * (A * n20 + B * n11), xg * yg + 4 * n11, -(1 + pow(xg, 2) + 4 * n20), yg;
        L_yg << 0, -1 / Z0, (yg / Z0) + 4 * (A * n11 + B * n02), 1 + pow(yg, 2) + 4 * n02, -xg * yg - 4 * n11, -xg;
        L_area << -area * A, -area * B, area * ((3 / Z0) - C), 3 * area * yg, -3 * area * xg, 0;
        L_angle << angle_ux, angle_uy, angle_uz, angle_wx, angle_wy, -1;

        // cout << "L_xg = " << L_xg << endl;
        // cout << "L_yg = " << L_yg << endl;
        // cout << "L_area = " << L_area << endl;
        // cout << "L_angle = " << L_angle << endl;

        Eigen::MatrixXd Int_matrix;
        Int_matrix.setZero(dim_s, dim_inputs);
        Int_matrix << -1 / Z0, 0, (xg / Z0) + 4 * (A * n20 + B * n11), yg,
            0, -1 / Z0, (yg / Z0) + 4 * (A * n11 + B * n02), -xg,
            -area * A, -area * B, area * ((3 / Z0) - C), 0,
            angle_ux, angle_uy, angle_uz, -1;

        Eigen::MatrixXd model_coefficients;
        model_coefficients.setIdentity(dim_s, dim_inputs);

        model_coefficients(0, 0) = 1.0;
        model_coefficients(1, 1) = 0.1;
        model_coefficients(2, 2) = 0.1;
        model_coefficients(3, 3) = 0.1;

        // cout << "Int_matrix shape: (" << Int_matrix.rows() << "," << Int_matrix.cols() << ")" << endl;
        // cout << "Int_matrix = \n"
        // << Int_matrix << endl;
        // cout << "model_coefficients * Int_matrix = \n" << model_coefficients * Int_matrix << endl;

        return model_coefficients * Int_matrix;
    }

} // namespace vsc_uav_target_tracking
