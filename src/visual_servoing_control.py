#!/usr/bin/env python

from __future__ import print_function
import roslib
roslib.load_manifest('vsc_uav_target_tracking')
import rospy
import numpy as np
import time
import matplotlib.pyplot as plt
from mavros_msgs.msg import PositionTarget
from numpy.linalg import norm
from math import cos, sin, tan, sqrt, exp, pi, atan2, acos, asin
import tf
import yaml

from vsc_uav_target_tracking.msg import VSCdata, IBVSdata, EKFdata
from ros_communication import ROSCommunication
from visual_servoing_utils import VisualServoingUtils
from ekf_estimation import EKFEstimation



class VisualServoingControl:
    def __init__(self):
        """
        Initialize the VisualServoingControl class.

        This method sets up the necessary components and parameters for visual servoing control.
        """
        
        self.ros_comms = ROSCommunication()
        self.visual_servoing_utils = VisualServoingUtils(dt=0.03, window_size=5)
        self.ekf_estimator = EKFEstimation(dt=0.03)  # Pass the same dt to EKFEstimation
        
        # Fetch parameters dynamically
        self.use_deriv_error_estimation = rospy.get_param("~use_deriv_error_estimation", True)
        self.use_moving_average_filter = rospy.get_param("~use_moving_average_filter", True)
        self.use_ekf_estimator = rospy.get_param("~use_ekf_estimator", True)

        # Load gains from YAML file
        gains_file_path = rospy.get_param("controller_gains_file", "")
        with open(gains_file_path, 'r') as stream:
            gains = yaml.safe_load(stream)["controller_gains"]

        self.forward_gain_Kc = gains["forward_gain_Kc"]
        self.thrust_gain_Kc = gains["thrust_gain_Kc"]
        self.sway_gain_Kc = gains["sway_gain_Kc"]
        self.yaw_gain_Kc = gains["yaw_gain_Kc"]

        self.forward_gain_Ke = gains["forward_gain_Ke"]
        self.thrust_gain_Ke = gains["thrust_gain_Ke"]
        self.sway_gain_Ke = gains["sway_gain_Ke"]
        self.yaw_gain_Ke = gains["yaw_gain_Ke"]
        
        # uav state variables
        self.uav_vel_body = np.array([0.0, 0.0, 0.0, 0.0])
        self.er_pix_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(8,1)
        self.vel_uav = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ZED stereo camera translation and rotation variables
        self.transCam = [0.0, 0.0, 0.14]
        self.rotCam = [0.0, -1.57, 0.0]
        self.phi_cam = self.rotCam[0]
        self.theta_cam = self.rotCam[1]
        self.psi_cam = self.rotCam[2]
        
        # ZED stereocamera intrinsic parameters
        self.cu = 360.5
        self.cv = 240.5
        self.ax = 252.07
        self.ay = 252.07
        
        # Variables initialization
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0
        self.phi_imu = 0.0
        self.theta_imu = 0.0 
        self.psi_imu = 0.0
        self.w_imu = 0.0
        self.t = 0.0
        self.dt = 0.03
        self.a = 0.2
        self.time = rospy.Time.now().to_sec()

    # Function calling the feature transformation from the image plane on a virtual image plane
    def features_transformation(self, mp, phi, theta):
        """
        Perform feature transformation from the image plane to a virtual image plane.

        Args:
            mp (numpy.ndarray): 12-element array representing four 3D points in the image.
            phi (float): Roll angle.
            theta (float): Pitch angle.

        Returns:
            numpy.ndarray: Transformed 3D points in the virtual image plane.
        """
        Rphi = np.array([[1.0, 0.0, 0.0],[0.0, cos(phi), -sin(phi)],[0.0, sin(phi), cos(phi)]]).reshape(3,3)
        Rtheta = np.array([[cos(theta), 0.0, sin(theta)],[0.0, 1.0, 0.0],[-sin(theta), 0.0, cos(theta)]]).reshape(3,3)
        Rft = np.dot(Rphi, Rtheta)
        mpv0 = np.dot(Rft, mp[0:3])
        mpv1 = np.dot(Rft, mp[3:6])
        mpv2 = np.dot(Rft, mp[6:9])
        mpv3 = np.dot(Rft, mp[9:12])
        mpv = np.hstack((mpv0, mpv1, mpv2, mpv3))
        
        return mpv    
    
    # Function forming the image plane features and forming the interaction matrices for all the features
    def calculate_interaction_matrix(self, mpv, mp_des, cu, cv, ax, ay):
        """
        Calculate the interaction matrix and image plane features.

        Args:
            mpv (numpy.ndarray): Transformed 3D points in the virtual image plane.
            mp_des (numpy.ndarray): Desired image plane features.
            cu (float): Image center x-coordinate.
            cv (float): Image center y-coordinate.
            ax (float): X focal length.
            ay (float): Y focal length.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Interaction matrix and error in image plane.
        """
        x_0 = (mpv[0]-cu)/ax
        y_0 = (mpv[1]-cv)/ay
        Z_0 = mpv[2]

        x_1 = (mpv[3]-cu)/ax
        y_1 = (mpv[4]-cv)/ay
        Z_1 = mpv[5]
        
        x_2 = (mpv[6]-cu)/ax
        y_2 = (mpv[7]-cv)/ay
        Z_2 = mpv[8]
            
        x_3 = (mpv[9]-cu)/ax
        y_3 = (mpv[10]-cv)/ay
        Z_3 = mpv[11]

        xd_0 = (mp_des[0]-cu)/ax
        yd_0 = (mp_des[1]-cv)/ay
        Zd_0 = mp_des[2]

        xd_1 = (mp_des[3]-cu)/ax
        yd_1 = (mp_des[4]-cv)/ay
        Zd_1 = mp_des[5]

        xd_2 = (mp_des[6]-cu)/ax
        yd_2 = (mp_des[7]-cv)/ay
        Zd_2 = mp_des[8]

        xd_3 = (mp_des[9]-cu)/ax
        yd_3 = (mp_des[10]-cv)/ay
        Zd_3 = mp_des[11]
                    
        Lm0 = np.array([[-1.0/Z_0, 0.0, x_0/Z_0, x_0*y_0, -(1.0+x_0*x_0), y_0],
                        [0.0, -1.0/Z_0, y_0/Z_0, 1.0+y_0*y_0, -x_0*y_0, -x_0]]).reshape(2,6)
        Lm1 = np.array([[-1.0/Z_1, 0.0, x_1/Z_1, x_1*y_1, -(1.0+x_1*x_1), y_1],
                        [0.0, -1.0/Z_1, y_1/Z_1, 1.0+y_1*y_1, -x_1*y_1, -x_1]]).reshape(2,6)
        Lm2 = np.array([[-1.0/Z_2, 0.0, x_2/Z_2, x_2*y_2, -(1.0+x_2*x_2), y_2],
                        [0.0, -1.0/Z_2, y_2/Z_2, 1.0+y_2*y_2, -x_2*y_2, -x_2]]).reshape(2,6)
        Lm3 = np.array([[-1.0/Z_3, 0.0, x_3/Z_3, x_3*y_3, -(1.0+x_3*x_3), y_3],
                        [0.0, -1.0/Z_3, y_3/Z_3, 1.0+y_3*y_3, -x_3*y_3, -x_3]]).reshape(2,6)
        Lm = np.concatenate((Lm0, Lm1, Lm2, Lm3), axis=0)
        er_pix = np.array([x_0-xd_0, y_0-yd_0, x_1-xd_1, y_1-yd_1, x_2-xd_2, y_2-yd_2, x_3-xd_3, y_3-yd_3 ]).reshape(8,1) #ax=ay=252.07
        
        return Lm, er_pix
    
    def cartesian_from_pixel(self, mp_pixel, cu, cv, ax, ay):
        """
        Convert pixel coordinates to Cartesian coordinates.

        Args:
            mp_pixel (numpy.ndarray): Image coordinates of 3D points.
            cu (float): Image center x-coordinate.
            cv (float): Image center y-coordinate.
            ax (float): X focal length.
            ay (float): Y focal length.

        Returns:
            numpy.ndarray: Cartesian coordinates of the 3D points.
        """
        Z_0 = mp_pixel[2]
        X_0 = Z_0*((mp_pixel[0]-cu)/ax)
        Y_0 = Z_0*((mp_pixel[1]-cv)/ay)
        
        Z_1 = mp_pixel[5]
        X_1 = Z_1*((mp_pixel[3]-cu)/ax)
        Y_1 = Z_1*((mp_pixel[4]-cv)/ay)
        
        Z_2 = mp_pixel[8]
        X_2 = Z_2*((mp_pixel[6]-cu)/ax)
        Y_2 = Z_2*((mp_pixel[7]-cv)/ay)
        
        Z_3 = mp_pixel[11]    
        X_3 = Z_3*((mp_pixel[9]-cu)/ax)
        Y_3 = Z_3*((mp_pixel[10]-cv)/ay)
                
        mp_cartesian = np.array([X_0, Y_0, Z_0, X_1, Y_1, Z_1, X_2, Y_2, Z_2, X_3, Y_3, Z_3])
        
        return mp_cartesian
    
    def pixels_from_cartesian(self, mp_cartesian, cu, cv, ax, ay):
        """ 
        Convert Cartesian coordinates to pixel coordinates.

        Args:
            mp_cartesian (numpy.ndarray): Cartesian coordinates of 3D points.
            cu (float): Image center x-coordinate.
            cv (float): Image center y-coordinate.
            ax (float): X focal length.
            ay (float): Y focal length.

        Returns:
            numpy.ndarray: Image coordinates of the 3D points.
        """
        u_0 = (mp_cartesian[0]/mp_cartesian[2])*ax + cu
        v_0 = (mp_cartesian[1]/mp_cartesian[2])*ay + cv
        
        u_1 = (mp_cartesian[3]/mp_cartesian[5])*ax + cu
        v_1 = (mp_cartesian[4]/mp_cartesian[5])*ay + cv
        
        u_2 = (mp_cartesian[6]/mp_cartesian[8])*ax + cu
        v_2 = (mp_cartesian[7]/mp_cartesian[8])*ay + cv
        
        u_3 = (mp_cartesian[9]/mp_cartesian[11])*ax + cu
        v_3 = (mp_cartesian[10]/mp_cartesian[11])*ay + cv
        
        mp_pixel = np.array([u_0, v_0, mp_cartesian[2], u_1, v_1, mp_cartesian[5], u_2, v_2, mp_cartesian[8], u_3, v_3, mp_cartesian[11]])        
    
        return mp_pixel

    def quadrotor_vs_control(self, Lm, er_pix, ew, vel_camera):
        """
        Perform Visual Servoing control for quadrotor.

        Args:
            Lm (numpy.ndarray): Interaction matrix.
            er_pix (numpy.ndarray): Error in image plane.
            ew (numpy.ndarray): Derivative error estimation.
            vel_camera (numpy.ndarray): Velocity in the camera frame.

        Returns:
            numpy.ndarray: Control commands for the quadrotor.
        """
        Kc = np.identity(6)
        Kc[0][0] = self.thrust_gain_Kc
        Kc[1][1] = self.sway_gain_Kc
        Kc[2][2] = self.forward_gain_Kc
        Kc[3][3] = self.yaw_gain_Kc
        Kc[4][4] = 0.0
        Kc[5][5] = 0.0

        Ke = np.identity(6)
        Ke[0][0] = self.thrust_gain_Ke
        Ke[1][1] = self.sway_gain_Ke
        Ke[2][2] = self.forward_gain_Ke
        Ke[3][3] = self.yaw_gain_Ke
        Ke[4][4] = 0.0
        Ke[5][5] = 0.0
        
        Ucmd = -np.dot(Kc,np.dot(np.linalg.pinv(Lm), er_pix))+np.dot(np.linalg.pinv(Lm), np.array([0.0, self.a, 0.0, self.a, 0.0, self.a, 0.0, self.a]).reshape(8,1)) - np.dot(Ke, np.dot(np.linalg.pinv(Lm), ew) )
        
        return Ucmd
    
    def get_feature_box(self, box, angle):
        """
        Get feature box for visual servoing based on detected box and angle.

        Args:
            box (List[List[float]]): Detected box coordinates.
            angle (float): Angle of the detected box.

        Returns:
            List[float]: Feature box coordinates.
        """
        if angle >= 0:
            return [box[0][0], box[0][1], self.z, box[1][0], box[1][1], self.z, box[2][0], box[2][1], self.z, box[3][0], box[3][1], self.z]
        else:
            return [box[3][0], box[3][1], self.z, box[0][0], box[0][1], self.z, box[1][0], box[1][1], self.z, box[2][0], box[2][1], self.z]

    def get_transformation_matrix(self, R_y, sst):
        """
        Get the transformation matrix for the camera.

        Args:
            R_y (numpy.ndarray): Rotation matrix.
            sst (numpy.ndarray): Transformation matrix.

        Returns:
            numpy.ndarray: Transformation matrix T.
        """
        T = np.zeros((6, 6), dtype=float)
        T[0:3, 0:3] = R_y
        T[3:6, 3:6] = R_y
        T[0:3, 3:6] = np.dot(sst, R_y)
        return T   
    
    def transform_body_to_camera(self, theta_cam, transCam):
        """
        Transform body coordinates to camera coordinates.

        Args:
            theta_cam (float): Camera pitch angle.
            transCam (List[float]): Camera translation.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: Rotation matrix R_y and transformation matrix sst.
        """
        R_y = np.array([[cos(theta_cam), 0.0, sin(theta_cam)],
                    [0.0, 1.0, 0.0],
                    [-sin(theta_cam), 0.0, cos(theta_cam)]]).reshape(3,3)
        sst = np.array([[0.0, -transCam[2], transCam[1]],
                             [transCam[2], 0.0, -transCam[0]],
                             [-transCam[1], transCam[0], 0.0]]).reshape(3,3)        
        return R_y, sst
    
    def control_execution(self, uav_vel_body):
        """
        Execute the control by publishing velocity commands.

        Args:
            uav_vel_body (numpy.ndarray): Body velocity commands.
        """
        twist = PositionTarget()
        #twist.header.stamp = 1
        twist.header.frame_id = 'world'
        twist.coordinate_frame = 8
        twist.type_mask = 1479
        twist.velocity.x = uav_vel_body[0]
        twist.velocity.y = uav_vel_body[1]
        twist.velocity.z = uav_vel_body[2]
        twist.yaw_rate = uav_vel_body[3]               
        # self.ros_comms.pub_vel.publish(twist)    
    
    def vsc_data_publishing(self, uav_vel_body, er_pix, t_vsc):
        """
        Publish Visual Servoing Control data.

        Args:
            uav_vel_body (numpy.ndarray): Body velocity commands.
            er_pix (numpy.ndarray): Error in image plane.
            t_vsc (float): Time of Visual Servoing Control.
        """
        ibvs_msg = IBVSdata()
        ibvs_msg.errors = er_pix
        ibvs_msg.cmds = uav_vel_body
        # print("ibvs_msg.cmds: ", ibvs_msg.cmds)
        ibvs_msg.time = t_vsc
        self.ros_comms.pub_ibvs_data.publish(ibvs_msg)
        
    def create_ekf_message(self, e_m, e_m_dot, u_bc, v_bc, t_vsc):
        """
        Create and publish Extended Kalman Filter (EKF) data message.

        Args:
            e_m (float): Mean error in image plane.
            e_m_dot (float): Derivative of mean error.
            u_bc (float): X-coordinate of the bounding box center.
            v_bc (float): Y-coordinate of the bounding box center.
            t_vsc (float): Time of Visual Servoing Control.
        """
        ekf_msg = EKFdata()
        ekf_msg.ekf_output = self.ekf_estimator.x_est
        ekf_msg.e_m = e_m
        ekf_msg.e_m_dot = e_m_dot
        ekf_msg.u_bc = u_bc
        ekf_msg.v_bc = v_bc
        ekf_msg.time = t_vsc
        self.ros_comms.pub_ekf_data.publish(ekf_msg)
    
    
    # Detect the line and piloting
    def detection_processing(self, box, angle):
        """
        Process object detection and perform Visual Servoing Control.

        Args:
            box (List[List[float]]): Detected box coordinates.
            angle (float): Angle of the detected box.
        """
        t_vsc = rospy.Time.now().to_sec() - self.time
        
        mp = self.get_feature_box(box, angle)
        mp_cartesian = self.cartesian_from_pixel(mp, self.cu, self.cv, self.ax, self.ay)
        mp_cartesian_v = self.features_transformation(mp_cartesian, self.phi_imu, self.theta_imu)
        mp_pixel_v = self.pixels_from_cartesian(mp_cartesian_v, self.cu, self.cv, self.ax, self.ay)

        mp_des = np.array([420, 472, self.z, 367, 483, self.z, 327, 2+self.a*self.t, self.z, 377, 0+self.a*self.t, self.z]) 
        # print("mp_des: ", mp_des)            
        
        R_y, sst = self.transform_body_to_camera(self.theta_cam, self.transCam)
        
        T = self.get_transformation_matrix(R_y, sst)
        Lm, er_pix = self.calculate_interaction_matrix(mp_pixel_v, mp_des, self.cu, self.cv, self.ax, self.ay) #TRANSFORM FEATURES
        
        u_bc = (box[0][0]+box[1][0]+box[2][0]+box[3][0])/4
        v_bc = (box[0][1]+box[1][1]+box[2][1]+box[3][1])/4
        
        velocity_camera = np.dot(np.linalg.inv(T), self.ros_comms.vel_uav)  
           
        ew = None
        if self.use_deriv_error_estimation:
            ew = self.visual_servoing_utils.deriv_error_estimation(er_pix, self.er_pix_prev)
        ew_filtered = None
        if self.use_moving_average_filter:
            ew_filtered = self.visual_servoing_utils.moving_average_filter(np.array(ew).reshape(8, 1))
        ew_filtered_odometry = ew_filtered - np.array(np.dot(Lm, velocity_camera)).reshape(8,1)
        wave_estimation_final = None
        if self.use_ekf_estimator:
            e_m = (er_pix[0] + er_pix[2] + er_pix[4] + er_pix[6]) / 4
            e_m_dot = (ew_filtered_odometry[0] + ew_filtered_odometry[2] + ew_filtered_odometry[4] + ew_filtered_odometry[6]) / 4
            wave_est_control_input = self.ekf_estimator.estimate(e_m, e_m_dot)
            wave_estimation_final = np.array([wave_est_control_input, [self.a], wave_est_control_input, [self.a],
                                         wave_est_control_input, [self.a], wave_est_control_input, [self.a]]).reshape(8, 1)
        
        UVScmd = self.quadrotor_vs_control(Lm, er_pix, wave_estimation_final, velocity_camera)
        UVScmd = np.dot(T, UVScmd.reshape(-1, 1))
        
        # Make sure UVScmd is a NumPy array and its elements are scalars
        UVScmd = np.asarray(UVScmd).reshape(-1)
                             
        self.uav_vel_body[0] = UVScmd[0]
        self.uav_vel_body[1] = UVScmd[1]
        self.uav_vel_body[2] = UVScmd[2]
        self.uav_vel_body[3] = UVScmd[5]
        
        self.control_execution(self.uav_vel_body)
        self.vsc_data_publishing(self.uav_vel_body, er_pix, t_vsc)
        self.create_ekf_message(e_m, e_m_dot, u_bc, v_bc, t_vsc)        
                        
        self.ros_comms.publish_error(er_pix)
        self.ros_comms.publish_angle(angle)
            
        self.t = self.t+self.dt    
