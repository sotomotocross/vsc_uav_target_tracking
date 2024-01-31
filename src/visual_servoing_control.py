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
from vsc_uav_target_tracking.msg import VSCdata
from ros_communication import ROSCommunication


class VisualServoingControl:
    def __init__(self):
        
        self.ros_comms = ROSCommunication()
        
        # uav state variables
        self.landed = 0
        self.takeoffed = 1
        self.uav_vel_body = np.array([0.0, 0.0, 0.0, 0.0])
        self.er_pix_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(8,1)

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

    def quadrotor_vs_control(self, Lm, er_pix):
        
        forward_gain_Kc = -2.2
        thrust_gain_Kc = 0.0
        sway_gain_Kc = 1.0
        yaw_gain_Kc = -2.5
        Kc = np.identity(6)
        Kc[0][0] = thrust_gain_Kc
        Kc[1][1] = sway_gain_Kc
        Kc[2][2] = forward_gain_Kc
        Kc[3][3] = yaw_gain_Kc
        Kc[4][4] = 0.0
        Kc[5][5] = 0.0

        Ucmd = -np.dot(Kc,np.dot(np.linalg.pinv(Lm), er_pix))+1.0*np.dot(np.linalg.pinv(Lm), np.array([0.0, self.a, 0.0, self.a, 0.0, self.a, 0.0, self.a]).reshape(8,1))
        
        return Ucmd
    
    def transform_body_to_camera(self, theta_cam, transCam):
        
        R_y = np.array([[cos(self.theta_cam), 0.0, sin(self.theta_cam)],
                    [0.0, 1.0, 0.0],
                    [-sin(self.theta_cam), 0.0, cos(self.theta_cam)]]).reshape(3,3)
        sst = np.array([[0.0, -self.transCam[2], self.transCam[1]],
                             [self.transCam[2], 0.0, -self.transCam[0]],
                             [-self.transCam[1], self.transCam[0], 0.0]]).reshape(3,3)
        
        return R_y, sst
    
    def control_execution(self, UVScmd, er_pix, t_vsc):
        
        twist = PositionTarget()
        #twist.header.stamp = 1
        twist.header.frame_id = 'world'
        twist.coordinate_frame = 8
        twist.type_mask = 1479
        twist.velocity.x = self.uav_vel_body[0]
        twist.velocity.y = self.uav_vel_body[1]
        twist.velocity.z = self.uav_vel_body[2]
        twist.yaw_rate = self.uav_vel_body[3]        

        vsc_msg = VSCdata()
        vsc_msg.errors = er_pix
        vsc_msg.cmds = self.uav_vel_body
        
        vsc_msg.time = t_vsc
        self.ros_comms.pub_vsc_data.publish(vsc_msg)
        # self.ros_comms.pub_vel.publish(twist)                     
    
    # Detect the line and piloting
    def detection_processing(self, box, angle):
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
        # print("Error pixel: ", er_pix)
        UVScmd = self.quadrotor_vs_control(Lm, er_pix)
        UVScmd = np.dot(T, UVScmd)
        print("UVScmd is: ",  UVScmd)
             
        self.uav_vel_body[0] = UVScmd[0]
        self.uav_vel_body[1] = UVScmd[1]
        self.uav_vel_body[2] = UVScmd[2]
        self.uav_vel_body[3] = UVScmd[5]
        
        self.control_execution(UVScmd, er_pix, t_vsc)
        
        self.ros_comms.publish_error(er_pix)
        self.ros_comms.publish_angle(angle)
            
        self.t = self.t+self.dt
    
    
    def get_feature_box(self, box, angle):
        if angle >= 0:
            return [box[0][0], box[0][1], self.z, box[1][0], box[1][1], self.z, box[2][0], box[2][1], self.z, box[3][0], box[3][1], self.z]
        else:
            return [box[3][0], box[3][1], self.z, box[0][0], box[0][1], self.z, box[1][0], box[1][1], self.z, box[2][0], box[2][1], self.z]

    def get_transformation_matrix(self, R_y, sst):
        T = np.zeros((6, 6), dtype=float)
        T[0:3, 0:3] = R_y
        T[3:6, 3:6] = R_y
        T[0:3, 3:6] = np.dot(sst, R_y)
        return T
