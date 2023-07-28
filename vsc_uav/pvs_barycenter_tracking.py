#!/usr/bin/env python
from __future__ import print_function
from re import I

from numpy.core.fromnumeric import size
from numpy.core.numeric import ones
import roslib
roslib.load_manifest('coastline_tracking')
import sys
import rospy
import cv2
import numpy as np
import numpy.matlib
import time
import json
import os
import matplotlib.pyplot as plt
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import Twist, Point, Vector3
from std_msgs.msg import Empty, Int16, Float32, Bool, UInt16MultiArray, UInt32MultiArray, UInt64MultiArray
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Vector3Stamped, TwistStamped
from numpy.linalg import norm
# from math import cos, sin, tan, sqrt, exp, pi, atan2, acos, asin
from math import *
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from coastline_tracking.msg import VSCdata, PVSdata, EKFdata, IBVSdata
from operator import itemgetter



class image_converter:
  
    def __init__(self):
        #Create publishers
        self.pub_vel = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.pub_error = rospy.Publisher('error', Int16, queue_size=10)
        self.pub_angle = rospy.Publisher('angle', Int16, queue_size=10)
        self.pub_vsc_data = rospy.Publisher("/vsc_data", VSCdata, queue_size=1000)
        self.pub_ekf_data = rospy.Publisher("/ekf_data", EKFdata, queue_size=1000)
        self.pub_ibvs_data = rospy.Publisher("/ibvs_data", IBVSdata, queue_size=1000)
        self.pub_pvs_data = rospy.Publisher("/pvs_data", PVSdata, queue_size=1000)
        self.pub_im = rospy.Publisher('im', Image, queue_size=10)
        self.ros_image_pub = rospy.Publisher("image_bounding_box", Image, queue_size=10)
        self.bridge = CvBridge()
        
        #Create subscribers
        self.image_sub = rospy.Subscriber("/image_raww", Image, self.callback)
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.updateImu)
        self.pos_sub = rospy.Subscriber("/mavros/global_position/local", Odometry, self.OdomCb)
        self.vel_uav = rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.VelCallback)
        
        # uav state variables
        self.landed = 0
        self.takeoffed = 1
        self.uav_vel_body = np.array([0.0, 0.0, 0.0, 0.0])
        self.vel_uav = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.er_pix_prev = np.array([0.0, 0.0]).reshape(2,1)
        self.mp_cartesian = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        self.mp_pixel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ZED stereo camera translation and rotation variables
        self.transCam = [0.0, 0.0, -0.14]
        self.rotCam = [0.0, 1.57, 0.0]
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
        self.psi = 0.0
        self.t = 0.0
        self.dt = 0.0335
        self.a = 1.0
        self.a = 1.4
        self.time = rospy.Time.now().to_sec()
        self.alpha_des = 0.0
        self.sigma_des = 14850.0
        self.alpha = 10.0
        self.sigma = 10.0

        # Definition of moving average filter parameters
        self.av_window = []
        self.cntr = 0
        self.window_size = 4
        self.values = []
        self.sum = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(8,1)    

        # Definition of Extended Kalman Filter parameters
        # self.P = 10.0*np.eye(3, dtype=float)
        # self.x_est = np.array([0.1, 0.1, 0.1],  dtype=float).reshape(3,1)
        self.P = 100.0*np.eye(4, dtype=float)
        self.x_est = np.array([0.4, 0.4, 0.4, 0.4],  dtype=float).reshape(4,1)

    
    def ekf_estimation(self, e_m, e_m_dot):
        
        #e_m: centroid error (e_m = x - xd = (u-ud)/ax)
        #e_dot_m : gradient (velocity) after removing the camera velocity
        #Ts: sampling time e.g. Ts = 0.1
        #P initial filter covariance matrix
        #x_est initial estimate to be update by Kalman
        
        # print("e_m in the EKF:", e_m)
        # print("e_m_dot in the EKF:", e_m_dot)

        PHI_S1 = 0.001 #Try to change these e.g. 0.1
        PHI_S2 = 0.001 #Try to change these e.g. 0.1

        F_est = np.zeros((4, 4), dtype=float)
        F_est [0,1] = 1.0
        F_est [1,0] = -self.x_est[2]**2
        F_est [1,2] = -2.0*self.x_est[2]*self.x_est[0] + 2.0*self.x_est[2]*self.x_est[3]
        F_est [1,3] = self.x_est[2]**2
        # print("F_est: ", F_est)
        
        PHI_k = np.eye(4,4) + F_est*self.dt
        # print("PHI_k: ", PHI_k)
        
        Q_k = np.array([[0.001, 0.0, 0.0, 0.0],
                        [0.0, 0.5*PHI_S1*(-2.0*self.x_est[2]*self.x_est[0]+2.0*self.x_est[2]*self.x_est[3])**2*(self.dt**2)+0.333*self.x_est[2]**4*(self.dt**3)*PHI_S2, 0.0, 0.5*self.x_est[2]**2*(self.dt**2)*PHI_S2],
                        [0.0, 0.5*PHI_S1*(-2.0*self.x_est[2]*self.x_est[0]+2.0*self.x_est[2]*self.x_est[3])**2*(self.dt**2), PHI_S1*self.dt, 0.0],
                        [0.0, 0.5*PHI_S2*(self.dt**2)*self.x_est[2]**2, 0.0, PHI_S2*self.dt]], dtype=float).reshape(4,4)
        # print("Q_k: ", Q_k)

        H_k = np.array([[1.0, 0.0, 0.0, 0.0], 
                        [0.0, 1.0, 0.0, 0.0]], dtype=float).reshape(2,4)
        # print("H_k: ", H_k)
        
        R_k = 0.1*np.eye(2, dtype=float)
        # print("R_k: ", R_k)

        M_k = np.dot(np.dot(PHI_k,self.P), PHI_k.transpose()) + Q_k
        # print("M_k: ", M_k)
        K_k = np.dot(np.dot(M_k, H_k.transpose()), np.linalg.inv(np.dot(np.dot(H_k, M_k), H_k.transpose()) + R_k) )
        # print("K_k: ", K_k)
        self.P = np.dot((np.eye(4) - np.dot(K_k, H_k)), M_k)
        # print("P: ", P)

        xdd = -((self.x_est[2]**2)*self.x_est[0]) + (self.x_est[2]**2)*self.x_est[3]
        # print("xdd: ", xdd)
        xd = self.x_est[1] + self.dt*xdd
        # print("xd: ", xd)
        x = self.x_est[0] + self.dt*xd
        # print("x: ", x)

        x_dash_k = np.array([x, xd, self.x_est[2], self.x_est[3]]).reshape(4,1)
        # print("x_dash_k: ", x_dash_k)
        
        z = np.array([e_m, e_m_dot]).reshape(2,1)
        
        self.x_est = np.dot(PHI_k, x_dash_k) + np.dot(K_k, z - np.dot(H_k, np.dot(PHI_k, x_dash_k)))
        # print("self.x_est: ", self.x_est)

        # return self.x_est # [estimated wave position, estimated wave velocity, estimated frequency] --> you only need x_est[1]: velocity

    # Function calling a Moving Average Filter
    def moving_average_filter(self, value):
        self.values.append(value)

        self.sum = [a + b for a, b in zip(value, self.sum)]
        
        if len(self.values) > self.window_size:
            self.sum = [a - b for a, b in zip(self.sum, self.values.pop(0))]
            
        ew_filtered = [float(a)/len(self.values) for a in self.sum]
        ew_filtered = np.array(ew_filtered).reshape(2,1)

        return ew_filtered

    
    # Callback function updating the IMU measurements (rostopic /mavros/imu/data)
    def updateImu(self, msg):
        self.phi_imu = msg.orientation.x
        self.theta_imu = msg.orientation.y
        self.psi_imu = msg.orientation.z
        self.w_imu = msg.orientation.w
        self.phi_imu, self.theta_imu, self.psi_imu = euler_from_quaternion ([self.phi_imu, self.theta_imu, self.psi_imu, self.w_imu])
    
    
    # Callback function updating the Odometry measurements (rostopic /mavros/global_position/local)
    def OdomCb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        # print("Message position: ", msg.pose.pose.position)


    # Callback function updating the Velocity measurements (rostopic /mavros/local_position/velocity_body)
    def VelCallback(self, msg):
    
        self.vel_uav[0] = msg.twist.linear.x
        self.vel_uav[1] = msg.twist.linear.y
        self.vel_uav[2] = msg.twist.linear.z
        self.vel_uav[3] = msg.twist.angular.x
        self.vel_uav[4] = msg.twist.angular.y
        self.vel_uav[5] = msg.twist.angular.z
        # print("Message uav velocity: ", self.vel_uav)
    
    
    # Function calling the feature transformation from the image plane on a virtual image plane
    def featuresTransformation(self, mp, phi, theta):
        
        Rphi = np.array([[1.0, 0.0, 0.0],[0.0, cos(phi), -sin(phi)],[0.0, sin(phi), cos(phi)]]).reshape(3,3)
        Rtheta = np.array([[cos(theta), 0.0, sin(theta)],[0.0, 1.0, 0.0],[-sin(theta), 0.0, cos(theta)]]).reshape(3,3)
        Rft = np.dot(Rphi, Rtheta)
        mpv0 = np.dot(Rft, mp[0:3])
        mpv1 = np.dot(Rft, mp[3:6])
        mpv2 = np.dot(Rft, mp[6:9])
        mpv3 = np.dot(Rft, mp[9:12])
        mpv = np.hstack((mpv0, mpv1, mpv2, mpv3))
        
        return mpv
    

    def featuresTransformation_barycenter(self, mp, barycenter_cartesian, phi, theta):
        
        Rphi = np.array([[1.0, 0.0, 0.0],[0.0, cos(phi), -sin(phi)],[0.0, sin(phi), cos(phi)]]).reshape(3,3)
        Rtheta = np.array([[cos(theta), 0.0, sin(theta)],[0.0, 1.0, 0.0],[-sin(theta), 0.0, cos(theta)]]).reshape(3,3)
        Rft = np.dot(Rphi, Rtheta)
        mpv0 = np.dot(Rft, mp[0:3])
        
        barycenter_cartesian_virtual = np.dot(Rft, barycenter_cartesian)
        
        return barycenter_cartesian_virtual


    def calculateIM_barycenter(self, mpv, mp_des, cu, cv, ax, ay, barycenter_pixel_virtual):
        
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
                    
        Lm0 = np.array([[-1.0/Z_0, 0.0, x_0/Z_0, x_0*y_0, -(1.0+x_0*x_0), y_0],
                        [0.0, -1.0/Z_0, y_0/Z_0, 1.0+y_0*y_0, -x_0*y_0, -x_0]]).reshape(2,6)
        Lm1 = np.array([[-1.0/Z_1, 0.0, x_1/Z_1, x_1*y_1, -(1.0+x_1*x_1), y_1],
                        [0.0, -1.0/Z_1, y_1/Z_1, 1.0+y_1*y_1, -x_1*y_1, -x_1]]).reshape(2,6)
        Lm2 = np.array([[-1.0/Z_2, 0.0, x_2/Z_2, x_2*y_2, -(1.0+x_2*x_2), y_2],
                        [0.0, -1.0/Z_2, y_2/Z_2, 1.0+y_2*y_2, -x_2*y_2, -x_2]]).reshape(2,6)
        Lm3 = np.array([[-1.0/Z_3, 0.0, x_3/Z_3, x_3*y_3, -(1.0+x_3*x_3), y_3],
                        [0.0, -1.0/Z_3, y_3/Z_3, 1.0+y_3*y_3, -x_3*y_3, -x_3]]).reshape(2,6)
        Lm = np.concatenate((Lm0, Lm1, Lm2, Lm3), axis=0)
        # print("Lm: ", Lm)
                    
        
        Lm0_cal = np.array([[-1.0/Z_0, 0.0],
                        [0.0, -1.0/Z_0]]).reshape(2,2)
        Lm1_cal = np.array([[-1.0/Z_1, 0.0],
                        [0.0, -1.0/Z_1]]).reshape(2,2)
        Lm2_cal = np.array([[-1.0/Z_2, 0.0],
                        [0.0, -1.0/Z_2]]).reshape(2,2)
        Lm3_cal = np.array([[-1.0/Z_3, 0.0],
                        [0.0, -1.0/Z_3]]).reshape(2,2)
        Lm_cal = np.concatenate((Lm0_cal, Lm1_cal, Lm2_cal, Lm3_cal), axis=0)
        print("Lm_cal: ", Lm_cal)  
        
        x_barycenter = (barycenter_pixel_virtual[0]-cu)/ax
        y_barycenter = (barycenter_pixel_virtual[1]-cv)/ay
        Z_barycenter = barycenter_pixel_virtual[2]     
                
        xd_barycenter = (cu-cu)/ax
        yd_barycenter = (cv-cv)/ay
        Zd_barycenter = barycenter_pixel_virtual[2]
        
        er_pix = np.array([x_barycenter-xd_barycenter, y_barycenter-yd_barycenter]).reshape(2,1)
        print("er_pix: ", er_pix)
        
        return Lm_cal, er_pix
    
    
    def cartesian_from_pixel_barycenter(self, mp_pixel, cX, cY, cu, cv, ax, ay):
        
        Z_barycenter = mp_pixel[2]
        X_barycenter = Z_barycenter*((cX-cu)/ax)
        Y_barycenter = Z_barycenter*((cY-cv)/ay)
        
        barycenter_cartesian = np.array([X_barycenter, Y_barycenter, Z_barycenter])            
        
        return barycenter_cartesian
    
    
    def pixels_from_cartesian_barycenter(self, barycenter_cartesian_virtual, cu, cv, ax, ay):
                
        cX_pixel_virtual = (barycenter_cartesian_virtual[0]/barycenter_cartesian_virtual[2])*ax + cu
        cY_pixel_virtual = (barycenter_cartesian_virtual[1]/barycenter_cartesian_virtual[2])*ay + cv
        
        barycenter_pixel_virtual = np.array([cX_pixel_virtual, cY_pixel_virtual, barycenter_cartesian_virtual[2]])
        
        return barycenter_pixel_virtual
       
    
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


    # Function calculating the derivative of the error of the pixels on the image plane
    def deriv_error_estimation(self, er_pix, er_pix_prev):
        
        ew = (er_pix - er_pix_prev)/self.dt 

        return ew


    # Function calculating the control law both for target tracking and trajectory planning
    def quadrotorPVS_tracking_control(self, L_bar, er_pix, ew):
                
        # ----- Home Ubuntu Desktop Tuning ----
        forward_gain_Kc = 0.4
        # forward_gain_Kc = 0.0
        sway_gain_Kc = 5.0
        Kc = np.identity(2)
        Kc[0][0] = forward_gain_Kc
        Kc[1][1] = sway_gain_Kc        
        # --------------------------------------
        lvz = 0.0 # thrust gain
        l_om_z = 0.1  # angular z-axes velocity
        # --------------------------------------      
        v_z = lvz*log(self.sigma_des/self.sigma)
        # print("v_z = ", v_z)
        omega_z = l_om_z*(self.alpha_des - self.alpha)
        # print("omega_z = ", omega_z)
        
        PVS_cmd_1 = -np.dot(Kc,np.dot(np.linalg.pinv(L_bar), er_pix))
        # print("PVS_cmd_1: ", PVS_cmd_1)
        # PVS_cmd_2 = -np.dot(L_bar,np.dot(np.dot(self.Ical,np.linalg.inv()),ew))
        # print("PVS_cmd_2: ", PVS_cmd_2)

        first_calculation = np.dot(L_bar, np.transpose(L_bar))
        # print("first_calculation: ", first_calculation)
        second_calculation = -np.dot(L_bar,first_calculation)
        # print("second_calculation: ", second_calculation)
        third_calculation = np.dot(second_calculation,self.Ical)
        # print("third_calculation: ", third_calculation)
        fourth_calculation = np.dot(third_calculation,ew)
        # print("forth_calculation: ", fourth_calculation)
        PVS_cmd_2 = fourth_calculation
        
        PVScmd = PVS_cmd_1 + PVS_cmd_2
        # PVScmd = PVS_cmd_1
        # print("PVScmd: ", PVScmd)
        
        PVScmd_final = np.array([v_z, PVScmd[1][0], PVScmd[0][0], omega_z, 0.0, 0.0]).reshape(6,1)
        
        return PVScmd_final
    
    
    # Function calculating the control law both for target tracking and trajectory planning
    def quadrotorPVS_no_tracking_control(self, L_bar, er_pix):
                
        # ----- Home Ubuntu Desktop Tuning ----
        forward_gain_Kc = 10.0
        sway_gain_Kc = 1.2
        Kc = np.identity(2)
        Kc[0][0] = forward_gain_Kc
        Kc[1][1] = sway_gain_Kc        
        # --------------------------------------        
        lvz = 0.0 # thrust gain
        l_om_z = 0.1  # angular z-axes velocity
        # --------------------------------------
        v_z = lvz*log(self.sigma_des/self.sigma)
        # print("v_z = ", v_z)
        omega_z = l_om_z*(self.alpha_des - self.alpha)
        # print("omega_z = ", omega_z)
        
        PVScmd = -np.dot(Kc,np.dot(np.transpose(L_bar),er_pix))
        PVScmd_final = np.array([v_z, PVScmd[1][0], PVScmd[0][0], omega_z, 0.0, 0.0]).reshape(6,1)
        
        return PVScmd_final
    

    # Detect the line and piloting
    def line_detect(self, cv_image):
        t_vsc = rospy.Time.now().to_sec() - self.time
        # Create a mask
        # cv_image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(cv_image, (130, 130, 130), (255, 255, 255))
        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=3)
        mask = cv2.dilate(mask, kernel, iterations=3)
        contours_blk, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # _, contours_blk, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_blk.sort(key=cv2.minAreaRect)

        
        if len(contours_blk) > 0 and cv2.contourArea(contours_blk[0]) > 200:
            # Box creation for the detected coastline
            blackbox = cv2.minAreaRect(contours_blk[0])
            (x_min, y_min), (w_min, h_min), angle = blackbox            
            box = cv2.boxPoints(blackbox)
            box = np.int0(box)
            
            M = cv2.moments(contours_blk[0])
            self.cX = int(M["m10"] / M["m00"])
            # print("cX = ", self.cX)
            self.cY = int(M["m01"] / M["m00"])
            # print("cY = ", self.cY)
            
            # Sorting of the orientation of the detected coastline
            if angle < -45:
                angle = 90 + angle
            if w_min < h_min and angle > 0:
                angle = (90 - angle) * -1
            if w_min > h_min and angle < 0:
                angle = 90 + angle

            self.alpha = angle
            # print("angle of the contour: ", self.alpha)
            self.sigma = cv2.contourArea(contours_blk[0])
            # print("area of the contour: ", self.sigma)
            
            # Recreation of the feature box
            if angle >= 0:
                mp = [box[0][0], box[0][1], self.z, box[1][0], box[1][1], self.z, box[2][0], box[2][1], self.z, box[3][0], box[3][1], self.z]
                # print("mp: ", mp)
            else:
                mp = [box[3][0], box[3][1], self.z, box[0][0], box[0][1], self.z, box[1][0], box[1][1], self.z, box[2][0], box[2][1], self.z]
                # print("mp: ", mp)    

            mp_des = np.array([420, 472+self.a*self.t, self.z, 367, 483+self.a*self.t, self.z, 327, 2+self.a*self.t, self.z, 377, 0+self.a*self.t, self.z])  
            
            cv2.drawContours(cv_image, [box], 0, (0, 0, 255), 1)
            cv2.line(cv_image, (int(x_min), 54), (int(x_min), 74), (255, 0, 0), 1)

            R_y = np.array([[cos(self.theta_cam), 0.0, sin(self.theta_cam)],
                    [0.0, 1.0, 0.0],
                    [-sin(self.theta_cam), 0.0, cos(self.theta_cam)]]).reshape(3,3)
            sst = np.array([[0.0, -self.transCam[2], self.transCam[1]],
                             [self.transCam[2], 0.0, -self.transCam[0]],
                             [-self.transCam[1], self.transCam[0], 0.0]]).reshape(3,3)
            T = np.zeros((6,6), dtype = float)
            T[0:3, 0:3] = R_y
            T[3:6, 3:6] = R_y
            T[0:3, 3:6] = np.dot(sst, R_y)
            # print("From body to camera transformation: ", T) 
            
            lateral_features = np.array([[mp[0], mp[3], mp[6], mp[9]],
                                         [mp[1], mp[4], mp[7], mp[10]]]).reshape(2,4)
            # print("size of lateral_features: ", np.size(lateral_features,1))
            self.Ical = (1.0/(np.size(lateral_features,1)))*np.matlib.repmat(np.eye(2),1,np.size(lateral_features,1))
            # print("Ical = " , Ical)
            
            barycenter = [self.cX, self.cY, self.z]
            # Interaction matrix, error of pixels and velocity commands calculation (a.k.a control execution)
            Lm_cal, er_pix = self.calculateIM_barycenter(mp, mp_des, self.cu, self.cv, self.ax, self.ay, barycenter) #TRANSFORM FEATURES
            L_bar = np.dot(self.Ical,Lm_cal)
            # print("L_bar: ", L_bar)
            
            velocity_camera = np.dot(T, self.vel_uav)
            # print("velocity_camera: ", velocity_camera)
            
            u_bc = (box[0][0]+box[1][0]+box[2][0]+box[3][0])/4
            # print("u of centroid: ", u_bc)
            v_bc = (box[0][1]+box[1][1]+box[2][1]+box[3][1])/4
            # print("v of centroid: ", v_bc)
                        
            ew = self.deriv_error_estimation(er_pix, self.er_pix_prev)
            # print("ew: ", ew)
            ew_filtered = self.moving_average_filter(np.array(ew).reshape(2,1))
            # print("ew_filtered: ", ew_filtered)          
            ew_filtered_odometry = ew_filtered - np.array(np.dot(L_bar, np.array([velocity_camera[0], velocity_camera[1]]))).reshape(2,1)

            # e_m = (er_pix[0]+er_pix[2]+er_pix[4]+er_pix[6])/4
            e_m = er_pix[0]
            # print("e_m: ", e_m)
            # e_m_dot = (ew_filtered_odometry[0]+ew_filtered_odometry[2]+ew_filtered_odometry[4]+ew_filtered_odometry[6])/4
            e_m_dot = ew_filtered_odometry[0]
            # print("e_m_dot: ", e_m_dot)
            self.ekf_estimation(e_m, e_m_dot)
            # print("Extended kalman filter estimation: ", self.x_est)
            # print("Extended kalman filter velocity estimation: ", wave_est[1])
            # wave_estimation_final = np.array([wave_est[1], [0.0], wave_est[1], [0.0], wave_est[1], [0.0], wave_est[1], [0.0]]).reshape(8,1)
            wave_est_control_input = self.x_est[1]
            wave_estimation_final = np.array([wave_est_control_input, [self.a], wave_est_control_input, [self.a], wave_est_control_input, [self.a], wave_est_control_input, [self.a]]).reshape(8,1)
            # print("final estimation: ", wave_estimation_final)
            
            
            PVScmd = self.quadrotorPVS_tracking_control(L_bar, er_pix, wave_estimation_final)
            # PVScmd = self.quadrotorPVS_no_tracking_control(L_bar, er_pix)
            # print("PVScmd = ", PVScmd)
            PVScmd = np.dot(np.linalg.inv(T), PVScmd)
            # print("transformed PVScmd = ", PVScmd)
            self.er_pix_prev = er_pix            
            
            self.uav_vel_body[0] = PVScmd[0]
            # self.uav_vel_body[0] = 0.0
            self.uav_vel_body[1] = PVScmd[1]
            # self.uav_vel_body[2] = PVScmd[2]
            self.uav_vel_body[2] = 0.0
            self.uav_vel_body[3] = PVScmd[5]
            
            twist = PositionTarget()
            #twist.header.stamp = 1
            twist.header.frame_id = 'world'
            twist.coordinate_frame = 8
            twist.type_mask = 1479
            # twist.velocity.x = -self.uav_vel_body[0]+self.a
            twist.velocity.x = self.a
            # twist.velocity.x = self.uav_vel_body[0]
            twist.velocity.y = self.uav_vel_body[1]
            twist.velocity.z = self.uav_vel_body[2]
            twist.yaw_rate = self.uav_vel_body[3]
            
            ekf_msg = EKFdata()
            ekf_msg.ekf_output = self.x_est
            ekf_msg.e_m = e_m
            ekf_msg.e_m_dot = e_m_dot
            ekf_msg.u_bc = u_bc
            ekf_msg.v_bc = v_bc
            ekf_msg.time = t_vsc
            self.pub_ekf_data.publish(ekf_msg)
            
            pvs_msg = PVSdata()
            pvs_msg.errors = er_pix
            pvs_msg.cmds = self.uav_vel_body
            print("pvs_msg.cmds: ", pvs_msg.cmds)
            pvs_msg.alpha = self.alpha
            pvs_msg.alpha_des = self.alpha_des
            pvs_msg.sigma = self.sigma
            pvs_msg.sigma_des = self.sigma_des
            pvs_msg.time = t_vsc
            self.pub_pvs_data.publish(pvs_msg)            
            
            # self.pub_vel.publish(twist)         
            
        ros_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.pub_im.publish(ros_msg)
        
        self.t = self.t+self.dt
        
        
  

  # Image processing @ 10 FPS
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            
        if self.takeoffed and (not self.landed):
            self.line_detect(cv_image)
            cv_image = cv2.resize(cv_image, (720, 480))
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            ros_image.header.stamp = data.header.stamp
            self.ros_image_pub.publish(ros_image)
            cv2.imshow("Image window", cv_image)
            cv2.waitKey(1) & 0xFF


def main(args):
    rospy.init_node('image_converter', anonymous=True)
    # t0 = rospy.Time.now().to_sec()
    ic = image_converter()
    # time.sleep(0.01)
    rospy.sleep(0.03)
    #ic.cam_down()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
if __name__ == '__main__':
    main(sys.argv)