#!/usr/bin/env python
from __future__ import print_function

from numpy.core.fromnumeric import size
from numpy.core.numeric import ones
import roslib
roslib.load_manifest('coastline_tracking')
import sys
import rospy
import cv2
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import Twist, Point, Vector3
from std_msgs.msg import Empty, Int16, Float32, Bool, UInt16MultiArray, UInt32MultiArray, UInt64MultiArray
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Vector3Stamped, TwistStamped
from numpy.linalg import norm
# from math import cos, sin, tan, sqrt, exp, pi, atan2, acos, asin
import math
from math import *
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from coastline_tracking.msg import IBVSdata
from operator import itemgetter



class image_converter:
  
    def __init__(self):
        #Create publishers
        self.pub_vel = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.pub_ibvs_data = rospy.Publisher("/ibvs_data", IBVSdata, queue_size=1000)
        # self.bridge = CvBridge()
        
        #Create subscribers
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.updateImu)
        self.pos_sub = rospy.Subscriber("/mavros/global_position/local", Odometry, self.OdomCb)
        self.vel_uav = rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.VelCallback)
        self.feat_sub = rospy.Subscriber("/floats", Floats, self.callback)
        
        # uav state variables
        self.landed = 0
        self.takeoffed = 1
        self.uav_vel_body = np.array([0.0, 0.0, 0.0, 0.0])
        self.vel_uav = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.er_pix_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(8,1)
        self.mp_cartesian = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        self.mp_pixel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # ZED stereo camera translation and rotation variables
        self.transCam = [0.0, 0.0, -0.34]
        self.rotCam = [math.pi, 0.0, -(math.pi)/2]
        self.phi_cam = self.rotCam[0]
        self.theta_cam = self.rotCam[1]
        self.psi_cam = self.rotCam[2]
        
        # ZED 2 stereocamera intrinsic parameters
        self.cu = 64.0
        self.cv = 64.0
        # self.ax = 0.0065
        # self.ay = 0.0065
        self.ax = 264.38
        self.ay = 264.38
        
        # Variables initialization
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0
        self.phi_imu = 0.0
        self.theta_imu = 0.0 
        self.psi = 0.0
        self.t = 0.0
        self.dt = 0.0335
        self.a = 0.2
        # self.a = 0.18
        self.time = rospy.Time.now().to_sec()
        


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
    
    def calculateIM(self, mpv, mp_des, cu, cv, ax, ay):
        
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
        # print("Lm = ", Lm)

        # Lm0 = np.array([[-cu/Z_0, 0.0, -mpv[0]/Z_0, (mpv[0]*mpv[1])/cu, (cu**2+mpv[0]**2)/cu, -mpv[1]],
        #                 [0.0, cu/Z_0, -mpv[1]/Z_0, -(cu**2+mpv[1]**2)/cu, (mpv[0]*mpv[1])/cu, mpv[0]]]).reshape(2,6)
        # Lm1 = np.array([[-cu/Z_1, 0, -mpv[2]/Z_1, mpv[2]*mpv[3]/cu, (cu**2+mpv[2]**2)/cu, -mpv[3]],
        #                 [0, cu/Z_1, -mpv[3]/Z_1, -(cu**2+mpv[3]**2)/cu, mpv[2]*mpv[3]/cu, mpv[2]]]).reshape(2,6)
        # Lm2 = np.array([[-cu/Z_2, 0, -mpv[4]/Z_2, mpv[4]*mpv[5]/cu, (cu**2+mpv[4]**2)/cu, -mpv[5]],
        #                 [0, cu/Z_2, -mpv[5]/Z_1, -(cu**2+mpv[5]**2)/cu, mpv[4]*mpv[5]/cu, mpv[4]]]).reshape(2,6)
        # Lm3 = np.array([[-cu/Z_3, 0, -mpv[6]/Z_1, mpv[6]*mpv[7]/cu, (cu**2+mpv[6]**2)/cu, -mpv[7]],
        #                 [0, cu/Z_1, -mpv[7]/Z_1, -(cu**2+mpv[7]**2)/cu, mpv[6]*mpv[7]/cu, mpv[6]]]).reshape(2,6)
        
        # Lm_alternate = np.concatenate((Lm0, Lm1, Lm2, Lm3), axis=0)
        # print("Lm_alternate = ", Lm_alternate)
        
        er_pix = np.array([x_0-xd_0, y_0-yd_0, x_1-xd_1, y_1-yd_1, x_2-xd_2, y_2-yd_2, x_3-xd_3, y_3-yd_3 ]).reshape(8,1) #ax=ay=252.07
        # print("er_pix = ", er_pix)
        
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


    # Function calculating the control law both for target tracking and trajectory planning
    def quadrotorVSControl_tracking(self, Lm, er_pix):

        # ----- NVIDIA Jetson AGX Xavier Developer Kit Tuning - Tracking testing ----
        forward_gain_Kc = 0.0001
        thrust_gain_Kc = 0.0
        sway_gain_Kc = 0.004
        yaw_gain_Kc = 0.013
        Kc = np.identity(6)
        Kc[0][0] = sway_gain_Kc
        Kc[1][1] = forward_gain_Kc
        Kc[2][2] = thrust_gain_Kc
        Kc[3][3] = 0.0
        Kc[4][4] = 0.0
        Kc[5][5] = yaw_gain_Kc
        # --------------------------------------

        first_term = -np.dot(Kc,np.dot(np.linalg.pinv(Lm), er_pix))
        second_term = np.dot(np.linalg.pinv(Lm), np.array([0.0, self.a, 0.0, self.a, 0.0, self.a, 0.0, self.a]).reshape(8,1)) 
        
        Ucmd = first_term + second_term
        
        # Ucmd = -np.dot(Kc,np.dot(np.linalg.pinv(Lm), er_pix))+np.dot(np.linalg.pinv(Lm), np.array([0.0, self.a, 0.0, self.a, 0.0, self.a, 0.0, self.a]).reshape(8,1)) 
        # print("1st control term: ", Kc*np.dot(np.linalg.pinv(Lm), er_pix))
        # print("1st control term: ", -np.dot(Kc,np.dot(np.linalg.pinv(Lm), er_pix)))
        # print("2nd control term: ", np.dot(np.linalg.pinv(Lm), np.array([0.0, self.a, 0.0, self.a, 0.0, self.a, 0.0, self.a]).reshape(8,1)))
        # print("3rd control term: ", Ke*np.dot(np.linalg.pinv(Lm), ew))
        # print ("Final control law calculation: ", Ucmd)
        
        return Ucmd
    

    # Detect the line and piloting
    def line_detect(self, u_1, v_1, u_2, v_2, u_3, v_3, u_4, v_4):
        t_vsc = rospy.Time.now().to_sec() - self.time
        # Create a mask
        # cv_image_hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        mp = [u_1, v_1, self.z, u_2, v_2, self.z, u_3, v_3, self.z, u_4, v_4, self.z]
        # print("mp: ", mp)
        mp_cartesian = self.cartesian_from_pixel(mp, self.cu, self.cv, self.ax, self.ay)
        # print("mp_cartesian: ", mp_cartesian)
        mp_cartesian_v = self.featuresTransformation(mp_cartesian, self.phi_imu, self.theta_imu)
        # print("mp_cartesian_v: ", mp_cartesian_v)
        mp_pixel_v = self.pixels_from_cartesian(mp_cartesian_v, self.cu, self.cv, self.ax, self.ay)
        # print("mp_pixel_v: ", mp_pixel_v)
        mp_des = np.array([69, 128+self.a*self.t, self.z, 59, 128+self.a*self.t, self.z, 59, 0+self.a*self.t, self.z, 69, 128+self.a*self.t, self.z]) 
        # print("mp_des: ", mp_des)        

        R_x = np.array([[1.0, 0.0, 0.0], 
                        [0.0, cos(self.rotCam[0]), -sin(self.rotCam[0])],
                        [0.0, sin(self.rotCam[0]), cos(self.rotCam[0])]]).reshape(3,3)

        R_y = np.array([[cos(self.rotCam[1]), 0.0, sin(self.rotCam[1])],
                        [0.0, 1.0, 0.0],
                        [-sin(self.rotCam[1]), 0.0, cos(self.rotCam[1])]]).reshape(3,3)

        R_z = np.array([[cos(self.rotCam[2]), -sin(self.rotCam[2]), 0.0], 
                        [sin(self.rotCam[2]), cos(self.rotCam[2]), 0.0],
                        [0.0, 0.0, 1.0]]).reshape(3,3)

        sst = np.array([[0.0, -self.transCam[2], self.transCam[1]],
                        [self.transCam[2], 0.0, -self.transCam[0]],
                        [-self.transCam[1], self.transCam[0], 0.0]]).reshape(3,3)

        R = np.dot(R_z,R_x)
        T = np.zeros((6,6), dtype = float)
        T[0:3, 0:3] = R
        T[3:6, 3:6] = R
        T[0:3, 3:6] = np.dot(sst, R)
        # print("From body to camera transformation: ", T)                   
        # Interaction matrix, error of pixels and velocity commands calculation (a.k.a control execution)
        Lm, er_pix = self.calculateIM(mp_pixel_v, mp_des, self.cu, self.cv, self.ax, self.ay) #TRANSFORM FEATURES
        print("Error pixel: ", er_pix)
        
        UVScmd = self.quadrotorVSControl_tracking(Lm, er_pix)
        # print("UVScmd: ", UVScmd) 
        # UVScmd = np.dot(T, UVScmd)
        UVScmd = np.dot(np.linalg.inv(T), UVScmd)
        self.er_pix_prev = er_pix
        # print("er_pix_prev: ", self.er_pix_prev)    
        # print("UVScmd_after_transform: ", UVScmd) 

        if UVScmd[0] >= 0.5:
            UVScmd[0] = 0.3
        
        if UVScmd[0] <= -0.5:
            UVScmd[0] = -0.3

        if UVScmd[1] >= 0.5:
            UVScmd[1] = 0.4
            
        if UVScmd[1] <= -0.5:
            UVScmd[1] = -0.4

        if UVScmd[3] >= 0.3:
            UVScmd[3] = 0.2 

        if UVScmd[3] <= -0.3:
            UVScmd[3] = -0.2    
             
        self.uav_vel_body[0] = UVScmd[0]
        # self.uav_vel_body[0] = 0.0
        self.uav_vel_body[1] = UVScmd[1]
        #self.uav_vel_body[2] = UVScmd[2]
        self.uav_vel_body[2] = 0.0
        self.uav_vel_body[3] = UVScmd[5]
            
        twist = PositionTarget()
        #twist.header.stamp = 1
        twist.header.frame_id = 'world'
        twist.coordinate_frame = 8
        twist.type_mask = 1479
        twist.velocity.x = self.uav_vel_body[0]
        twist.velocity.y = self.uav_vel_body[1]
        twist.velocity.z = self.uav_vel_body[2]
        twist.yaw_rate = self.uav_vel_body[3]
        
        ibvs_msg = IBVSdata()
        ibvs_msg.errors = er_pix
        ibvs_msg.cmds = self.uav_vel_body
        print("ibvs_msg.cmds: ", ibvs_msg.cmds)
        ibvs_msg.time = t_vsc
        self.pub_ibvs_data.publish(ibvs_msg)
        
        self.pub_vel.publish(twist)   
        
        self.t = self.t+self.dt
    
    def callback(self, msg):
        try:
            u_1 = msg.data[0]
            # print("self.u_1 = ", self.u_1)
            v_1 = msg.data[1]
            # print("self.v_1 = ", self.v_1)
            u_2 = msg.data[2]
            # print("self.u_2 = ", self.u_2)
            v_2 = msg.data[3]
            # print("self.v_2 = ", self.v_2)
            u_3 = msg.data[4]
            # print("self.u_3 = ", self.u_3)
            v_3 = msg.data[5]
            # print("self.v_3 = ", self.v_3)
            u_4 = msg.data[6]
            # print("self.u_4 = ", self.u_4)
            v_4 = msg.data[7]
            # print("self.v_4 = ", self.v_4)
        except CvBridgeError as e:
            print(e)
        
        if self.takeoffed and (not self.landed):
            self.line_detect(u_1, v_1, u_2, v_2, u_3, v_3, u_4, v_4)
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
