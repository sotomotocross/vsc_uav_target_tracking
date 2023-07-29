#!/usr/bin/env python
from __future__ import print_function

from numpy.core.fromnumeric import size
from numpy.core.numeric import ones
import roslib
roslib.load_manifest('vsc_uav_target_tracking')
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
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Vector3Stamped, TwistStamped
from numpy.linalg import norm
# from math import cos, sin, tan, sqrt, exp, pi, atan2, acos, asin
from math import *
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry
from vsc_uav_target_tracking.msg import VSCdata, PVSdata, EKFdata, IBVSdata
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
        # self.image_sub = rospy.Subscriber("/image_raww", Image, self.callback)
        # self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.updateImu)
        # self.pos_sub = rospy.Subscriber("/mavros/global_position/local", Odometry, self.OdomCb)
        # self.vel_uav = rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.VelCallback)
        
        # uav state variables
        self.landed = 0
        self.takeoffed = 1
        self.uav_vel_body = np.array([0.0, 0.0, 0.0, 0.0])
        self.vel_uav = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.er_pix_prev = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(8,1)
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
        self.a = 0.01
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


    # Function calculating the derivative of the error of the pixels on the image plane
    def deriv_error_estimation(self, er_pix, er_pix_prev):
        
        ew = (er_pix - er_pix_prev)/self.dt 

        return ew


    # Function calculating the control law both for target tracking and trajectory planning
    def quadrotorVSControl_tracking(self, Lm, er_pix, ew):
               
        # ----- Home Ubuntu Desktop Tuning ----
        forward_gain_Kc = 0.5
        thrust_gain_Kc = 0.0
        sway_gain_Kc = 0.35
        yaw_gain_Kc = 1.6
        Kc = np.identity(6)
        Kc[0][0] = thrust_gain_Kc
        Kc[1][1] = sway_gain_Kc
        Kc[2][2] = forward_gain_Kc
        Kc[3][3] = yaw_gain_Kc
        Kc[4][4] = 0.0
        Kc[5][5] = 0.0

        Ke = np.identity(6)
        forward_gain_Ke = 0.5
        thrust_gain_Ke = 0.0
        sway_gain_Ke = 0.35
        yaw_gain_Ke = -2.0
        Ke = np.identity(6)
        Ke[0][0] = thrust_gain_Ke
        Ke[1][1] = sway_gain_Ke
        Ke[2][2] = forward_gain_Ke
        Ke[3][3] = yaw_gain_Ke
        Ke[4][4] = 0.0
        Ke[5][5] = 0.0
        # --------------------------------------

        Ucmd = -np.dot(Kc,np.dot(np.linalg.pinv(Lm), er_pix))+np.dot(np.linalg.pinv(Lm), np.array([0.0, self.a, 0.0, self.a, 0.0, self.a, 0.0, self.a]).reshape(8,1)) - np.dot(Ke, np.dot(np.linalg.pinv(Lm), ew) )
        
        return Ucmd
    

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

            # Sorting of the orientation of the detected coastline
            if angle < -45:
                angle = 90 + angle
            if w_min < h_min and angle > 0:
                angle = (90 - angle) * -1
            if w_min > h_min and angle < 0:
                angle = 90 + angle
            
            self.alpha = angle
            self.sigma = cv2.contourArea(contours_blk[0])

            # Recreation of the feature box
            if angle >= 0:
                # print("1st control choice")
                # print("Angle:", angle)
                mp = [box[0][0], box[0][1], self.z, box[1][0], box[1][1], self.z, box[2][0], box[2][1], self.z, box[3][0], box[3][1], self.z]
                # print("mp: ", mp)
            else:
                # print("2nd control choice") 
                # print("Angle:", angle)
                mp = [box[3][0], box[3][1], self.z, box[0][0], box[0][1], self.z, box[1][0], box[1][1], self.z, box[2][0], box[2][1], self.z]
                # print("mp: ", mp)    

            # print("mp: ", mp)
            mp_cartesian = self.cartesian_from_pixel(mp, self.cu, self.cv, self.ax, self.ay)
            # print("mp_cartesian: ", mp_cartesian)
            mp_cartesian_v = self.featuresTransformation(mp_cartesian, self.phi_imu, self.theta_imu)
            # print("mp_cartesian_v: ", mp_cartesian_v)
            mp_pixel_v = self.pixels_from_cartesian(mp_cartesian_v, self.cu, self.cv, self.ax, self.ay)
            # print("mp_pixel_v: ", mp_pixel_v)

            mp_des = np.array([420, 472+self.a*self.t, self.z, 367, 483+self.a*self.t, self.z, 327, 2+self.a*self.t, self.z, 377, 0+self.a*self.t, self.z]) 
            # print("mp_des: ", mp_des)
            
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
            
            Lm, er_pix = self.calculateIM(mp_pixel_v, mp_des, self.cu, self.cv, self.ax, self.ay) #TRANSFORM FEATURES
            velocity_camera = np.dot(T, self.vel_uav)
            
            u_bc = (box[0][0]+box[1][0]+box[2][0]+box[3][0])/4
            # print("u of centroid: ", u_bc)
            v_bc = (box[0][1]+box[1][1]+box[2][1]+box[3][1])/4
            # print("v of centroid: ", v_bc)
            
            ew = self.deriv_error_estimation(er_pix, self.er_pix_prev)
            # print("ew: ", ew)
            ew_filtered = self.moving_average_filter(np.array(ew).reshape(8,1))
            # print("ew_filtered: ", ew_filtered)          
            ew_filtered_odometry = ew_filtered - np.array(np.dot(Lm, velocity_camera)).reshape(8,1)

            e_m = (er_pix[0]+er_pix[2]+er_pix[4]+er_pix[6])/4
            # print("e_m: ", e_m)
            e_m_dot = (ew_filtered_odometry[0]+ew_filtered_odometry[2]+ew_filtered_odometry[4]+ew_filtered_odometry[6])/4
            # print("e_m_dot: ", e_m_dot)
            self.ekf_estimation(e_m, e_m_dot)
            wave_est_control_input = self.x_est[1]
            wave_estimation_final = np.array([wave_est_control_input, [self.a], wave_est_control_input, [self.a], wave_est_control_input, [self.a], wave_est_control_input, [self.a]]).reshape(8,1)
            # print("final estimation: ", wave_estimation_final)
            
            UVScmd = self.quadrotorVSControl_tracking(Lm, er_pix, wave_estimation_final)
            # UVScmd = np.dot(T, UVScmd)
            UVScmd = np.dot(np.linalg.inv(T), UVScmd)
            self.er_pix_prev = er_pix
            # print("er_pix_prev: ", self.er_pix_prev)    
            # print("UVScmd: ", UVScmd)        
             
            self.uav_vel_body[0] = UVScmd[0]
            self.uav_vel_body[1] = UVScmd[1]
            self.uav_vel_body[2] = UVScmd[2]
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
            # twist.velocity.x = 0.0
            # twist.velocity.y = 0.0
            # twist.velocity.z = 0.0
            # twist.yaw_rate = 0.0
            
            ekf_msg = EKFdata()
            ekf_msg.ekf_output = self.x_est
            ekf_msg.e_m = e_m
            ekf_msg.e_m_dot = e_m_dot
            ekf_msg.u_bc = u_bc
            ekf_msg.v_bc = v_bc
            ekf_msg.time = t_vsc
            self.pub_ekf_data.publish(ekf_msg)
            
            ibvs_msg = IBVSdata()
            ibvs_msg.errors = er_pix
            ibvs_msg.cmds = self.uav_vel_body
            print("ibvs_msg.cmds: ", ibvs_msg.cmds)
            ibvs_msg.time = t_vsc
            self.pub_ibvs_data.publish(ibvs_msg)
            
            # pvs_msg = PVSdata()
            # pvs_msg.alpha = self.alpha
            # pvs_msg.alpha_des = self.alpha_des
            # pvs_msg.sigma = self.sigma
            # pvs_msg.sigma_des = self.sigma_des
            # pvs_msg.time = t_vsc
            # self.pub_pvs_data.publish(pvs_msg) 

            self.pub_vel.publish(twist)         
            
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