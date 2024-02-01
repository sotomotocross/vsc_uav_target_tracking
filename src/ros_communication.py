#!/usr/bin/env python

from __future__ import print_function
import roslib
roslib.load_manifest('vsc_uav_target_tracking')
import rospy
import numpy as np
import time
import matplotlib.pyplot as plt
from mavros_msgs.msg import PositionTarget
from std_msgs.msg import Int16
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped
from numpy.linalg import norm
from math import cos, sin, tan, sqrt, exp, pi, atan2, acos, asin
import tf
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from vsc_uav_target_tracking.msg import VSCdata, IBVSdata, EKFdata

class ROSCommunication:
    def __init__(self):
        """
        Initialize ROSCommunication class, setting up ROS node and publishers/subscribers.

        Args:
            None

        Returns:
            None
        """
        rospy.init_node('vsc_uav_target_tracking', anonymous=True)
        
        self.pub_vel = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=1)
        self.pub_error = rospy.Publisher('error', Int16, queue_size=10)
        self.pub_angle = rospy.Publisher('angle', Int16, queue_size=10)
        self.pub_vsc_data = rospy.Publisher("/vsc_data", VSCdata, queue_size=1000)
        self.pub_ekf_data = rospy.Publisher("/ekf_data", EKFdata, queue_size=1000)
        self.pub_ibvs_data = rospy.Publisher("/ibvs_data", IBVSdata, queue_size=1000)
        
        #Create subscribers
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.update_imu)
        self.pos_sub = rospy.Subscriber("/mavros/global_position/local", Odometry, self.odometry_callback)
        self.vel_uav = rospy.Subscriber("/mavros/local_position/velocity_body", TwistStamped, self.velocity_callback)
        
        self.phi_imu = 0.0
        self.theta_imu = 0.0 
        self.psi_imu = 0.0
        self.w_imu = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 1.0       
        self.vel_uav = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def publish_error(self, error_value):
        error_msg = Int16()
        error_msg.data = error_value
        self.pub_error.publish(error_msg)

    def publish_angle(self, angle_value):
        angle_msg = Int16()
        angle_msg.data = angle_value
        self.pub_angle.publish(angle_msg)

    def update_imu(self, msg):
        self.phi_imu = msg.orientation.x
        self.theta_imu = msg.orientation.y
        self.psi_imu = msg.orientation.z
        self.w_imu = msg.orientation.w
        self.phi_imu, self.theta_imu, self.psi_imu = euler_from_quaternion ([self.phi_imu, self.theta_imu, self.psi_imu, self.w_imu])
    
    def odometry_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        # print("Message position: ", msg.pose.pose.position)

    def velocity_callback(self, msg):    
        self.vel_uav[0] = msg.twist.linear.x
        self.vel_uav[1] = msg.twist.linear.y
        self.vel_uav[2] = msg.twist.linear.z
        self.vel_uav[3] = msg.twist.angular.x
        self.vel_uav[4] = msg.twist.angular.y
        self.vel_uav[5] = msg.twist.angular.z
        # print("Message uav velocity: ", self.vel_uav)
