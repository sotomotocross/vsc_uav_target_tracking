#!/usr/bin/env python
import roslib
roslib.load_manifest('coastline_tracking')

import rospy
import numpy as np
from rospy_tutorials.msg import Floats


def talker():
    pub = rospy.Publisher('floats', Floats, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    r = rospy.Rate(10) # 10hz
    i = 0
    a = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    while not rospy.is_shutdown():  
        a = np.load('/home/sotiris/ROS_workspaces/uav_simulator_ws/prediction_test.npy')
        pub.publish(a[i])
        print("row is: ", a[i])
        i += 1
        r.sleep()

if __name__ == '__main__':
    talker()