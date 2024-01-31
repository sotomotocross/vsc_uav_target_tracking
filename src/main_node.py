#!/usr/bin/env python

import rospy
from ros_communication import ROSCommunication
from visual_servoing_control import VisualServoingControl
from img_seg_cnn.msg import PredData
import numpy as np

def detection_callback(data):
    """
    Callback function for processing detected data.

    Args:
        data (PredData): Detected data from the topic '/pred_data'.

    Returns:
        None
    """
    box_1 = data.box_1
    box_2 = data.box_2
    box_3 = data.box_3
    box_4 = data.box_4
    cX = data.cX
    cY = data.cY
    angle = data.alpha
    sigma = data.sigma
    sigma_square = data.sigma_square
    sigma_square_log = data.sigma_square_log

    box = np.array([box_1, box_2, box_3, box_4], dtype=int)

    visual_servoing_control = VisualServoingControl()
    visual_servoing_control.detection_processing(box, angle)

def main():
    """
    Main function to initialize the ROS node and subscribe to the detection data.

    Args:
        None

    Returns:
        None
    """
    rospy.init_node('vsc_uav_target_tracking', anonymous=True)

    ros_communication = ROSCommunication()
    feature_subscriber = rospy.Subscriber("/pred_data", PredData, detection_callback)

    # Start the main loop or other necessary logic
    rospy.spin()

if __name__ == '__main__':
    main()
