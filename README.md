# Visual Servo control for target tracking using UAVs
This is a ROS python package of various Visual servo controllers for UAV aiming on target tracking. The general application of these controllers is a dynamic coastline in the presenc of waves.
The different version of visual servoing will be analyzed below.
The controllers were all developed in a UAV synthetic simulation environment: https://github.com/sotomotocross/UAV_simulator_ArduCopter.git

## Classic IBVS method for target tracking
We initially implemented a classical IBVS strategy for an underactuated UAV aiming target tracking which in our case is a constantly moving coastline in the presence of waves.
All the various cases of IBVS for target tracking are attempts of estimating the wave motion (velocity) and incorporating it in the implemented controller.
The basic controller of this attempt is an IBVS target tracking scheme incoporating an appropriately formulated EKF, based on the Gerstner wave models for the motion of the waves.
The project is organized into the following files:

- `main_node.py`: Orchestrates the entire system.
- `ros_communication.py`: Handles ROS communication.
- `visual_servoing.py`: Implements the core visual servoing logic.
- `visual_servoing_utils.py`: Provides visual servoing utilities.
- `controller_gains.yaml`: Specify controller gains in this YAML file.
- `my_controller_params.yaml`: Specify boolean parameters of which of the various controller version you want to run.
```
$ roslaunch vsc_uav_target_tracking my_controller.launch
```
This is the core of [[1]](#1).

## Partitioned Visual Servo Control strategy (Deprecated - To be updated)
This is a PVS implementation for the same application considering the decoupling between translational and rotational velocities.
```
$ rosrun vsc_uav_target_tracking part_vs_track_ekf_est.py
```
In this case we managed to extend the target motion state estimation module incorporating the [Flownet 2](https://github.com/lmb-freiburg/ROS-packages.git) implementation and a hybrid model-based and data-driven framework (named [KalmanNet](https://github.com/KalmanNet/KalmanNet_TSP.git)) estimating again the velocity of the waves incorporated in our PVS controller.
```
$ rosrun vsc_uav_target_tracking part_vs_track_knet_est.py
```
This is the core of [[2]](#2).

## Combination of Image moments with Visual Servoing (Deprecated - To be updated)
This in implementation of IBVS for target tracking utilizing [image moments](10.1109/TRO.2004.829463) as a statistical target descriptor.
```
$ rosrun vsc_uav_target_tracking img_moments_ibvs.py
```

## Neuromorphic implementation of perception module and control implementation
Here we implemented an event-based tracking control framework for detection, tracking and surveillance of dynamic coastlines using a multirorotor UAV.
Based on [interfacing of a DVS to SpiNN-3 Neuromorphic platform](https://github.com/ntouev/spinn_aer_if.git) and [a framework for DVS event streams manipulation and contour-based areas](https://github.com/ntouev/ev_snn_percept.git)

```
$ rosrun vsc_uav_target_tracking part_vs_track_knet_est.py
```
The controller was not developed on a synthetic environment. Due to the presence of the hardware framework it was developed and implemented directly on an octorotor UAV featuring Pixhawk and Ardupilot.

This is the core of diploma thesis [[3]](#3) and of paper [[4]](#4) that is accepted and will be presented on IROS 2023.

## References
<a id="1">[1]</a> 
S. N. Aspragkathos, G. C. Karras, and K. J. Kyriakopoulos, “A visual servoing strategy for coastline tracking using an unmanned aerial vehicle", in 2022 30th Mediterranean Conference on Control and
        Automation (MED), pp. 375–381, IEEE, 2022, [10.1109/MED54222.2022.9837275](10.1109/MED54222.2022.9837275)

<a id="2">[2]</a> 
S. N. Aspragkathos, G. C. Karras, and K. J. Kyriakopoulos, “A Hybrid Model and Data-Driven Vision-Based Framework for the Detection, Tracking and Surveillance of Dynamic Coastlines Using a Multirotor UAV", in 2022 30th Mediterranean Conference on Control and
Automation (MED), pp. 375–381, IEEE, 2022, [https://doi.org/10.3390/drones6060146](https://doi.org/10.3390/drones6060146)

<a id="3">[3]</a> 
E. Ntouros, “Multicopter control using dynamic vision and neuromorphic computing", Athens, October 2022, [https://dspace.lib.ntua.gr/xmlui/bitstream/handle/123456789/56541/thesis.pdf?sequence=1](https://dspace.lib.ntua.gr/xmlui/bitstream/handle/123456789/56541/thesis.pdf?sequence=1)

<a id="4">[4]</a> 
S. N. Aspragkathos, E. Ntouros, G. C. Karras, B. Linares-Barranco, T. Serrano-Gotarredona and K. J. Kyriakopoulos, “An Event-Based Tracking Control Framework for Multirotor Aerial Vehicles Using a Dynamic Vision Sensor and Neuromorphic Hardware", Accepted on IEEE/RSJ 2023 International Conference on Intelligent Robots and Systems (IROS), IEEE, 2023

