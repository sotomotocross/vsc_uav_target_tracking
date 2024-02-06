cmake_minimum_required(VERSION 3.0.2)
project(vsc_uav_target_tracking)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  mavros_msgs
  message_runtime
  rospy
  roslib
  sensor_msgs
  std_msgs  
  message_generation
  img_seg_cnn
  roscpp
  mavros    
)


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  VSCdata.msg
  KNet_pos_input.msg
  PVSdata.msg
  IBVSdata.msg
  EKFdata.msg
#   Message1.msg
#   Message2.msg
)


## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

catkin_package(
 INCLUDE_DIRS include
#  LIBRARIES coastline_tracking
CATKIN_DEPENDS cv_bridge geometry_msgs roslib rospy sensor_msgs std_msgs message_runtime message_generation img_seg_cnn roscpp mavros mavros_msgs  
    DEPENDS system_lib eigen3
)

# Include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
)

# Declare the library sources
set(LIB_SOURCES
    src/roscpp_node/Controller.cpp
    src/roscpp_node/FeatureData.cpp
    src/roscpp_node/VelocityTransformer.cpp
    src/roscpp_node/DynamicsCalculator.cpp    
)

# Declare a library
add_library(${PROJECT_NAME}
    ${LIB_SOURCES}
)

# Add library dependencies
add_dependencies(${PROJECT_NAME}
    ${catkin_EXPORTED_TARGETS}
)

# Link the library
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
)

# Declare an executable
add_executable(
    ${PROJECT_NAME}_node
    src/roscpp_node/node.cpp
)

# Add executable dependencies
add_dependencies(${PROJECT_NAME}_node
    ${catkin_EXPORTED_TARGETS}
)

# Link the executable with the library
target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
)