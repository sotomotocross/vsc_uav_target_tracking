#include "Controller.hpp"

int main(int argc, char* argv[])
{
  // Initialize ROS
  ros::init(argc, argv, "vsc_uav_target_tracking");

  // Create a ROS node handle
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  // Print a message indicating the start of the node
  std::cout << "Starting ibvs image moments node" << std::endl;

  // Create an instance of the Controller class
  vsc_uav_target_tracking::Controller controller(nh, pnh);

  // Enter the ROS spin loop
  ros::spin();

  return 0;
}
