cmake_minimum_required(VERSION 3.0.2)
project(vehicle_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  std_msgs
)

add_message_files(
  FILES
  State.msg
  Input.msg
  DynObs.msg
  DynObsPt.msg
  Point2d.msg
  # Guideline.msg
  Scenario.msg
  LocalGuidelinePoint.msg
)

add_service_files(
  FILES
  ReqMotionPrediction.srv
  UpdateVehicleStates.srv
  GetAction.srv
  ResetMP.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES vehicle_msgs
 CATKIN_DEPENDS geometry_msgs message_runtime nav_msgs std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

