import matplotlib.pyplot as plt
import numpy as np
import os
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Point
from std_msgs.msg import ColorRGBA
from tf.transformations import quaternion_from_euler

color_map = {
    "red": [1, 0, 0],
    "blue": [0.36, 0.74, 0.89],
    "green": [0, 1, 0],
    "black": [0, 0, 0],
    "yellow": [1, 1, 0],
    "purple": [1, 0, 1],
}

def draw_centerlines(waypoints_list, marker_array_pub):
    marker_array = MarkerArray()
    for i, waypoints in enumerate(waypoints_list):
        marker = get_centerline_marker(i, waypoints, color_map["black"], Marker.LINE_STRIP)
        marker_array.markers.append(marker)
    marker_array_pub.publish(marker_array)

def get_centerline_marker(id, waypoints, color, line_type):    
    marker = Marker()
    marker.header.stamp = rospy.Time.now()
    marker.id = id
    marker.header.frame_id = "map"   # Set the frame ID according to your needs
    marker.pose.orientation.w = 1.0  # Identity quaternion
    marker.type = line_type
    marker.action = Marker.ADD
    marker.scale.x = 0.08  # Line width
    for point in waypoints:
        p = Point()
        p.x = point[0]
        p.y = point[1]
        p.z = 0
        marker.points.append(p)
        pt_color = ColorRGBA()
        pt_color.r = color[0]
        pt_color.g = color[1]
        pt_color.b = color[2]
        pt_color.a = 1
        marker.colors.append(pt_color)

    return marker 

vehicle_height = 1.5

def draw_vehicles(marker_array_pub, multi_vehicle_states, vehicle_sizes):
    marker_array = MarkerArray()
    # Delete all markers at the beginning of each frame
    marker_del = Marker()
    marker_del.header.frame_id = "map"   
    marker_del.action = Marker.DELETEALL
    marker_array.markers.append(marker_del)
    
    for vehicle_id, vehicle_state in enumerate(multi_vehicle_states):
        vehicle_color = 'blue' if vehicle_id == 0 else 'red'
        length = vehicle_sizes[vehicle_id][0]
        width = vehicle_sizes[vehicle_id][1]
        pose = (vehicle_state[0], vehicle_state[1], vehicle_state[2])
        marker_array.markers.append(
            get_vehicle_marker(id=vehicle_id, 
                               pose=pose,
                               width=width,
                               length=length,
                               color=vehicle_color,
                               alpha=0.5))

        marker_array.markers.append(
                get_vehicle_id_marker(id=vehicle_id, text=vehicle_id, pose=pose, color='black'))
    
    marker_array_pub.publish(marker_array)

def visualize_closed_loop_trajectories(marker_array_pub, cl_trajs, vehicle_size):
    marker_array = MarkerArray()
    num_vehicle = len(cl_trajs)
    len_traj = cl_trajs[0].shape[0]
    count = 0
    for i in range(num_vehicle): 
        if i == 0:
            color = 'blue'
        elif i == 1:
            color = 'red'
        else:
            color = 'green'
        
        for k in range(0, len_traj, 12):            
            pose = (cl_trajs[i][k, 0], cl_trajs[i][k, 1], cl_trajs[i][k, 2])
            marker_array.markers.append(
                get_vehicle_marker(id= count, 
                                   pose=pose,
                                   width=vehicle_size[1],
                                   length=vehicle_size[0],
                                   color=color,
                                   alpha=0.5))
            if i != 1:
                marker_array.markers.append(
                    get_vehicle_id_marker(id=count, text= round(0.1*k, 1), pose=pose, color='black'))
            count += 1
    marker_array_pub.publish(marker_array)

def get_vehicle_marker(id, pose, width, length, color, alpha=0.5):
    x, y, yaw = pose
    z = vehicle_height / 2
    marker_msg = Marker()
    marker_msg.header.frame_id = "map"  # Set the frame ID according to your coordinate system
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.ns = "vehicle"
    marker_msg.id = id
    marker_msg.type = Marker.CUBE
    marker_msg.action = Marker.ADD

    marker_msg.pose.position.x = x
    marker_msg.pose.position.y = y
    marker_msg.pose.position.z = z

    quat = quaternion_from_euler(0, 0, yaw)  # Convert yaw angle to quaternion
    marker_msg.pose.orientation = Quaternion(*quat)

    marker_msg.scale.x = length
    marker_msg.scale.y = width
    marker_msg.scale.z = vehicle_height
    
    color = color_map[color]
    marker_msg.color.a = alpha
    marker_msg.color.r = color[0]
    marker_msg.color.g = color[1]
    marker_msg.color.b = color[2]
    
    marker_msg.header.stamp = rospy.Time.now()  # Update the timestamp

    return marker_msg 

def get_vehicle_id_marker(id, text, pose, color):
    x, y, yaw = pose
    z = vehicle_height / 2
    marker_msg = Marker()
    marker_msg.header.frame_id = "map"  # Set the frame ID according to your coordinate system
    marker_msg.header.stamp = rospy.Time.now()
    marker_msg.ns = "vehicle_id"
    marker_msg.id = id
    marker_msg.text = f"{text}"
    marker_msg.type = Marker.TEXT_VIEW_FACING
    marker_msg.action = Marker.ADD

    marker_msg.pose.position.x = x
    marker_msg.pose.position.y = y
    marker_msg.pose.position.z = z

    # quat = quaternion_from_euler(0, 0, yaw)  # Convert yaw angle to quaternion
    # marker_msg.pose.orientation = Quaternion(*quat)

    # marker_msg.scale.x = length
    # marker_msg.scale.y = width
    # marker_msg.scale.z = vehicle_height
    marker_msg.scale.z = 1.2
    
    color = color_map[color]
    marker_msg.color.a = 0.9
    marker_msg.color.r = color[0]
    marker_msg.color.g = color[1]
    marker_msg.color.b = color[2]
    
    marker_msg.header.stamp = rospy.Time.now()  # Update the timestamp

    return marker_msg 

def draw_guidelines(
    guidelines, 
    left_bounds,
    right_bounds,
    marker_array_pub
):
    """
    Draws the guidelines in rviz.
    """
    marker_array = MarkerArray()
    id = 0
    for guideline in guidelines:
        guideline_waypoints = guideline.get_waypoints()
        # guideline_waypoints = guideline.get_local_points() 
        type_dict = dict(marker_type=Marker.LINE_STRIP, color=color_map["red"], scale=[0.1, 0.0, 0.0])
        marker_array.markers.append(get_road_marker(id, guideline_waypoints[:, 0], guideline_waypoints[:, 1], type_dict))
        id += 1
        type_dict = dict(marker_type=Marker.SPHERE_LIST, color=color_map["red"], scale=[0.3, 0.3, 0.3])
        marker_array.markers.append(get_road_marker(id, guideline_waypoints[:, 0], guideline_waypoints[:, 1], type_dict))
        id += 1
    
    for left_bound in left_bounds:
        left_bound_waypoints = left_bound.get_waypoints()
        type_dict = dict(marker_type=Marker.LINE_STRIP, color=color_map["blue"], scale=[0.1, 0.0, 0.0])
        marker_array.markers.append(get_road_marker(id, left_bound_waypoints[:, 0], left_bound_waypoints[:, 1], type_dict))
        id += 1
        type_dict = dict(marker_type=Marker.SPHERE_LIST, color=color_map["blue"], scale=[0.3, 0.3, 0.3])
        marker_array.markers.append(get_road_marker(id, left_bound_waypoints[:, 0], left_bound_waypoints[:, 1], type_dict))
        id += 1
    
    for right_bound in right_bounds:
        right_bound_waypoints = right_bound.get_waypoints()
        type_dict = dict(marker_type=Marker.LINE_STRIP, color=color_map["blue"], scale=[0.1, 0.0, 0.0])
        marker_array.markers.append(get_road_marker(id, right_bound_waypoints[:, 0], right_bound_waypoints[:, 1], type_dict))
        id += 1
        type_dict = dict(marker_type=Marker.SPHERE_LIST, color=color_map["blue"], scale=[0.3, 0.3, 0.3])
        marker_array.markers.append(get_road_marker(id, right_bound_waypoints[:, 0], right_bound_waypoints[:, 1], type_dict))
        id += 1
    
    marker_array_pub.publish(marker_array)

def draw_local_road_bounds(local_guideline, marker_array_pub):
    """
    Draws the road bounds in rviz.
    """
    marker_array = MarkerArray()
    id = 0
    num_waypoint = len(local_guideline)
    waypoints = np.zeros((num_waypoint, 2))
    bound_points_left = np.zeros((num_waypoint, 2))
    bound_points_right = np.zeros((num_waypoint, 2))
    for k in range(num_waypoint):
        waypoint = np.array([local_guideline[k].waypoint.x, local_guideline[k].waypoint.y]) 
        lane_width_left = local_guideline[k].lane_width_left
        lane_width_right = local_guideline[k].lane_width_right
        norm_vec = np.array([local_guideline[k].norm_vec.x, local_guideline[k].norm_vec.y])

        waypoints[k, :] = waypoint
        bound_points_left[k, :] = waypoint + norm_vec * lane_width_left
        bound_points_right[k, :] = waypoint - norm_vec * lane_width_right

    type_dict = dict(marker_type=Marker.SPHERE_LIST, color=color_map["green"], scale=[0.2, 0.2, 0.2])
    marker_array.markers.append(get_road_marker(id, waypoints[:, 0], waypoints[:, 1], type_dict))    
    id += 1
    type_dict = dict(marker_type=Marker.SPHERE_LIST, color=color_map["green"], scale=[0.2, 0.2, 0.2])
    marker_array.markers.append(get_road_marker(id, bound_points_left[:, 0], bound_points_left[:, 1], type_dict))
    id += 1
    type_dict = dict(marker_type=Marker.SPHERE_LIST, color=color_map["green"], scale=[0.2, 0.2, 0.2])
    marker_array.markers.append(get_road_marker(id, bound_points_right[:, 0], bound_points_right[:, 1], type_dict))
    
    marker_array_pub.publish(marker_array)  


def draw_lanelet_map(laneletmap, marker_array_pub):
    """
    Draws the lanelet map in rviz.
    """
    marker_array = MarkerArray()
    id = 0
    for ls in laneletmap.lineStringLayer:
        if "type" not in ls.attributes.keys():
            raise RuntimeError("ID " + str(id) + ": Linestring type must be specified")
        elif ls.attributes["type"] == "curbstone":
            type_dict = dict(marker_type=Marker.LINE_STRIP, color=color_map["black"], scale=[0.5, 0.0, 0.0])
            ls_points_x = [pt.x for pt in ls]
            ls_points_y = [pt.y for pt in ls]
            marker_array.markers.append(get_road_marker(id, ls_points_x, ls_points_y, type_dict))
            id += 1
        # elif ls.attributes["type"] == "line_thin":
        #     if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
        #         type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
        #     else:
        #         type_dict = dict(color="white", linewidth=1, zorder=10)
        # elif ls.attributes["type"] == "line_thick":
        #     if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
        #         type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
        #     else:
        #         type_dict = dict(color="white", linewidth=2, zorder=10)
        # elif ls.attributes["type"] == "pedestrian_marking":
        #     type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        # elif ls.attributes["type"] == "bike_marking":
        #     type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        # elif ls.attributes["type"] == "stop_line":
        #     type_dict = dict(color="white", linewidth=3, zorder=10)
        
        # elif ls.attributes["type"] == "virtual":
        #     ls_points_x = [pt.x for pt in ls]
        #     ls_points_y = [pt.y for pt in ls]
        #     type_dict = dict(marker_type=Marker.LINE_STRIP, color=color_map["blue"], scale=[0.2, 0.2, 0.2])
        #     marker_array.markers.append(get_road_marker(id, ls_points_x, ls_points_y, type_dict))
        #     id += 1
        #     type_dict = dict(marker_type=Marker.SPHERE_LIST, color=color_map["blue"], scale=[0.3, 0.3, 0.3])
        #     marker_array.markers.append(get_road_marker(id, ls_points_x, ls_points_y, type_dict))
        #     id += 1
        
        # elif ls.attributes["type"] == "road_border":
        #     type_dict = dict(color="black", linewidth=1, zorder=10)
        # elif ls.attributes["type"] == "guard_rail":
        #     type_dict = dict(color="black", linewidth=1, zorder=10)
        # elif ls.attributes["type"] == "traffic_sign":
        #     continue
        # elif ls.attributes["type"] == "building":
        #     type_dict = dict(color="pink", zorder=1, linewidth=5)
        # elif ls.attributes["type"] == "spawnline":
        #     if ls.attributes["spawn_type"] == "start":
        #         type_dict = dict(color="green", zorder=11, linewidth=2)
        #     elif ls.attributes["spawn_type"] == "end":
        #         type_dict = dict(color="red", zorder=11, linewidth=2)

        # else:
        #     if ls.attributes["type"] not in unknown_linestring_types:
        #         unknown_linestring_types.append(ls.attributes["type"])
        #     continue
    marker_array_pub.publish(marker_array)
       
    
def get_road_marker(id, pts_x, pts_y, type_dict):
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "road"
    marker.id = id
    marker.type = type_dict["marker_type"]
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = type_dict["scale"][0]
    marker.scale.y = type_dict["scale"][1]
    marker.scale.z = type_dict["scale"][2]
    
    for i in range(len(pts_x)):
        marker.points.append(Point(x=pts_x[i], y=pts_y[i], z=0.0))
        pt_color = ColorRGBA()
        pt_color.r = type_dict["color"][0]
        pt_color.g = type_dict["color"][1]
        pt_color.b = type_dict["color"][2]
        pt_color.a = 1.0
                
        marker.colors.append(pt_color)
    
    return marker