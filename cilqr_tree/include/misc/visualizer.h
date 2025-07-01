#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include "cilqr_tree/trajectory_tree_data.h"
#include <Eigen/Dense>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class Visualizer
{
private:
    ros::NodeHandle nh;
    ros::Publisher waypoints_pub;
    ros::Publisher obstacles_marker_array_pub; 
    ros::Publisher vehicle_marker_array_pub;

public:
    Visualizer(ros::NodeHandle &nh_) : nh(nh_)
    {
        waypoints_pub = nh.advertise<visualization_msgs::Marker>("/visualizer/waypoints", 10);
        obstacles_marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("/visualizer/obstacles_marker_array", 10);
        vehicle_marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("/visualizer/vehicle_marker_array", 10);
    }

    inline void visualize(
        cilqr_tree::TrajTree<6, 2> &traj_tree,
        const std::vector<std::vector<std::vector<Vector4d>>> &pred_trajs,
        const double vehicle_length,
        const double vehicle_width,
        const double inter_axle_length
    )
    {
        visualization_msgs::Marker waypoints_marker;
        visualization_msgs::MarkerArray obstacles_marker_array;
        visualization_msgs::MarkerArray vehicle_marker_array;

        waypoints_marker.id = 0;
        waypoints_marker.type = visualization_msgs::Marker::SPHERE_LIST;
        waypoints_marker.header.stamp = ros::Time::now();
        waypoints_marker.header.frame_id = "map";
        waypoints_marker.pose.orientation.w = 1.00;
        waypoints_marker.action = visualization_msgs::Marker::ADD;
        waypoints_marker.ns = "waypoints";
        waypoints_marker.color.r = 1.00;
        waypoints_marker.color.g = 0.00;
        waypoints_marker.color.b = 0.00;
        waypoints_marker.color.a = 0.50;
        waypoints_marker.scale.x = 0.10;
        waypoints_marker.scale.y = 0.10;
        waypoints_marker.scale.z = 0.10;

        for (size_t i = 0; i < pred_trajs.size(); ++i)
        {
            for (size_t j = 0; j < pred_trajs[i].size(); ++j)
            {
                for (size_t k = 0; k < pred_trajs[i][j].size(); ++k)
                {
                    visualization_msgs::Marker obstacle_marker;
                    tf2::Quaternion q;
                    q.setRPY(0, 0, pred_trajs[i][j][k](2));

                    obstacle_marker.id = i * pred_trajs[i].size() * pred_trajs[i][j].size() + j * pred_trajs[i][j].size() + k;
                    obstacle_marker.type = visualization_msgs::Marker::CUBE;
                    obstacle_marker.header.stamp = ros::Time::now();
                    obstacle_marker.header.frame_id = "map";
                    obstacle_marker.action = visualization_msgs::Marker::ADD;
                    obstacle_marker.ns = "obstacles";
                    obstacle_marker.color.r = 1.0;
                    obstacle_marker.color.g = 0.0;
                    obstacle_marker.color.b = 0.0;
                    obstacle_marker.color.a = 0.01;
                    obstacle_marker.scale.x = vehicle_length;
                    obstacle_marker.scale.y = vehicle_width;
                    obstacle_marker.scale.z = 1.0;

                    obstacle_marker.pose.orientation = tf2::toMsg(q);
                    obstacle_marker.pose.position.x = pred_trajs[i][j][k](0);
                    obstacle_marker.pose.position.y = pred_trajs[i][j][k](1);
                    obstacle_marker.pose.position.z = 0.5;
                    obstacles_marker_array.markers.push_back(obstacle_marker);
                }
            }
        }

        for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
        {
            for (int k = 0; k < traj_tree.N + 1; ++k)
            {
                geometry_msgs::Point point;
                auto state = traj_tree(k, br_id).x;
                point.x = state(0, 0);
                point.y = state(1, 0);
                point.z = 0.2;
                waypoints_marker.points.push_back(point);
                
                visualization_msgs::Marker vehicle_marker;
                tf2::Quaternion q;
                q.setRPY(0, 0, state(2, 0));

                vehicle_marker.id = br_id * traj_tree.N + k;
                vehicle_marker.type = visualization_msgs::Marker::CUBE;
                vehicle_marker.header.stamp = ros::Time::now();
                vehicle_marker.header.frame_id = "map";
                vehicle_marker.action = visualization_msgs::Marker::ADD;
                vehicle_marker.ns = "vehicle";
                vehicle_marker.color.r = 0.0;
                vehicle_marker.color.g = 0.0;
                vehicle_marker.color.b = 1.0;
                // vehicle_marker.color.a = 1.0 / double(k + 1);
                vehicle_marker.color.a = 0.01;
                vehicle_marker.scale.x = vehicle_length;
                vehicle_marker.scale.y = vehicle_width;
                vehicle_marker.scale.z = 1.0;

                vehicle_marker.pose.orientation = tf2::toMsg(q);
                vehicle_marker.pose.position.x = state(0, 0) + inter_axle_length * cos(state(2, 0)) / 2.0;
                vehicle_marker.pose.position.y = state(1, 0) + inter_axle_length * sin(state(2, 0)) / 2.0;
                vehicle_marker.pose.position.z = 0.5;
                vehicle_marker_array.markers.push_back(vehicle_marker);
                
            }
        }
        waypoints_pub.publish(waypoints_marker);
        obstacles_marker_array_pub.publish(obstacles_marker_array);
        vehicle_marker_array_pub.publish(vehicle_marker_array);
        
    }


    inline void visualize(
        cilqr_tree::TrajTree<6, 2> &traj_tree,
        const std::vector<std::vector<Vector4d>> &pred_trajs_1,
        const std::vector<std::vector<Vector4d>> &pred_trajs_2,
        const double vehicle_length,
        const double vehicle_width,
        const double inter_axle_length
    )
    {
        visualization_msgs::Marker waypoints_marker;
        visualization_msgs::MarkerArray obstacles_marker_array;
        visualization_msgs::MarkerArray vehicle_marker_array;

        waypoints_marker.id = 0;
        waypoints_marker.type = visualization_msgs::Marker::SPHERE_LIST;
        waypoints_marker.header.stamp = ros::Time::now();
        waypoints_marker.header.frame_id = "map";
        waypoints_marker.pose.orientation.w = 1.00;
        waypoints_marker.action = visualization_msgs::Marker::ADD;
        waypoints_marker.ns = "waypoints";
        waypoints_marker.color.r = 1.00;
        waypoints_marker.color.g = 0.00;
        waypoints_marker.color.b = 0.00;
        waypoints_marker.color.a = 0.50;
        waypoints_marker.scale.x = 0.10;
        waypoints_marker.scale.y = 0.10;
        waypoints_marker.scale.z = 0.10;

        for (size_t i = 0; i < pred_trajs_1.size(); ++i)
        {
            for (size_t k = 0; k < pred_trajs_1[i].size(); ++k)
            {
                visualization_msgs::Marker obstacle_marker;
                tf2::Quaternion q;
                q.setRPY(0, 0, pred_trajs_1[i][k](2));

                obstacle_marker.id = i * pred_trajs_1[i].size() + k;
                obstacle_marker.type = visualization_msgs::Marker::CUBE;
                obstacle_marker.header.stamp = ros::Time::now();
                obstacle_marker.header.frame_id = "map";
                obstacle_marker.action = visualization_msgs::Marker::ADD;
                obstacle_marker.ns = "obstacles";
                obstacle_marker.color.r = 0.0;
                obstacle_marker.color.g = 1.0;
                obstacle_marker.color.b = 0.0;
                obstacle_marker.color.a = 0.01;
                obstacle_marker.scale.x = vehicle_length;
                obstacle_marker.scale.y = vehicle_width;
                obstacle_marker.scale.z = 1.0;

                obstacle_marker.pose.orientation = tf2::toMsg(q);
                obstacle_marker.pose.position.x = pred_trajs_1[i][k](0);
                obstacle_marker.pose.position.y = pred_trajs_1[i][k](1);
                obstacle_marker.pose.position.z = 0.5;
                obstacles_marker_array.markers.push_back(obstacle_marker);
            }
        }

        for (size_t i = 0; i < pred_trajs_2.size(); ++i)
        {
            for (size_t k = 0; k < pred_trajs_2[i].size(); ++k)
            {
                visualization_msgs::Marker obstacle_marker;
                tf2::Quaternion q;
                q.setRPY(0, 0, pred_trajs_2[i][k](2));

                obstacle_marker.id = pred_trajs_1.size() * pred_trajs_1[0].size() + i * pred_trajs_2[i].size() + k;
                obstacle_marker.type = visualization_msgs::Marker::CUBE;
                obstacle_marker.header.stamp = ros::Time::now();
                obstacle_marker.header.frame_id = "map";
                obstacle_marker.action = visualization_msgs::Marker::ADD;
                obstacle_marker.ns = "obstacles";
                obstacle_marker.color.r = 0.0;
                obstacle_marker.color.g = 1.0;
                obstacle_marker.color.b = 0.0;
                obstacle_marker.color.a = 0.01;
                obstacle_marker.scale.x = vehicle_length;
                obstacle_marker.scale.y = vehicle_width;
                obstacle_marker.scale.z = 1.0;

                obstacle_marker.pose.orientation = tf2::toMsg(q);
                obstacle_marker.pose.position.x = pred_trajs_2[i][k](0);
                obstacle_marker.pose.position.y = pred_trajs_2[i][k](1);
                obstacle_marker.pose.position.z = 0.5;
                obstacles_marker_array.markers.push_back(obstacle_marker);
            }
        }

        for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
        {
            for (int k = 0; k < traj_tree.N + 1; ++k)
            {
                geometry_msgs::Point point;
                auto state = traj_tree(k, br_id).x;
                point.x = state(0, 0);
                point.y = state(1, 0);
                point.z = 0.2;
                waypoints_marker.points.push_back(point);
                
                visualization_msgs::Marker vehicle_marker;
                tf2::Quaternion q;
                q.setRPY(0, 0, state(2, 0));

                vehicle_marker.id = br_id * traj_tree.N + k;
                vehicle_marker.type = visualization_msgs::Marker::CUBE;
                vehicle_marker.header.stamp = ros::Time::now();
                vehicle_marker.header.frame_id = "map";
                vehicle_marker.action = visualization_msgs::Marker::ADD;
                vehicle_marker.ns = "vehicle";
                vehicle_marker.color.r = 0.0;
                vehicle_marker.color.g = 0.0;
                vehicle_marker.color.b = 1.0;
                // vehicle_marker.color.a = 1.0 / double(k + 1);
                vehicle_marker.color.a = 0.01;
                vehicle_marker.scale.x = vehicle_length;
                vehicle_marker.scale.y = vehicle_width;
                vehicle_marker.scale.z = 1.0;

                vehicle_marker.pose.orientation = tf2::toMsg(q);
                vehicle_marker.pose.position.x = state(0, 0) + inter_axle_length * cos(state(2, 0)) / 2.0;
                vehicle_marker.pose.position.y = state(1, 0) + inter_axle_length * sin(state(2, 0)) / 2.0;
                vehicle_marker.pose.position.z = 0.5;
                vehicle_marker_array.markers.push_back(vehicle_marker);
                
            }
        }
        waypoints_pub.publish(waypoints_marker);
        obstacles_marker_array_pub.publish(obstacles_marker_array);
        vehicle_marker_array_pub.publish(vehicle_marker_array);
    }

    inline void visualize(
        cilqr_tree::TrajTree<6, 2> &traj_tree,
        const double vehicle_length,
        const double vehicle_width,
        const double inter_axle_length
    )
    {
        visualization_msgs::Marker waypoints_marker;
        visualization_msgs::MarkerArray vehicle_marker_array;

        waypoints_marker.id = 0;
        waypoints_marker.type = visualization_msgs::Marker::SPHERE_LIST;
        waypoints_marker.header.stamp = ros::Time::now();
        waypoints_marker.header.frame_id = "map";
        waypoints_marker.pose.orientation.w = 1.00;
        waypoints_marker.action = visualization_msgs::Marker::ADD;
        waypoints_marker.ns = "waypoints";
        waypoints_marker.color.r = 1.00;
        waypoints_marker.color.g = 0.00;
        waypoints_marker.color.b = 0.00;
        waypoints_marker.color.a = 0.50;
        waypoints_marker.scale.x = 0.10;
        waypoints_marker.scale.y = 0.10;
        waypoints_marker.scale.z = 0.10;

        for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
        {
            for (int k = 0; k < traj_tree.N + 1; ++k)
            {
                geometry_msgs::Point point;
                auto state = traj_tree(k, br_id).x;
                point.x = state(0, 0);
                point.y = state(1, 0);
                point.z = 0.2;
                waypoints_marker.points.push_back(point);
                
                visualization_msgs::Marker vehicle_marker;
                tf2::Quaternion q;
                q.setRPY(0, 0, state(2, 0));

                vehicle_marker.id = br_id * traj_tree.N + k;
                vehicle_marker.type = visualization_msgs::Marker::CUBE;
                vehicle_marker.header.stamp = ros::Time::now();
                vehicle_marker.header.frame_id = "map";
                vehicle_marker.action = visualization_msgs::Marker::ADD;
                vehicle_marker.ns = "vehicle";
                vehicle_marker.color.r = 0.0;
                vehicle_marker.color.g = 0.0;
                vehicle_marker.color.b = 1.0;
                // vehicle_marker.color.a = 1.0 / double(k + 1);
                vehicle_marker.color.a = 0.01;
                vehicle_marker.scale.x = vehicle_length;
                vehicle_marker.scale.y = vehicle_width;
                vehicle_marker.scale.z = 1.0;

                vehicle_marker.pose.orientation = tf2::toMsg(q);
                vehicle_marker.pose.position.x = state(0, 0) + inter_axle_length * cos(state(2, 0)) / 2.0;
                vehicle_marker.pose.position.y = state(1, 0) + inter_axle_length * sin(state(2, 0)) / 2.0;
                vehicle_marker.pose.position.z = 0.5;
                vehicle_marker_array.markers.push_back(vehicle_marker);
                
            }
        }
        waypoints_pub.publish(waypoints_marker);
        vehicle_marker_array_pub.publish(vehicle_marker_array);
    }
};

#endif