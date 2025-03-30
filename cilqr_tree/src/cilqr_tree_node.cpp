#include <ros/ros.h>
#include <ros/console.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include <Eigen/Dense>
#include <vehicle_msgs/ReqMotionPrediction.h>
#include <vehicle_msgs/GetAction.h>
#include <vehicle_msgs/ResetMP.h>
#include <vehicle_msgs/State.h>

#include "cilqr_tree/ilqr_tree.h"
#include "cilqr_tree/ilqr_params.h"
#include "cilqr_tree/trajectory_tree_data.h"
#include "misc/visualizer.h"
#include "cilqr_tree/cilqr_tree.h"

using namespace Eigen;

class CiLQRNode {
public:
    CiLQRNode() {}
    ~CiLQRNode() {}

    void init(ros::NodeHandle &nh) 
    {       
        _motion_prediction_client = nh.serviceClient<vehicle_msgs::ReqMotionPrediction>("/get_motion_prediction");
        _reset_srv = nh.advertiseService("/motion_planner/reset", &CiLQRNode::reset_callback, this);
        _get_action_srv = nh.advertiseService("/motion_planner/get_ev_action", &CiLQRNode::get_action_callback, this);
        _visualizer_ptr = std::make_shared<Visualizer>(nh);
        _planner_ptr = std::make_shared<cilqr_tree::BranchMPC>(nh, _local_guidelines, _visualizer_ptr);
    }

    bool reset_callback(
        vehicle_msgs::ResetMP::Request &req,
        vehicle_msgs::ResetMP::Response &res) 
    {
        _sim_id = req.sim_id;
        _planner_type = req.planner_type;   
        ROS_INFO("Simulation %d", _sim_id);
        ROS_INFO("Planner type %d", _planner_type);
        ROS_INFO("Reset motion planner");
        _ev_action.setZero();
        return true;
    }

    bool get_action_callback(
        vehicle_msgs::GetAction::Request &req,
        vehicle_msgs::GetAction::Response &res) 
    {
        ROS_INFO("Start to compute control input");
        _local_guidelines.clear();
        _ev_pred_trajs.clear();
        _multi_sv_pred_trajs.clear();

        _vehicle_state(0) = req.ego_state.x;
        _vehicle_state(1) = req.ego_state.y;
        _vehicle_state(2) = req.ego_state.heading;
        _vehicle_state(3) = req.ego_state.vel;
        // augmented state
        _vehicle_state(4) = _ev_action(0);
        _vehicle_state(5) = _ev_action(1);

        get_predicted_trajectories(req);

        // We assume all vehicles have the same length, width, and inter_axle_length.
        double length = req.ego_state.length;
        double width = req.ego_state.width;
        double inter_axle_length = req.ego_state.inter_axle_length;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        _planner_ptr->solve(_vehicle_state, _sim_id, _planner_type);
        _ev_action = _planner_ptr->traj_tree.get_control();

        std::vector<Vector4d> ev_trajs;
        _planner_ptr->traj_tree.get_state_trajectories(ev_trajs);
        for (int i = 0; i < ev_trajs.size(); ++i)
        {
            vehicle_msgs::State state;
            state.x = ev_trajs[i](0);
            state.y = ev_trajs[i](1);
            state.heading = ev_trajs[i](2);
            state.vel = ev_trajs[i](3);
            res.ego_trajs.push_back(state);
        }
        res.acc = _ev_action(0);
        res.steer = _ev_action(1);
        _visualizer_ptr->visualize(_planner_ptr->traj_tree, _multi_sv_pred_trajs, length, width, inter_axle_length);

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        int time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        std::cout << "Time difference = " << time_diff << "[ms]" << std::endl;
        res.computation_time = time_diff;
        res.num_total_iter = _planner_ptr->solver_data_.total_iter;
        res.success = _planner_ptr->solver_data_.success;
        res.alpha = _planner_ptr->solver_data_.alpha;

        ROS_INFO("Control input computed.");
        return true;
    }

private:
    void get_predicted_trajectories(vehicle_msgs::GetAction::Request &req)
    {
        ROS_INFO("Start to get motion predictions");
        Eigen::Vector2d lane_widths;
        Eigen::Vector2d norm_vec;

        int num_scenario = req.scenarios.size();
        int num_surrounding_vehicle = req.num_surrounding_vehicle;
        _multi_sv_pred_trajs.resize(num_surrounding_vehicle);
        for (int sv_id = 0; sv_id < num_surrounding_vehicle; ++sv_id)
        {
            _multi_sv_pred_trajs[sv_id].resize(num_scenario);
        }
        _local_guidelines.resize(num_scenario);

        /* num_scenario == num_branch */   
        for (int br_id = 0; br_id < num_scenario; ++br_id)
        {
            std::vector<Vector4d> ev_pred_traj;
            for (int k = 0; k < req.scenarios[br_id].predictions[0].trajs.size(); ++k)
            {              
                for (int sv_id = 0; sv_id < num_surrounding_vehicle; ++sv_id)
                {
                    Vector4d state_tmp;
                    state_tmp << req.scenarios[br_id].predictions[sv_id].trajs[k].x,
                                 req.scenarios[br_id].predictions[sv_id].trajs[k].y,
                                 req.scenarios[br_id].predictions[sv_id].trajs[k].heading,
                                 req.scenarios[br_id].predictions[sv_id].trajs[k].speed;  
                    _multi_sv_pred_trajs[sv_id][br_id].push_back(state_tmp);
                }
                ev_pred_traj.push_back(Vector4d::Zero()); 
            }
            _ev_pred_trajs.push_back(ev_pred_traj);

            for (int k = 0; k < req.scenarios[br_id].reference_input_sequence.size(); ++k)
            {
                Vector2d ref_input;
                ref_input << req.scenarios[br_id].reference_input_sequence[k].acc,
                             req.scenarios[br_id].reference_input_sequence[k].steer;
                _planner_ptr->update_reference_input(ref_input, k, br_id);
            }
           
            for (int k = 0; k < req.scenarios[br_id].local_guideline.size(); ++k)
            {
                _local_guidelines[br_id].emplace_back(req.scenarios[br_id].local_guideline[k].waypoint.x, req.scenarios[br_id].local_guideline[k].waypoint.y);

                lane_widths << req.scenarios[br_id].local_guideline[k].lane_width_left, req.scenarios[br_id].local_guideline[k].lane_width_right;
                norm_vec << req.scenarios[br_id].local_guideline[k].norm_vec.x, req.scenarios[br_id].local_guideline[k].norm_vec.y;
                
                _planner_ptr->traj_tree(k, br_id).A_lin.row(0) =  norm_vec.transpose();
                _planner_ptr->traj_tree(k, br_id).A_lin.row(1) = -norm_vec.transpose();
                _planner_ptr->traj_tree(k, br_id).b_lin(0) =  norm_vec.transpose() * _local_guidelines[br_id][k] + lane_widths(0);
                _planner_ptr->traj_tree(k, br_id).b_lin(1) = -norm_vec.transpose() * _local_guidelines[br_id][k] + lane_widths(1);

                _planner_ptr->update_branch_probability(req.scenarios[br_id].prob, k, br_id);
            }

        }
         /* TODO: */
        _planner_ptr->add_predicted_trajectories(req.scenarios[0].predictions[0].length, 
                                                 req.scenarios[0].predictions[0].width, 
                                                 0.75*req.scenarios[0].predictions[0].length, 
                                                 _ev_pred_trajs);

        for (int sv_id = 0; sv_id < num_surrounding_vehicle; ++sv_id)
        {
            _planner_ptr->add_predicted_trajectories(req.scenarios[0].predictions[sv_id].length,
                                                     req.scenarios[0].predictions[sv_id].width,
                                                     0.75*req.scenarios[0].predictions[sv_id].length,
                                                     _multi_sv_pred_trajs[sv_id]);
        }

    }

private:
    std::vector<std::vector<Vector4d>> _ev_pred_trajs;
    std::vector<std::vector<std::vector<Vector4d>>> _multi_sv_pred_trajs;
    
    Eigen::Matrix<double, 6, 1> _vehicle_state; // num_state: 6
    Eigen::Vector2d _ev_action;

    std::vector<std::vector<Eigen::Vector2d>> _local_guidelines;
    
    ros::ServiceClient _motion_prediction_client;
    ros::ServiceServer _get_action_srv;
    ros::ServiceServer _reset_srv;

    std::shared_ptr<cilqr_tree::BranchMPC> _planner_ptr;
    std::shared_ptr<Visualizer> _visualizer_ptr;

    int _sim_id;
    int _planner_type;
    
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "cilqr_tree_node");
    ros::NodeHandle nh_("~"); // private node handler
    ros::Rate loop_rate(10);
    
    CiLQRNode cilqr_node;
    cilqr_node.init(nh_);
    ros::spin();    
}