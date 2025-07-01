from utils.lat_ctrl import PurePursuit
from utils.lon_ctrl import LongitudinalController
from utils.vehicle import Vehicle
from copy import deepcopy
import numpy as np
import math 
import matplotlib.pyplot as plt


# weight_vel = 0.5
# weight_acc = 0.0
# weight_steer = 0.0
# weight_safety = 1000
# threshold = 1000.0


weight_vel = 0.5
weight_acc = 120.0
weight_steer = 0.0
weight_safety = 1000
threshold = 1200.0

vel_des = 5.0
r_prox = 4.0

class BehaviorPlanner:
    def __init__(self, lon_action_set, guideline_waypoints_set):
        self._lat_ctrl = PurePursuit(K_dd=1.) #TODO: put K_dd in config file
        self._lon_ctrl = LongitudinalController()
                
        self._planning_horizon = 50
        self._dt = 0.1
        self._wheel_base = 3.0
        
        self._vehicle = Vehicle(dt=self._dt, wheel_base=self._wheel_base)
        
        self._lon_action_set = lon_action_set
        self._guideline_waypoints_set = guideline_waypoints_set
                
        self._alpha_acc = 1.0
        self._alpha_steer = 1.0
    
    def _forward_simulation(self, vehicle_state, lon_action, guideline_id):
        acc_pre = 0.
        steer_pre = 0.
        self._vehicle.state = vehicle_state
        state_traj = np.zeros((self._planning_horizon + 1, 4))
        state_traj[0] = self._vehicle.state
        input_traj = np.zeros((self._planning_horizon, 2))
        for k in range(self._planning_horizon):
            acc = self._lon_ctrl.compute_acceleration(self._vehicle.speed, lon_action)
            steer = self._lat_ctrl.compute_steering_angle(self._vehicle.state_rear, self._guideline_waypoints_set[guideline_id])
                        
            acc = self._alpha_acc * acc + (1 - self._alpha_acc) * acc_pre
            steer = self._alpha_steer * steer + (1 - self._alpha_steer) * steer_pre
                        
            self._vehicle.step(acc, steer)
            state_traj[k+1] = self._vehicle.state
            input_traj[k] = np.array([acc, steer])
            
            acc_pre = acc
            steer_pre = steer
        
        return state_traj, input_traj
    
    def _evaluate_cost(self, state_traj, input_traj, sv_state_traj):
        cost = 0.
        for k in range(self._planning_horizon):
            cost_progress = weight_vel * (state_traj[k, 3] - vel_des)**2
            cost_effort = weight_acc * input_traj[k, 0]**2 + weight_steer * input_traj[k, 1]**2
            dist = np.linalg.norm(state_traj[k, :2] - sv_state_traj[k, :2])
            cost_safety = 0.
            if dist < r_prox:
                cost_safety = weight_safety * math.exp(-dist**2 / (2 * r_prox**2))
            cost += cost_progress + cost_effort + cost_safety
        
        cost_progress = weight_vel * (state_traj[-1, 3] - vel_des)**2
        dist = np.linalg.norm(state_traj[-1, :2] - sv_state_traj[-1, :2])
        cost_safety = 0.
        if dist < r_prox:
            cost_safety = weight_safety * math.exp(-dist**2 / (2 * r_prox**2))
        cost += cost_progress + cost_safety
        
        return cost
    
    def get_behavior_plan(self, vehicle_state, sv_state_trajs):
        state_traj_all = []
        input_traj_all = [] 
        min_cost = float('inf')
        best_action_id = 0
        num_lon_action = len(self._lon_action_set)
        
        for guideline_id in range(len(self._guideline_waypoints_set)): 
            for lon_action_id, lon_action in enumerate(self._lon_action_set):
                state_traj, input_traj = self._forward_simulation(vehicle_state, lon_action, guideline_id)
                
                state_traj_all.append(state_traj)
                input_traj_all.append(input_traj)
                cost = 0.
                for sv_state_traj in sv_state_trajs:
                    cost += self._evaluate_cost(state_traj, input_traj, sv_state_traj)
                
                print(f"guideline_id: {guideline_id}, lon_action_id: {lon_action_id}, cost: {cost}")
                
                if cost < min_cost and cost < threshold:
                    min_cost = cost
                    best_action_id = guideline_id * num_lon_action + lon_action_id
            
        return state_traj_all[best_action_id], input_traj_all[best_action_id], best_action_id
    
    def get_action(self, id):
        guideline_id = id // len(self._lon_action_set)
        lon_action_id = id % len(self._lon_action_set) 
        lon_action = self._lon_action_set[lon_action_id]
        
        return lon_action, guideline_id