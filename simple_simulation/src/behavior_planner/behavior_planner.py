import numpy as np
import math
import rospy
from utils.lat_ctrl import PurePursuit
from utils.lon_ctrl import LongitudinalController
from utils.vehicle import Vehicle

class BehaviorPlanner:
    def __init__(self, lon_action_set, guideline_waypoints_set, config=None):
        # Load configuration parameters with defaults
        if config is None:
            rospy.logerr("Configuration for BehaviorPlanner is not provided!")
        
        # Cost function weights
        self._weight_vel = config['weight_vel']
        self._weight_acc = config['weight_acc']
        self._weight_steer = config['weight_steer']
        self._weight_safety = config['weight_safety']
        
        # Planning constraints and thresholds
        self._threshold = config['threshold']
        self._vel_des = config['vel_des']
        self._r_prox = config['r_prox']
        
        # Planning horizon and dynamics
        self._planning_horizon = config['planning_horizon']
        self._dt = config['dt']
        self._wheel_base = config['wheel_base']
        
        # Control smoothing parameters
        self._alpha_acc = config['alpha_acc']
        self._alpha_steer = config['alpha_steer']

        self._lat_ctrl = PurePursuit(K_dd=config['K_dd'])
        self._lon_ctrl = LongitudinalController()
        
        self._vehicle = Vehicle(dt=self._dt, wheel_base=self._wheel_base)
        
        self._lon_action_set = lon_action_set
        self._guideline_waypoints_set = guideline_waypoints_set
    
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
            cost_progress = self._weight_vel * (state_traj[k, 3] - self._vel_des)**2
            cost_effort = self._weight_acc * input_traj[k, 0]**2 + self._weight_steer * input_traj[k, 1]**2
            dist = np.linalg.norm(state_traj[k, :2] - sv_state_traj[k, :2])
            cost_safety = 0.
            if dist < self._r_prox:
                cost_safety = self._weight_safety * math.exp(-dist**2 / (2 * self._r_prox**2))
            cost += cost_progress + cost_effort + cost_safety
        
        cost_progress = self._weight_vel * (state_traj[-1, 3] - self._vel_des)**2
        dist = np.linalg.norm(state_traj[-1, :2] - sv_state_traj[-1, :2])
        cost_safety = 0.
        if dist < self._r_prox:
            cost_safety = self._weight_safety * math.exp(-dist**2 / (2 * self._r_prox**2))
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
                
                if cost < min_cost and cost < self._threshold:
                    min_cost = cost
                    best_action_id = guideline_id * num_lon_action + lon_action_id
            
        return state_traj_all[best_action_id], input_traj_all[best_action_id], best_action_id
    
    def get_action(self, id):
        guideline_id = id // len(self._lon_action_set)
        lon_action_id = id % len(self._lon_action_set) 
        lon_action = self._lon_action_set[lon_action_id]
        
        return lon_action, guideline_id