from utils.lat_ctrl import PurePursuit
from utils.vehicle import Vehicle
from copy import deepcopy
import numpy as np

class MotionControl:
    def __init__(
        self,
        decision, 
        wheel_base,
        Kp_lon,
        acc_max,
        steer_max,
        K_dd,
        dt,
        prediction_horizon,
    ):
        self._decision = decision
        self._lat_ctrl = PurePursuit(K_dd=K_dd, wheel_base=wheel_base)
        self._wheel_base = wheel_base
        self._Kp_lon = Kp_lon
        self._acc_max = acc_max
        self._steer_max = steer_max
        
        self._horizon = prediction_horizon
        self._dt = dt

    def compute_acceleration(self, state_rear, t):
        v = state_rear[3]
        switch_time = self._decision[2]
        v_des_stage1, v_des_stage2 = self._decision[0], self._decision[1]
        v_des = v_des_stage1 if t < switch_time else v_des_stage2
        acc = self._Kp_lon * (v_des - v)
        
        return np.clip(acc, -self._acc_max, self._acc_max)
        
    def get_action(self, state_rear, centerline_waypoints, t):
        acc = self.compute_acceleration(state_rear, t)
        steer = self._lat_ctrl.compute_steering_angle(state_rear, centerline_waypoints)        
        
        return acc, steer
    
    def get_predicted_trajectories(self, init_state, policies):
        predicted_trajectories = []
        dim_x = init_state.shape[0]
        for policy in policies: 
            vehicle = Vehicle(dt=self._dt, wheel_base=self._wheel_base)
            vehicle.state = deepcopy(init_state)
            predicted_trajectory = np.zeros((self._horizon + 1, dim_x))
            predicted_trajectory[0] = deepcopy(vehicle.state)
            for k in range(self._horizon):
                acc = self._Kp_lon * (policy["lon"] - vehicle.speed)
                acc = np.clip(acc, -self._acc_max, self._acc_max)                
                steer = self._lat_ctrl.compute_steering_angle(vehicle.state_rear, policy["lat"])
                vehicle.step(acc, steer)
                predicted_trajectory[k + 1] = deepcopy(vehicle.state)
            
            predicted_trajectories.append(predicted_trajectory)
            
        return predicted_trajectories

    # def get_reference_input_sequence(self, init_state, guideline, T, dt, v_des):
    #     vehicle = Vehicle(dt=dt, wheel_base=self._wheel_base)
    #     vehicle.state = deepcopy(init_state)
    #     reference_input_sequences = np.zeros((T, 2))
    #     for k in range(T):
    #         acc = self._lon_ctrl.compute_acceleration(vehicle.speed, v_des)
    #         steer = self._lat_ctrl.compute_steering_angle(vehicle.state_rear, guideline)
    #         reference_input_sequences[k] = np.array([acc, steer])
                        
    #         vehicle.step(acc, steer)

    #     return reference_input_sequences

