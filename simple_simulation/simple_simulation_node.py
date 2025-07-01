import rospy
import os
import rospkg
import numpy as np
import itertools
from vehicle_msgs.msg import State, Input, Point2d, Scenario, DynObs, DynObsPt, LocalGuidelinePoint
from vehicle_msgs.srv import GetAction, GetActionRequest, UpdateVehicleStates, UpdateVehicleStatesRequest, ResetMP, ResetMPRequest
from utils.visualization import *
from utils.guideline import Guideline
from utils.vehicle import Vehicle  
from motion_control.motion_control import MotionControl
from behavior_planner.behavior_planner import BehaviorPlanner
import yaml
import argparse


class SimpleSimulationNode:
    def __init__(
        self, 
        centerlines,
        num_vehicles,
        vehicle_sizes,
        multi_vehicle_states,
        sv_predicted_decision_set,
        ego_lon_decision_set,
        config
    ):
        self._centerlines_vis_pub = rospy.Publisher("centerlines", MarkerArray, queue_size=1)
        self._vehicle_vis_pubs = rospy.Publisher("vehicles", MarkerArray, queue_size=10)
        self._timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback)

        self._centerlines = centerlines
        self._centerlines_waypoints = [centerline.get_waypoints() for centerline in centerlines]
        self._num_vehicles = num_vehicles
        self._vehicle_sizes = vehicle_sizes
        self._multi_vehicle_states = multi_vehicle_states
        self._sv_predicted_decision_set = sv_predicted_decision_set
        self._config = config
        
        self._vehicles = []
        self._behavior_planner = BehaviorPlanner(ego_lon_decision_set, self._centerlines_waypoints)

    def _timer_callback(self, event):
        draw_centerlines(self._centerlines_waypoints, self._centerlines_vis_pub)
        
    def _reset_MP(self, sim_id):
        rospy.wait_for_service("/motion_planner/reset", timeout=1.)
        reset_srv = rospy.ServiceProxy("/motion_planner/reset", ResetMP)
        req = ResetMPRequest()
        req.sim_id = sim_id        
        req.planner_type = int(self._config["motion_planner_type"])
        
        res = reset_srv(req)
        
    def _create_vehicles(self, sv_decisions):
        self._vehicles = []
        self._motion_controllers = dict()
        for i in range(self._num_vehicles):
            vehicle = Vehicle(id=f"vehicle_{i}")
            vehicle.length = self._vehicle_sizes[i][0]
            vehicle.width = self._vehicle_sizes[i][1]
            vehicle.wheel_base = self._vehicle_sizes[i][2]  
            vehicle.dt = self._config["sim_dt"]
            vehicle.state = self._multi_vehicle_states[i]
            self._vehicles.append(vehicle)
    
            if i > 0:
                motion_control = MotionControl(
                    sv_decisions[i-1], 
                    wheel_base=vehicle.wheel_base,
                    Kp_lon=self._config["Kp_lon"],
                    acc_max=self._config["acc_max"],
                    steer_max=self._config["steer_max"],
                    K_dd=self._config["K_dd"],
                    dt=self._config["plan_dt"],
                    prediction_horizon=self._config["prediction_horizon"],
                )
                self._motion_controllers[i] = motion_control
                
    def _get_local_guideline(self, ref_state_traj, target_lane_id):
        
        planning_horizon = self._config["prediction_horizon"]
        lane_width = self._config["lane_width"]
        scale = self._config["lane_width_scale"]
        lane_width_left = lane_width / 2
        lane_width_right = lane_width / 2
        idx = 1
        local_guideline = []
        for k in range(planning_horizon + 1):
            s, _ = self._centerlines[target_lane_id].project(ref_state_traj[k, :2])
            point, idx = self._centerlines[target_lane_id].get_point(s, idx)
            local_guideline.append(LocalGuidelinePoint())            
            local_guideline[k].waypoint = Point2d(x=point[0], y=point[1])
            local_guideline[k].lane_width_left = abs(lane_width_left) * scale 
            local_guideline[k].lane_width_right = abs(lane_width_right) * scale
            
            if k > 0:
                tangent_vec = point - point_prev
                norm_vec = np.array([-tangent_vec[1], tangent_vec[0]])
                if np.linalg.norm(norm_vec) > 0.:
                    norm_vec = norm_vec / np.linalg.norm(norm_vec)
                    local_guideline[k-1].norm_vec = Point2d(x=norm_vec[0], y=norm_vec[1])
                else:
                    local_guideline[k-1].norm_vec = local_guideline[k-2].norm_vec                    
            point_prev = point
            
        local_guideline[-1].norm_vec = local_guideline[-2].norm_vec
        
        return local_guideline
    
    def _get_decisions(self, time_step):
        decisions_all = []
        if time_step < self._sv_predicted_decision_set["decision_switch_time"]:
            stage_id = 0
        else:
            stage_id = 1
            
        for i in range(self._num_vehicles - 1):
            decisions_tmp = []
            for decision in self._sv_predicted_decision_set["decisions"][stage_id][i]:
                decisions_tmp.append(
                    {"lon": decision["speed"], "lat": self._centerlines_waypoints[decision["lane_id"]]}
                )
            decisions_all.append(decisions_tmp)
        
        return decisions_all
    
    def _get_predicted_trajectories(self, time_step):
        decisions_all =  self._get_decisions(time_step)
        num_scenario = 1
        decision_ids_all = []
        predicted_trajectories_all = []
        for i in range(1, self._num_vehicles):
            num_decision = len(decisions_all[i-1])
            num_scenario *= num_decision
            decision_ids_all.append(range(num_decision))
            predicted_trajectories = self._motion_controllers[i].get_predicted_trajectories(self._multi_vehicle_states[i], decisions_all[i-1])
            predicted_trajectories_all.append(predicted_trajectories)
        
        self._scenarios = []
        for decision_comb in itertools.product(*decision_ids_all):
            scenario = Scenario()
            scenario.prob = 1.0 / num_scenario  #TODO: need to set the probability properly
            
            predicted_trajectories_one_scenario = []
            for i in range(1, self._num_vehicles):
                scenario.predictions.append(DynObs(length=self._vehicle_sizes[i][0], width=self._vehicle_sizes[i][1]))        
                predicted_trajectory = predicted_trajectories_all[i - 1][decision_comb[i - 1]]
                predicted_trajectories_one_scenario.append(predicted_trajectory)
                for state in predicted_trajectory:
                    scenario.predictions[i-1].trajs.append(DynObsPt(x=state[0], y=state[1], heading=state[2], speed=state[3]))
            
            # Ego vehicle
            best_state_traj, best_input_traj, best_action_id = self._behavior_planner.get_behavior_plan(self._multi_vehicle_states[0], predicted_trajectories_one_scenario)
            scenario.reference_input_sequence = []
            for input in best_input_traj:
                scenario.reference_input_sequence.append(Input(acc=input[0], steer=input[1]))
                
            lon_speed_des, target_lane_id = self._behavior_planner.get_action(best_action_id)            
            local_guideline = self._get_local_guideline(best_state_traj, target_lane_id)            
            scenario.local_guideline = local_guideline
            
            self._scenarios.append(scenario)  

            print(f"desired longitudinal speed: {lon_speed_des}, target lane id: {target_lane_id}")
    
    def _get_ev_action(self):             
        rospy.wait_for_service("/motion_planner/get_ev_action", timeout=1.)
        get_action_srv = rospy.ServiceProxy("/motion_planner/get_ev_action", GetAction)        
        req = GetActionRequest()
        state_msg = State()
        state_msg.x = self._multi_vehicle_states[0][0]
        state_msg.y = self._multi_vehicle_states[0][1]   
        state_msg.heading = self._multi_vehicle_states[0][2]    
        state_msg.vel = self._multi_vehicle_states[0][3]
        state_msg.length = self._vehicle_sizes[0][0]
        state_msg.width = self._vehicle_sizes[0][1]
        state_msg.inter_axle_length = self._vehicle_sizes[0][2]
        req.ego_state = state_msg
 
        req.scenarios = self._scenarios
        
        req.num_surrounding_vehicle = self._num_vehicles - 1
        
        res = get_action_srv(req)
        
        ego_trajs_np = np.zeros((len(res.ego_trajs), 4))
        for i, traj in enumerate(res.ego_trajs):
            ego_trajs_np[i, 0] = traj.x
            ego_trajs_np[i, 1] = traj.y
            ego_trajs_np[i, 2] = traj.heading
            ego_trajs_np[i, 3] = traj.vel
     
        return res.acc, res.steer, res.alpha
        
    def _update(self, time_step):
        sim_dt = self._config["sim_dt"]
        plan_dt = self._config["plan_dt"]
        for i in range(self._num_vehicles):
            if i == 0:
                if self._count < self._ratio_dt:
                    acc = self._acc_ego
                    steer = self._steer_ego
                else: # Replan 
                    rospy.loginfo(f"Replan at time step: {time_step}")
                    self._count = 0
                    self._get_predicted_trajectories(time_step)
                    acc, steer, alpha = self._get_ev_action()               
                    self._acc_ego = acc
                    self._steer_ego = steer
                self._count += 1
            else:
                acc, steer = self._motion_controllers[i].get_action(self._vehicles[i].state_rear, 
                                                                    self._centerlines_waypoints[i], time_step * sim_dt)   
            
            self._vehicles[i].step(acc, steer)
            self._multi_vehicle_states[i] = self._vehicles[i].state
    
    def run_one_simulation(self, sv_decisions):
        self._ratio_dt = int(self._config["plan_dt"] / self._config["sim_dt"])
        self._count = self._ratio_dt
        self._acc_ego = 0.
        self._steer_ego = 0.
        
        num_sim_step = self._config["num_sim_steps"]
        dt = self._config["sim_dt"]
        self._create_vehicles(sv_decisions)
        rate = rospy.Rate(int(1.0 / dt))
        
        for time_step in range(num_sim_step):
            draw_vehicles( self._vehicle_vis_pubs , self._multi_vehicle_states, self._vehicle_sizes)            
            self._update(time_step)
            rate.sleep()
            
    def run(
        self, 
        sv_decisions,
        num_simulation=1,
    ):        
        for sim_id in range(num_simulation):
            print(f"----------Simulation: {sim_id}----------")  
            self._reset_MP(sim_id)          
            self.run_one_simulation(sv_decisions)
            
        rospy.spin()
        

def load_example_config(name, config_dir):
    config_path = os.path.join(config_dir, f"{name}.yaml")
    
    if not os.path.exists(config_path):
        rospy.logerr(f"Scenario config file not found: {config_path}")
        return None
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config['centerline_waypoints'] = [np.array(waypoints) for waypoints in config['centerline_waypoints']]
    config['multi_vehicle_states'] = [np.array(state) for state in config['multi_vehicle_states']]

    return config
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario')
    args, unknown = parser.parse_known_args()
      
    rospy.init_node("simple_simulation_node")
    sim_config = rospy.get_param("simple_simulation")
    
    print(f"sim_config:\n {sim_config}")
    
    root_dir = os.path.dirname(rospkg.RosPack().get_path('simple_simulation'))
    
    # Load example configuration
    config_dir = os.path.join(root_dir, "simple_simulation", "config")
    example_config = load_example_config(args.scenario, config_dir)
    
    if example_config is None:
        rospy.logerr("Failed to load example configuration!")
        exit(1)
    
    print(f"Loaded example: {example_config['example_name']}")
    
    centerline_waypoints = example_config['centerline_waypoints']
    centerlines = [Guideline(waypoints) for waypoints in centerline_waypoints]
    
    num_vehicles = example_config['num_vehicles']
    multi_vehicle_states = example_config['multi_vehicle_states']
    vehicle_sizes = example_config['vehicle_sizes']
    
    sv_predicted_decision_set = {
        "decision_switch_time": example_config['sv_predicted_decision_set']['decision_switch_time'],
        "decisions": [
            list(example_config['sv_predicted_decision_set']['decisions']['stage_1']),
            list(example_config['sv_predicted_decision_set']['decisions']['stage_2'])
        ]
    }
    sv_decisions = example_config['sv_decisions']
    
    ego_lon_decision_set = example_config['ego_lon_decision_set']
        
    # # hard-coded lanes
    # centerline_waypoints = [
    #     np.array([
    #         [0., -100.], 
    #         [0.,  0.], 
    #         [0.,  500.]]),
    #     np.array([
    #         [-100., 0.], 
    #         [ 0.,   0.], 
    #         [500.,  0.]]),
    # ]
    # centerlines = [
    #     Guideline(centerline_waypoints[0]),
    #     Guideline(centerline_waypoints[1]),
    # ]
    
    
    # num_vehicles = 2
    # multi_vehicle_states = [
    #     np.array([1., -50., np.pi / 2, 5.]),
    #     np.array([-35, 1.5, 0., 5.]),
    # ]
    # vehicle_sizes = [
    #     [4.0, 2.0, 3.0],
    #     [4.0, 2.0, 3.0],
    # ]
    # sv_predicted_decision_set = {
    #     "decision_switch_time": 1.0,
    #     "decisions": 
    #     [
    #         # Stage 1
    #         [
    #             [{"speed": 5.0, "lane_id": 1}, {"speed": 1.0, "lane_id": 1}],   # SV1
    #             # [],   SV2 
    #         ],
    #         # Stage 2
    #         [
    #             [{"speed": 5.0, "lane_id": 1}, {"speed": 1.0, "lane_id": 1}],  
    #         ]
    #     ]
    # }
    # sv_decisions = [
    #     [5., 5., 1.]    # v_des_stage1, v_des_stage2, switch_time
    # ]
    
    sim_node = SimpleSimulationNode(
        centerlines,
        num_vehicles,
        vehicle_sizes,
        multi_vehicle_states,
        sv_predicted_decision_set,
        ego_lon_decision_set,
        sim_config
    )
    sim_node.run(sv_decisions)    
    