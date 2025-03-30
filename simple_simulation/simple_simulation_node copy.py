import rospy
import rospkg
import os
import rospkg
import lanelet2
import numpy as np
import math
import itertools
from vehicle_msgs.msg import State, Input, Point2d, Scenario, DynObs, DynObsPt, LocalGuidelinePoint
from vehicle_msgs.srv import GetAction, GetActionRequest, UpdateVehicleStates, UpdateVehicleStatesRequest, ResetMP, ResetMPRequest
from utils.visualization import *
from utils.guideline import get_guideline
from utils.vehicle import Vehicle  
from copy import deepcopy
from motion_control.motion_control import MotionControl
from behavior_planner.behavior_planner import BehaviorPlanner
from random import uniform, seed
import matplotlib.pyplot as plt
import json

SIM_RATE = 10
WAIT_TIME = 1.0

# -----------
# 0: Risk-neutral, 1: Risk-aware
MOTION_PLANNER_TYPE = 0
# MOTION_PLANNER_TYPE = 1

# TEST_SCENARIO = 1
TEST_SCENARIO = 2

# NUM_SIMULATION = 1
# NUM_SIMULATION = 2
# NUM_SIMULATION = 100
NUM_SIMULATION = 500

NUM_SIM_STEP = 1
# NUM_SIM_STEP = 5
# NUM_SIM_STEP = 55
# NUM_SIM_STEP = 100

# TIME_STEP_POLICY_CHANGE = 10
# TIME_STEP_POLICY_CHANGE = 13
# TIME_STEP_POLICY_CHANGE = 16
TIME_STEP_POLICY_CHANGE = 30

# scale = 1.0 # for monte carlo simulation TS1
scale = 1.1 # TS2
# scale = 100 

class SimpleSimulationNode:
    def __init__(
        self, 
        laneletmap, 
        num_vheicles, 
        nominal_multi_vehicle_states, 
        vehicle_sizes,
        num_sim_step,
        sv_predicted_policies,
        sv_actual_policies,  
    ):
        self._road_vis_pub = rospy.Publisher("roads", MarkerArray, queue_size=10)
        self._road_bouunds_vis_pub = rospy.Publisher("road_bounds", MarkerArray, queue_size=10)
        self._guidelines_vis_pub = rospy.Publisher("guidelines", MarkerArray, queue_size=10)
        self._vehicle_vis_pubs = rospy.Publisher("vehicles", MarkerArray, queue_size=10)
        self._timer = rospy.Timer(rospy.Duration(0.1), self._timer_callback)
        
        self._cl_trajs_pub = rospy.Publisher("cl_trajs", MarkerArray, queue_size=10)
        
        self._laneletmap = laneletmap
        self._guidelines = []
        self._left_bounds = []
        self._right_bounds = []
        self._get_guidelines()
        
        self._behavior_planner = BehaviorPlanner(self._guidelines[0].get_waypoints())
                
        self._motion_ctrl_ego = MotionControl()
        self._motion_ctrl = MotionControl()
        self._scenarios = []
        
        # TODO:
        self._planning_horizon = 50
        self._dt = 0.1
        self._num_sim_step = num_sim_step
         
        self._num_vehicles = num_vheicles
        self._nominal_multi_vehicle_states = nominal_multi_vehicle_states
        self._vehicle_sizes = vehicle_sizes

        self._multi_vehicle_states = deepcopy(self._nominal_multi_vehicle_states)
        
        self._sv_predicted_policies = sv_predicted_policies
        self._sv_actual_policies = sv_actual_policies
    
    def _reset_MP(self, sim_id, planner_type):
        rospy.wait_for_service("/motion_planner/reset", timeout=1.)
        reset_srv = rospy.ServiceProxy("/motion_planner/reset", ResetMP)
        req = ResetMPRequest()
        req.sim_id = sim_id        
        req.planner_type = int(planner_type)
        
        res = reset_srv(req)
        
    def _get_policies(self, time_step):
        policies_all = []
        if time_step < self._sv_predicted_policies["time_step_policy_change"]:
            for i in range(self._num_vehicles - 1):
                policies_all_tmp = []
                for policy in self._sv_predicted_policies["policies"][0][i]:
                    policies_all_tmp.append(
                        {"lon": policy["speed"], "lat": self._guidelines[policy["lane_id"]].get_waypoints()}
                    )
                policies_all.append(policies_all_tmp)
        else:
            for i in range(self._num_vehicles - 1):
                policies_all_tmp = []
                for policy in self._sv_predicted_policies["policies"][1][i]:
                    policies_all_tmp.append(
                        {"lon": policy["speed"], "lat": self._guidelines[policy["lane_id"]].get_waypoints()}
                    )
                policies_all.append(policies_all_tmp)
        
        return policies_all
    
    def _create_vehicles(self):
        self._vehicles = []
        for i in range(self._num_vehicles):
            vehicle = Vehicle(id=f"vehicle_{i}")
            vehicle.length = self._vehicle_sizes[i][0]
            vehicle.width = self._vehicle_sizes[i][1]
            vehicle.wheel_base = self._vehicle_sizes[i][2]  
            vehicle.dt = self._dt
            vehicle.state = self._multi_vehicle_states[i]
            self._vehicles.append(vehicle)
                       
    def _timer_callback(self, event):
        draw_lanelet_map(self._laneletmap, self._road_vis_pub)
        draw_guidelines(self._guidelines, self._left_bounds, self._right_bounds, self._guidelines_vis_pub)
            
    def _get_guidelines(self):
        # Hard-coded lanelet ids
        lanelet_ids_list = [
            [10, 38, 45, 47], 
            [17, 21, 27, 28, 29],
            [58, 22, 45],
            [58, 37, 11],
        ]
        for lanelet_ids in lanelet_ids_list:
            guideline, left_bound, right_bound = get_guideline(self._laneletmap, lanelet_ids)
            self._guidelines.append(guideline)
            self._left_bounds.append(left_bound)
            self._right_bounds.append(right_bound)
            
    def _get_local_guideline(self, ref_state_traj):
        # EV
        id = 1
        local_guideline = []
        for k in range(self._planning_horizon + 1):
            s, _ = self._guidelines[0].project(ref_state_traj[k, :2])
            point, id = self._guidelines[0].get_point(s, id)
            local_guideline.append(LocalGuidelinePoint())
            local_guideline[k].waypoint = Point2d(x=point[0], y=point[1])
            
            _, lane_width_left = self._left_bounds[0].project(point)
            local_guideline[k].lane_width_left = abs(lane_width_left) * scale 
        
            _, lane_width_right = self._right_bounds[0].project(point)
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
        # print("----")
        local_guideline[-1].norm_vec = local_guideline[-2].norm_vec
        # draw_local_road_bounds(local_guideline, self._road_bouunds_vis_pub)
        
        return local_guideline
    
                             
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
        # TODO: Path
        if SAVE_DATA:
            np.savetxt("/home/luyao/DataDrive/catkin_ws_trajectory_tree/src/simple_simulation/data/ego_state_trajs/ego_trajs.txt", ego_trajs_np)
        
        solver_data = {"success": res.success, 
                       "computation_time": res.computation_time,
                       "total_iter": res.num_total_iter}
        if SAVE_DATA:
            with open('/home/luyao/DataDrive/catkin_ws_trajectory_tree/src/simple_simulation/data/paper_results/solver_data.txt', 'a') as file: 
                file.write(json.dumps(solver_data))
                file.write(os.linesep)

        
        return res.acc, res.steer, res.alpha
        
    def _get_predicted_trajectories(self, time_step):
        
        policies_all = self._get_policies(time_step)
        
        num_scenario = 1
        policy_ids_all = []
        predicted_trajectories_all = []
        for i in range(1, self._num_vehicles):
            num_policy = len(policies_all[i-1])
            num_scenario *= num_policy
            policy_ids_all.append(range(num_policy))

            predicted_trajectories = self._motion_ctrl.get_predicted_trajectories(self._multi_vehicle_states[i], policies_all[i-1])
            predicted_trajectories_all.append(predicted_trajectories)

        self._scenarios = []
        # self._vel_des_all = []
        for policy_comb in itertools.product(*policy_ids_all):                        
            scenario = Scenario()
            
            # TODO: Need to set the probability properly
            scenario.prob = 1.0 / num_scenario
            
            predicted_trajectories_one_scenario = []
            for i in range(1, self._num_vehicles):
                scenario.predictions.append(DynObs(length=self._vehicle_sizes[i][0], width=self._vehicle_sizes[i][1]))        
                predicted_trajectory = predicted_trajectories_all[i - 1][policy_comb[i - 1]]
                predicted_trajectories_one_scenario.append(predicted_trajectory)
                
                for state in predicted_trajectory:
                    scenario.predictions[i-1].trajs.append(DynObsPt(x=state[0], y=state[1], heading=state[2], speed=state[3]))
            
            # Ego vehicle
            best_state_traj, best_input_traj, best_action_id = self._behavior_planner.get_behavior_plan(self._multi_vehicle_states[0], predicted_trajectories_one_scenario)
            scenario.reference_input_sequence = []
            for input in best_input_traj:
                scenario.reference_input_sequence.append(Input(acc=input[0], steer=input[1]))
                
            vel_des = self._behavior_planner.get_action(best_action_id)
            print(f"vel_des: {vel_des}")
            # local_guideline = self._get_local_guideline(np.array([self._multi_vehicle_states[0][0], self._multi_vehicle_states[0][1]]), vel_des)
            local_guideline = self._get_local_guideline(best_state_traj)
            scenario.local_guideline = local_guideline
           
            # self._vel_des_all.append(self._behavior_planner.get_action(best_action_id))
            
            print(f"best_action_id: {best_action_id}")
            
            # scenario.reference_input_sequence = []
            # reference_input_sequence = self._motion_ctrl_ego.get_reference_input_sequence(self._multi_vehicle_states[0], self._guidelines[0].get_waypoints(), self._planning_horizon, self._dt, self._vel_des)
            # for input in reference_input_sequence:
            #     scenario.reference_input_sequence.append(Input(acc=input[0], steer=input[1]))
            
            self._scenarios.append(scenario)   
    
    @staticmethod
    def generate_random_initial_states(init_state, num_samples):
        initial_states = []
        vel = init_state[3]
        seed(100)
        for _ in range(num_samples):
            initial_state_tmp = init_state + np.array([uniform(-1.0, 1.0), uniform(-3.0, 3.0), 0.0, uniform(-0.1*vel, 0.1*vel)])
            initial_states.append(initial_state_tmp)
        
        root_dir = os.path.dirname(rospkg.RosPack().get_path('simple_simulation'))
        data_dir = os.path.join(root_dir, "simple_simulation/data")
        np.savetxt(data_dir + "/ego_initial_states.txt", np.vstack(initial_states))
                       
        return initial_states
    
    def run_one_simulation(self, planner_type, save_traj=False):
        rospy.sleep(WAIT_TIME)
        self._create_vehicles()
        rate = rospy.Rate(SIM_RATE)
        
        cl_ev_state_traj = np.zeros((self._num_sim_step + 1, 4))
        cl_ev_input_traj = np.zeros((self._num_sim_step, 2))
        cl_ev_state_traj[0, :] = self._multi_vehicle_states[0]
        
        cl_sv_state_trajs = []
        for i in range(1, self._num_vehicles):
            cl_sv_state_trajs.append(np.zeros((self._num_sim_step + 1, 4)))
            cl_sv_state_trajs[i-1][0, :] = self._multi_vehicle_states[i]
        
        for step in range(self._num_sim_step):
            print("Step: ", step)
            update_objects_plot( self._vehicle_vis_pubs , self._multi_vehicle_states, self._vehicle_sizes)            
            for i in range(self._num_vehicles):
                if i == 0:
                    self._get_predicted_trajectories(time_step=step)
                    print(f"ego state: {self._multi_vehicle_states[i]}")
                    acc, steer, alpha = self._get_ev_action()
                    print(f"acc: {acc}, steer: {steer}")
                    # acc = 0.
                    # steer = 0.
                    # self._vehicles[i].step(acc, steer)    
                    # self._multi_vehicle_states[i] = self._vehicles[i].state
                else:
                    print(f"vehicle state: {self._multi_vehicle_states[i]}")
                    # acc, steer = self._motion_ctrl.get_action(self._vehicles[i].state_rear, self._guidelines[i].get_waypoints()) 
                    acc, steer = self._motion_ctrl.get_action(self._vehicles[i].state_rear,
                                                              self._sv_actual_policies[i-1]["speed"], 
                                                              self._guidelines[self._sv_actual_policies[i-1]["lane_id"]].get_waypoints()) 

                self._vehicles[i].step(acc, steer)    
                self._multi_vehicle_states[i] = self._vehicles[i].state
                
                cl_sv_state_trajs[i-1][step + 1, :] = self._multi_vehicle_states[i]
            
            cl_ev_state_traj[step + 1, :] = self._multi_vehicle_states[0]
            
            rate.sleep()
        
        if save_traj:
            root_dir = os.path.dirname(rospkg.RosPack().get_path('simple_simulation'))
            data_dir = os.path.join(root_dir, "simple_simulation/data")
            if planner_type == 0:
                np.savetxt(data_dir + f"/ego_state_trajs_cl/planner_type_{planner_type}_ego_state_trajs_cl.txt", cl_ev_state_traj)
            elif planner_type == 1:
                np.savetxt(data_dir + f"/ego_state_trajs_cl/planner_type_{planner_type}_{alpha}_ego_state_trajs_cl.txt", cl_ev_state_traj)
            else:
                assert False, "Invalid planner type"
            for i in range(1, self._num_vehicles):
                np.savetxt(data_dir + f"/ego_state_trajs_cl/sv_state_trajs_cl_{i-1}.txt", cl_sv_state_trajs[i-1])
    
    def _empty_files(self):
        # TODO: Path
        with open('/home/luyao/DataDrive/catkin_ws_trajectory_tree/src/simple_simulation/data/paper_results/solver_data.txt', 'w') as file: 
            pass
                   
    def run(self, 
            ego_init_states=[], 
            num_simulation=1,
            motion_planner_type=0):
             
        if num_simulation > 1:
            rospy.loginfo("Generating random initial states for the ego vehicle.")
            ego_init_states = self.generate_random_initial_states(self._multi_vehicle_states[0], num_simulation)
        else:
            if len(ego_init_states) == 0:
                rospy.logerr("Please provide an initial state for the ego vehicle.")
        
        # if len(ego_init_states) == 0:
        #     ego_init_states = self.generate_random_initial_states(self._multi_vehicle_states[0], num_simulation)            
        
        for sim_id in range(num_simulation):
            print(f"----------Simulation: {sim_id}----------")
            
            self._reset_MP(sim_id, motion_planner_type)
            
            self._multi_vehicle_states[0] = ego_init_states[sim_id]
            for i in range(1, self._num_vehicles):
                self._multi_vehicle_states[i] = self._nominal_multi_vehicle_states[i]
            
            self.run_one_simulation(motion_planner_type, save_traj=True)
            
        rospy.spin()

         
if __name__ == "__main__":
    rospy.init_node("simple_simulation_node")
    root_dir = os.path.dirname(rospkg.RosPack().get_path('simple_simulation'))

    # hard-coded lanes
    centerlines = [
        np.array([
            [0., -100.], 
            [0.,  0.], 
            [0.,  500.]]),
        np.array([
            [-100., 0.], 
            [ 0.,   0.], 
            [500.,  0.]]),
    ]
    
  
    
    # ----------------------------------------------------------------
    # 2 vehicles
    # num_vehicles = 2
    # nominal_multi_vehicle_states = [
    #     np.array([ 997.52057136, 1004.31917628,   -1.60576457,    4.85370839]),
    #     np.array([1015.5, 987., np.deg2rad(175.0), 5.0]),
    # ]
    # vehicle_sizes = [(4.0, 2.0, 3.0), (4.0, 2.0, 3.0)]
    
    # sv_predicted_policies = {
    #     "time_step_policy_change": 10,
    #     "policies": [
    #         [
    #             [{"speed": 5.0, "lane_id": 1}, {"speed": 1.0, "lane_id": 1}],    
    #         ],
    #         # [
    #         #     [{"speed": 5.0, "lane_id": 1}, {"speed": 5.0, "lane_id": 1}],   
    #         # ]
    #         [
    #             [{"speed": 0.0, "lane_id": 1}, {"speed": 0.0, "lane_id": 1}],   
    #         ]
    #     ]
    # }
    # sv_actual_policies = [
    #     {"speed": 0.0, "lane_id": 1}, 
    # ]
    # ----------------------------------------------------------------
    
    # ----------------------------------------------------------------
    # 3 vehicles
    num_vehicles = 3
    nominal_multi_vehicle_states = [
        # np.array([997.6, 1025.8, -math.pi/2, 5.0]),
        # np.array([997.6, 1005.8, -math.pi/2, 5.0]),
        
        # np.array([998.96345612, 989.1057916, -1.40729123, 3.42226529]),
        
        # np.array([ 997.52057136, 1004.31917628,   -1.60576457,    4.85370839]),
        np.array([997.43, 1003.0, -1.61, 4.93]),
    
        # np.array([ 997.52057136, 1004.31917628,   -1.60576457,    3]),

        np.array([1015.5, 986.5, np.deg2rad(175.0), 5.0]),
        np.array([975.0, 985.0, np.deg2rad(-5.0), 5.0]),
    ]
    vehicle_sizes = [(4.0, 2.0, 3.0), 
                     (4.0, 2.0, 3.0), 
                     (4.0, 2.0, 3.0)]
    
    if TEST_SCENARIO == 1:
        sv_predicted_policies = {
            "time_step_policy_change": TIME_STEP_POLICY_CHANGE,
            "policies": [
                [
                    [{"speed": 5.0, "lane_id": 1}, {"speed": 1.0, "lane_id": 1}],
                    [{"speed": 5.0, "lane_id": 2}, {"speed": 1.0, "lane_id": 2}],        
                ],
                [
                    # 1
                    [{"speed": 0.0, "lane_id": 1}, {"speed": 0.0, "lane_id": 1}],
                    [{"speed": 5.0, "lane_id": 2}, {"speed": 5.0, "lane_id": 2}],      
                    # 2
                    # [{"speed": 5.0, "lane_id": 1}, {"speed": 5.0, "lane_id": 1}],
                    # [{"speed": 5.0, "lane_id": 2}, {"speed": 5.0, "lane_id": 2}],      
                ]
            ]
        }
        sv_actual_policies = [
            # 1 
            {"speed": 0.0, "lane_id": 1},
            {"speed": 5.0, "lane_id": 2},  
            # 2
            # {"speed": 5.0, "lane_id": 1},
            # {"speed": 5.0, "lane_id": 2},  
        ]
    elif TEST_SCENARIO == 2: 
        sv_predicted_policies = {
            "time_step_policy_change": TIME_STEP_POLICY_CHANGE,
            "policies": [
                [
                    [{"speed": 5.0, "lane_id": 1}, {"speed": 1.0, "lane_id": 1}],
                    [{"speed": 5.0, "lane_id": 2}, {"speed": 5.0, "lane_id": 3}],       
                    # [{"speed": 1.0, "lane_id": 1}, {"speed": 1.0, "lane_id": 1}],
                    # [{"speed": 5.0, "lane_id": 3}, {"speed": 5.0, "lane_id": 3}],        
                ],
                [
                    [{"speed": 0.0, "lane_id": 1}, {"speed": 0.0, "lane_id": 1}],
                    [{"speed": 5.0, "lane_id": 2}, {"speed": 5.0, "lane_id": 2}],      
                ]
            ]
        }
        
        sv_actual_policies = [
            {"speed": 0.0, "lane_id": 1},
            {"speed": 5.0, "lane_id": 2},  
        ]
    else:
        assert False, "Invalid test scenario"
    
    # ----------------------------------------------------------------  
    ego_init_states = [        
        # Used for closed-loop trajectory simulation
        # np.array([9.983241560120058011e+02, 1.002012104655640087e+03, -1.610000000000000098e+00, 5.8])
        # --------------------------
        
        # 105
        # np.array([9.967526272206013118e+02, 1.000297953681930153e+03, -1.610000000000000098e+00, 5.133277435253204679e+00])
        # 205
        # np.array([9.966620760936378929e+02, 1.000204485684453061e+03, -1.610000000000000098e+00, 5.093114023976141880e+00])

        # np.array([9.968244180090987356e+02, 1.000383016882980996e+03, -1.610000000000000098e+00, 5.334093818516103980e+00])
        # np.array([9.968188834487409622e+02, 1.000390446166143761e+03, -1.610000000000000098e+00, 5.290059652066903872e+00])
        # np.array([9.967547730423597159e+02, 1.001203209786476691e+03, -1.610000000000000098e+00, 5.320688982039294856e+00])

        # np.array([9.970515235325585763e+02, 1.000428677267196917e+03, -1.610000000000000098e+00, 5.058130374966241405e+00])
        # np.array([9.974831564012209810e+02, 1.000452937568310176e+03, -1.610000000000000098e+00, 5.371438562096799707e+00])
        # np.array([9.967114044861940329e+02, 1.000333068819523078e+03, -1.610000000000000098e+00, 5.061067449319119227e+00])

        # np.array([9.967526272206013118e+02, 1.000297953681930153e+03, -1.610000000000000098e+00, 5.133277435253204679e+00])

        np.array([9.973069085501022073e+02, 1.000206162831567553e+03, -1.610000000000000098e+00, 5.414375525491722740e+00
])
        

    ]
    # ego_init_states = []
    
            
    sim_node = SimpleSimulationNode(laneletmap, 
                                    num_vehicles, 
                                    nominal_multi_vehicle_states, 
                                    vehicle_sizes, NUM_SIM_STEP, 
                                    sv_predicted_policies,
                                    sv_actual_policies)
    
    sim_node.run(ego_init_states, 
                 NUM_SIMULATION,
                 MOTION_PLANNER_TYPE)

    # sim_node.visualize()
    
    