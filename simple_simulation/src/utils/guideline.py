import numpy as np
import numpy.linalg as LA
import math
from math import sqrt
from copy import deepcopy
import itertools

class Guideline:
    def __init__(self, waypoints):
        self._waypoints = waypoints
        self._local_points = None
        self._num_waypoints = waypoints.shape[0]
        self._lengths = np.zeros(self._num_waypoints)
        for i in range(1, self._num_waypoints):
            self._lengths[i] = self._lengths[i-1] + LA.norm(self._waypoints[i] - self._waypoints[i-1])
    
    def get_point(self, s, id_start=1):
        if s < 0:
            return self._waypoints[0]
        elif s > self._lengths[-1]:
            return self._waypoints[-1]
        else:
            for i in range(id_start, self._num_waypoints):
                if s <= self._lengths[i]:
                    if s > self._lengths[i-1]:
                        id = i
                        break
                    else:
                        for j in range(id_start-1, -1, -1):
                            if s >= self._lengths[j]:
                                id = j + 1
                                break
            delta_s = s - self._lengths[id-1]
            point = delta_s * (self._waypoints[id] - self._waypoints[id-1]) / LA.norm(self._waypoints[id] - self._waypoints[id-1]) + self._waypoints[id-1]
            return point, id
    
    def set_local_points(self, local_points):
        self._local_points = local_points
    
    def get_local_points(self):
        return self._local_points
        
    def get_waypoints(self):
        return self._waypoints
    
    def add_waypoint(self, waypoint):
        self._waypoints = np.vstack((self._waypoints, waypoint))
    
    def _project_to_line_segment(self, po, po1, po2):
        # vector from po1 to query.
        relative = po - po1
        # compute the unit direction of this line segment
        direction = po2 - po1
        direction /= LA.norm(direction)
        
        projection = np.dot(relative, direction)
        cross      = np.cross(relative, direction)
        cross_sign = math.copysign(1, cross)
        if projection < 0:
            signed_dist = cross_sign * LA.norm(relative)
            length = 0
            ppo = po1
        elif projection > LA.norm(po2-po1):
            signed_dist = cross_sign * LA.norm(po-po2)
            length = LA.norm(po2-po1)
            ppo = po2
        else:
            signed_dist = cross
            length = projection
            ppo = projection*direction + po1
            
        return signed_dist, length, ppo

    def project(self, po):
        # a simple line search
        min_signed_dist = float('inf')
        for i in range(1, self._num_waypoints):
            signed_dist, length_tmp, ppo_tmp = self._project_to_line_segment(po, 
                                                          self._waypoints[i-1], 
                                                          self._waypoints[i])
            if  abs(signed_dist) < abs(min_signed_dist):
                min_signed_dist = signed_dist
                length = self._lengths[i-1] + length_tmp
                          
        return length, min_signed_dist

def create_laneletmap(lanelet_map_file):
    lat_origin = 0.
    lon_origin = 0.
    print("Loading map...")
    projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
    laneletmap = lanelet2.io.load(lanelet_map_file, projector) 
    
    return laneletmap   

def get_guideline(laneletmap, lanelet_ids):
    # guideline_waypoints = []
    # left_bound_waypoints = [] 
    # right_bound_waypoints = []
    
    guideline_waypoints = [[] for _ in range(len(lanelet_ids))]
    left_bound_waypoints = [[] for _ in range(len(lanelet_ids))]
    right_bound_waypoints = [[] for _ in range(len(lanelet_ids))]
    
    
    # def get_waypoints(line, waypoints):
    #     for p in line:
    #         if waypoints == []:
    #             waypoints.append([p.x, p.y])
    #         else:
    #             p_ = waypoints[-1]
    #             # remove duplicated points
    #             if sqrt((p.x-p_[0])**2 + (p.y-p_[1])**2) > 0.1:
    #                 waypoints.append([p.x, p.y])
    
    def get_waypoints(line, waypoints, id):        
        for p in line:
            if waypoints[id] == []:
                waypoints[id].append([p.x, p.y])
            else:
                p_ = waypoints[id][-1]
                # remove duplicated points
                if sqrt((p.x-p_[0])**2 + (p.y-p_[1])**2) > 0.1:
                    waypoints[id].append([p.x, p.y])
    
    def remove_duplicated_points(waypoints):
        for i in range(1, len(waypoints)):
            p = waypoints[i][0]
            p_ = waypoints[i-1][-1]
            if sqrt((p[0]-p_[0])**2 + (p[1]-p_[1])**2) < 0.1:
               waypoints[i-1].pop(-1)
          
    for id, ll in  enumerate(laneletmap.laneletLayer):
        if id in lanelet_ids:
            cl = ll.centerline
            lb = ll.leftBound
            rb = ll.rightBound
            get_waypoints(cl, guideline_waypoints, lanelet_ids.index(id))
            get_waypoints(lb, left_bound_waypoints, lanelet_ids.index(id))
            get_waypoints(rb, right_bound_waypoints, lanelet_ids.index(id))
    
    remove_duplicated_points(guideline_waypoints)
    remove_duplicated_points(left_bound_waypoints)
    remove_duplicated_points(right_bound_waypoints)
    
    guideline_waypoints = list(itertools.chain.from_iterable(guideline_waypoints))
    left_bound_waypoints = list(itertools.chain.from_iterable(left_bound_waypoints))
    right_bound_waypoints = list(itertools.chain.from_iterable(right_bound_waypoints))
        
    guideline = Guideline(np.array(guideline_waypoints))
    left_bound = Guideline(np.array(left_bound_waypoints))
    right_bound = Guideline(np.array(right_bound_waypoints))    
        
    return guideline, left_bound, right_bound

if __name__ == "__main__":
    waypoints = np.array([
        [0., 0.],
        [1., 0.],
        [1.+math.sqrt(3), 1.]
    ])
    guideline = Guideline(waypoints)
    
    # p = np.array([0.5, 0.2])
    p = np.array([0.5, -2])
    
    # p = np.array([1., 0.2])
    # p = np.array([1., -0.2])
    # p = np.array([1.+math.sqrt(3)/3, 1.])
    s, d = guideline.project(p)
    print(s)
    print(d)