import numpy as np
from dataclasses import dataclass

@dataclass
class Vehicle:
    """
    The dynamics of a kinematic bicycle model (rear wheel) with state
    - 'px': x position
    - 'py': y position
    - 'heading': heading angle
    - 'speed': speed
    """
    # id
    id: str = "" # ['ego_vehicle', 'vehicle_1', 'vehicle_2', ...]
    
    # state 
    px: float = 0.
    py: float = 0.
    heading: float = 0.
    speed: float = 0.
    
    px_rear: float = 0.
    py_rear: float = 0.
    
    # control
    acc: float = 0.
    steering_angle: float = 0.
    
    # parameters
    # wheel_base: float = wheel_base
    dt: float = None
    wheel_base: float = None
    length: float = None
    width: float = None
    
    @property
    def state(self):
        # change coordinate
        self.px = self.px_rear + np.cos(self.heading)*(self.wheel_base/2)
        self.py = self.py_rear + np.sin(self.heading)*(self.wheel_base/2) 
        return np.array([float(self.px),
                         float(self.py),
                         float(self.heading),
                         float(self.speed)])
        
    @property
    def state_rear(self):
        return np.array([float(self.px_rear),
                         float(self.py_rear),
                         float(self.heading),
                         float(self.speed)])
        
    # @property
    # def state_mid(self):
    #     length_difference = (wheel_base + overhang_rear + overhang_front) / 2 - overhang_rear 
    #     return np.array([float(self.px + np.cos(self.heading) * length_difference),
    #                      float(self.py + np.sin(self.heading) * length_difference),
    #                      float(self.heading),
    #                      float(self.speed)])
    
    @state.setter
    def state(self, state):
        self.px      = float(state[0])
        self.py      = float(state[1])
        self.heading = float(state[2])
        self.speed   = float(state[3])
        
        self.px_rear = self.px + np.cos(self.heading)*(-self.wheel_base/2)
        self.py_rear = self.py + np.sin(self.heading)*(-self.wheel_base/2)
    
    @state_rear.setter
    def state_rear(self, state):
        self.px_rear = float(state[0])
        self.py_rear = float(state[1])
        self.heading = float(state[2])
        self.speed   = float(state[3])
        
    def step(self, 
             acc: float, 
             steering: float):
        # RK4
        # k1 = self._sys_dynamics(self.state, acc, steering)
        # k2 = self._sys_dynamics(self.state + self.dt/2*k1, acc, steering)
        # k3 = self._sys_dynamics(self.state + self.dt/2*k2, acc, steering)
        # k4 = self._sys_dynamics(self.state + self.dt*k3, acc, steering)
        # self.state += self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
        
        k1 = self._sys_dynamics(self.state_rear, acc, steering)
        k2 = self._sys_dynamics(self.state_rear + self.dt/2*k1, acc, steering)
        k3 = self._sys_dynamics(self.state_rear + self.dt/2*k2, acc, steering)
        k4 = self._sys_dynamics(self.state_rear + self.dt*k3, acc, steering)
        self.state_rear += self.dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
    def _sys_dynamics(self, 
                      state: np.ndarray, 
                      acc: float, 
                      steering: float):
        state_dot = np.zeros_like(state)
        state_dot[0] = state[3] * np.cos(state[2])
        state_dot[1] = state[3] * np.sin(state[2])
        state_dot[2] = state[3] * np.tan(steering) / self.wheel_base
        state_dot[3] = acc
        return state_dot