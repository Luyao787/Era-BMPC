
class LongitudinalController:
    def __init__(self, Ks=0.6, Kv=0.5, acc_min=-1.6, acc_max=2.1, dt=0.1):
        self._Ks = Ks
        self._Kv = Kv
        self._acc_min = acc_min
        self._acc_max = acc_max
        self._dt = dt
        
    def compute_acceleration(self, v, v_des):
                
        acc = self._Kv * (v_des - v)
        acc =  max(self._acc_min, min(acc, self._acc_max))
       
        # TODO(Luyao): Make sense?
        # if ds == 0 and acc < 0:
        #     acc = 0.
        # print(f"acc: {acc}")
        
        # if acc < 0:
            # acc = max(acc, -0.9*ds/self._dt)
            
        return acc