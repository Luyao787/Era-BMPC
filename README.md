# Era-BMPC
This repository contains the code for the paper titled "[An Efficient Risk-aware Branch MPC for Automated Driving that is
Robust to Uncertain Vehicle Behaviors](https://ieeexplore.ieee.org/document/10886383)", accepted for presentation at CDC 2024.

## Run
```
roslaunch cilqr_tree cilqr_tree_node.launch 
```

Highway

```
roslaunch simple_simulation simulation.launch scenario:=highway
```

Intersection

```
roslaunch simple_simulation simulation.launch scenario:=intersection
```

## TODO
- [ ] Better README.md
- [ ] Cleanup
- [ ] Better implementation