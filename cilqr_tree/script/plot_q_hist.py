import numpy as np
import matplotlib.pyplot as plt

num_simulation = 1
# num_simulation = 3
# num_simulation = 100
# num_simulation = 500
for i in range(num_simulation):
    q_hist = np.loadtxt('/home/luyao/DataDrive/catkin_ws_trajectory_tree/src/cilqr_tree_new/data/q_hist_' + str(i) + '.txt')
    plt.figure()
    plt.plot(q_hist[:, 0], '-o')
    plt.plot(q_hist[:, 1], '-+')
    plt.plot(q_hist[:, 2], '-*')
    plt.plot(q_hist[:, 3], '-x')
    plt.ylim([-0.2, 1.0])
    plt.legend(['q1', 'q2', 'q3', 'q4'])
    plt.title('q_hist_' + str(i))
    plt.xlabel('iteration')

    plt.savefig('/home/luyao/DataDrive/catkin_ws_trajectory_tree/src/cilqr_tree_new/data/plot/q_hist_' + str(i) + '.png')

    plt.close()