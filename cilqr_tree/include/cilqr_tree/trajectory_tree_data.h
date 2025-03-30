#pragma once
 
#include <vector>
#include "eigen_types.h"
#include "knotpoint_data.h"

namespace cilqr_tree
{

template<int n, int m>
class TrajTree
{ 
private:
    std::vector<std::vector<KnotpointData<n, m>>> traj_tree_;

public:
    int N;
    int N_sh;
    int num_branch;
    
    double num_state = n;
    double num_input = m;

public:
    TrajTree() = default;
    TrajTree(int num_branch,
             int N, 
             int N_sh) : N(N), N_sh(N_sh), num_branch(num_branch)
    {
        for (int br_id = 0; br_id < num_branch; ++br_id)
        {
            traj_tree_.emplace_back(std::vector<KnotpointData<n, m>>());
            for (int k = 0; k < N + 1; ++k)
            {
                traj_tree_[br_id].emplace_back(k == N, k, br_id);
            }
        }
    }

    inline KnotpointData<n, m>& operator() (int k, int branch_id) { return traj_tree_[branch_id][k]; }

    inline void update_state_input_trajectory_trees()
    {
        for (int br_id = 0; br_id < num_branch; ++br_id)
        {
            for (int k = 0; k < N + 1; ++k)
            {
                traj_tree_[br_id][k].x = traj_tree_[br_id][k].x_tmp;
                traj_tree_[br_id][k].u = traj_tree_[br_id][k].u_tmp;            
            }
        }
    }

    inline void reset_state_input_trajectory_trees()
    {
        for (int br_id = 0; br_id < num_branch; ++br_id)
        {
            for (int k = 0; k < N + 1; ++k)
            {
                traj_tree_[br_id][k].x_tmp = traj_tree_[br_id][k].x;
                traj_tree_[br_id][k].u_tmp = traj_tree_[br_id][k].u;            
            }
        }
    }

    inline void set_reference_trajectories(std::vector<VecState<n>> &x_ref, std::vector<VecInput<m>> &u_ref)
    {

        for (int br_id = 0; br_id < num_branch; ++br_id)
        {
            for (int k = 0; k < N; ++k)
            {
                traj_tree_[br_id][k].x_ref = x_ref[br_id];
                traj_tree_[br_id][k].u_ref = u_ref[br_id];
            }
            traj_tree_[br_id][N].x_ref = x_ref[br_id];
        }
    }

    inline void set_branch_probabilities(double* branch_probs)
    {
        for (int br_id = 0; br_id < num_branch; ++br_id)
        {
            for (int k = 0; k < N; ++k)
            {
                traj_tree_[br_id][k].prob = branch_probs[br_id];
            }
        }
    }

    inline void set_initial_state(const VecState<n> &x0)
    {
        for (int br_id = 0; br_id < num_branch; ++br_id) { traj_tree_[br_id][0].x = x0; }
    }

    inline VecInput<m> get_control() { return traj_tree_[0][0].u; }
    
    inline void get_state_trajectories(std::vector<VecState<n-m>> &state_trajs)
    {
        state_trajs.clear();
        state_trajs.resize(num_branch * (N + 1));
        for (int br_id = 0; br_id < num_branch; ++br_id)
        {
            for (int k = 0; k < N + 1; ++k)
            {
                state_trajs[br_id * (N + 1) + k] = traj_tree_[br_id][k].x.head(n - m);
            }
        }
    }
};

template<int n>
class PredictedTrajectories
{
private:
    double vehicle_length_;
    double vehicle_width_;
    double inter_axle_length_;
    int num_branch_;
    int traj_length_;
    std::vector<std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>>> discs_all_; // (px, py, radius)
    Eigen::Matrix<double, 3, Eigen::Dynamic> discs_axis_aligned_; // (px, py, radius)

public:
    std::vector<std::vector<VecState<n>>> state_trajs_;

    PredictedTrajectories() = default;

    PredictedTrajectories(const double vehicle_length, 
                          const double vehicle_width,
                          const double inter_axle_length, 
                          const std::vector<std::vector<VecState<n>>> &state_trajs) : 
                          vehicle_length_(vehicle_length), vehicle_width_(vehicle_width), inter_axle_length_(inter_axle_length), state_trajs_(state_trajs)
    {
        num_branch_ = state_trajs_.size();
        traj_length_ = state_trajs_[0].size();
        discs_all_.resize(num_branch_, std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>>(traj_length_));
        generate_axis_aligned_bounding_discs();
        generate_bounding_discs();
    }

    inline void generate_axis_aligned_bounding_discs()
    {
        double radius = sqrt(2.) * vehicle_width_ / 2.;
        double dist_btw_centers = vehicle_length_ - vehicle_width_;
        int num_discs = ceil(dist_btw_centers / radius) + 1;
        double h = (num_discs == 1) ? 0. : dist_btw_centers / (num_discs - 1);
        double center_x_init = -vehicle_length_ / 2. + vehicle_width_ / 2.;
        double center_y_init = 0.;
        discs_axis_aligned_.resize(3, num_discs);

        for (int i = 0; i < num_discs; ++i)
        {
            discs_axis_aligned_(0, i) = center_x_init + i * h;
            discs_axis_aligned_(1, i) = center_y_init;
            discs_axis_aligned_(2, i) = radius;
        }
    }

    inline void generate_bounding_discs()
    { 
        for (int br_id = 0; br_id < num_branch_; ++br_id)
        {
            for (size_t k = 0; k < state_trajs_[br_id].size(); ++k)
            {
                double px = state_trajs_[br_id][k][0];
                double py = state_trajs_[br_id][k][1];
                double theta = state_trajs_[br_id][k][2];
                Eigen::Matrix<double, 3, Eigen::Dynamic> discs = discs_axis_aligned_;
                for (int i = 0; i < discs.cols(); ++i)
                {
                    // Coordinate transformation
                    discs(0, i) = px + cos(theta) * discs_axis_aligned_(0, i) - sin(theta) * discs_axis_aligned_(1, i);
                    discs(1, i) = py + sin(theta) * discs_axis_aligned_(0, i) + cos(theta) * discs_axis_aligned_(1, i);
                }
                discs_all_[br_id][k] = discs;
            }
        }

    }

    inline void get_bouding_discs(const int k,
                                  const int branch_id,
                                  Eigen::Matrix<double, 3, Eigen::Dynamic> &discs) const // Safe?
    {
        discs = discs_all_[branch_id][k];
    }

    inline void get_axis_aligned_bouding_discs(Eigen::Matrix<double, 3, Eigen::Dynamic> &discs) const // Safe?
    {
        discs = discs_axis_aligned_;
    }
    
    inline double get_inter_axle_length() const { return inter_axle_length_; }

};

}   // namespace cilqr_tree