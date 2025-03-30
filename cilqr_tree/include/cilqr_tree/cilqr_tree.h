#pragma once

#include <ros/ros.h>
#include <Eigen/Dense>
#include <cmath>
#include <string>

#include "ilqr_params.h"
#include "eigen_types.h"
#include "constraints.h"
#include "trajectory_tree_data.h"
#include "ilqr_tree.h"
#include "misc/visualizer.h"

namespace cilqr_tree {

static constexpr int num_state = 6;
static constexpr int num_input = 2;

class BranchMPC
{
private:
    ros::NodeHandle nh_;
    iLQRParams params_;
    double num_branch_, N_;
    Constraints constraints_;
    double rho_;
    std::vector<std::vector<Eigen::Vector2d>>& local_guidelines_;
    
    bool first_run = true;

    static inline double stage_cost(void *ptr, 
                                    cilqr_tree::KnotpointData<num_state, num_input> &knotpoint, 
                                    const cilqr_tree::iLQRParams &params,
                                    bool include_augmented_cost);

    static inline void stage_cost_expansion(void *ptr, 
                                            const cilqr_tree::iLQRParams &params,
                                            cilqr_tree::KnotpointData<num_state, num_input> &knotpoint);
    
    static inline void discrete_dynamics(const VecState<num_state> &x, 
                                         const VecInput<num_input> &u, 
                                         VecState<num_state> &x_next);
    
    static inline void discrete_dynamics_expansion(const VecState<num_state> &x, 
                                                   const VecInput<num_input> &u, 
                                                   MatState<num_state> &A, 
                                                   MatStateInput<num_state, num_input> &B);

    static inline void transform_to_rear(VecState<num_state> &x, double inter_axle_length)
    {
        x[0] -= inter_axle_length * cos(x[2]) / 2.0;
        x[1] -= inter_axle_length * sin(x[2]) / 2.0;
    }


    inline double update_dual_variables()
    {
        double cons_vio = 0.;
        for (int br_id = 0; br_id < num_branch_; ++br_id)
        {
            for (int k = 1; k < N_; ++k)    // ignore k = 0
            {
                KnotpointData<num_state, num_input> &knotpoint = traj_tree(k, br_id);
                cons_vio = std::max(cons_vio, 
                                    constraints_.update_mu_x(knotpoint.x_tmp, rho_, knotpoint.hx, knotpoint.mu_x));
                cons_vio = std::max(cons_vio,
                                    constraints_.update_mu_u(knotpoint.u_tmp, rho_, knotpoint.hu, knotpoint.mu_u));
                cons_vio = std::max(cons_vio,
                                    constraints_.update_mu_x_ineq(this, k, knotpoint.branch_id, knotpoint.x_tmp, rho_, knotpoint.hx_ineq, knotpoint.mu_x_ineq));

                // std::cout << "k: " << k << std::endl;
                // std::cout << "mu_x: " << knotpoint.mu_x.transpose() << std::endl;
                // std::cout << "mu_u: " << knotpoint.mu_u.transpose() << std::endl;
                // std::cout << "mu_x_ineq:\n" << knotpoint.mu_x_ineq.transpose() << std::endl;
            }
            KnotpointData<num_state, num_input> &knotpoint = traj_tree(N_, br_id);
            cons_vio = std::max(cons_vio, 
                                constraints_.update_mu_x(knotpoint.x_tmp, rho_, knotpoint.hx, knotpoint.mu_x));
            cons_vio = std::max(cons_vio,
                                constraints_.update_mu_x_ineq(this, N_, knotpoint.branch_id, knotpoint.x_tmp, rho_, knotpoint.hx_ineq, knotpoint.mu_x_ineq));
            // std::cout << "mu_x: " << knotpoint.mu_x.transpose() << std::endl;
            // std::cout << "mu_x_ineq: " << knotpoint.mu_x_ineq.transpose() << std::endl;
        }
        return cons_vio;
    }

    inline double safety_cost(const VecState<num_state-num_input> &x_sub, const int k, const int branch_id);
    
    inline void safety_cost_expansion(const VecState<num_state-num_input> &x,
                                      const int k,
                                      const int branch_id,
                                      MatState<num_state> &l_xx,
                                      VecState<num_state> &l_x);    

    inline void update_reference_trajectories()
    {
        /* shared part */
        int num_surrounding_vehicle = multi_vehicle_pred_trajs_.size() - 1;

        VecInput<num_input> u_sh;
        for (int k = 0; k < traj_tree.N_sh; ++k)
        {
            u_sh.setZero();
            for (int br_id = 0; br_id < num_branch_; ++br_id)
            {
                u_sh += traj_tree(k, br_id).prob * traj_tree(k, br_id).u;
            }
            for (int br_id = 0; br_id < num_branch_; ++br_id)
            {
                traj_tree(k, br_id).u = u_sh;
            }
        }

        for (int br_id = 0; br_id < num_branch_; ++br_id)
        {
            for (int k = 0; k < N_; ++k)
            {
                traj_tree(k, br_id).x_ref.setZero();
                traj_tree(k, br_id).x_ref.head(2) = local_guidelines_[br_id][k];                
                traj_tree(k, br_id).x_ref(3) = 5.0; // TODO: hand-crafted
        
                traj_tree(k, br_id).u_ref.setZero();
                
                discrete_dynamics(traj_tree(k, br_id).x, 
                                  traj_tree(k, br_id).u, 
                                  traj_tree(k + 1, br_id).x);

                traj_tree(k, br_id).mu_x.setZero();
                traj_tree(k, br_id).mu_u.setZero();
                // TODO: hand-crafted
                traj_tree(k, br_id).mu_x_ineq.resize(2 * 4 + 3 * 3 * num_surrounding_vehicle);
                traj_tree(k, br_id).mu_x_ineq.setZero();
            }
            traj_tree(N_, br_id).x_ref.setZero();
            traj_tree(N_, br_id).x_ref.head(2) = local_guidelines_[br_id][N_];
            traj_tree(N_, br_id).x_ref(3) = 5.0;

            traj_tree(N_, br_id).mu_x.setZero();
            traj_tree(N_, br_id).mu_u.setZero();
            traj_tree(N_, br_id).mu_x_ineq.resize(2 * 4 + 3 * 3 * num_surrounding_vehicle);
            traj_tree(N_, br_id).mu_x_ineq.setZero();
        }
        first_run = false;
    }

public:
    TrajTree<num_state, num_input> traj_tree;
    std::vector<double> branch_costs;
    double vehicle_length_, vehicle_width_, inter_axle_length_;
    std::shared_ptr<Visualizer>& visualizer_ptr;
    std::vector<PredictedTrajectories<num_state-num_input>> multi_vehicle_pred_trajs_;
    SolverData solver_data_;

public:
    BranchMPC(ros::NodeHandle& nh,
              std::vector<std::vector<Vector2d>>& local_guidelines, 
              std::shared_ptr<Visualizer>& visualizer_ptr) : 
              nh_(nh), params_(iLQRParams(nh)), constraints_(params_), local_guidelines_(local_guidelines), visualizer_ptr(visualizer_ptr)
    {
        num_branch_ = params_.num_branch;
        N_ = params_.N;
        traj_tree = TrajTree<num_state, num_input>(params_.num_branch, params_.N, params_.N_sh);
        branch_costs.resize(num_branch_);

        // TODO: need to properly initialize
        vehicle_length_ = 4.0;
        vehicle_width_ = 2.0;
        inter_axle_length_ = 3.0;
    }

    inline void update_reference_input(VecInput<num_input> &u_ref, const int k, const int branch_id)
    {
        traj_tree(k, branch_id).u = u_ref;    
    }

    inline void update_branch_probability(double prob, const int k, const int branch_id)
    {
        traj_tree(k, branch_id).prob = prob;
        traj_tree(k, branch_id).prob_norm = prob;
    }

    inline void safety_constraints(const VecState<num_state-num_input> &x_sub, 
                                   const int k, 
                                   const int branch_id,
                                   Eigen::VectorXd &hx);

    inline void safety_constraints_jac(const VecState<num_state-num_input> &x_sub, 
                                       const int k, 
                                       const int branch_id,
                                       Eigen::MatrixXd &jac_hx);

    static inline double angle_diff(double x, double y, double angle,
                                    double x_ref, double y_ref)
    {
        double dx = x_ref - x;
        double dy = y_ref - y;
        double angle_ref = std::atan2(dy, dx);
        double diff = angle - angle_ref;

        return diff;
    }
    
    static inline void D_angle_diff(double x, double y, double angle,
                                    double x_ref, double y_ref,
                                    Eigen::Matrix<double, 3, 1> &D_angle)
    {
        double dx = x_ref - x;
        double dy = y_ref - y;

        D_angle(0) = -dy / (dx * dx + dy * dy);
        D_angle(1) = dx / (dx * dx + dy * dy);
        D_angle(2) = 1.0;
    }

    inline void get_vehicle_vertices(const VecState<num_state> &x, 
                                     std::vector<VecState<2>> &vertices)
    {
        VecState<num_state-num_input> x_sub = x.head(num_state-num_input);
        vertices.resize(4);
        double cos_theta = cos(x(2));
        double sin_theta = sin(x(2));
        double length_, width_;
        width_ = vehicle_width_ / 2.;

        length_ = vehicle_length_ / 2. + inter_axle_length_ / 2.;
        vertices[0] << x_sub(0) + length_ * cos_theta + width_ * sin_theta,
                       x_sub(1) + length_ * sin_theta - width_ * cos_theta;

        vertices[1] << x_sub(0) + length_ * cos_theta - width_ * sin_theta,
                       x_sub(1) + length_ * sin_theta + width_ * cos_theta;

        length_ = vehicle_length_ / 2. - inter_axle_length_ / 2.;
        vertices[2] << x_sub(0) - length_ * cos_theta - width_ * sin_theta,
                       x_sub(1) - length_ * sin_theta + width_ * cos_theta;

        vertices[3] << x_sub(0) - length_ * cos_theta + width_ * sin_theta,
                       x_sub(1) - length_ * sin_theta - width_ * cos_theta; 
    }

    inline void add_predicted_trajectories(const double vehicle_length, 
                                           const double vehicle_width,
                                           const double inter_axle_length, 
                                           const std::vector<std::vector<VecState<num_state-num_input>>> &state_trajs)
    {
        multi_vehicle_pred_trajs_.emplace_back(vehicle_length, vehicle_width, inter_axle_length, state_trajs);
    }

    inline void solve(const VecState<num_state> &x0, 
                      int sim_id,
                      int planner_type)
    {
        double cons_vio;
        VecState<num_state> x0_rear;
        x0_rear = x0;

        double inter_axle_length = params_.L;
        transform_to_rear(x0_rear, inter_axle_length);
        rho_ = params_.rho_init;
        traj_tree.set_initial_state(x0_rear);
        
        update_reference_trajectories();

        /* Solver Data */
        solver_data_.q_hist.clear();
        solver_data_.q_pre.resize(traj_tree.num_branch);
        solver_data_.total_iter = 0;
        solver_data_.success = false;
        solver_data_.alpha = params_.alpha;

        for (int i = 0; i < traj_tree.num_branch; ++i)
        {
            solver_data_.q_pre[i] = traj_tree(0, i).prob;
        } 

        int ret;
        for (int iter = 0; iter < params_.MAX_OUTER_ITER; ++iter)
        {
            if (planner_type == 0) // risk-neutral
            {
                ret = ilqr_optimize(this,
                                    stage_cost,
                                    stage_cost_expansion,
                                    discrete_dynamics,
                                    discrete_dynamics_expansion,
                                    params_,
                                    traj_tree,
                                    solver_data_,
                                    visualizer_ptr);
            }
            else if (planner_type == 1) // risk-aware
            {
                ret = risk_ilqr_optimize(this,
                                         stage_cost,
                                         stage_cost_expansion,
                                         discrete_dynamics,
                                         discrete_dynamics_expansion,
                                         params_,
                                         traj_tree,
                                         solver_data_,
                                         visualizer_ptr);
            }
            else
            {
                ROS_ERROR("Invalid planner type!");
            }
            
            if (ret >= 0)
            {
                cons_vio = update_dual_variables();
                
                std::cout << "cons_vio: " << cons_vio << std::endl;
                std::cout  << "rho: " << rho_ << std::endl;
                
                if (cons_vio < params_.constr_vio_tol)
                {
                    solver_data_.success = true;
                    std::cout << "q: " << solver_data_.q_pre.transpose() << std::endl;
                    std::cout << "CiLQR converged!" << std::endl;
                    break;
                }
                else
                {
                    rho_ = std::min(params_.rho_max, rho_ * params_.rho_scaling);
                    
                }
            }   
            else
            {
                std::cout << "iLQR failed!" << std::endl;
                break;
            }
        }

        multi_vehicle_pred_trajs_.clear();

        if (cons_vio >= params_.constr_vio_tol)
        {
            ROS_ERROR("Optimization failed!");
        }
    }

};

inline double BranchMPC::stage_cost(void *ptr,
                                    cilqr_tree::KnotpointData<num_state, num_input> &knotpoint, // remove const
                                    const cilqr_tree::iLQRParams &params,
                                    bool include_augmented_cost)
{
    BranchMPC &obj = *static_cast<BranchMPC*>(ptr);

    double cost = 0.;
    int num_state_ = num_state - num_input;
    if (!knotpoint.is_final)
    {
        Vector4d x = knotpoint.x_tmp.head(num_state_);

        Vector2d u_pre = knotpoint.x_tmp.tail(num_input);
        Vector4d dx = x - knotpoint.x_ref.head(num_state_);
        dx(2) = obj.angle_diff(x(0), x(1), x(2), knotpoint.x_ref(0), knotpoint.x_ref(1));

        Vector2d du = knotpoint.u_tmp - knotpoint.u_ref;

        double cost_track = (0.5 * dx.transpose() * params.Q * dx)(0) + (0.5 * du.transpose() * params.R * du)(0);    // Any better way?
        double cost_comfort = (0.5 * (knotpoint.u_tmp - u_pre).transpose() * params.R_com * (knotpoint.u_tmp - u_pre))(0);
        double cost_safety = obj.safety_cost(x, knotpoint.k, knotpoint.branch_id);
      
        cost = cost_track + cost_comfort + cost_safety;
        cost *= knotpoint.prob;
        if (knotpoint.k > 0 && include_augmented_cost)
        {
            cost += obj.constraints_.compute_augmented_cost(
                ptr,
                knotpoint.k,
                knotpoint.branch_id,
                knotpoint.x_tmp, 
                knotpoint.u_tmp, 
                knotpoint.mu_x, 
                knotpoint.mu_u, 
                knotpoint.mu_x_ineq, 
                obj.rho_,
                knotpoint.hx,
                knotpoint.hu,
                knotpoint.hx_ineq,
                knotpoint.mask_x,
                knotpoint.mask_u,
                knotpoint.mask_x_ineq
            );
        }
        if (!include_augmented_cost)
        {
            // cost += obj.constraints_.compute_augmented_cost(
            //     ptr,
            //     knotpoint.k,
            //     knotpoint.branch_id,
            //     knotpoint.x_tmp, 
            //     knotpoint.u_tmp, 
            //     knotpoint.mu_x, 
            //     knotpoint.mu_u, 
            //     knotpoint.mu_x_ineq, 
            //     obj.rho_
            // );
        }    
    }
    else
    {
        Vector4d x = knotpoint.x_tmp.head(num_state_);
        Vector4d dx = x - knotpoint.x_ref.head(num_state_);
        dx(2) = obj.angle_diff(x(0), x(1), x(2), knotpoint.x_ref(0), knotpoint.x_ref(1));

        cost = (0.5 * dx.transpose() * params.Qf * dx)(0) + obj.safety_cost(x, knotpoint.k, knotpoint.branch_id);    
        
        cost *= knotpoint.prob;
        if (include_augmented_cost)
        {
            cost += obj.constraints_.compute_augmented_cost(
                ptr,
                knotpoint.k,
                knotpoint.branch_id,
                knotpoint.x_tmp, 
                knotpoint.mu_x, 
                knotpoint.mu_x_ineq, 
                obj.rho_,
                knotpoint.hx,
                knotpoint.hx_ineq,
                knotpoint.mask_x,
                knotpoint.mask_x_ineq
            );
        }

        if (!include_augmented_cost)
        {
            // cost += knotpoint.prob * obj.constraints_.compute_augmented_cost(
            //     ptr,
            //     knotpoint.k,
            //     knotpoint.branch_id,
            //     knotpoint.x_tmp, 
            //     knotpoint.mu_x, 
            //     knotpoint.mu_x_ineq, 
            //     obj.rho_
            // );

            double aug_cost = obj.constraints_.compute_augmented_cost(
                ptr,
                knotpoint.k,
                knotpoint.branch_id,
                knotpoint.x_tmp, 
                knotpoint.mu_x, 
                knotpoint.mu_x_ineq, 
                obj.rho_,
                knotpoint.hx,
                knotpoint.hx_ineq,
                knotpoint.mask_x,
                knotpoint.mask_x_ineq
            );
        }

    }
    return cost;

}

inline void BranchMPC::stage_cost_expansion(void *ptr,
                                            const cilqr_tree::iLQRParams &params,
                                            cilqr_tree::KnotpointData<num_state, num_input> &knotpoint)
{
    BranchMPC &obj = *static_cast<BranchMPC*>(ptr);

    int num_state_ = num_state - num_input;

    if (!knotpoint.is_final)
    {
        Vector4d x = knotpoint.x_tmp.head(num_state_);
        Vector2d u_pre = knotpoint.x_tmp.tail(num_input);
        Vector4d dx = x - knotpoint.x_ref.head(num_state_);
        dx(2) = obj.angle_diff(x(0), x(1), x(2), knotpoint.x_ref(0), knotpoint.x_ref(1));

        Vector2d du = knotpoint.u_tmp - knotpoint.u_ref;

        knotpoint.l_xx.setZero();
        knotpoint.l_ux.setZero();
        knotpoint.l_uu.setZero();
        knotpoint.l_x.setZero();
        knotpoint.l_u.setZero();
        
        knotpoint.l_xx.block(0, 0, num_state_, num_state_) = params.Q;
        knotpoint.l_xx.block(num_state_, num_state_, num_input, num_input) = params.R_com;

        knotpoint.l_ux.block(0, num_state_, num_input, num_input) = -params.R_com;

        knotpoint.l_uu = params.R + params.R_com;

        knotpoint.l_x.head(num_state_) = params.Q * dx;
        knotpoint.l_x(2) = 0.;
        Eigen::Matrix<double, 3, 1> D_angle;
        obj.D_angle_diff(x(0), x(1), x(2), knotpoint.x_ref(0), knotpoint.x_ref(1), D_angle);
        knotpoint.l_x.head(num_state_-1) += params.Q(2, 2) * D_angle * dx(2);
        knotpoint.l_xx(2, 2) = 0.;
        knotpoint.l_xx.block(0, 0, num_state_-1, num_state_-1) += params.Q(2, 2) * D_angle * D_angle.transpose();

        knotpoint.l_x.tail(num_input) = params.R_com * (u_pre - knotpoint.u_tmp);
        knotpoint.l_u = params.R * du + params.R_com * (knotpoint.u_tmp - u_pre);

        MatState<num_state> l_xx_safety;
        VecState<num_state> l_x_safety;
        obj.safety_cost_expansion(x, knotpoint.k, knotpoint.branch_id, l_xx_safety, l_x_safety);

        knotpoint.l_xx += l_xx_safety;
        knotpoint.l_x += l_x_safety;
   
        knotpoint.l_xx *= knotpoint.prob;
        knotpoint.l_ux *= knotpoint.prob;
        knotpoint.l_uu *= knotpoint.prob;
        knotpoint.l_x *= knotpoint.prob;
        knotpoint.l_u *= knotpoint.prob;
        if (knotpoint.k > 0)
        {
            obj.constraints_.compute_augmented_cost_expansion(
                ptr,
                knotpoint.k,
                knotpoint.branch_id,
                knotpoint.x_tmp, 
                knotpoint.u_tmp, 
                knotpoint.mu_x, 
                knotpoint.mu_u, 
                knotpoint.mu_x_ineq, 
                obj.rho_,
                knotpoint.hx,
                knotpoint.hu,
                knotpoint.hx_ineq,
                knotpoint.mask_x,
                knotpoint.mask_u,
                knotpoint.mask_x_ineq,
                knotpoint.jac_hx,
                knotpoint.jac_hu,
                knotpoint.jac_hx_ineq,
                knotpoint.l_xx, 
                knotpoint.l_uu, 
                knotpoint.l_x, 
                knotpoint.l_u
            );
        }
    }
    else
    {
        Vector4d x = knotpoint.x_tmp.head(num_state_);
        Vector4d dx = x - knotpoint.x_ref.head(num_state_);
        dx(2) = obj.angle_diff(x(0), x(1), x(2), knotpoint.x_ref(0), knotpoint.x_ref(1));

        knotpoint.l_xx.setZero();
        knotpoint.l_x.setZero();
        knotpoint.l_xx.block(0, 0, num_state_, num_state_) = params.Qf;

        knotpoint.l_x.head(num_state_) = params.Qf * dx;
        knotpoint.l_x(2) = 0.;
        Eigen::Matrix<double, 3, 1> D_angle;
        obj.D_angle_diff(x(0), x(1), x(2), knotpoint.x_ref(0), knotpoint.x_ref(1), D_angle);
        knotpoint.l_x.head(num_state_-1) += params.Qf(2, 2) * D_angle * dx(2);
        knotpoint.l_xx(2, 2) = 0.;
        knotpoint.l_xx.block(0, 0, num_state_-1, num_state_-1) += params.Qf(2, 2) * D_angle * D_angle.transpose();
        
        MatState<num_state> l_xx_safety;
        VecState<num_state> l_x_safety;
        obj.safety_cost_expansion(x, knotpoint.k, knotpoint.branch_id, l_xx_safety, l_x_safety);

        knotpoint.l_xx += l_xx_safety;
        knotpoint.l_x += l_x_safety;
        
        knotpoint.l_xx *= knotpoint.prob;
        knotpoint.l_x *= knotpoint.prob;
        obj.constraints_.compute_augmented_cost_expansion(
            ptr,
            knotpoint.k,
            knotpoint.branch_id,
            knotpoint.x_tmp, 
            knotpoint.mu_x, 
            knotpoint.mu_x_ineq, 
            obj.rho_,
            knotpoint.hx,
            knotpoint.hx_ineq,
            knotpoint.mask_x,
            knotpoint.mask_x_ineq,
            knotpoint.jac_hx,
            knotpoint.jac_hx_ineq,
            knotpoint.l_xx, 
            knotpoint.l_x
        );

        knotpoint.V_xx = knotpoint.l_xx;
        knotpoint.V_x = knotpoint.l_x;
    }
}

inline void dynamics(const VecState<num_state-num_input> &x, 
                     const VecInput<num_input> &u, 
                     VecState<num_state-num_input> &x_dot)
{
    double L = 3.0;
    double theta = x[2];
    double v = x[3];
    double a = u[0];
    double delta = u[1];
    x_dot << v * cos(theta), 
             v * sin(theta), 
             v / L * tan(delta), 
             a;
}

inline void BranchMPC::discrete_dynamics(const VecState<num_state> &x, 
                                         const VecInput<num_input> &u, 
                                         VecState<num_state> &x_next)
{
    double L = 3.0;
    double dt = 0.1;
    double theta = x[2];
    double v = x[3];
    double a = u[0];
    double delta = u[1];
    Vector4d x_dot;  
    // x_dot << v * cos(theta), 
    //          v * sin(theta), 
    //          v / L * tan(delta), 
    //          a;  

    // Foward Euler
    dynamics(x.head(4), u, x_dot);
    x_next.head(4) = x.head(4) + x_dot * dt;

    // RK2
    // VecState<num_state-num_input> k1, k2;
    // dynamics(x.head(4), u, k1);
    // dynamics(x.head(4) + k1 * dt, u, k2);
    // x_next.head(4) = x.head(4) + (k1 + k2) * dt / 2.0;

    x_next[4] = a;
    x_next[5] = delta;
}

inline void dynamics_expansion(
    const VecState<num_state-num_input> &x,
    const VecInput<num_input> &u,
    MatState<num_state-num_input> &Fx,
    MatStateInput<num_state-num_input, num_input> &Fu
)
{
    double L = 3.0;
    double theta = x[2];
    double v = x[3];
    double a = u[0];
    double delta = u[1];
    Fx.setZero();
    Fu.setZero();
    Fx(0, 2) = -v * sin(theta);
    Fx(0, 3) = cos(theta);
    Fx(1, 2) =  v * cos(theta);
    Fx(1, 3) = sin(theta);
    Fx(2, 3) = tan(delta) / L;

    Fu(2, 1) = v / L / cos(delta) / cos(delta);
    Fu(3, 0) = 1.0;
}

inline void BranchMPC::discrete_dynamics_expansion(const VecState<num_state> &x, 
                                                   const VecInput<num_input> &u, 
                                                   MatState<num_state> &A, 
                                                   MatStateInput<num_state, num_input> &B)
{
    double L = 3.0;
    double dt = 0.1;
    double theta = x[2];
    double v = x[3];
    // double a = u[0];
    double delta = u[1];

    Matrix4d A_;
    Matrix<double, 4, 2> B_;
    // A_.setIdentity();
    // B_.setZero();
    // A_(0, 2) = -v * sin(theta) * dt;
    // A_(0, 3) = cos(theta) * dt;
    // A_(1, 2) =  v * cos(theta) * dt;
    // A_(1, 3) = sin(theta) * dt;
    // A_(2, 3) = tan(delta) / L * dt;
    // B_(2, 1) = v / L / cos(delta) / cos(delta) * dt;
    // B_(3, 0) = dt;

    // Foward Euler
    dynamics_expansion(x.head(4), u, A_, B_);
    A_ *= dt;
    A_ += Matrix4d::Identity();
    B_ *= dt;

    // RK2
    // VecState<num_state-num_input> k1;
    // dynamics(x.head(4), u, k1);

    // MatState<num_state-num_input> Fx1, Fx2;
    // MatStateInput<num_state-num_input, num_input> Fu1, Fu2;
    // dynamics_expansion(x.head(4), u, Fx1, Fu1);
    // dynamics_expansion(x.head(4) + k1 * dt, u, Fx2, Fu2);

    // MatState<num_state-num_input> Kx2;
    // MatStateInput<num_state-num_input, num_input> Ku2;
    // Kx2 = Fx2 + Fx2 * Fx1 * dt;
    // A_.setIdentity();
    // A_ += (Fx1 + Kx2) * dt / 2.0;

    // B_.setZero();
    // Ku2 = Fx2 * Fu1 * dt + Fu2;
    // B_ += (Fu1 + Ku2) * dt / 2.0;

    /* ---------- */
    A.setZero();
    B.setZero();
    A.block(0, 0, 4, 4) = A_;
    B.block(0, 0, 4, 2) = B_;
    B.block(4, 0, 2, 2).setIdentity();
}

inline double BranchMPC::safety_cost(const VecState<num_state-num_input> &x,
                                     const int k,
                                     const int branch_id)
{
    // TODO: 
    double r_saf = params_.r_saf;
    double cost = 0.;
    int num_vehicle = multi_vehicle_pred_trajs_.size();
    double half_inter_axle_length = multi_vehicle_pred_trajs_[0].get_inter_axle_length() / 2.;
    for (int veh_i = 1; veh_i < num_vehicle; ++veh_i)
    {
        Eigen::Vector2d p_sv_cen = multi_vehicle_pred_trajs_[veh_i].state_trajs_[branch_id][k].head(2);
        Eigen::Vector2d p_ego_cen;
        p_ego_cen << x[0] + half_inter_axle_length * cos(x[2]), 
                     x[1] + half_inter_axle_length * sin(x[2]); 
        double dist = (p_ego_cen - p_sv_cen).norm();
        if (dist < r_saf)
        {
            cost += 0.5 * (dist - r_saf) * (dist - r_saf);
        }
    }
    cost *= params_.safety_weight;
    return cost;
}

inline void BranchMPC::safety_constraints(const VecState<num_state-num_input> &x,
                                          const int k,
                                          const int branch_id,
                                          Eigen::VectorXd &hx)
{
    std::vector<double> hx_tmp;

    int num_vehicle = multi_vehicle_pred_trajs_.size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> discs_ego;
    multi_vehicle_pred_trajs_[0].get_axis_aligned_bouding_discs(discs_ego);
    double half_inter_axle_length = multi_vehicle_pred_trajs_[0].get_inter_axle_length() / 2.;

    for (int veh_i = 1; veh_i < num_vehicle; ++veh_i)
    {
        Eigen::Matrix<double, 3, Eigen::Dynamic> discs_sv;
        multi_vehicle_pred_trajs_[veh_i].get_bouding_discs(k, branch_id, discs_sv);
        for (int disc_i = 0; disc_i < discs_ego.cols(); ++disc_i)
        {
            Eigen::Vector2d p_dc; // Disc center in world frame
            double r_ego = discs_ego(2, disc_i);
            p_dc << x[0] + (half_inter_axle_length + discs_ego(0, disc_i)) * cos(x[2]),
                    x[1] + (half_inter_axle_length + discs_ego(0, disc_i)) * sin(x[2]);
            for (int disc_j = 0; disc_j < discs_sv.cols(); ++disc_j)
            {
                Eigen::Vector2d p_sv = discs_sv.col(disc_j).head(2);
                double r_sv = discs_sv(2, disc_j);
                double dist = (p_dc - p_sv).norm();
                double r_total = r_ego + r_sv + params_.safety_margin;
                hx_tmp.push_back(r_total - dist);
            }
        }
    }
    hx.resize(hx_tmp.size());
    for (size_t i = 0; i < hx_tmp.size(); ++i)
    {
        hx(i) = hx_tmp[i];
    }

}

inline void BranchMPC::safety_cost_expansion(const VecState<num_state-num_input> &x,
                                             const int k,
                                             const int branch_id,
                                             MatState<num_state> &l_xx,
                                             VecState<num_state> &l_x)
{
    double r_saf = params_.r_saf;

    l_xx.setZero(); l_x.setZero();
    int num_vehicle = multi_vehicle_pred_trajs_.size();
    double half_inter_axle_length = multi_vehicle_pred_trajs_[0].get_inter_axle_length() / 2.;

    for (int veh_i = 1; veh_i < num_vehicle; ++veh_i)
    {
        Eigen::Vector2d p_sv_cen = multi_vehicle_pred_trajs_[veh_i].state_trajs_[branch_id][k].head(2);
        Eigen::Vector2d p_ego_cen;
        p_ego_cen << x[0] + half_inter_axle_length * cos(x[2]), 
                     x[1] + half_inter_axle_length * sin(x[2]);
        Vector2d vec_diff = p_ego_cen - p_sv_cen;
        double vec_diff_norm = vec_diff.norm();

        if (vec_diff_norm < r_saf)
        {
            VecState<num_state> l_x_tmp;
            MatState<num_state> l_xx_tmp;
            l_x_tmp << vec_diff[0] / vec_diff_norm,
                       vec_diff[1] / vec_diff_norm,
                      -vec_diff[0] / vec_diff_norm * half_inter_axle_length * sin(x[2]) 
                      +vec_diff[1] / vec_diff_norm * half_inter_axle_length * cos(x[2]),
                       0., 0., 0.;
            l_xx_tmp = l_x_tmp * l_x_tmp.transpose();
            l_x_tmp = (vec_diff_norm - r_saf) * l_x_tmp;
            l_x += l_x_tmp;
            l_xx += l_xx_tmp;
        }
    }
    l_x *= params_.safety_weight;
    l_xx *= params_.safety_weight;
}

inline void BranchMPC::safety_constraints_jac(
    const VecState<num_state-num_input> &x,
    const int k,
    const int branch_id,
    Eigen::MatrixXd &jac_hx)
{
    std::vector<VecState<num_state>> jac_hx_tmp;
    int num_vehicle = multi_vehicle_pred_trajs_.size();
    Eigen::Matrix<double, 3, Eigen::Dynamic> discs_ego;
    multi_vehicle_pred_trajs_[0].get_axis_aligned_bouding_discs(discs_ego);
    double half_inter_axle_length = multi_vehicle_pred_trajs_[0].get_inter_axle_length() / 2.;
    for (int veh_i = 1; veh_i < num_vehicle; ++veh_i)
    {
        Eigen::Matrix<double, 3, Eigen::Dynamic> discs_sv;
        multi_vehicle_pred_trajs_[veh_i].get_bouding_discs(k, branch_id, discs_sv);
        for (int disc_i = 0; disc_i < discs_ego.cols(); ++disc_i)
        {
            Eigen::Vector2d p_dc; // Disc center in world frame
            double r_ego = discs_ego(2, disc_i);
            p_dc << x[0] + (half_inter_axle_length + discs_ego(0, disc_i)) * cos(x[2]),
                    x[1] + (half_inter_axle_length + discs_ego(0, disc_i)) * sin(x[2]);                
            for (int disc_j = 0; disc_j < discs_sv.cols(); ++disc_j)
            {
                Eigen::Vector2d p_sv = discs_sv.col(disc_j).head(2);
                double r_sv = discs_sv(2, disc_j);
                Vector2d vec_diff = p_dc - p_sv;
                double vec_diff_norm = vec_diff.norm();
                double r_total = r_ego + r_sv + params_.safety_margin;
                VecState<num_state> l_x_tmp;
                l_x_tmp << vec_diff[0] / vec_diff_norm,
                            vec_diff[1] / vec_diff_norm,
                            -vec_diff[0] / vec_diff_norm * (half_inter_axle_length + discs_ego(0, disc_i)) * sin(x[2]) 
                            +vec_diff[1] / vec_diff_norm * (half_inter_axle_length + discs_ego(0, disc_i)) * cos(x[2]),
                            0., 0., 0.;
                // TODO: Need a more efficient way to calculate the Jacobian
                jac_hx_tmp.push_back(-l_x_tmp);
            }
        }
    }
    jac_hx.resize(jac_hx_tmp.size(), num_state);
    for (size_t i = 0; i < jac_hx_tmp.size(); ++i)
    {
        jac_hx.row(i) = jac_hx_tmp[i].transpose();
    }
}

} // namespace cilqr_tree

