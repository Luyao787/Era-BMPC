#pragma once

#include "trajectory_tree_data.h"
#include "ilqr_params.h"
#include "eigen_types.h"
#include <omp.h>
#include "proj_hyperplane_box.h"
#include "solver_data.h"
#include "misc/visualizer.h"
namespace cilqr_tree
{

/* Return values */
enum
{    
    ILQR_MAX_ITER_REACHED = -1,
    ILQR_CONVERGED,
};

template<int n, int m>
using ilqr_stage_cost_t = double (*)(void *instance,
                                     KnotpointData<n, m> &knotpoint,
                                     const iLQRParams &params,
                                     bool include_augmented_cost);

template<int n, int m>
using ilqr_stage_cost_expansion_t = void (*)(void *instance,
                                             const iLQRParams &params,
                                             KnotpointData<n, m> &knotpoint);

template<int n, int m>
using ilqr_dynamics_t = void (*)(const VecState<n> &x, 
                                 const VecInput<m> &u, 
                                 VecState<n> &x_next);

template<int n, int m>
using ilqr_dynamics_expansion_t = void (*)(const VecState<n> &x, 
                                           const VecInput<m> &u, 
                                           MatState<n> &A, 
                                           MatStateInput<n, m> &B);


template<int n, int m>
inline double compute_trajectory_cost(TrajTree<n, m> &traj_tree,    // constant reference?
                                      void *instance,
                                      ilqr_stage_cost_t<n, m> stage_cost,
                                      const iLQRParams &params)
{
    double cost = 0.;
    for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
    {
        for (int k = 0; k < traj_tree.N + 1; ++k)
        {
            KnotpointData<n, m>& knotpoint = traj_tree(k, br_id);
            cost += stage_cost(instance, knotpoint, params, true);
        }
    }
    return cost;
}

template<int n, int m>
inline double compute_trajectory_cost(TrajTree<n, m> &traj_tree,    // constant reference?
                                      void *instance,
                                      ilqr_stage_cost_t<n, m> stage_cost,
                                      int br_id,
                                      const iLQRParams &params)
{
    double cost = 0.;
    for (int k = 0; k < traj_tree.N + 1; ++k)
    {
        KnotpointData<n, m>& knotpoint = traj_tree(k, br_id);
        cost += stage_cost(instance, knotpoint, params, false) / knotpoint.prob;
    }
    return cost;
}

template<int n, int m>
inline void compute_linear_policy(const MatState<n> &A, const MatStateInput<n, m> &B,
                                  const MatState<n> &l_xx, 
                                  const MatInputState<n, m> &l_ux,
                                  const MatInput<m> &l_uu, 
                                  const VecState<n> &l_x,
                                  const VecInput<m> &l_u,
                                  const MatState<n> &V_xx_next, const VecState<n> &V_x_next,
                                  const double regu,
                                  MatState<n> &V_xx, VecState<n> &V_x,
                                  MatInputState<n, m> &K, VecInput<m> &d, 
                                  double* exp_cost_redu)
{
    VecState<n> Q_x_;
    VecInput<m> Q_u_;
    MatState<n> Q_xx_;
    MatInput<m> Q_uu_;
    MatInputState<n, m> Q_ux_;

    // MatState<n> V_xx_next_regu;
    // V_xx_next_regu = V_xx_next + regu * MatState<n>::Identity();

    MatState<n> V_xx_next_regu;
    V_xx_next_regu = V_xx_next;
    V_xx_next_regu.diagonal().array() += regu;

    // VecState<n> Q_x = l_x + A.transpose() * V_x_next;
    // VecInput<m> Q_u = l_u + B.transpose() * V_x_next;
    // MatState<n> Q_xx = l_xx + A.transpose() * V_xx_next * A;
    // MatInput<m> Q_uu = l_uu + B.transpose() * V_xx_next * B;
    // MatInputState<n, m> Q_ux = l_ux + B.transpose() * V_xx_next * A;

    // MatInput<m> Q_uu_regu = l_uu + B.transpose() * V_xx_next_regu * B;
    // MatInputState<n, m> Q_ux_regu = l_ux + B.transpose() * V_xx_next_regu * A;

    VecState<n> Q_x = l_x;
    Q_x_.noalias() = A.transpose() * V_x_next;
    Q_x.noalias() += Q_x_;
    VecInput<m> Q_u = l_u;
    Q_u_.noalias() = B.transpose() * V_x_next;
    Q_u.noalias() += Q_u_;

    MatState<n> Q_xx = l_xx;
    Q_xx_.noalias() =  A.transpose() * V_xx_next;
    Q_xx.noalias() += Q_xx_ * A;
    MatInput<m> Q_uu = l_uu;
    Q_ux_.noalias() = B.transpose() * V_xx_next;
    Q_uu.noalias() += Q_ux_ * B;
    MatInputState<n, m> Q_ux = l_ux;
    Q_ux.noalias() += Q_ux_ * A;

    MatInput<m> Q_uu_regu = l_uu;
    Q_ux_ = B.transpose() * V_xx_next_regu;
    Q_uu_regu.noalias() += Q_ux_ * B;
    MatInputState<n, m> Q_ux_regu = l_ux;
    Q_ux_regu.noalias() += Q_ux_ * A;


    // VecState<n> Q_x = l_x + A.transpose() * V_x_next;
    // VecInput<m> Q_u = l_u + B.transpose() * V_x_next;
    // MatState<n> Q_xx = l_xx + A.transpose() * V_xx_next * A;
    // MatInput<m> Q_uu = l_uu + B.transpose() * V_xx_next * B;
    // MatInputState<n, m> Q_ux = l_ux + B.transpose() * V_xx_next * A;

    // MatInput<m> Q_uu_regu = l_uu + B.transpose() * V_xx_next_regu * B;
    // MatInputState<n, m> Q_ux_regu = l_ux + B.transpose() * V_xx_next_regu * A;


    // auto H = Q_uu_regu.llt();
    // K = -H.solve(Q_ux_regu);
    // d = -H.solve(Q_u);

    K = -Q_ux_regu;
    d = -Q_u;
    Eigen::LLT<Eigen::Ref<MatInput<m>>> H(Q_uu_regu);
    H.solveInPlace(K);
    H.solveInPlace(d);

    // V_xx = Q_xx + K.transpose() * Q_uu * K + Q_ux.transpose() * K + K.transpose() * Q_ux;
    // V_x  = Q_x  + K.transpose() * Q_uu * d + Q_ux.transpose() * d + K.transpose() * Q_u; 

    V_xx = Q_xx;
    Q_ux_.noalias() = Q_uu * K;
    Q_xx_.noalias() = K.transpose() * Q_ux;
    // Q_x_.noalias () = K.transpose() * Q_u;
    V_xx.noalias() += Q_ux_.transpose() * K;
    V_xx.noalias() += Q_xx_;
    V_xx.noalias() += Q_xx_.transpose();

    V_x  = Q_x;
    V_x.noalias() += Q_ux_.transpose() * d;
    V_x.noalias() += K.transpose() * Q_u;
    V_x.noalias() += Q_ux.transpose() * d;

    Q_u_.noalias() = Q_uu * d; 
    
    exp_cost_redu[0] = -Q_u.transpose() * d;
    exp_cost_redu[1] = -0.5 * d.transpose() * Q_u_;

}

template<int n, int m>
inline void backward_pass(const double regu,
                          const iLQRParams &params,
                          TrajTree<n, m> &traj_tree,
                          void *instance,
                          ilqr_stage_cost_expansion_t<n, m> stage_cost_expansion,
                          ilqr_dynamics_expansion_t<n, m> dynamics_expansion,
                          double* total_expected_cost_reduction)
{
    for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
    {
        int N = traj_tree.N;
        KnotpointData<n, m>& knotpoint = traj_tree(N, br_id);
        stage_cost_expansion(instance, params, knotpoint);        
    }
    double total_expected_cost_reduction_tmp1 = 0.;
    double total_expected_cost_reduction_tmp2 = 0.;

    {
        omp_set_num_threads(traj_tree.num_branch);
        #pragma omp parallel for reduction(+:total_expected_cost_reduction_tmp1, total_expected_cost_reduction_tmp2)
        // #pragma omp for reduction(+:total_expected_cost_reduction_tmp1, total_expected_cost_reduction_tmp2)
        for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
        {
            double exp_cost_redu1_thread = 0.0;
            double exp_cost_redu2_thread = 0.0;

            for (int k = traj_tree.N - 1; k >= traj_tree.N_sh; --k)        
            {
                KnotpointData<n, m>& knotpoint = traj_tree(k, br_id);
                stage_cost_expansion(instance, params, knotpoint);
                dynamics_expansion(knotpoint.x, knotpoint.u, knotpoint.A, knotpoint.B);    
                
                compute_linear_policy(knotpoint.A, knotpoint.B,
                                    knotpoint.l_xx, knotpoint.l_ux, knotpoint.l_uu, 
                                    knotpoint.l_x, knotpoint.l_u,
                                    traj_tree(k + 1, br_id).V_xx, traj_tree(k + 1, br_id).V_x,
                                    regu,
                                    knotpoint.V_xx, knotpoint.V_x,
                                    knotpoint.K, knotpoint.d,
                                    knotpoint.exp_cost_redu);
        
                exp_cost_redu1_thread += knotpoint.exp_cost_redu[0];
                exp_cost_redu2_thread += knotpoint.exp_cost_redu[1];
            }

            total_expected_cost_reduction_tmp1 += exp_cost_redu1_thread;
            total_expected_cost_reduction_tmp2 += exp_cost_redu2_thread;

        }
    }

    total_expected_cost_reduction[0] = total_expected_cost_reduction_tmp1;
    total_expected_cost_reduction[1] = total_expected_cost_reduction_tmp2;

    int N_sh = traj_tree.N_sh;
    MatState<n> V_xx_sh; 
    VecState<n> V_x_sh; 
    MatState<n> l_xx_sh;
    MatInputState<n, m> l_ux_sh;
    MatInput<m> l_uu_sh;
    VecState<n> l_x_sh;
    VecInput<m> l_u_sh;
    
    for (int k = N_sh - 1; k >= 0; --k)
    {
        if (k == N_sh - 1)
        {
            V_xx_sh.setZero(); 
            V_x_sh.setZero();
            for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
            {
                V_xx_sh += traj_tree(k + 1, br_id).V_xx;
                V_x_sh  += traj_tree(k + 1, br_id).V_x;
            }
        }
        else
        {
            V_xx_sh = traj_tree(k + 1, 0).V_xx;
            V_x_sh  = traj_tree(k + 1, 0).V_x;
        }
        l_xx_sh.setZero();
        l_ux_sh.setZero();
        l_uu_sh.setZero();
        l_x_sh.setZero();
        l_u_sh.setZero();
        for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
        {
            KnotpointData<n, m>& knotpoint = traj_tree(k, br_id);
            stage_cost_expansion(instance, params, knotpoint);
            l_xx_sh += knotpoint.l_xx;
            l_ux_sh += knotpoint.l_ux;
            l_uu_sh += knotpoint.l_uu;
            l_x_sh  += knotpoint.l_x;
            l_u_sh  += knotpoint.l_u;
        }
        KnotpointData<n, m>& knotpoint = traj_tree(k, 0);
        dynamics_expansion(knotpoint.x, knotpoint.u, knotpoint.A, knotpoint.B);
        compute_linear_policy(knotpoint.A, knotpoint.B,
                              l_xx_sh, l_ux_sh, l_uu_sh, 
                              l_x_sh, l_u_sh,
                              V_xx_sh, V_x_sh,
                              regu,
                              knotpoint.V_xx, knotpoint.V_x,
                              knotpoint.K, knotpoint.d,
                              knotpoint.exp_cost_redu);
        for (int br_id = 1; br_id < traj_tree.num_branch; ++br_id)
        {
            traj_tree(k, br_id).K = traj_tree(k, 0).K;
            traj_tree(k, br_id).d = traj_tree(k, 0).d;
        }
        total_expected_cost_reduction[0] += knotpoint.exp_cost_redu[0];
        total_expected_cost_reduction[1] += knotpoint.exp_cost_redu[1];
    }
}

template<int n, int m>
inline bool forward_pass(const double &cost,    
                         const double* total_exp_cost_redu,
                         const iLQRParams &params,
                         void *instance,
                         ilqr_stage_cost_t<n, m> stage_cost,
                         ilqr_dynamics_t<n, m> dynamics,
                         TrajTree<n, m> &traj_tree, 
                         double &cost_new,
                         double &regu)
{  
    bool update_accepted = false;
    double alpha = 1.;
    for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
    {
        traj_tree(0, br_id).x_tmp = traj_tree(0, br_id).x;
    }    
    for (int iter = 0; iter < params.MAX_LS_ITER; ++iter)
    {
        for (int k = 0; k < traj_tree.N + 1; ++k)
        {            
            for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
            {                
                KnotpointData<n, m>& knotpoint = traj_tree(k, br_id);
                knotpoint.u_tmp = knotpoint.u + 
                                  knotpoint.K * (knotpoint.x_tmp - knotpoint.x) + 
                                  alpha * knotpoint.d;
                dynamics(knotpoint.x_tmp, knotpoint.u_tmp, traj_tree(k + 1, br_id).x_tmp);       
            }
        }
        cost_new = compute_trajectory_cost(traj_tree, instance, stage_cost, params);

        if(cost_new <= cost - params.ls_accept_ratio * (alpha * total_exp_cost_redu[0] + alpha * alpha * total_exp_cost_redu[1]))
        {
            update_accepted = true;
            traj_tree.update_state_input_trajectory_trees();
            if (alpha == 1.)
                regu = std::max(0.1 * regu, params.regu_min);
            return update_accepted;
        }
        alpha *= params.ls_decay_rate;
    }
    regu = std::min(10. * regu, params.regu_max);
    cost_new = cost;
    traj_tree.reset_state_input_trajectory_trees();
    std::cout << "Line search failed!" << std::endl;
    return update_accepted;
}

template<int n, int m>
inline bool max_optimize(
    void *instance,
    ilqr_stage_cost_t<n, m> stage_cost,
    const iLQRParams &params,
    TrajTree<n, m> &traj_tree,
    SolverData &solver_data
)
{
    double alpha = params.alpha;
    double beta_init = params.beta;
    double beta;
    double gamma = params.gamma;
    bool converged = false;
    Eigen::Matrix<double, 4, 1> q_;
    std::vector<double> probs(traj_tree.num_branch);
    std::vector<double> branch_costs(traj_tree.num_branch);
    Eigen::Matrix<double, 4, 1> lb;
    Eigen::Matrix<double, 4, 1> ub;
    Eigen::Matrix<double, 4, 1> c;
    double prob_norm;
    for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
    {
        prob_norm = traj_tree(0, br_id).prob_norm;
        branch_costs[br_id] = compute_trajectory_cost(traj_tree, instance, stage_cost, br_id, params);
        probs[br_id] = traj_tree(0, br_id).prob;

        beta = beta_init / std::sqrt(solver_data.total_iter + 1);    // useful? add sqrt?
        q_(br_id) = std::max((1 - beta * gamma), 1e-2) * probs[br_id] + gamma * branch_costs[br_id]; // To avoid too small step size

        lb(br_id) = 0.01;
        ub(br_id) = 1.0 / alpha * prob_norm;
        c(br_id) = 1.0;
    }
    proj::proj_hyperplane_box<4>(q_, c, lb, ub);

    if ((q_ - solver_data.q_pre).norm() < 1e-3)
    {
        converged = true;
    }
    solver_data.q_pre = q_;
    solver_data.q_hist.back().push_back(q_);

    for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
    {
        for (int k = 0; k < traj_tree.N + 1; ++k)
        {
            traj_tree(k, br_id).prob = q_(br_id);
        }
    }

    return converged;
}

template<int n, int m>
inline double gradient(TrajTree<n, m> &traj_tree)
{
    double avg_grad = 0.0;
    for (int br_id = 0; br_id < traj_tree.num_branch; ++br_id)
    {
        for (int k = 0; k < traj_tree.N; ++k)
        {
            double u_max = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < traj_tree.num_input; ++i)
            {
                u_max = std::max(u_max, 
                                 std::abs(traj_tree(k, br_id).d[i]) / (std::abs(traj_tree(k, br_id).u[i]) + 1));
            }
            avg_grad += u_max;
        }
    }
    avg_grad /= (traj_tree.num_branch * traj_tree.N);
    return avg_grad;
}

/* Risk-neutral */
template<int n, int m>
inline int ilqr_optimize(
    void *instance,
    ilqr_stage_cost_t<n, m> stage_cost,
    ilqr_stage_cost_expansion_t<n, m> stage_cost_expansion,
    ilqr_dynamics_t<n, m> dynamics,
    ilqr_dynamics_expansion_t<n, m> dynamics_expansion,
    const iLQRParams &params,
    TrajTree<n, m> &traj_tree,
    SolverData &solver_data,
    std::shared_ptr<Visualizer>& visualizer_ptr
)

{
    double cost = 0.;
    double cost_new = 0.;
    bool update_accepted = false;
    bool ilqr_converged = false;
    double total_expected_cost_reduction[2] = {0., 0.};
    double regu = params.regu_init;
    traj_tree.reset_state_input_trajectory_trees();
    solver_data.q_hist.push_back(std::vector<Eigen::VectorXd>());

    for (int iter = 0; iter < params.MAX_ITER; ++iter)
    {
        cost = compute_trajectory_cost(traj_tree, instance, stage_cost, params);
        backward_pass(regu,
                      params,
                      traj_tree, 
                      instance,
                      stage_cost_expansion, 
                      dynamics_expansion, 
                      total_expected_cost_reduction);
        update_accepted = forward_pass(cost, 
                                       total_expected_cost_reduction, 
                                       params, 
                                       instance,
                                       stage_cost, 
                                       dynamics, 
                                       traj_tree, 
                                       cost_new, 
                                       regu);
        double avg_grad = gradient(traj_tree);
        // std::cout << "cost: " << cost << ", cost_new: " << cost_new << ", grad: " << avg_grad << std::endl;
        ilqr_converged = (
            update_accepted && 
            (std::abs(cost - cost_new) <= params.cost_redu_tol * (1. + cost_new) || avg_grad <= params.grad_tol)
            // (std::abs(cost - cost_new) < params.cost_redu_tol || avg_grad <= params.grad_tol)
        );
        if (ilqr_converged)
        {
            std::cout << "iter: " << iter << std::endl;  
            // std::cout << "Converged!" << std::endl;
            
            solver_data.total_iter += 1;

            return ILQR_CONVERGED;
        }
        else if (!update_accepted)
        {
            // std::cout << "Update rejected!" << std::endl;
        }
        else
        {
            double length = 4.0;
            double width = 2.0;
            double inter_axle_length = 3.0;
            // visualizer_ptr->visualize(traj_tree, length, width, inter_axle_length);
            // solver_data.total_iter += 1;
        }
        solver_data.total_iter += 1;

    }
    
    return ILQR_MAX_ITER_REACHED;
}

/* Risk-aware */
template<int n, int m>
inline int risk_ilqr_optimize(
    void *instance,
    ilqr_stage_cost_t<n, m> stage_cost,
    ilqr_stage_cost_expansion_t<n, m> stage_cost_expansion,
    ilqr_dynamics_t<n, m> dynamics,
    ilqr_dynamics_expansion_t<n, m> dynamics_expansion,
    const iLQRParams &params,
    TrajTree<n, m> &traj_tree,
    SolverData &solver_data,
    std::shared_ptr<Visualizer>& visualizer_ptr
)

{
    double cost = 0.;
    double cost_new = 0.;
    bool update_accepted = false;
    bool ilqr_converged = false;
    bool max_opt_converged = false;

    double total_expected_cost_reduction[2] = {0., 0.};
    double regu = params.regu_init;
    traj_tree.reset_state_input_trajectory_trees();
    
    solver_data.q_hist.push_back(std::vector<Eigen::VectorXd>());

    for (int iter = 0; iter < params.MAX_ITER; ++iter)
    {
        cost = compute_trajectory_cost(traj_tree, instance, stage_cost, params);
        backward_pass(regu,
                      params,
                      traj_tree, 
                      instance,
                      stage_cost_expansion, 
                      dynamics_expansion, 
                      total_expected_cost_reduction);
        update_accepted = forward_pass(cost, 
                                       total_expected_cost_reduction, 
                                       params, 
                                       instance,
                                       stage_cost, 
                                       dynamics, 
                                       traj_tree, 
                                       cost_new, 
                                       regu);
        double avg_grad = gradient(traj_tree);
        // std::cout << "cost_diff: " << std::abs(cost_new - cost) << ", grad: " << avg_grad << std::endl;
        ilqr_converged = (
            update_accepted && 
            (std::abs(cost - cost_new) <= params.cost_redu_tol * (1. + cost_new) || avg_grad <= params.grad_tol) && 
            // (std::abs(cost - cost_new) <= params.cost_redu_tol || avg_grad <= params.grad_tol) && 
            // (std::abs(cost - cost_new) <= params.cost_redu_tol && avg_grad <= params.grad_tol) && 
            max_opt_converged
        );
        if (ilqr_converged)
        {
            std::cout << "iter: " << iter << std::endl;  
            // std::cout << "Converged!" << std::endl;
            solver_data.total_iter += 1;

            return ILQR_CONVERGED;
        }
        else if (!update_accepted)
        {
            // std::cout << "Update rejected!" << std::endl;
        }
        else
        {
            // max_opt_converged = max_optimize(instance, stage_cost, params, traj_tree, solver_data);
    
            // double length = 4.0;
            // double width = 2.0;
            // double inter_axle_length = 3.0;
            // visualizer_ptr->visualize(traj_tree, length, width, inter_axle_length);

            // solver_data.total_iter += 1;
        }
        solver_data.total_iter += 1;

    }

    return ILQR_MAX_ITER_REACHED;
}

}   // namespace cilqr_tree