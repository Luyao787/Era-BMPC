#pragma once

#include <Eigen/Dense>
#include <ros/ros.h>

using namespace Eigen;

namespace cilqr_tree
{

struct  iLQRParams
{
    iLQRParams() = default;

    iLQRParams(const ros::NodeHandle &nh)
    {
        std::cout << ".....Loading iLQR parameters....." << std::endl;
        nh.getParam("L", L);
        nh.getParam("dt", dt);
        nh.getParam("N", N);
        nh.getParam("N_sh", N_sh);
        nh.getParam("alpha", alpha);
        nh.getParam("num_branch", num_branch);
        nh.getParam("num_state", num_state);
        nh.getParam("num_aug_state", num_aug_state);
        nh.getParam("num_input", num_input);
        nh.getParam("weight_state", weight_state);
        nh.getParam("weight_input", weight_input);
        nh.getParam("weight_comfort", weight_comfort);
        nh.getParam("weight_safety", safety_weight);
        nh.getParam("state_max", state_max);
        nh.getParam("state_min", state_min);
        nh.getParam("input_max", input_max);
        nh.getParam("input_min", input_min);
        nh.getParam("safety_margin", safety_margin);
        nh.getParam("r_saf", r_saf);
        nh.getParam("rho_init", rho_init);
        nh.getParam("rho_max", rho_max);
        nh.getParam("rho_scaling", rho_scaling);
        nh.getParam("MAX_ITER", MAX_ITER);
        nh.getParam("MAX_LS_ITER", MAX_LS_ITER);
        nh.getParam("MAX_OUTER_ITER", MAX_OUTER_ITER);
        nh.getParam("ls_accept_ratio", ls_accept_ratio);
        nh.getParam("ls_decay_rate", ls_decay_rate);
        nh.getParam("regu_init", regu_init);
        nh.getParam("regu_max", regu_max);
        nh.getParam("regu_min", regu_min);
        nh.getParam("cost_redu_tol", cost_redu_tol);
        nh.getParam("grad_tol", grad_tol);
        nh.getParam("constr_vio_tol", constr_vio_tol);
        nh.getParam("beta", beta);
        nh.getParam("gamma", gamma);

        Q.setZero();
        Q(0,0) = weight_state[0];
        Q(1,1) = weight_state[1];
        Q(2,2) = weight_state[2];
        Q(3,3) = weight_state[3];
        Qf = 10. * Q;
        // Qf = 5 * Q;
        R.setZero();
        R(0,0) = weight_input[0];
        R(1,1) = weight_input[1];
        R_com.setZero();
        R_com(0,0) = weight_comfort[0];
        R_com(1,1) = weight_comfort[1];

        x_max << state_max[0], state_max[1], state_max[2], state_max[3];
        x_min << state_min[0], state_min[1], state_min[2], state_min[3];
        u_max << input_max[0], input_max[1];
        u_min << input_min[0], input_min[1];
    }

    double L;   
    double dt;

    int N;
    int N_sh;
    int num_branch;
    double alpha;

    int num_state;
    int num_aug_state;
    int num_input;
    
    std::vector<double> weight_state;
    std::vector<double> weight_input;
    std::vector<double> weight_comfort;
    Matrix4d Q;
    Matrix4d Qf;
    Matrix2d R;
    Matrix2d R_com;
    double safety_weight;

    std::vector<double> state_max;
    std::vector<double> state_min;
    std::vector<double> input_max;
    std::vector<double> input_min;
    Vector4d x_max;
    Vector4d x_min;
    Vector2d u_max;
    Vector2d u_min;

    double safety_margin;
    double r_saf;

    double rho_init;
    double rho_max;
    double rho_scaling;

    int MAX_ITER;
    int MAX_LS_ITER;
    int MAX_OUTER_ITER;

    double ls_accept_ratio;
    double ls_decay_rate;

    double regu_init;
    double regu_max;
    double regu_min;

    double cost_redu_tol;
    double grad_tol;
    double constr_vio_tol;

    double beta;
    double gamma;
};


} // namespace ilqr