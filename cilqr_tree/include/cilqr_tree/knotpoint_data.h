#pragma once

#include <vector>
#include "eigen_types.h"

namespace cilqr_tree
{

template<int n, int m>
class KnotpointData
{
public:
    VecState<n> x;
    VecInput<m> u;
    VecState<n> x_tmp;
    VecInput<m> u_tmp;

    MatInputState<n, m> K;
    VecInput<m> d;

    MatState<n> V_xx;
    VecState<n> V_x;
    MatState<n> l_xx;
    MatInputState<n, m> l_ux;
    MatInput<m> l_uu;
    VecState<n> l_x;
    VecInput<m> l_u;

    MatState<n> A;
    MatStateInput<n, m> B;

    // TODO: more generic way?
    Eigen::Matrix<double, 2, 2> A_lin;
    Eigen::Matrix<double, 2, 1> b_lin;

    // TODO: more generic way?
    VecState<2*(n-m)> mu_x;
    VecInput<2*m> mu_u;
    Eigen::VectorXd mu_x_ineq;  // related to road boundary constraints
  
    VecState<2*(n-m)> hx;
    VecInput<2*m> hu;
    Eigen::VectorXd hx_ineq;

    Eigen::Matrix<double, 2*(n-m), n> jac_hx;
    Eigen::Matrix<double, 2*m, m> jac_hu;
    Eigen::MatrixXd jac_hx_ineq;
    MatState<2*(n-m)> mask_x; 
    MatInput<2*m> mask_u;
    Eigen::MatrixXd mask_x_ineq;

    
    double exp_cost_redu[2] = {0., 0.};
    bool is_final;
    int k, branch_id;

    VecState<n> x_ref;
    VecInput<m> u_ref;

    double prob = 1.0;
    double prob_norm = 1.0;

public:
    KnotpointData() = default;
    KnotpointData(bool is_final, int k, int branch_id): is_final(is_final), k(k), branch_id(branch_id)
    {
        x.setZero();
        u.setZero();
        x_tmp.setZero();
        u_tmp.setZero();

        K.setZero();
        d.setZero();
        V_xx.setZero();
        V_x.setZero();
        l_xx.setZero();
        l_ux.setZero();
        l_uu.setZero();
        l_x.setZero();
        l_u.setZero();
        A.setZero();
        B.setZero();

        A_lin.setZero();
        b_lin.setZero();

        x_ref.setZero();
        u_ref.setZero();

        mu_x.setZero();
        mu_u.setZero();
    };

private:


};

}
