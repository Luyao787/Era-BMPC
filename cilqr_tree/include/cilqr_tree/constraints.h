#pragma once

namespace cilqr_tree
{

class Constraints
{
static constexpr int num_state = 6;
static constexpr int num_input = 2;

private:
    VecState<num_state-num_input> x_max_, x_min_;
    VecInput<num_input> u_max_, u_min_;
    const int num_state_sub_ = num_state - num_input;

public:
    Constraints() = default;
    Constraints(const iLQRParams& params): 
        x_max_(params.x_max), x_min_(params.x_min), 
        u_max_(params.u_max), u_min_(params.u_min) { }

    inline void eval_box_cons_x(const VecState<num_state> &x, 
                                VecState<2*(num_state-num_input)> &hx)
    {
        VecState<num_state-num_input> x_sub = x.head(num_state_sub_);
        hx.head(num_state_sub_) = x_sub - x_max_;
        hx.tail(num_state_sub_) = x_min_ - x_sub;
    }

    inline void eval_box_cons_u(const VecInput<num_input> &u, 
                                VecInput<2*num_input> &hu)
    {
        hu.head(num_input) = u - u_max_;
        hu.tail(num_input) = u_min_ - u;
    }

    inline void eval_box_cons_x_jac(Eigen::Matrix<double, 2*(num_state-num_input), num_state> &jac_hx)
    {
        jac_hx.setZero();
        jac_hx.block(0, 0, num_state_sub_, num_state_sub_).setIdentity();
        jac_hx.block(num_state_sub_, 0, num_state_sub_, num_state_sub_) = -Matrix<double, num_state-num_input, num_state-num_input>::Identity();
    }

    inline void eval_box_cons_u_jac(Eigen::Matrix<double, 2*num_input, num_input> &jac_hu)
    {
        jac_hu.setZero();
        jac_hu.block(0, 0, num_input, num_input).setIdentity();
        jac_hu.block(num_input, 0, num_input, num_input) = -Matrix<double, num_input, num_input>::Identity();
    }

    void eval_ineq_cons(void *ptr, int k, int branch_id, const VecState<num_state> &x, Eigen::VectorXd &hx);

    void eval_ineq_cons_jac(void *ptr, int k, int branch_id, const VecState<num_state> &x, Eigen::MatrixXd &jac_hx);
    
    inline void eval_mask(const VecState<2*(num_state-num_input)> &mu, 
                          const VecState<2*(num_state-num_input)> &h,
                          MatState<2*(num_state-num_input)> &mask)
    {
        mask.setZero();
        for (int i = 0; i < mu.size(); ++i)
        {
            if (mu(i) > 0. || h(i) > 0.) 
                mask(i, i) = 1.;
        }
    }

    inline void eval_mask(const VecState<2*num_input> &mu, 
                          const VecState<2*num_input> &h,
                          MatState<2*num_input> &mask)
    {
        mask.setZero();
        for (int i = 0; i < mu.size(); ++i)
        {
            if (mu(i) > 0. || h(i) > 0.) 
                mask(i, i) = 1.;
        }
    }

    inline void eval_mask(const Eigen::VectorXd &mu, 
                          const Eigen::VectorXd &h,
                          Eigen::MatrixXd &mask)
    {
        mask.resize(mu.size(), mu.size());
        mask.setZero();
        for (int i = 0; i < mu.size(); ++i)
        {
            if (mu(i) > 0. || h(i) > 0.) 
                mask(i, i) = 1.;
        }
    }

    inline double compute_augmented_cost(
        void *ptr,
        int k, 
        int branch_id,
        const VecState<num_state> &x, 
        const VecInput<num_input> &u, 
        const VecState<2*(num_state-num_input)> &mu_x, 
        const VecInput<2*num_input> &mu_u,
        const Eigen::VectorXd &mu_x_ineq,
        const double rho,
        VecState<2*(num_state-num_input)> &hx,
        VecInput<2*num_input> &hu,
        Eigen::VectorXd &hx_ineq,
        MatState<2*(num_state-num_input)> &mask_x,
        MatInput<2*num_input> &mask_u,
        Eigen::MatrixXd &mask_x_ineq
    )
    {
        double cost = 0.;
        eval_box_cons_x(x, hx);
        eval_mask(mu_x, hx, mask_x);
        cost += (mu_x.transpose() * hx + 0.5 * rho * hx.transpose() * mask_x * hx)(0);

        eval_box_cons_u(u, hu);
        eval_mask(mu_u, hu, mask_u);
        cost += (mu_u.transpose() * hu + 0.5 * rho * hu.transpose() * mask_u * hu)(0);

        eval_ineq_cons(ptr, k, branch_id, x, hx_ineq);
        eval_mask(mu_x_ineq, hx_ineq, mask_x_ineq);

        cost += (mu_x_ineq.transpose() * hx_ineq + 0.5 * rho * hx_ineq.transpose() * mask_x_ineq * hx_ineq)(0);

        return cost;
    }   

    inline double compute_augmented_cost(
        void *ptr,
        int k,
        int branch_id,
        const VecState<num_state> &x, 
        const VecState<2*(num_state-num_input)> &mu_x, 
        const Eigen::VectorXd &mu_x_ineq,
        const double rho,
        VecState<2*(num_state-num_input)> &hx,
        Eigen::VectorXd &hx_ineq,
        MatState<2*(num_state-num_input)> &mask_x,
        Eigen::MatrixXd &mask_x_ineq
    )
    {
        double cost;
        eval_box_cons_x(x, hx);
        eval_mask(mu_x, hx, mask_x);
        cost = (mu_x.transpose() * hx + 0.5 * rho * hx.transpose() * mask_x * hx)(0);

        eval_ineq_cons(ptr, k, branch_id, x, hx_ineq);
        eval_mask(mu_x_ineq, hx_ineq, mask_x_ineq);
        cost += (mu_x_ineq.transpose() * hx_ineq + 0.5 * rho * hx_ineq.transpose() * mask_x_ineq * hx_ineq)(0);

        return cost;
    }

    inline void compute_augmented_cost_expansion(
        void *ptr,
        int k,
        int branch_id,
        const VecState<num_state> &x, 
        const VecInput<num_input> &u, 
        const VecState<2*(num_state-num_input)> &mu_x, 
        const VecInput<2*num_input> &mu_u,
        const Eigen::VectorXd &mu_x_ineq,
        const double rho,
        VecState<2*(num_state-num_input)> &hx,
        VecInput<2*num_input> &hu,
        Eigen::VectorXd &hx_ineq,
        MatState<2*(num_state-num_input)> &mask_x,
        MatInput<2*num_input> &mask_u,
        Eigen::MatrixXd &mask_x_ineq,
        Eigen::Matrix<double, 2*(num_state-num_input), num_state> &jac_hx, 
        Eigen::Matrix<double, 2*num_input, num_input> &jac_hu,
        Eigen::MatrixXd &jac_hx_ineq,  
        MatState<num_state> &l_xx,
        MatInput<num_input> &l_uu,
        VecState<num_state> &l_x,
        VecInput<num_input> &l_u
    )
    {
        eval_box_cons_x(x, hx);
        eval_mask(mu_x, hx, mask_x);
        eval_box_cons_x_jac(jac_hx);
        l_x += jac_hx.transpose() * (mu_x + rho * mask_x * hx);
        l_xx += rho * jac_hx.transpose() * mask_x * jac_hx;

        eval_box_cons_u(u, hu);
        
        eval_mask(mu_u, hu, mask_u);
        eval_box_cons_u_jac(jac_hu);
        l_u += jac_hu.transpose() * (mu_u + rho * mask_u * hu);
        l_uu += rho * jac_hu.transpose() * mask_u * jac_hu;

        eval_ineq_cons(ptr, k, branch_id, x, hx_ineq);
        eval_mask(mu_x_ineq, hx_ineq, mask_x_ineq);
        eval_ineq_cons_jac(ptr, k, branch_id, x, jac_hx_ineq);
        l_x += jac_hx_ineq.transpose() * (mu_x_ineq + rho * mask_x_ineq * hx_ineq);
        l_xx += rho * jac_hx_ineq.transpose() * mask_x_ineq * jac_hx_ineq;
    }

    inline void compute_augmented_cost_expansion(
        void *ptr,
        int k,
        int branch_id,
        const VecState<num_state> &x, 
        const VecState<2*(num_state-num_input)> &mu_x, 
        const Eigen::VectorXd &mu_x_ineq,
        const double rho,
        VecState<2*(num_state-num_input)> &hx,
        Eigen::VectorXd &hx_ineq,
        MatState<2*(num_state-num_input)> &mask_x,
        Eigen::MatrixXd &mask_x_ineq,
        Eigen::Matrix<double, 2*(num_state-num_input), num_state> &jac_hx,
        Eigen::MatrixXd &jac_hx_ineq,
        MatState<num_state> &l_xx,
        VecState<num_state> &l_x
    )
    {
        eval_box_cons_x(x, hx);
        eval_mask(mu_x, hx, mask_x);
        eval_box_cons_x_jac(jac_hx);
        l_x += jac_hx.transpose() * (mu_x + rho * mask_x * hx);
        l_xx += rho * jac_hx.transpose() * mask_x * jac_hx;
        eval_ineq_cons(ptr, k, branch_id, x, hx_ineq);        
        eval_mask(mu_x_ineq, hx_ineq, mask_x_ineq);        
        eval_ineq_cons_jac(ptr, k, branch_id, x, jac_hx_ineq);
        l_x += jac_hx_ineq.transpose() * (mu_x_ineq + rho * mask_x_ineq * hx_ineq);
        l_xx += rho * jac_hx_ineq.transpose() * mask_x_ineq * jac_hx_ineq;
    }

    inline double update_mu_x(const VecState<num_state> &x,
                              const double rho,
                              VecState<2*(num_state-num_input)> &hx,
                              VecState<2*(num_state-num_input)> &mu_x)
    {
        eval_box_cons_x(x, hx);
        for (int i = 0; i < mu_x.size(); ++i)
        {
            if (mu_x(i) > 0. || hx(i) > 0.)
                mu_x(i) = std::max(0., mu_x(i) + rho * hx(i));

        }
        return ((hx + hx.cwiseAbs())/2.).cwiseAbs().maxCoeff();
    }

    inline double update_mu_u(const VecInput<num_input> &u,
                              const double rho,
                              VecInput<2*num_input> &hu,
                              VecInput<2*num_input> &mu_u)
    {        
        eval_box_cons_u(u, hu);
        for (int i = 0; i < mu_u.size(); ++i)
        {
            if (mu_u(i) > 0. || hu(i) > 0.)
                mu_u(i) = std::max(0., mu_u(i) + rho * hu(i));
        }
        return ((hu + hu.cwiseAbs())/2.).cwiseAbs().maxCoeff();
    }

    inline double update_mu_x_ineq(
        void *ptr,
        int k,
        int branch_id,
        const VecState<num_state> &x,
        const double rho,
        Eigen::VectorXd &hx_ineq,
        Eigen::VectorXd &mu_x_ineq
    )
    {
        eval_ineq_cons(ptr, k, branch_id, x, hx_ineq);
        for (int i = 0; i < mu_x_ineq.size(); ++i)
        {
            if (mu_x_ineq(i) > 0. || hx_ineq(i) > 0.)
                mu_x_ineq(i) = std::max(0., mu_x_ineq(i) + rho * hx_ineq(i));
        }
        return ((hx_ineq + hx_ineq.cwiseAbs())/2.).cwiseAbs().maxCoeff();
    }

};

} // namespace cilqr_tree