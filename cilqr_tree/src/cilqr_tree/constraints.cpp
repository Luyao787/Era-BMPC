#include "cilqr_tree/cilqr_tree.h"
#include "cilqr_tree/constraints.h"
#include "cilqr_tree/eigen_types.h"

namespace cilqr_tree
{
    void Constraints::eval_ineq_cons(
        void *ptr, 
        int k,
        int branch_id,
        const VecState<num_state> &x, 
        Eigen::VectorXd &hx)
    {
        BranchMPC &obj = *static_cast<BranchMPC*>(ptr);
        std::vector<VecState<2>> vertices;
        obj.get_vehicle_vertices(x, vertices);
        int num_vertices = vertices.size();
        // TODO: hand-crafted
        Eigen::VectorXd hx_lin(2 * num_vertices);
        hx_lin.setZero();

        KnotpointData<num_state, num_input> &knotpoint = obj.traj_tree(k, branch_id);
        for (int i = 0; i < num_vertices; ++i)
        {
            hx_lin.segment(2 * i, 2) = knotpoint.A_lin * vertices[i] - knotpoint.b_lin;
        }

        Eigen::VectorXd hx_nonlin;
        obj.safety_constraints(x.head(num_state-num_input), k, branch_id, hx_nonlin);

        // if (branch_id == 0 && k == 49)
        // {
        //     std::cout << hx_lin.size() << ", " << hx_nonlin.size() << std::endl;
        //     std::cout << "hx_nonlin: " << hx_nonlin.transpose() << std::endl;
        // }

        hx.resize(hx_lin.size() + hx_nonlin.size());
        hx << hx_lin, hx_nonlin;
        // std::cout << "---" << std::endl;
    }
    
    void Constraints::eval_ineq_cons_jac(
        void *ptr,
        int k, 
        int branch_id,
        const VecState<num_state> &x, 
        Eigen::MatrixXd &jac_hx)
    {
        BranchMPC &obj = *static_cast<BranchMPC*>(ptr);
        std::vector<VecState<2>> vertices;
        obj.get_vehicle_vertices(x, vertices);
        int num_vertices = vertices.size();
        
        double length_ , width_;
        width_ = obj.vehicle_width_ / 2.;

        Eigen::MatrixXd jac_hx_lin(2 * num_vertices, num_state);
        jac_hx_lin.setZero();
        KnotpointData<num_state, num_input> &knotpoint = obj.traj_tree(k, branch_id);
        for (int i = 0; i < num_vertices; ++i)
        {
            Eigen::Matrix<double, 2, num_state> jac_tmp;
            jac_tmp.setZero();
            jac_tmp.block(0, 0, 2, 2).setIdentity();
            // TODO: hand-crafted   
            if (i == 0)
            {
                length_ = obj.vehicle_length_ / 2. + obj.inter_axle_length_ / 2.;
                jac_tmp(0, 2) = -length_ * sin(x(2)) + width_ * cos(x(2));
                jac_tmp(1, 2) =  length_ * cos(x(2)) + width_ * sin(x(2));
            }
            else if (i == 1)
            {
                length_ = obj.vehicle_length_ / 2. + obj.inter_axle_length_ / 2.;
                jac_tmp(0, 2) = -length_ * sin(x(2)) - width_ * cos(x(2));
                jac_tmp(1, 2) =  length_ * cos(x(2)) - width_ * sin(x(2));
            }
            else if (i == 2)
            {
                length_ = obj.vehicle_length_ / 2. - obj.inter_axle_length_ / 2.;
                jac_tmp(0, 2) =  length_ * sin(x(2)) - width_ * cos(x(2));
                jac_tmp(1, 2) = -length_ * cos(x(2)) - width_ * sin(x(2));
            }
            else if (i == 3)
            {
                length_ = obj.vehicle_length_ / 2. - obj.inter_axle_length_ / 2.;
                jac_tmp(0, 2) =  length_ * sin(x(2)) + width_ * cos(x(2));
                jac_tmp(1, 2) = -length_ * cos(x(2)) + width_ * sin(x(2));
            }

            jac_hx_lin.block(2 * i, 0, 2, num_state) = knotpoint.A_lin * jac_tmp;

        }

        Eigen::MatrixXd jac_hx_nonlin;
        obj.safety_constraints_jac(x.head(num_state-num_input), k, branch_id, jac_hx_nonlin);

        // std::cout << "jac_hx_lin: " << jac_hx_lin.rows() << " " << jac_hx_lin.cols() << std::endl;
        // std::cout << "jac_hx_nonlin: " << jac_hx_nonlin.rows() << " " << jac_hx_nonlin.cols() << std::endl;

        jac_hx.resize(jac_hx_lin.rows() + jac_hx_nonlin.rows(), num_state);
        jac_hx << jac_hx_lin, jac_hx_nonlin;

    }

}   // namespace cilqr_tree