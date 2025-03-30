#pragma once
#include <vector>
#include <Eigen/Dense>

namespace cilqr_tree
{

struct SolverData
{
    std::vector<std::vector<Eigen::VectorXd>> q_hist; 
    Eigen::VectorXd q_pre;
    int total_iter;
    bool success = false;
    double alpha;
};

} // namespace cilqr_tree