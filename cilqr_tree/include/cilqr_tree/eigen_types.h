#pragma once

#include <Eigen/Dense>

namespace cilqr_tree 
{
    template <int n>
    using VecState = Eigen::Matrix<double, n, 1>;
    
    template <int m>
    using VecInput = Eigen::Matrix<double, m, 1>;

    template <int n>
    using MatState = Eigen::Matrix<double, n, n>;

    template <int m>
    using MatInput = Eigen::Matrix<double, m, m>;

    template <int n, int m>
    using MatInputState = Eigen::Matrix<double, m, n>;

    template <int n, int m>
    using MatStateInput = Eigen::Matrix<double, n, m>;

}   // namespace cilqr_tree