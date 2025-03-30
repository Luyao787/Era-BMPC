#pragma once

#include <Eigen/Dense>

namespace proj
{
    template <int d>
    inline void proj_box(
        Eigen::Matrix<double, d, 1>& x, 
        const Eigen::Matrix<double, d, 1>& lb, 
        const Eigen::Matrix<double, d, 1>& ub)
    {
        for (int i = 0; i < d; i++)
        {
            if (x(i) < lb(i))   
                x(i) = lb(i);
            else if (x(i) > ub(i))
                x(i) = ub(i);
        }
    }
    
    template <int d>
    inline double fun(const double mu, 
                      const Eigen::Matrix<double, d, 1>& x, 
                      const Eigen::Matrix<double, d, 1>& c,
                      const Eigen::Matrix<double, d, 1>& lb,
                      const Eigen::Matrix<double, d, 1>& ub)
    {
        Eigen::Matrix<double, d, 1> y;
        y = x - mu * c;
        proj_box(y, lb, ub);
        return c.transpose() * y - 1;
    }

    template <int d>
    inline double bisection(double a, 
                            double b, 
                            double tol,
                            const Eigen::Matrix<double, d, 1>& x,
                            const Eigen::Matrix<double, d, 1>& c,
                            const Eigen::Matrix<double, d, 1>& lb,
                            const Eigen::Matrix<double, d, 1>& ub)
    {
        double fa = fun(a, x, c, lb, ub);
        double fb = fun(b, x, c, lb, ub);
        if (fa * fb > 0)
        {
            std::cout << "Error: f(a) and f(b) have the same sign." << std::endl;
            return 0;
        }
        double m = a;
        while ((b - a) >= tol)
        {
            m = (a + b) / 2;
            double fm = fun(m, x, c, lb, ub);
            if (fm == 0)
                break;
            else if (fa * fm < 0)
            {
                b = m;
                fb = fm;
            }
            else
            {
                a = m;
                fa = fm;
            }
        }
        return m;
    }
    
    template <int d>
    inline void proj_hyperplane_box(
        Eigen::Matrix<double, d, 1>& x, 
        const Eigen::Matrix<double, d, 1>& c, 
        const Eigen::Matrix<double, d, 1>& lb, 
        const Eigen::Matrix<double, d, 1>& ub)
    {
        double bisec_lower = -1;
        double bisec_upper =  1;
        while (fun(bisec_lower, x, c, lb, ub) < 0)
        {
            bisec_lower *= 2;
        }
        while (fun(bisec_upper, x, c, lb, ub) > 0)
        {
            bisec_upper *= 2;
        }

        double mu = bisection(bisec_lower, bisec_upper, 1e-6, x, c, lb, ub);
        Eigen::Matrix<double, d, 1> y;
        y = x - mu * c;
        proj_box(y, lb, ub);
        x = y;
    }

} // namespace proj
