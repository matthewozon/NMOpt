"""
    BFGS_param

    Nbfgs:               maximum number of iterations
    alpha_min,alpha_max: line search maximum range
    mu:                  line search parameter
    M:                   number of stored vectors (limit memory usage compared with the computation of the inverse of the Hessian matrix)
    Nsearch:             maximum number of iteration for the line search
    tol:                 relative tolerance (stopping criteria: norm of the gradient difference (y), norm of the step (s) and cos(y,s))
"""
mutable struct BFGS_param
    # parameters
    Nbfgs::Int64
    alpha_min::Cdouble
    alpha_max::Cdouble
    mu::Cdouble
    M::Int64
    Nsearch::Int64
    tol::Cdouble

    # constructors
    function BFGS_param(Niter::Int64)
        new(Niter,-4.0,4.0,0.4,10,50,1.0e-8)
    end
    function BFGS_param(ws::BFGS_param)
        # new(ws.Niter,ws.alpha_min,ws.alpha_max,ws.mu,ws.M,ws.Nsearch,ws.tol)
        new(ws...)
    end
end

