#------------------------------------------------------------------------------
#
# This file is part of the NMOpt module which is licensed under the MIT "Expat" License:
#
# Copyright (C) 2022,  Matthew Ozon.
#
#------------------------------------------------------------------------------

"""

    BFGS(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false,path::Bool=true)
    BFGS(X0::Array{Cdouble,1},H0::Array{Cdouble,2},F::Function,Fgrad::Function,ws::BFGS_param;verbose::Bool=false,path::Bool=true)
    BFGS(X0::Cdouble,H0::Cdouble,Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false,path::Bool=true)
    BFGS(X0::Cdouble,H0::Cdouble,F::Function,Fgrad::Function,ws::BFGS_param;verbose::Bool=false,path::Bool=true)

    compute ̂x̂ ∈ argmin{F(x)}  using the quasi-Newton method BFGS (see [BFGS wikipedia](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm))

    X0:                  starting point
    H0:                  initial inverse Hessian matrix
    Nbfgs:               maximum number of iterations
    alpha_min,alpha_max: line search maximum range
    mu:                  line search parameter
    Nsearch:             maximum number of iteration for the line search
    F:                   cost function to minimize (x::Array{Cdouble}->F(x))
    Fgrad:               gradient of the cost function (x::Array{Cdouble}->Fgrad(x))
    tol:                 relative tolerance (stopping criteria: norm of the gradient difference (y), norm of the step (s) and cos(y,s))

    ws:                  BFGS_param structure with the BFGS parameters [`BFGS_param`](@ref)
    
    optional arguments:

      - verbose:             verbose if set to true
      - path:                keep track of all iterations if set to true (if false: Xpath=nothing)

    output:

      - X:     final state of the optimization process (in the vincinity of argmin{F(x)})
      - H:     estimate of the inverse Hessian matrix 
      - Xpath: steps between the initial point X0 and the arrival point X, or nothing 
      - Nlast: number of iteration 

"""
function BFGS(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false,path::Bool=true)
    # allocate some variables and init TO DO: set as global variables
    p = Array{Cdouble,1}(undef,length(X0))
    y = Array{Cdouble,1}(undef,length(X0))
    y_tmp = Array{Cdouble,1}(undef,length(X0))
    H = copy(H0)
    X = copy(X0)
    y_tmp = Fgrad(X)
    if path
        Xpath = Array{Cdouble,2}(undef,Nbfgs+1,length(X0))
        Xpath[1,:] = X0
    else
        Xpath = nothing
    end
    Nlast = Nbfgs+1

    #BFGS iterations
    for k in 1:Nbfgs
        # descent direction
        p = -H*y_tmp
        # step length
        alpha_k = line_search(X,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
        # set step and new state
        s = alpha_k*p
        X = X + s
        # keep track of the path
        if path
            Xpath[k+1,:] = X
        end
        # set gradient difference
        y = Fgrad(X) - y_tmp
        y_tmp = y + y_tmp
        # stopping criteria 
        if verbose
            println("iteration: ", k)
            println("\n")
        end
        normS = norm(s,2)
        normY = norm(y,2)
        cosSY = sum(y.*s)/(norm(s,2)*norm(y,2))
        if ( (normS<tol) | (normY<tol) | any(isnan.([normS;normY])) | any(isinf.([normS;normY])))
            if verbose
                println("CVG: norm of S or Y")
                println("\n")
            end
            Nlast = k+1
            break
        end
        if ( (abs(cosSY)<tol) | isnan.(cosSY) | isinf.(cosSY) )
            if verbose
                println("CVG: orthogonality of S and Y")
                println("\n")
            end
            Nlast = k+1
            break
        end
        # update the approximation of the inverse Hessian matrix
        H = H + ((s'*y + y'*H*y)/((s'*y)^2))*s*s' - (1.0/(s'*y))*(H*y*s'+s*y'*H')
    end
    X,H,Xpath,Nlast
end
function BFGS(X0::Array{Cdouble,1},H0::Array{Cdouble,2},F::Function,Fgrad::Function,ws::BFGS_param;verbose::Bool=false,path::Bool=true)
    BFGS(X0,H0,ws.Nbfgs,ws.alpha_min,ws.alpha_max,ws.mu,F,Fgrad,ws.Nsearch,ws.tol;verbose=verbose,path=path)
end


function BFGS(X0::Cdouble,H0::Cdouble,Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false,path::Bool=true)
    # allocate some variables and init TO DO: set as global variables
    p = 0.0
    y = 0.0
    y_tmp = 0.0
    H = copy(H0)
    X = copy(X0)
    y_tmp = Fgrad(X)
    Xpath = Array{Cdouble,1}(undef,Nbfgs+1)
    Xpath[1] = X0
    if path
        Xpath = Array{Cdouble,1}(undef,Nbfgs+1)
        Xpath[1] = X0
    else
        Xpath = nothing
    end
    Nlast = Nbfgs+1

    #BFGS iterations
    for k in 1:Nbfgs
        # descent direction
        p = -H*y_tmp
        # step length
        alpha_k = line_search(X,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
        # set step and new state
        s = alpha_k*p
        X = X + s
        # keep track of the path
        if path
            Xpath[k+1] = X
        end
        # set gradient difference
        y = Fgrad(X) - y_tmp
        y_tmp = y + y_tmp
        # stopping criteria 
        if verbose
            println("iteration: ", k)
            println("\n")
        end
        normS = abs(s)
        normY = abs(y)
        if ( (normS<tol) | (normY<tol) | any(isnan.([normS;normY])) | any(isinf.([normS;normY])))
            if verbose
                println("CVG: norm of S or Y")
                println("\n")
            end
            Nlast = k+1
            break
        end
        # update the approximation of the inverse of the second derivative
        H = H + ((s*y + y*H*y)/((s*y)^2))*s*s - (1.0/(s*y))*(H*y*s+s*y*H)
    end
    X,H,Xpath,Nlast
end
function BFGS(X0::Cdouble,H0::Cdouble,F::Function,Fgrad::Function,ws::BFGS_param;verbose::Bool=false,path::Bool=true)
    BFGS(X0,H0,ws.Nbfgs,ws.alpha_min,ws.alpha_max,ws.mu,F,Fgrad,ws.Nsearch,ws.tol;verbose=verbose,path=path)
end