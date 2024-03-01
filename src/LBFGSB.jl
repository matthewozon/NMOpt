#------------------------------------------------------------------------------
#
# This file is part of the NMOpt module which is licensed under the MIT "Expat" License:
#
# Copyright (C) 2022,  Matthew Ozon.
#
#------------------------------------------------------------------------------

"""

    LBFGSB(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,M::Int64,lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false,path::Bool=true)
    LBFGSB(X0::Array{Cdouble,1},H0::Array{Cdouble,2},lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,ws::BFGS_param;verbose::Bool=false,path::Bool=true)

    LBFGSB(X0::Array{Cdouble,1},p0::Array{Cdouble,1},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,M::Int64,lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false,path::Bool=true)
    LBFGSB(X0::Array{Cdouble,1},p0::Array{Cdouble,1},lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,ws::BFGS_param;verbose::Bool=false,path::Bool=true)

    compute ̂x̂ ∈ argmin{F(x) | x.>= lx and x.<=ux}  using the limitted memory quasi-Newton method L-BFGS  (see [BFGS wikipedia](https://en.wikipedia.org/wiki/Limited-memory_BFGS))

    X0:                  starting point
    H0:                  initial inverse Hessian matrix (this is a dense matrix that may become to heavy, it is only used for )
    p0:                  initial descent direction (-H0*Fgrad(X0))
    Nbfgs:               maximum number of iterations
    alpha_min,alpha_max: line search maximum range
    mu:                  line search parameter
    M:                   number of stored vectors (limit memory usage compared with the computation of the inverse of the Hessian matrix)
    Nsearch:             maximum number of iteration for the line search
    F:                   cost function to minimize (x::Array{Cdouble}->F(x))
    Fgrad:               gradient of the cost function (x::Array{Cdouble}->Fgrad(x))
    lx,ux:               lower and upper boundary of ̂x 
    tol:                 relative tolerance (stopping criteria: norm of the gradient difference (y), norm of the step (s) and cos(y,s))

    ws:                  BFGS_param structure with the BFGS parameters [`BFGS_param`](@ref)

    optional arguments:

      - verbose: by default set to false 
      - path:    default true, keep track of all the steps between the initial point X0 and the arrival point X (if false: Xpath=nothing)

    output:

      - X:     final state of the optimization process (in the vincinity of argmin{F(x)})
      - Xpath: steps between the initial point X0 and the arrival point X
      - Nlast: number of iteration 
"""
function LBFGSB(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,M::Int64,lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false,path::Bool=true)
    # M is the number of vectors stored
    # init
    if path
        Xpath = Array{Cdouble,2}(undef,Nbfgs+1,length(X0))
        Xpath[1,:] = X0
    else
        Xpath = nothing
    end
    y_tmp = Fgrad(X0)
    p = -H0*y_tmp
    # enforce the constraints
    for idx_e in 1:length(X0)
        if (X0[idx_e]<=lx[idx_e]) & (y_tmp[idx_e]>0.0)
            p[idx_e] = 0.0
        end
        if (X0[idx_e]>=ux[idx_e]) & (y_tmp[idx_e]<0.0)
            p[idx_e] = 0.0
        end
    end
    alpha_k = line_search(X0,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
    s = alpha_k*p
    X = X0 + s
    # enforce the constraints
    for idx_e in eachindex(X) # 1:length(X)
        if (X[idx_e]<=lx[idx_e])
            X[idx_e] = lx[idx_e]
        end
        if (X[idx_e]>=ux[idx_e])
            X[idx_e] = ux[idx_e]
        end
    end
    y = Fgrad(X) - y_tmp
    y_tmp = y + y_tmp
    S_storage = Array{Cdouble,2}(undef,length(X0),M)
    Y_storage = Array{Cdouble,2}(undef,length(X0),M)
    S_storage[:,1] = s
    Y_storage[:,1] = y
    cosSY = ones(Cdouble,M) # keep track of the norm and orthogonality of the stored vectors
    normS = ones(Cdouble,M)
    normY = ones(Cdouble,M)
    normS[1] = norm(s,2)
    normY[1] = norm(y,2)
    cosSY[1] = sum(s.*y)/(normS[1]*normY[1])
    gam = Array{Cdouble,1}(undef,M)
    km = 0.0
    Nlast = Nbfgs+1



    # iteration
    for k=1:Nbfgs
        # compute the direction without computing the Hessian matrix
        if k<M
            km = k
        else
            km = M
        end
        qm = y_tmp
        for i = km:-1:1
            gam[i] = (1.0/(S_storage[:,i]'*Y_storage[:,i])[1])*(S_storage[:,i]'*qm)[1]
            qm = qm - gam[i]*Y_storage[:,i]
        end
        p = H0*qm
        for i = 1:km
            beta_j = (1.0/(S_storage[:,i]'*Y_storage[:,i])[1])*(Y_storage[:,i]'*p)[1]
            p = p + (gam[i] - beta_j)*S_storage[:,i]
        end
        p = -p # because it's a descent
        # enforce the constraints
        for idx_e in eachindex(X) # 1:length(X)
            if (X[idx_e]<=lx[idx_e]) & (y_tmp[idx_e]>0.0)
            #if (X[idx_e]<=lx[idx_e]) & (p[idx_e]<0.0)
                p[idx_e] = 0.0
            end
            if (X[idx_e]>=ux[idx_e]) & (y_tmp[idx_e]<0.0)
            #if (X[idx_e]>=ux[idx_e]) & (p[idx_e]>0.0)
                p[idx_e] = 0.0
            end
        end

        # compute the line serach
        alpha_k = line_search(X,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
        # evolution step
        s = alpha_k*p
        # new state
        X = X + s
        # enforce the constraints
        for idx_e in eachindex(X) # 1:length(X)
            if (X[idx_e]<=lx[idx_e])
                X[idx_e] = lx[idx_e]
            end
            if (X[idx_e]>=ux[idx_e])
                X[idx_e] = ux[idx_e]
            end
        end
        # keep track of the path
        if path
            Xpath[k+1,:] = X
        end
        # gradient difference
        y = Fgrad(X) - y_tmp
        y_tmp = y + y_tmp
        if k<M
            S_storage[:,k+1] = s
            Y_storage[:,k+1] = y
            normS[k+1] = norm(s,2)
            normY[k+1] = norm(y,2)
            cosSY[k+1] = sum(s.*y)/(normS[k+1]*normY[k+1])
        else
            # move the stored vectors
            S_storage[:,1:M-1] = S_storage[:,2:M]
            Y_storage[:,1:M-1] = Y_storage[:,2:M]
            normS[1:M-1] = normS[2:M]
            normY[1:M-1] = normY[2:M]
            cosSY[1:M-1] = cosSY[2:M]
            # push the new one in last position
            S_storage[:,M] = s
            Y_storage[:,M] = y
            normS[M] = norm(s,2)
            normY[M] = norm(y,2)
            cosSY[M] = sum(s.*y)/(normS[M]*normY[M])
        end
        if verbose
            println("iteration: ", k)
            # println(X)
            if k<M
                println([normS[k+1]; normY[k+1]; cosSY[k+1]])
            else
                println([normS[M]; normY[M]; cosSY[M]])
            end
            println("\n")
        end
        # if at least one of the vectors of Y_storage or S_storage is null, we should stop the algorithm
        if ((any(normS.<=tol)) | (any(normY.<=tol)) | (any(isnan.(normY))) | (any(isinf.(normY))) | (any(isnan.(normS))) | (any(isinf.(normS))) )
            if verbose
                println("CVG: norm of S or Y")
                println("\n")
            end
            Nlast = k+1
            break
        end
        # if a pair of vector in S_storage and Y_storage are orthogonal, we should stop the algo
        if (any(abs.(cosSY).<tol) | (any(isnan.(cosSY))) )
            if verbose
                println("CVG: orthogonality of S and Y")
                println("\n")
            end
            Nlast = k+1
            break
        end
    end
    X,Xpath,Nlast
end
function LBFGSB(X0::Array{Cdouble,1},H0::Array{Cdouble,2},lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,ws::BFGS_param;verbose::Bool=false,path::Bool=true)
    LBFGSB(X0,H0,ws.Nbfgs,ws.alpha_min,ws.alpha_max,ws.mu,ws.M,lx,ux,F,Fgrad,ws.Nsearch,ws.tol;verbose=verbose,path=path)
end






function LBFGSB(X0::Array{Cdouble,1},p0::Array{Cdouble,1},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,M::Int64,lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false,path::Bool=true)
    # M is the number of vectors stored
    # init
    if path
        Xpath = Array{Cdouble,2}(undef,Nbfgs+1,length(X0))
        Xpath[1,:] = X0
    else
        Xpath = nothing
    end
    y_tmp = Fgrad(X0)
    p = p0[:] # -H0*y_tmp
    # enforce the constraints
    for idx_e in 1:length(X0)
        if (X0[idx_e]<=lx[idx_e]) & (y_tmp[idx_e]>0.0)
            p[idx_e] = 0.0
        end
        if (X0[idx_e]>=ux[idx_e]) & (y_tmp[idx_e]<0.0)
            p[idx_e] = 0.0
        end
    end
    alpha_k = line_search(X0,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
    s = alpha_k*p
    X = X0 + s
    # enforce the constraints
    for idx_e in eachindex(X) # 1:length(X)
        if (X[idx_e]<=lx[idx_e])
            X[idx_e] = lx[idx_e]
        end
        if (X[idx_e]>=ux[idx_e])
            X[idx_e] = ux[idx_e]
        end
    end
    y = Fgrad(X) - y_tmp
    y_tmp = y + y_tmp
    S_storage = Array{Cdouble,2}(undef,length(X0),M)
    Y_storage = Array{Cdouble,2}(undef,length(X0),M)
    S_storage[:,1] = s
    Y_storage[:,1] = y
    cosSY = ones(Cdouble,M) # keep track of the norm and orthogonality of the stored vectors
    normS = ones(Cdouble,M)
    normY = ones(Cdouble,M)
    normS[1] = norm(s,2)
    normY[1] = norm(y,2)
    cosSY[1] = sum(s.*y)/(normS[1]*normY[1])
    gam = Array{Cdouble,1}(undef,M)
    km = 0.0
    Nlast = Nbfgs+1



    # iteration
    for k=1:Nbfgs
        # compute the direction without computing the Hessian matrix
        if k<M
            km = k
        else
            km = M
        end
        qm = y_tmp
        for i = km:-1:1
            gam[i] = (1.0/(S_storage[:,i]'*Y_storage[:,i])[1])*(S_storage[:,i]'*qm)[1]
            qm = qm - gam[i]*Y_storage[:,i]
        end
        p = H0*qm
        for i = 1:km
            beta_j = (1.0/(S_storage[:,i]'*Y_storage[:,i])[1])*(Y_storage[:,i]'*p)[1]
            p = p + (gam[i] - beta_j)*S_storage[:,i]
        end
        p = -p # because it's a descent
        # enforce the constraints
        for idx_e in eachindex(X) # 1:length(X)
            if (X[idx_e]<=lx[idx_e]) & (y_tmp[idx_e]>0.0)
            #if (X[idx_e]<=lx[idx_e]) & (p[idx_e]<0.0)
                p[idx_e] = 0.0
            end
            if (X[idx_e]>=ux[idx_e]) & (y_tmp[idx_e]<0.0)
            #if (X[idx_e]>=ux[idx_e]) & (p[idx_e]>0.0)
                p[idx_e] = 0.0
            end
        end

        # compute the line serach
        alpha_k = line_search(X,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
        # evolution step
        s = alpha_k*p
        # new state
        X = X + s
        # enforce the constraints
        for idx_e in eachindex(X) # 1:length(X)
            if (X[idx_e]<=lx[idx_e])
                X[idx_e] = lx[idx_e]
            end
            if (X[idx_e]>=ux[idx_e])
                X[idx_e] = ux[idx_e]
            end
        end
        # keep track of the path
        if path
            Xpath[k+1,:] = X
        end
        # gradient difference
        y = Fgrad(X) - y_tmp
        y_tmp = y + y_tmp
        if k<M
            S_storage[:,k+1] = s
            Y_storage[:,k+1] = y
            normS[k+1] = norm(s,2)
            normY[k+1] = norm(y,2)
            cosSY[k+1] = sum(s.*y)/(normS[k+1]*normY[k+1])
        else
            # move the stored vectors
            S_storage[:,1:M-1] = S_storage[:,2:M]
            Y_storage[:,1:M-1] = Y_storage[:,2:M]
            normS[1:M-1] = normS[2:M]
            normY[1:M-1] = normY[2:M]
            cosSY[1:M-1] = cosSY[2:M]
            # push the new one in last position
            S_storage[:,M] = s
            Y_storage[:,M] = y
            normS[M] = norm(s,2)
            normY[M] = norm(y,2)
            cosSY[M] = sum(s.*y)/(normS[M]*normY[M])
        end
        if verbose
            println("iteration: ", k)
            # println(X)
            if k<M
                println([normS[k+1]; normY[k+1]; cosSY[k+1]])
            else
                println([normS[M]; normY[M]; cosSY[M]])
            end
            println("\n")
        end
        # if at least one of the vectors of Y_storage or S_storage is null, we should stop the algorithm
        if ((any(normS.<=tol)) | (any(normY.<=tol)) | (any(isnan.(normY))) | (any(isinf.(normY))) | (any(isnan.(normS))) | (any(isinf.(normS))) )
            if verbose
                println("CVG: norm of S or Y")
                println("\n")
            end
            Nlast = k+1
            break
        end
        # if a pair of vector in S_storage and Y_storage are orthogonal, we should stop the algo
        if (any(abs.(cosSY).<tol) | (any(isnan.(cosSY))) )
            if verbose
                println("CVG: orthogonality of S and Y")
                println("\n")
            end
            Nlast = k+1
            break
        end
    end
    X,Xpath,Nlast
end
function LBFGSB(X0::Array{Cdouble,1},p0::Array{Cdouble,1},lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,ws::BFGS_param;verbose::Bool=false,path::Bool=true)
    LBFGSB(X0,p0,ws.Nbfgs,ws.alpha_min,ws.alpha_max,ws.mu,ws.M,lx,ux,F,Fgrad,ws.Nsearch,ws.tol;verbose=verbose,path=path)
end
