#
# optim.jl
#
# NMOpt.jl is a module implementing a simple versions of BFGS and two
# derivated algorithm with limited memory and constraints
#
#
#------------------------------------------------------------------------------
#
# This file is part of the NMOpt module which is licensed under the MIT "Expat" License:
#
# Copyright (C) 2022,  Matthew Ozon.
#
#------------------------------------------------------------------------------

# BFGS algorithm: compute Nbfgs iteration from the initial point X0 and the initial approximation of the inverse Hessian matrix H0

"""

    BFGS(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false)

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

"""
function BFGS(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false)
    # allocate some variables and init TO DO: set as global variables
    p = Array{Cdouble,1}(undef,length(X0))
    y = Array{Cdouble,1}(undef,length(X0))
    y_tmp = Array{Cdouble,1}(undef,length(X0))
    # H = Array{Cdouble,2}(undef,size(H0,1),size(H0,2))
    H = copy(H0)
    # X = Array{Cdouble,1}(undef,length(X0))
    X = copy(X0)
    y_tmp = Fgrad(X)
    Xpath = Array{Cdouble,2}(undef,Nbfgs+1,length(X0))
    Xpath[1,:] = X0
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
        Xpath[k+1,:] = X
        # set gradient difference
        y = Fgrad(X) - y_tmp
        y_tmp = y + y_tmp
        # update the approximation of the inverse Hessian matrix
        # H = H - (1.0/(y'*H*y)[1])*H*y*y'*H + (1.0/(y'*s)[1])*s*s' # TODO: check for the norm of s and y, and for the orthogonality of y and s
        # H = H - (1.0/(s'*H*s))*H*s*s'*H' + (1.0/(y'*s))*y*y'
        H = H + ((s'*y + y'*H*y)/((s'*y)^2))*s*s' - (1.0/(s'*y))*(H*y*s'+s*y'*H')
        if verbose
            println("iteration: ", k)
            # println(X)
            # println(H)
            println(norm(s,2))
            println(norm(y,2))
            println("\n")
        end
        normS = norm(s,2)
        normY = norm(y,2)
        cosSY = sum(y.*s)/(norm(s,2)*norm(y,2))
        if ( (normS<tol) | (normY<tol) | any(isnan.([normS;normY])) | any(isinf.([normS;normY])))
            if verbose
                # println(s)
                # println(y)
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
        # if norm(s,2)<1.0e-8
        #     Nlast = k+1
        #     break
        # end
    end
    X,H,Xpath,Nlast
end






"""

    BFGSB(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false)

    compute ̂x̂ ∈ argmin{F(x) | x.>= lx and x.<=ux}  using the quasi-Newton method BFGS (see [BFGS wikipedia](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm))

    X0:                  starting point
    H0:                  initial inverse Hessian matrix
    Nbfgs:               maximum number of iterations
    alpha_min,alpha_max: line search maximum range
    mu:                  line search parameter
    Nsearch:             maximum number of iteration for the line search
    F:                   cost function to minimize (x::Array{Cdouble}->F(x))
    Fgrad:               gradient of the cost function (x::Array{Cdouble}->Fgrad(x))
    lx,ux:               lower and upper boundary of ̂x 
    tol:                 relative tolerance (stopping criteria: norm of the gradient difference (y), norm of the step (s) and cos(y,s))

"""
function BFGSB(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false)
    # allocate some variables and init
    p = Array{Cdouble,1}(undef,length(X0))
    y = Array{Cdouble,1}(undef,length(X0))
    y_tmp = Array{Cdouble,1}(undef,length(X0))
    H = Array{Cdouble,2}(undef,size(H0,1),size(H0,2))
    H = H0
    X = Array{Cdouble,1}(undef,length(X0))
    X = X0
    y_tmp = Fgrad(X)
    Xpath = Array{Cdouble,2}(undef,Nbfgs+1,length(X0))
    Xpath[1,:] = X0
    Nlast = Nbfgs+1

    #BFGS iterations
    for k in 1:Nbfgs
        # descent direction
        p = -H*y_tmp
        # enforce the constraints
        for idx_e = 1:length(X)
            if ((X[idx_e]<=lx[idx_e]) & (y_tmp[idx_e]>0.0) ) # & (p[idx_e]<0.0))
            # if (X[idx_e]<=lx[idx_e]) & (p[idx_e]<0.0)
                p[idx_e] = 0.0
            end
            if ((X[idx_e]>=ux[idx_e]) & (y_tmp[idx_e]<0.0) ) # & (p[idx_e]>0.0))
            # if (X[idx_e]>=ux[idx_e]) & (p[idx_e]>0.0)
                p[idx_e] = 0.0
            end
        end
        # step length
        alpha_k = line_search(X,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
        # set step and new state
        s = alpha_k*p
        X = X + s
        # enforce the constraints
        for idx_e in eachindex(X)  #1:length(X)
            if (X[idx_e]<=lx[idx_e])
                X[idx_e] = lx[idx_e]
            end
            if (X[idx_e]>=ux[idx_e])
                X[idx_e] = ux[idx_e]
            end
        end
        # keep track of the path
        Xpath[k+1,:] = X
        # set gradient difference
        y = Fgrad(X) - y_tmp
        y_tmp = y + y_tmp
        # update the approximation of the inverse Hessian matrix
        # H = H - (1.0/(y'*H*y)[1])*H*y*y'*H + (1.0/(y'*s)[1])*s*s' # TODO: check for the norm of s and y, and for the orthogonality of y and s
        # H = H - (1.0/(s'*H*s))*H*s*s'*H' + (1.0/(y'*s))*y*y'
        H = H + ((s'*y + y'*H*y)/((s'*y)^2))*s*s' - (1.0/(s'*y))*(H*y*s'+s*y'*H')
        if verbose
            println("iteration: ", k)
            # println(X)
            println(norm(s,2))
            println(norm(y,2))
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
    end
    X,H,Xpath,Nlast
end

























function LBFGS(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,M::Int64,F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false)
    # M is the number of vectors stored
    # init
    Xpath = Array{Cdouble,2}(undef,Nbfgs+1,length(X0))
    Xpath[1,:] = X0
    y_tmp = Fgrad(X0)
    p = -H0*y_tmp
    alpha_k = line_search(X0,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
    s = alpha_k*p
    X = X0 + s
    Xpath[2,:] = X
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
        # compute the line serach
        alpha_k = line_search(X,p,alpha_min,alpha_max,mu,F,Fgrad,Nsearch;verbose=verbose)
        # evolution step
        s = alpha_k*p
        # new state
        X = X + s
        # keep track of the path
        Xpath[k+1,:] = X
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
        if (any(cosSY.<tol) | (any(isnan.(cosSY))) )
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



# L-BFGS with boundary constraints
function LBFGSB(X0::Array{Cdouble,1},H0::Array{Cdouble,2},Nbfgs::Int64,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,M::Int64,lx::Array{Cdouble,1},ux::Array{Cdouble,1},F::Function,Fgrad::Function,Nsearch::Int64=50,tol::Cdouble=1.0e-8;verbose::Bool=false)
    # M is the number of vectors stored
    # init
    Xpath = Array{Cdouble,2}(undef,Nbfgs+1,length(X0))
    Xpath[1,:] = X0
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
    Xpath[2,:] = X
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
        Xpath[k+1,:] = X
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