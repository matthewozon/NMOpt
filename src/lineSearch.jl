#
# lineSearch.jl
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

# line search optimization problem
function phi(x_::Array{Cdouble,1},alpha::Cdouble,p_::Array{Cdouble,1},F::Function)
    F(x_+alpha*p_)
end
function phi(x_::Cdouble,alpha::Cdouble,p_::Cdouble,F::Function)
    F(x_+alpha*p_)
end

function phi_deriv(x_::Array{Cdouble,1},alpha::Cdouble,p_::Array{Cdouble,1},Fgrad::Function)
    df = Fgrad(x_+alpha*p_)
    sum(df.*p_)
end
function phi_deriv(x_::Cdouble,alpha::Cdouble,p_::Cdouble,Fgrad::Function)
    df = Fgrad(x_+alpha*p_)
    df*p_
end




# line serach
function trial_alpha(alpha_l::Cdouble,alpha_u::Cdouble)
    0.5*(alpha_l+alpha_u) # but it should be much more complicated... yet it works
end

function interval_update(ft::Cdouble, fl::Cdouble,gt::Cdouble,alpha_l::Cdouble,alpha_t::Cdouble,alpha_u::Cdouble)
    # compute the modified updating algorithm  of More 1994 (Line Search Algorithms Sufficient Decrease)
    if ft>fl
        alpha_l_new = alpha_l
        alpha_u_new = alpha_t
    else
        if gt*(alpha_l-alpha_t)>=0.0
            alpha_l_new = alpha_t
            alpha_u_new = alpha_u
        else
            alpha_l_new = alpha_t
            alpha_u_new = alpha_l
        end
    end
    alpha_l_new,alpha_u_new
end


"""

    belong_to(x_::Array{Cdouble,1},p_::Array{Cdouble,1},alpha_::Cdouble,mu_::Cdouble,F::Function,Fgrad::Function)
    belong_to(x_::Cdouble,p_::Cdouble,alpha_::Cdouble,mu_::Cdouble,F::Function,Fgrad::Function)
    
    check the criterion 

    F(x+α*p)<F(x)+μ*α ⟨p,∂F/∂x(x)⟩

    |⟨p,∂F/∂x(x+α*p)⟩|⩽μ|⟨p,∂F/∂x(x)⟩|

    return true if both criterion are met, false otherwise 
"""
function belong_to(x_::Array{Cdouble,1},p_::Array{Cdouble,1},alpha_::Cdouble,mu_::Cdouble,F::Function,Fgrad::Function)
    # if ((phi(x_,alpha_,p_,F) < phi(x_,0.0,p_,F) + mu_*alpha_*phi_deriv(x_,0.0,p_,Fgrad)) & (abs(phi_deriv(x_,alpha_,p_,Fgrad))<=mu_*abs(phi_deriv(x_,0.0,p_,Fgrad))))
    #     it_belongs_to = true
    # else
    #     it_belongs_to = false
    # end
    # it_belongs_to
    (phi(x_,alpha_,p_,F) < phi(x_,0.0,p_,F) + mu_*alpha_*phi_deriv(x_,0.0,p_,Fgrad)) & (abs(phi_deriv(x_,alpha_,p_,Fgrad))<=mu_*abs(phi_deriv(x_,0.0,p_,Fgrad)))
end

function belong_to(x_::Cdouble,p_::Cdouble,alpha_::Cdouble,mu_::Cdouble,F::Function,Fgrad::Function)
    # if ((phi(x_,alpha_,p_,F) < phi(x_,0.0,p_,F) + mu_*alpha_*phi_deriv(x_,0.0,p_,Fgrad)) & (abs(phi_deriv(x_,alpha_,p_,Fgrad))<=mu_*abs(phi_deriv(x_,0.0,p_,Fgrad))))
    #     it_belongs_to = true
    # else
    #     it_belongs_to = false
    # end
    # it_belongs_to

    ((phi(x_,alpha_,p_,F) < phi(x_,0.0,p_,F) + mu_*alpha_*phi_deriv(x_,0.0,p_,Fgrad)) & (abs(phi_deriv(x_,alpha_,p_,Fgrad))<=mu_*abs(phi_deriv(x_,0.0,p_,Fgrad))))
end

"""
    line_search(x_::Array{Cdouble,1},p_::Array{Cdouble,1},alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50;verbose::Bool=false)

    compute ̂̂α ∈ argmin{F(x+α*p)}

    inputs:

      - x: point x
      - p: search direction 
      - α_min, and α_max:  minimum and maximum values for α
      - μ: weight for the criterion F(x+α*p)<F(x)+μ*α ⟨p,∂F/∂x(x)⟩, and  |⟨p,∂F/∂x(x+α*p)⟩|⩽μ|⟨p,∂F/∂x(x)⟩|
      - F and Fgrad: cost function and its gradient 
      - Nsearch: maximum number of iteration for the line search 

    optional argument:

      - verbose: verbose if true 

    output: 

      - α within the vincinity of the ̂α
      
"""
function line_search(x_::Array{Cdouble,1},p_::Array{Cdouble,1},alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50;verbose::Bool=false)
    # init the search
    if phi(x_,alpha_min,p_,F)<phi(x_,alpha_max,p_,F)
        alpha_l = alpha_min
        alpha_u = alpha_max
    else
        alpha_l = alpha_max
        alpha_u = alpha_min
    end
    alpha_t = trial_alpha(alpha_l,alpha_u)
    for i in 1:Nsearch
        # choose a trial step length
        alpha_t = trial_alpha(alpha_l,alpha_u)
        # check if it belongs to the acceptable set
        if belong_to(x_,p_,alpha_t,mu,F,Fgrad)
            # break the loop and return the current value of alpha
            if verbose
                println(i)
                println("final step length: ", alpha_t)
            end
            break
        else
            # continue the loop
            ft = phi(x_,alpha_t,p_,F)
            fl = phi(x_,alpha_l,p_,F)
            gt = phi_deriv(x_,alpha_t,p_,Fgrad)
            alpha_l,alpha_u = interval_update(ft,fl,gt,alpha_l,alpha_t,alpha_u)
        end
        if i==Nsearch
            if verbose
                println(i)
                println("no cvg yet, final step length: ", alpha_t)
            end
        end
    end
    alpha_t
end

function line_search(x_::Cdouble,p_::Cdouble,alpha_min::Cdouble,alpha_max::Cdouble,mu::Cdouble,F::Function,Fgrad::Function,Nsearch::Int64=50;verbose::Bool=false)
    # init the search
    if phi(x_,alpha_min,p_,F)<phi(x_,alpha_max,p_,F)
        alpha_l = alpha_min
        alpha_u = alpha_max
    else
        alpha_l = alpha_max
        alpha_u = alpha_min
    end
    alpha_t = trial_alpha(alpha_l,alpha_u)
    for i in 1:Nsearch
        # choose a trial step length
        alpha_t = trial_alpha(alpha_l,alpha_u)
        # check if it belongs to the acceptable set
        if belong_to(x_,p_,alpha_t,mu,F,Fgrad)
            # break the loop and return the current value of alpha
            if verbose
                println(i)
                println("final step length: ", alpha_t)
            end
            break
        else
            # continue the loop
            ft = phi(x_,alpha_t,p_,F)
            fl = phi(x_,alpha_l,p_,F)
            gt = phi_deriv(x_,alpha_t,p_,Fgrad)
            alpha_l,alpha_u = interval_update(ft,fl,gt,alpha_l,alpha_t,alpha_u)
        end
        if i==Nsearch
            if verbose
                println(i)
                println("no cvg yet, final step length: ", alpha_t)
            end
        end
    end
    alpha_t
end
