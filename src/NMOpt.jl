#
# NMOpt.jl --
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


# TODO: find a better line search algorithm

"""
This is the [`NMOpt`](@ref), it contains 
* [`NMOpt.BFGS`](@ref)
* [`NMOpt.BFGSB`](@ref)
* [`NMOpt.LBFGS`](@ref)
* [`NMOpt.LBFGSB`](@ref)
* [`NMOpt.belong_to`](@ref)
* [`NMOpt.line_search`](@ref)

ref:
  - More and Sorensen 1982 (Newton's method technical report): BFGS
  - More 1994 (Line Search Algorithms Sufficient Decrease): line search
  - Nocedal 1980 (Updating Quasi-Newton Matrices With Limited Storage): limited memory BFGS
  - Thiébaut (Optimization issues in blind deconvolution algorithms): bounded limited memory BFGS, a.k.a vmlmb
"""
module NMOpt # Newton based optimization Method 

using LinearAlgebra # norm is in there

export belong_to, line_search # those functions are not mean to be used by the end user and shouldn't be exported, but they are for the time being # trial_alpha, interval_update, 

# probably not the best names
export BFGS, BFGSB, LBFGS, LBFGSB # BFGS, constrained BFGS, limited memory BFGS and bounded limited memory BFGS

# implementation
include("lineSearch.jl")
include("optim.jl")

end
