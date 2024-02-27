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

# struct with all the BFGS parameters 
include("BFGS_struct.jl")

# BFGS algorithm: compute Nbfgs iteration from the initial point X0 and the initial approximation of the inverse Hessian matrix H0
include("BFGS.jl")

include("BFGSB.jl")

include("LBFGS.jl")

include("LBFGSB.jl")
