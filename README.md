[![NMOpt CI](https://github.com/matthewozon/NMOpt/actions/workflows/CI_NMOpt.yml/badge.svg)](https://github.com/matthewozon/NMOpt/actions/workflows/CI_NMOpt.yml)
[![Documentation](https://github.com/matthewozon/NMOpt/actions/workflows/documentation.yml/badge.svg)](https://github.com/matthewozon/NMOpt/actions/workflows/documentation.yml)

# NMOpt

**BFGS** is an simple implementation of the Newton' methods in Julia.
The main objectives of those methods is to solve convex optimization
problem, possibly with constraints and/or limited memory.

## Examples

**Loading the package**
```
using NMOpt
```

**Problem to optimize**

```
function F(x::Array{Cdouble,1})
    exp(x[1]-1.0) + exp(-x[2]+1.0) + (x[1]-x[2])^2 + 2.0*sin(pi*sum(x))
end

function Fgrad(x::Array{Cdouble,1})
    y = Array{Cdouble,1}(undef,length(x))
    y[1] =  exp(x[1]-1.0)  + 2.0(x[1]-x[2]) + 2.0*pi*cos(pi*sum(x))
    y[2] = -exp(-x[2]+1.0) - 2.0(x[1]-x[2]) + 2.0pi*cos(pi*sum(x))
    y_
end
```

**Initial conditions**
```
X0 = zeros(Cdouble,2)
X0[1] = 2.0 
X0[2] = -0.5
p0 = ones(Cdouble,2)          # first descent direction
alpha_min = -4.0              # smallest value of the length of the step
alpha_max = 4.0               # largest value of the length of the step 
mu = 0.4                      # <0.5 parameter for the line search algorithm
Nbfgs = 100                   # maximum number of iterations for BFGS main loop
Nsearch = 10                  # maximum number of iteration for the line search
Mmemory = 4                   # size of the memory
H0 = 0.01*[1.0 0.0; 0.0 1.0]  # initial Hessian matrix 
```

**BFGS**
```
Xend,Hend,Xpath,Nlast = BFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,F,Fgrad,Nsearch)
```

**L-BFGS**

```
Xend,Xpath,Nlast = LBFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,Mmemory,F,Fgrad,Nsearch)
```

**BFGS-B**
```
Xend,Hend,Xpath,Nlast = BFGSB(X0,H0,2Nbfgs,alpha_min,alpha_max,mu,lx,ux,F,Fgrad,Nsearch)
```

**L-BFGS-B**
```
Xend,Xpath,Nlast = LBFGSB(X0+[-0.5;0.0],H0,2Nbfgs,alpha_min,alpha_max,mu,Mmemory,lx,ux,F,Fgrad,Nsearch)
```


![example_optim_path](https://github.com/matthewozon/NMOpt/assets/7929598/13cec604-4986-4885-b02e-1997523a0bfb)


## Install
```
] add add https://github.com/matthewozon/NMOpt
```

## Related references

  - More and Sorensen 1982 (Newton's method technical report): BFGS
  - More 1994 (Line Search Algorithms Sufficient Decrease): line search
  - Nocedal 1980 (Updating Quasi-Newton Matrices With Limited Storage): limited memory BFGS
  - Thiébaut (Optimization issues in blind deconvolution algorithms): bounded limited memory BFGS, a.k.a vmlmb
