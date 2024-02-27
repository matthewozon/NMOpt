using Test
using NMOpt 

function test_belong_to()
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + (x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 2*(x[2].-1.5)]
    end

    μ = 0.4
    α1 = 1.0
    α2 = 2.0
    p = [1.0; 1.0]
    x = [0.0; 0.0]

    belong_to(x,p,α1,μ,F,Fgrad) & (!belong_to(x,p,α2,μ,F,Fgrad))
end

function test_line_search()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + (x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 2*(x[2].-1.5)]
    end

    X0 = -ones(Cdouble,2)
    p0 = ones(Cdouble,2)                 # first descent direction
    alpha_min = -4.0         # smallest value of the length of the step
    alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    mu = 0.4                 # <0.5 parameter for the line search algorithm

    α_searched =  line_search(X0,p0,alpha_min,alpha_max,mu,F,Fgrad)

    cond1 = (!isinf(α_searched)) & (!isnan(α_searched))
    cond2 = (α_searched<=alpha_max) & (alpha_min<=α_searched)

    # optimization problem
    function F_scalar(x::Cdouble)
        (x-1.5)^2
    end
    
    function Fgrad_scalar(x::Cdouble)
        2*(x-1.5)
    end


    X0 = -1.0
    p0 = 1.0                 # first descent direction
    alpha_min = -4.0         # smallest value of the length of the step
    alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    mu = 0.4                 # <0.5 parameter for the line search algorithm

    α_searched =  line_search(X0,p0,alpha_min,alpha_max,mu,F_scalar,Fgrad_scalar)

    cond3 = (!isinf(α_searched)) & (!isnan(α_searched))
    cond4 = (α_searched<=alpha_max) & (alpha_min<=α_searched)

    cond1 & cond2 & cond3 & cond4
end

@testset "BFGS algorithms" begin
    @test test_belong_to()
    @test test_line_search()
end

function test_BFGS()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + (x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 2*(x[2].-1.5)]
    end



    X0 = -ones(Cdouble,2)
    alpha_min = -4.0         # smallest value of the length of the step
    alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    mu = 0.4                 # <0.5 parameter for the line search algorithm
    Nbfgs = 100              # number of iterations
    Nsearch = 10             # maximum number of iteration for the line search
    H0 = [2.0 0.0; 0.0 2.0]                 # initial Hessian matrix #0.000001*

    Xend,Hend,Xpath,Nlast = BFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,F,Fgrad,Nsearch)

    cond1 = isapprox(sum((Xend-[1.5;1.5]).^2),0.0,atol=1.0e-15)
    cond2 = !any(isnan.(Hend)) & !any(isinf.(Hend))
    cond3 = (Nlast<=(Nbfgs+1))
    cond4 = !any(isnan.(Xpath[1:Nlast,:])) & !any(isinf.(Xpath[1:Nlast,:]))


    # optimization problem
    function F_scalar(x::Cdouble)
        (x-1.5)^2
    end
    
    function Fgrad_scalar(x::Cdouble)
        2*(x-1.5)
    end


    X0 = -1.0
    alpha_min = -4.0         # smallest value of the length of the step
    alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    mu = 0.4                 # <0.5 parameter for the line search algorithm
    Nbfgs = 100              # number of iterations
    Nsearch = 10             # maximum number of iteration for the line search
    H0 = 2.0                 # initial Hessian matrix #0.000001*

    Xend,Hend,Xpath,Nlast = BFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,F_scalar,Fgrad_scalar,Nsearch)

    cond5 = isapprox(abs(Xend-1.5),0.0,atol=1.0e-15)
    cond6 = !(isnan(Hend)) & !(isinf.(Hend))
    cond7 = (Nlast<=(Nbfgs+1))
    cond8 = !any(isnan.(Xpath[1:Nlast])) & !any(isinf.(Xpath[1:Nlast]))

    cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
end

function test_BFGS_struct()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + (x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 2*(x[2].-1.5)]
    end

    X0 = -ones(Cdouble,2)
    ws = BFGS_param(100)        # number of iterations = 100
    ws.alpha_min = -4.0         # smallest value of the length of the step
    ws.alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    ws.mu = 0.4                 # <0.5 parameter for the line search algorithm
    ws.Nsearch = 10             # maximum number of iteration for the line search
    H0 = [2.0 0.0; 0.0 2.0]                 # initial Hessian matrix #0.000001*

    Xend,Hend,Xpath,Nlast = BFGS(X0,H0,F,Fgrad,ws)

    cond1 = isapprox(sum((Xend-[1.5;1.5]).^2),0.0,atol=1.0e-15)
    cond2 = !any(isnan.(Hend)) & !any(isinf.(Hend))
    cond3 = (Nlast<=(ws.Nbfgs+1))
    cond4 = !any(isnan.(Xpath[1:Nlast,:])) & !any(isinf.(Xpath[1:Nlast,:]))


    # optimization problem
    function F_scalar(x::Cdouble)
        (x-1.5)^2
    end
    
    function Fgrad_scalar(x::Cdouble)
        2*(x-1.5)
    end


    X0 = -1.0
    H0 = 2.0                 # initial Hessian matrix #0.000001*

    Xend,Hend,Xpath,Nlast = BFGS(X0,H0,F_scalar,Fgrad_scalar,ws)

    cond5 = isapprox(abs(Xend-1.5),0.0,atol=1.0e-15)
    cond6 = !(isnan(Hend)) & !(isinf.(Hend))
    cond7 = (Nlast<=(ws.Nbfgs+1))
    cond8 = !any(isnan.(Xpath[1:Nlast])) & !any(isinf.(Xpath[1:Nlast]))

    cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
end


function test_BFGSB()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + (x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 2*(x[2].-1.5)]
    end


    lx = [-Inf, -Inf]
    ux = [Inf,0.0]
    X0 = -ones(Cdouble,2)
    alpha_min = -4.0         # smallest value of the length of the step
    alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    mu = 0.4                 # <0.5 parameter for the line search algorithm
    Nbfgs = 100              # number of iterations
    Nsearch = 10             # maximum number of iteration for the line search
    H0 = [2.0 0.0; 0.0 2.0]                 # initial Hessian matrix #0.000001*

    Xend,Hend,Xpath,Nlast = BFGSB(X0,H0,Nbfgs,alpha_min,alpha_max,mu,lx,ux,F,Fgrad,Nsearch)

    cond1 = isapprox(sum((Xend-[1.5;0.0]).^2),0.0,atol=1.0e-15)
    cond2 = !any(isnan.(Hend)) & !any(isinf.(Hend))
    cond3 = (Nlast<=(Nbfgs+1))
    cond4 = !any(isnan.(Xpath[1:Nlast,:])) & !any(isinf.(Xpath[1:Nlast,:]))


    # optimization problem
    function F_scalar(x::Cdouble)
        (x-1.5)^2
    end
    
    function Fgrad_scalar(x::Cdouble)
        2*(x-1.5)
    end


    lx = -Inf
    ux = 0.0
    X0 = -1.0
    alpha_min = -4.0         # smallest value of the length of the step
    alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    mu = 0.4                 # <0.5 parameter for the line search algorithm
    Nbfgs = 100              # number of iterations
    Nsearch = 10             # maximum number of iteration for the line search
    H0 = 2.0                 # initial Hessian matrix #0.000001*


    Xend,Hend,Xpath,Nlast = BFGSB(X0,H0,Nbfgs,alpha_min,alpha_max,mu,lx,ux,F_scalar,Fgrad_scalar,Nsearch)

    cond5 = isapprox(abs(Xend-0.0),0.0,atol=1.0e-15)
    cond6 = !(isnan(Hend)) & !(isinf.(Hend))
    cond7 = (Nlast<=(Nbfgs+1))
    cond8 = !any(isnan.(Xpath[1:Nlast])) & !any(isinf.(Xpath[1:Nlast]))

    cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
end

function test_BFGSB_struct()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + (x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 2*(x[2].-1.5)]
    end


    lx = [-Inf, -Inf]
    ux = [Inf,0.0]
    X0 = -ones(Cdouble,2)
    ws = BFGS_param(100)        # number of iterations = 100
    ws.alpha_min = -4.0         # smallest value of the length of the step
    ws.alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    ws.mu = 0.4                 # <0.5 parameter for the line search algorithm
    ws.Nsearch = 10             # maximum number of iteration for the line search
    H0 = [2.0 0.0; 0.0 2.0]                 # initial Hessian matrix #0.000001*

    Xend,Hend,Xpath,Nlast = BFGSB(X0,H0,lx,ux,F,Fgrad,ws)

    cond1 = isapprox(sum((Xend-[1.5;0.0]).^2),0.0,atol=1.0e-15)
    cond2 = !any(isnan.(Hend)) & !any(isinf.(Hend))
    cond3 = (Nlast<=(ws.Nbfgs+1))
    cond4 = !any(isnan.(Xpath[1:Nlast,:])) & !any(isinf.(Xpath[1:Nlast,:]))


    # optimization problem
    function F_scalar(x::Cdouble)
        (x-1.5)^2
    end
    
    function Fgrad_scalar(x::Cdouble)
        2*(x-1.5)
    end


    lx = -Inf
    ux = 0.0
    X0 = -1.0
    H0 = 2.0                 # initial Hessian matrix #0.000001*


    Xend,Hend,Xpath,Nlast = BFGSB(X0,H0,lx,ux,F_scalar,Fgrad_scalar,ws)

    cond5 = isapprox(abs(Xend-0.0),0.0,atol=1.0e-15)
    cond6 = !(isnan(Hend)) & !(isinf.(Hend))
    cond7 = (Nlast<=(ws.Nbfgs+1))
    cond8 = !any(isnan.(Xpath[1:Nlast])) & !any(isinf.(Xpath[1:Nlast]))

    cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
end

function test_LBFGS()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + 4.0*(x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 4.0*2*(x[2].-1.5)]
    end

    X0 = -ones(Cdouble,2)
    alpha_min = -4.0         # smallest value of the length of the step
    alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    mu = 0.4                 # <0.5 parameter for the line search algorithm
    Nbfgs = 100              # number of iterations
    Nsearch = 10             # maximum number of iteration for the line search
    Mmemory = 4              # number of stored vectors 
    H0 = [1.0 0.0; 0.0 2.0]                 # initial Hessian matrix #0.000001*

    Xend,Xpath,Nlast = LBFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,Mmemory,F,Fgrad,Nsearch)

    cond1 = isapprox(sum((Xend-[1.5;1.5]).^2),0.0,atol=1.0e-15)
    cond2 = (Nlast<=(Nbfgs+1))
    if (!isnothing(Xpath))
        cond3 = !any(isnan.(Xpath[1:Nlast,:])) & !any(isinf.(Xpath[1:Nlast,:]))
    else
        cond3 = true
    end

    cond1 & cond2 & cond3
end

function test_LBFGS_struct()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + 4.0*(x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 4.0*2*(x[2].-1.5)]
    end

    X0 = -ones(Cdouble,2)
    ws = BFGS_param(100)        # number of iterations = 100
    ws.alpha_min = -4.0         # smallest value of the length of the step
    ws.alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    ws.mu = 0.4                 # <0.5 parameter for the line search algorithm
    ws.Nsearch = 10             # maximum number of iteration for the line search
    ws.M = 4                    # number of stored vectors 
    H0 = [1.0 0.0; 0.0 2.0]     # initial Hessian matrix #0.000001*

    Xend,Xpath,Nlast = LBFGS(X0,H0,F,Fgrad,ws)

    cond1 = isapprox(sum((Xend-[1.5;1.5]).^2),0.0,atol=1.0e-15)
    cond2 = (Nlast<=(ws.Nbfgs+1))
    if (!isnothing(Xpath))
        cond3 = !any(isnan.(Xpath[1:Nlast,:])) & !any(isinf.(Xpath[1:Nlast,:]))
    else
        cond3 = true
    end

    cond1 & cond2 & cond3
end

function test_LBFGSB()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + 40.0*(x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 40.0*2*(x[2].-1.5)]
    end

    lx = [-Inf, -Inf]
    ux = [Inf,0.0]
    X0 = -ones(Cdouble,2)
    alpha_min = -4.0         # smallest value of the length of the step
    alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    mu = 0.4                 # <0.5 parameter for the line search algorithm
    Nbfgs = 100              # number of iterations
    Nsearch = 10             # maximum number of iteration for the line search
    Mmemory = 4              # number of stored vectors 
    H0 = [1.0 0.0; 0.0 2.0]  # initial Hessian matrix #0.000001*

    Xend,Xpath,Nlast = LBFGSB(X0,H0,Nbfgs,alpha_min,alpha_max,mu,Mmemory,lx,ux,F,Fgrad,Nsearch)

    cond1 = isapprox(sum((Xend-[1.5;0.0]).^2),0.0,atol=1.0e-15)
    cond2 = (Nlast<=(Nbfgs+1))
    if (!isnothing(Xpath))
        cond3 = !any(isnan.(Xpath[1:Nlast,:])) & !any(isinf.(Xpath[1:Nlast,:]))
    else
        cond3 = true
    end

    cond1 & cond2 & cond3
end

function test_LBFGSB_struct()
    # optimization problem
    function F(x::Array{Cdouble,1})
        (x[1]-1.5)^2 + 40.0*(x[2]-1.5)^2
    end
    
    function Fgrad(x::Array{Cdouble,1})
        [2*(x[1].-1.5); 40.0*2*(x[2].-1.5)]
    end

    lx = [-Inf, -Inf]
    ux = [Inf,0.0]
    X0 = -ones(Cdouble,2)
    ws = BFGS_param(100)        # number of iterations = 100
    ws.alpha_min = -4.0         # smallest value of the length of the step
    ws.alpha_max = 4.0          # smallest value of the length of the step 2000000.0
    ws.mu = 0.4                 # <0.5 parameter for the line search algorithm
    ws.Nsearch = 10             # maximum number of iteration for the line search
    ws.M = 4                    # number of stored vectors  
    H0 = [1.0 0.0; 0.0 2.0]  # initial Hessian matrix #0.000001*

    Xend,Xpath,Nlast = LBFGSB(X0,H0,lx,ux,F,Fgrad,ws)

    cond1 = isapprox(sum((Xend-[1.5;0.0]).^2),0.0,atol=1.0e-15)
    cond2 = (Nlast<=(ws.Nbfgs+1))
    if (!isnothing(Xpath))
        cond3 = !any(isnan.(Xpath[1:Nlast,:])) & !any(isinf.(Xpath[1:Nlast,:]))
    else
        cond3 = true
    end

    cond1 & cond2 & cond3
end

@testset "BFGS algorithms" begin
    @test test_BFGS()
    @test test_BFGS_struct()
    @test test_BFGSB()
    @test test_BFGSB_struct()
    @test test_LBFGS()
    @test test_LBFGS_struct()
    @test test_LBFGSB()
    @test test_LBFGSB_struct()
end