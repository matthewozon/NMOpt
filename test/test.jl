using PyPlot
using NMOpt


# optimization problem
function F(x_::Array{Cdouble,1})
    exp(x_[1]-1.0) + exp(-x_[2]+1.0) + (x_[1]-x_[2])^2 + 2.0*sin(pi*sum(x_))
end

function Fgrad(x_::Array{Cdouble,1})
    y_ = Array{Cdouble,1}(undef,length(x_))
    y_[1] =  exp(x_[1]-1.0)  + 2.0(x_[1]-x_[2]) + 2.0*pi*cos(pi*sum(x_))
    y_[2] = -exp(-x_[2]+1.0) - 2.0(x_[1]-x_[2]) + 2.0pi*cos(pi*sum(x_))
    y_
end

# other examples
# function F(x_::Array{Cdouble,1})
#     x_[1]^2+x_[2]^2
#     # sin(3.0x_[1]) + sin(3.0x_[2])
# end
# function Fgrad(x_::Array{Cdouble,1})
#     y_ = Array(Cdouble,length(x_))
#     y_[1] = 2.0*x_[1]
#     y_[2] = 2.0*x_[2]
#     # y_[1] = 3.0cos(3.0x_[1])
#     # y_[2] = 3.0cos(3.0x_[2])
#     y_
# end


# for display purposes
XX = collect(-1.0:0.1:3)
YY = collect(-1.0:0.1:3)
XXX = repeat(XX,1,length(XX))
YYY = repeat(YY',length(YY),1)
ZZZ = exp.(XXX.-1.0) .+ exp.(-YYY.+1.0) .+ (XXX.-YYY).^2 .+ 2.0*sin.(pi.*(XXX.+YYY)) # F(X,Y)
# other examples
# ZZZ = XXX.^2+YYY.^2
# ZZZ = sin(3.0XXX)+sin(3.0YYY)


function displayMap(fig::Int64)
    figure(fig)
    pcolormesh(repeat(XX,1,length(YY))',repeat(YY,1,length(XX)),ZZZ,edgecolors="face",shading="None",norm=matplotlib.colors.Normalize(vmin=0.0,vmax=maximum(ZZZ)))
    cbar=colorbar()
    cbar.set_label("intensity")
    cbar.formatter.set_powerlimits((-1,2))
    cbar.update_ticks()
end



X0 = zeros(Cdouble,2)
X0[1] = 2.0 #2.5
# X0[1] = 0.5
X0[2] = -0.5 #1.0 #-0.5
# X0[1] = 0.5461157658396215
# X0[2] = 0.964054633381516
p0 = ones(Cdouble,2)             # first descent direction
alpha_min = -4.0          # smallest value of the length of the step
alpha_max = 4.0          # smallest value of the length of the step 2000000.0
mu = 0.4                 # <0.5 parameter for the line search algorithm
Nbfgs = 100              # number of iterations
Nsearch = 10             # maximum number of iteration for the line search
Mmemory = 4              # size of the memory
H0 = [1.0 0.0; 0.0 1.0]  # initial Hessian matrix #0.000001*

# line_search(X0,p0,alpha_min,alpha_max,mu)

println("BFGS")
Xend,Hend,Xpath,Nlast = BFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,F,Fgrad,Nsearch)



# imshowData(5,XX,YY,ZZZ,_norm=:Normalize,_vmin=0.0,_vmax=maximum(ZZZ),_edgecolors="face")
displayMap(5)
scatter(Xpath[1:Nlast,2],Xpath[1:Nlast,1])
scatter(Xend[2],Xend[1])



println("L-BFGS")
Xend,Xpath,Nlast = LBFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,Mmemory,F,Fgrad,Nsearch)
#imshowData(6,XX,YY,log(ZZZ),_norm=:Normalize,_vmin=0.0,_vmax=maximum(log(ZZZ)),_edgecolors="face")
scatter(Xpath[1:Nlast,2],Xpath[1:Nlast,1])
scatter(Xend[2],Xend[1])






println("BFGS B")
# define the constraints
# lx = [0.75, -Inf]
lx = [-0.5, -Inf]
ux = [Inf,0.0*0.8]
# lx = [0.0, 0.0]
# ux = [Inf,Inf]
Xend,Hend,Xpath,Nlast = BFGSB(X0,H0,2Nbfgs,alpha_min,alpha_max,mu,lx,ux,F,Fgrad,Nsearch)
displayMap(7)
scatter(Xpath[1:Nlast,2],Xpath[1:Nlast,1])
scatter(Xend[2],Xend[1])
plot(YY,lx[1]*ones(length(YY)))
plot(lx[2]*ones(length(XX)),XX)
plot(YY,ux[1]*ones(length(YY)))
plot(ux[2]*ones(length(XX)),XX)
plot(Xpath[1:Nlast,2],Xpath[1:Nlast,1])




println("L-BFGS B")
# define the constraints
# lx = [0.75, -Inf]
# ux = [Inf,0.8]
# lx = [0.0, 0.0]
# ux = [Inf,Inf]
Xend,Xpath,Nlast = LBFGSB(X0,H0,2Nbfgs,alpha_min,alpha_max,mu,Mmemory,lx,ux,F,Fgrad,Nsearch)
# imshowData(7,XX,YY,ZZZ,_norm=:Normalize,_vmin=0.0,_vmax=maximum(ZZZ),_edgecolors="face")
# displayMap(7)
scatter(Xpath[1:Nlast,2],Xpath[1:Nlast,1])
scatter(Xend[2],Xend[1])
# plot(Xpath[1:Nlast,2],Xpath[1:Nlast,1])
# plot(YY,lx[1]*ones(length(YY)))
# plot(lx[2]*ones(length(XX)),XX)
# plot(YY,ux[1]*ones(length(YY)))
# plot(ux[2]*ones(length(XX)),XX)
