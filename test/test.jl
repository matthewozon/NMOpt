using PyPlot
rc("text", usetex=true)
using NMOpt

# for display purposes
XX = collect(-1.0:0.1:3)
YY = collect(-1.0:0.1:3)
XXX = repeat(XX,1,length(XX))
YYY = repeat(YY',length(YY),1)
ZZZ = exp.(XXX.-1.0) .+ exp.(-YYY.+1.0) .+ (XXX.-YYY).^2 .+ 2.0*sin.(pi.*(XXX.+YYY)) # F(X,Y)


function displayMap(fig::Int64,sub::Int64=111)
    figg = figure(fig)
    ax = subplot(sub)
    pcm = pcolormesh(repeat(XX,1,length(YY))',repeat(YY,1,length(XX)),ZZZ,edgecolors="face",shading="None",norm=matplotlib.colors.Normalize(vmin=0.0,vmax=maximum(ZZZ)))
    cbar=colorbar()
    # cbar.set_label("intensity")
    cbar.formatter.set_powerlimits((-1,2))
    cbar.update_ticks()
    figg,ax,pcm,cbar
end



# optimization problem
"""
    F(x,y) = e^{x-1.0} + e^{-y+1} + (x-y)^2 + 2*sin(π*(x+y))
"""
function F(x_::Array{Cdouble,1})
    exp(x_[1]-1.0) + exp(-x_[2]+1.0) + (x_[1]-x_[2])^2 + 2.0*sin(pi*sum(x_))
end

function Fgrad(x_::Array{Cdouble,1})
    y_ = Array{Cdouble,1}(undef,length(x_))
    y_[1] =  exp(x_[1]-1.0)  + 2.0(x_[1]-x_[2]) + 2.0*pi*cos(pi*sum(x_))
    y_[2] = -exp(-x_[2]+1.0) - 2.0(x_[1]-x_[2]) + 2.0pi*cos(pi*sum(x_))
    y_
end


# initial state and parameter for the optimization
X0 = zeros(Cdouble,2)
X0[1] = 2.0 #2.5
X0[2] = -0.5
p0 = ones(Cdouble,2)             # first descent direction
alpha_min = -4.0          # smallest value of the length of the step
alpha_max = 4.0          # smallest value of the length of the step 2000000.0
mu = 0.4                 # <0.5 parameter for the line search algorithm
Nbfgs = 100              # number of iterations
Nsearch = 10             # maximum number of iteration for the line search
Mmemory = 4              # size of the memory
H0 = 0.01*[1.0 0.0; 0.0 1.0]  # initial Hessian matrix #0.000001*

# define the constraints for each dimension lx=low, ux=up
lx = [-0.5, -Inf]
ux = [Inf,0.8]

# BFGS
println("BFGS")
Xend,Hend,Xpath,Nlast = BFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,F,Fgrad,Nsearch)

# plot 
figure(1,figsize=[10,4])
figg1,ax1,pcm1,cbar1 = displayMap(1,121)
xticks(fontsize=12)
yticks(fontsize=12)
scatter(Xpath[1:Nlast,2],Xpath[1:Nlast,1],label="path BFGS")
scatter(Xend[2],Xend[1],label="optimal state BFGS")


# limited memory BFGS
println("L-BFGS")
Xend,Xpath,Nlast = LBFGS(X0,H0,Nbfgs,alpha_min,alpha_max,mu,Mmemory,F,Fgrad,Nsearch)

# plot 
scatter(Xpath[1:Nlast,2],Xpath[1:Nlast,1],label="path L-BFGS")
scatter(Xend[2],Xend[1],label="optimal state L-BFGS")
legend(fontsize=12,borderpad=0.2,borderaxespad=0.2,handletextpad=0.5,handlelength=0.6,framealpha=0.4,loc="lower right")
xlim(-1.0,2.25)
ylim(-1.0,2.25)




# BFGS-B 
println("BFGS B")
Xend,Hend,Xpath,Nlast = BFGSB(X0,H0,2Nbfgs,alpha_min,alpha_max,mu,lx,ux,F,Fgrad,Nsearch)

# plot 
figg2,ax2,pcm2,cbar2 = displayMap(1,122)
xticks(fontsize=12)
yticks(fontsize=12)
scatter(Xpath[1:Nlast,2],Xpath[1:Nlast,1],label="path BFGS-B")
scatter(Xend[2],Xend[1],label="optimal state BFGS-B")
# plot the constraints
plot(YY,lx[1]*ones(length(YY)))
plot(lx[2]*ones(length(XX)),XX)
plot(YY,ux[1]*ones(length(YY)))
plot(ux[2]*ones(length(XX)),XX)
plot(Xpath[1:Nlast,2],Xpath[1:Nlast,1])



# limited memory BFGS-B 
println("L-BFGS B")
# define the constraints
Xend,Xpath,Nlast = LBFGSB(X0+[-0.5;0.0],H0,2Nbfgs,alpha_min,alpha_max,mu,Mmemory,lx,ux,F,Fgrad,Nsearch)

# plot
scatter(Xpath[1:Nlast,2],Xpath[1:Nlast,1],label="path L-BFGS-B")
scatter(Xend[2],Xend[1],label="optimal state L-BFGS-B")
legend(fontsize=12,borderpad=0.2,borderaxespad=0.2,handletextpad=0.5,handlelength=0.6,framealpha=0.4,loc="lower right")
xlim(-1.0,2.25)
ylim(-1.0,2.25)
tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
ax1.annotate("a)", xy=(3, 1),  xycoords="axes fraction", xytext=(-0.14, 0.975), textcoords="axes fraction", color="black",fontsize=14)
ax2.annotate("b)", xy=(3, 1),  xycoords="axes fraction", xytext=(-0.14, 0.975), textcoords="axes fraction", color="black",fontsize=14)

# savefig("example_optim_path.png")