using StatsBase
mutable struct LNMMSB <: AbstractMMSB
 K          ::Int64             #number of communities
 N          ::Int64             #number of individuals
 elbo       ::Float64           #ELBO
 newelbo    ::Float64           #new ELBO
 μ          ::Vector{Float64}   #true mean of the MVN
 μ_var      ::Matrix2d{Float64} #variational mean of the MVN
 m0         ::Vector{Float64}   #hyperprior on mean
 m          ::Vector{Float64}   #variational hyperprior on mean
 M0         ::Matrix2d{Float64} #hyperprior on precision
 M          ::Matrix2d{Float64} #hyperprior on variational precision
 Λ          ::Matrix2d{Float64} #true precision
 Λ_var      ::Matrix3d{Float64} #variational preicisoin
 l0         ::Float64           #df for Preicision in Wishart
 L0         ::Matrix2d{Float64} #scale for precision in Wishart
 l          ::Float64           #variational df
 L          ::Matrix2d{Float64} #variational scale
 lzeta      ::Vector{Float64}   #additional variation param
 ϕlinoutsum ::Float64           #sum of phi products for links
 ϕnlinoutsum::Float64           #sum of phi products for nonlinks
 η0         ::Float64           #hyperprior on beta
 η1         ::Float64           #hyperprior on beta
 b0         ::Vector{Float64}   #variational param for beta
 b1         ::Vector{Float64}   #variational param for beta
 network    ::Network{Int64}    #sparse view network
 mbsize     ::Int64             #minibatch size
 mbids      ::Vector{Int64}     #minibatch node ids
 nho       ::Float64           # no. of validation links
 function LNMMSB(network::Network{Int64}, K::Int64)
  network     =load("data/network.jld")["network"] # network loaded
  N           = size(network,1)                    #setting size of nodes
  elbo        =0.0                                 #init ELBO at zero
  newelbo     =0.0                                 #init new ELBO at zero
  μ           =zeros(Float64,K)                    #zero the mu vector
  μ_var       =zeros(Float64, (N,K))               # zero the mu_var vector
  m0          =zeros(Float64,K)                    #zero m0 vector
  m           =zeros(Float64,K)                    #zero m vector
  M0          =eye(Float64,K)                  #init M0 matrix
  M           =zeros(Float64,(K,K))                # zero M matrix
  Λ           =(1.0/K).*eye(Float64,K)         #init Lambda matrix
  Λ_var       =zeros(Float64,(N,K,K))              # zero Lambda_var matrix
  l0          =K*1.0                               #init the df l0
  L0          =(1.0/K).*eye(Float64,K)         #init the scale L0
  l           =K*1.0                               #init the df l
  L           =zeros(Float64,(K,K))                #zero the scale L
  ϕlinoutsum  =zero(Float64)                       #zero the phi link product sum
  ϕnlinoutsum =zero(Float64)                       #zero the phi nonlink product sum
  lzeta       =ones(Float64, N)                    #one additional variational param
  η0          =1.0                                 #one the beta param
  η1          =1.0                                 #one the beta param
  b0          =ones(Float64, K)                    #one the beta variational param
  b1          =ones(Float64, K)                    #one the beta variational param
  mbsize      =2                                   #number of nodes in the minibatch
  mbids       =zeros(Int64,mbsize)                 # to be extended
  nho        =nnz(network)*0.025                  #init nho

  model = new(K, N, elbo, newelbo, μ, μ_var, m0, m, M0, M, Λ, Λ_var, l0, L0, l,
   L, lzeta, ϕlinoutsum, ϕnlinoutsum, η0, η1, b0, b1, network, mbsize, mbids,nho)
  return model
 end
end


#Updates
function elogpmu(model::LNMMSB)
 .5*(-K*log(2*pi)-trace(inv(model.M))-model.m*transpose(model.m))
end
function elogpLambda(model::LNMMSB)
 .5*(-K*(K+1)*log(2)-.5*K*(K-1)*log(pi)+digamma(.5*model.l,model.K)-2*lgamma(.5*model.K, model.K)-
 model.l*model.K*trace(model.L)-logdet(model.L)+model.K*logdet(model.K.*eye(model.L0)))
end

function elogptheta(model::LNMMSB)

end
function elogpzout(model::LNMMSB)
 s = zero(Float64)
 for a in 1:model.N
  for b in 1:model.N
   if a == b
    continue;
   end
  end
 end
end
function elogpzin(model::LNMMSB)
end
function elogpbeta(model::LNMMSB)
end
function elogpnetwork(model::LNMMSB)
end
function elogqmu(model::LNMMSB)
end
function elogqLambda(model::LNMMSB)
end
function elogqtheta(model::LNMMSB)
end
function elogqbeta(model::LNMMSB)
end
function elogqzout(model::LNMMSB)
end
function elogqzin(model::LNMMSB)
end
function computeelbo!(model::LNMMSB)
 model.elbo=model.newelbo
 model.newelbo=0
 return model.elbo
end
function computenewelbo!(model::LNMMSB)
 model.newelbo += elogpmu(model)+elogpLambda(model)+elogptheta(model)+
 elogpzout(model)+elogpzin(model)+elogpbeta(model)+elogpnetwork(model)-
 elogqmu(model)-elogqLambda(model)-elogqtheta(model)-elogqbeta(model)-
 elogqzout(model)-elogqzin(model)
end
function updatem!(model::LNMMSB)
 model.m= inv(model.M0 .+ (model.N*model.l).*model.L)*(model.M0*model.m0 + model.l.*model.L*transpose(sum(model.μ_var,1)))
end
function updateM!(model::LNMMSB)
 model.M = model.M0 + (model.N*model.l).*model.L

end
function updatel!(model::LNMMSB)
 model.l = model.l0+model.N
end



function updateL!(model::LNMMSB)
end
function updatemua!(model::LNMMSB, a::Int64)
end
function updateLambdaa!(model::LNMMSB, a::Int64)
end
function updatemb0!(model::LNMMSB)
end
function updatemb1!(model::LNMMSB)
end
function updatemzetaa!(model::LNMMSB, a::Int64)
end
function updatephiout!(model::LNMMSB, a::Int64, b::Int64)
end
function updatephiin!(model::LNMMSB, a::Int64, b::Int64)
end

function train!(model::LNMMSB, iter::Int64, etol::Float64, niter::Int64, ntol::Float64, viter::Int64, vtol::Float64, elboevery::Int64)

end
println()
