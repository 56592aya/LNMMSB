using StatsBase
mutable struct LNMMSB <: AbstractMMSB
 K::Int64
 N::Int64
 elbo::Float64
 newelbo::Float64
 μ::Vector{Float64}
 μ_var::Matrix2d{Float64}
 m0::Vector{Float64}
 m::Vector{Float64}
 M0::Matrix2d{Float64}
 M::Matrix2d{Float64}
 Λ::Matrix2d{Float64}
 Λ_var::Matrix3d{Float64}
 l0::Float64
 L0::Matrix2d{Float64}
 l::Float64
 L::Matrix2d{Float64}
 lzeta::Vector{Float64}
 ϕlinoutsum::Float64
 ϕnlinoutsum::Float64
 η0::Float64
 η1::Float64
 b0::Vector{Float64}
 b1::Vector{Float64}
 network::Network{Int64}
 mbsize::Int64
 mbids::Vector{Int64}
 function LNMSSB(network::Network, K::Int64)
  network=load("data/network.jld")["network"]
  N = size(network,1)
  elbo=0
  newelbo=0
  μ=zeros(Float64,K)
  μ_var=zeros(Float64, (N,K))
  m0=zeros(Float64,K)
  m=zeros(Float64,K)
  M0=zeros(Float64,(K,K))
  M=zeros(Float64,(K,K))
  Λ=zeros(Float64,(K,K))
  Λ_var=zeros(Float64,(N,K,K))
  l0=zero(Float64)
  L0=zeros(Float64,(K,K))
  l=zero(Float64)
  L=zeros(Float64,(K,K))
  ϕlinoutsum=zero(Float64)
  ϕnlinoutsum=zero(Float64)
  lzeta=ones(Float64, N)
  η0=ones(Float64, K)
  η1=ones(Float64, K)
  b0=ones(Float64, K)
  b1=ones(Float64, K)
 end
 model = new()
 return model
end

#Updates
function elogpmu(model::LNMMSB)


end
function elogpLambda(model::LNMMSB)

end

function elogptheta(model::LNMMSB)

end
function elogpzout(model::LNMMSB)
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
#Train
# function train!(model::LNMMSB, iter::Int64, etol::Float64, niter::Int64, ntol::Float64, viter::Int64, vtol::Float64, elboevery::Int64)
# end
