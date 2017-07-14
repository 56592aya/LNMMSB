
#Updates
function elogpmu(model::LNMMSB)
 -.5*(model.K*log(2*pi)+trace(model.m*model.m') + trace(inv(model.M)))
end
function elogpLambda(model::LNMMSB)
 -.5*(-(model.K^2.0)*log(model.K) + logdet(model.L) + model.l*model.K*trace(model.L)+
 .5*model.K.*(model.K-1.0)*log(pi) + 2.0*lgamma(.5*model.K, model.K) + digamma(.5*model.l, model.K)+
 model.K*(model.K)*log(2.0))
end

function elogptheta(model::LNMMSB)
 s = zero(Float64)
 for a in model.mbids
  s += model.K*log(2pi) - digamma(.5*model.l, model.K) -model.K*log(2.0) - logdet(model.L)+
  model.l*trace(model.L*(inv(model.M) + inv(model.Λ_var[a,:,:])+(model.μ_var[a,:] - model.m)(model.μ_var[a,:] - model.m)'))
 end
 s*-.5
end
#think about how to keep track of phis
function elogpzout(model::LNMMSB, mb::MiniBatch)
 s = zero(Float64)
 for a in model.mbids
  for b in train_sinks[a]
   for k in 1:model.K

   end

  end
  for b in train_sources[a]
   for k in 1:model.K
   end
  end
  for b in mb.mbfnadj[a]
   for k in 1:model.K
   end
  end
  for b in mb.mbbnadj[a]
   for k in 1:model.K
   end
  end
 end
end
function elogpzin(model::LNMMSB)
end
function elogpbeta(model::LNMMSB)
 s = zero(Float64)
 for k in 1:model.K
  (model.η0-1)*digamma(model.b0[k]) - (model.η0-2)*digamma(model.b0[k]+model.b1[k])
 end
 s
end
function elogpnetwork(model::LNMMSB, mb::MiniBatch)
 s = zero(Float64)
 for link in mb.mblinks
  # a = link.src;b=link.dst;
  ϕout=link.ϕout;ϕin=link.ϕin;
  for k in 1:model.K
   s+=ϕout*ϕin*(digamma(model.b0[k])- digamma(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON)
  end
 end
 for nonlink in mb.mbnonlinks
  # a = nonlink.src;b=nonlink.dst;
  ϕout=nonlink.ϕout;ϕin=nonlink.ϕin;
  for k in 1:model.K
   s+=ϕout*ϕin*(digamma(model.b1[k])-digamma(model.b1[k]+model.b0[k])-log1p(-EPSILON))+log1p(-EPSILON)
  end
 end
 s
end
function elogqmu(model::LNMMSB)
 -.5*(model.K*log(2pi) + model.K - logdet(model.M))
end
function elogqLambda(model::LNMMSB)
 -.5*((model.K+1)*logdet(model.L) + model.K*(model.K+1)*log(2)+model.K*model.l+
 .5*model.K*(model.K-1)*log(pi) + 2*lgamma(.5*model.l, model.K) - (model.l-model.K-1)*digamma(.5*model.l, model.K))
end
function elogqtheta(model::LNMMSB)
 s = zero(Float64)
 for a in model.mbids
  s += model.K*log(2pi)-logdet(model.Λ_var[a,:,:]+model.K)
 end
 -.5*s
end
function elogqbeta(model::LNMMSB)
 s = zero(Float64)
 for k in 1:model.K
  s+=lbeta(model.b0[k],model.b1[k]) - (model.b0[k]-1)*digamma(model.b0[k]) -(model.b1[k]-1)*digamma(model.b1[k]) +(model.b0[k]+model.b1[k]-2)*digamma(model.b0[k]+model.b1[k])
 end
 -s
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
##Pay attention to the weightings of them accordint the the train/mb
function computenewelbo!(model::LNMMSB)
 model.newelbo += elogpmu(model)+elogpLambda(model)+elogptheta(model)+
 elogpzout(model)+elogpzin(model)+elogpbeta(model)+elogpnetwork(model)-
 elogqmu(model)-elogqLambda(model)-elogqtheta(model)-elogqbeta(model)-
 elogqzout(model)-elogqzin(model)
end
function updatem!(model::LNMMSB)
 model.m= inv(model.mbsize*model.l*model.L + eye(model.K))*model.l*model.L*model.μ_var'*ones(model.mbsize)
end
function updateM!(model::LNMMSB)
 model.M = eye(model.K) + model.mbsize*model.l*model.L
end
function updatel!(model::LNMMSB)
 model.l = model.K+model.mbsize
end
function updateL!(model::LNMMSB, mb::MiniBatch)
 s1=view((model.μ_var .- model.m')', :,model.mbids)*view((model.μ_var .- model.m'), model.mbids,:)
 s2 = reshape(sum(view(model.Λ_var, model.mbids, :, :),1),model.K, model.K)
 inv(model.K*eye(model.K)+model.mbsize*inv(M) + inv(s2)+s1)
end
function updatemua!(model::LNMMSB, a::Int64)
end
function updateLambdaa!(model::LNMMSB, a::Int64)
end
function updateb0!(model::LNMMSB)
 for k in 1:model.K
  model.b0[k]=model.η0 +
 end
end
function updateb1!(model::LNMMSB)
 for k in 1:model.K
  model.b1[k]=1.0 +
 end
end
function updatemzetaa!(model::LNMMSB, a::Int64)
end
function updatephiout!(model::LNMMSB, a::Int64, b::Int64)
end
function updatephiin!(model::LNMMSB, a::Int64, b::Int64)
end

function train!(model::LNMMSB, iter::Int64, etol::Float64, niter::Int64, ntol::Float64, viter::Int64, vtol::Float64, elboevery::Int64, mb::MiniBatch)##only initiated MiniBatch
    preparedata(model)

    
    mb

end
