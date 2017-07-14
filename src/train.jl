#negative cross entropies
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
		model.l*trace(model.L*(inv(model.M) + inv(model.Λ_var[a,:,:])+(model.μ_var[a,:] - model.m)*(model.μ_var[a,:] - model.m)'))
	end
	s*-.5
end
#think about how to keep track of phis
function elogpzlout(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s1+=mbl.ϕout[k]*model.μ_var[mbl.src,k]
		end
		for l in 1:model.K
			s2-= model.ζ[mbl.src]*exp(model.μ_var[mbl.src,l]+1.0/model.Λ_var[mbl.src,l,l]) - log(model.ζ[mbl.src])-1
		end
	end
	s1+s2
end
function elogpzlin(model::LNMMSB)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s1+=mbl.ϕin[k]*model.μ_var[mbl.dst,k]
		end
		for l in 1:model.K
			s2-= model.ζ[mbl.dst]*exp(model.μ_var[mbl.dst,l]+1.0/model.Λ_var[mbl.dst,l,l]) - log(model.ζ[mbl.dst])-1
		end
	end
	s1+s2
end
function elogpznlout(model::LNMMSB)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s1+=mbn.ϕout[k]*model.μ_var[mbn.src,k]
		end
		for l in 1:model.K
			s2-= model.ζ[mbn.src]*exp(model.μ_var[mbn.src,l]+1.0/model.Λ_var[mbn.src,l,l]) - log(model.ζ[mbn.src])-1
		end
	end
	s1+s2
end
function elogpznlin(model::LNMMSB)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s1+=mbn.ϕin[k]*model.μ_var[mbn.dst,k]
		end
		for l in 1:model.K
			s2-= model.ζ[mbn.dst]*exp(model.μ_var[mbn.dst,l]+1.0/model.Λ_var[mbn.dst,l,l]) - log(model.ζ[mbn.dst])-1
		end
	end
	s1+s2
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
	for mbl in mb.mblinks
		# a = link.src;b=link.dst;
		ϕout=mbl.ϕout;ϕin=mbl.ϕin;
		for k in 1:model.K
			s+=ϕout[k]*ϕin[k]*(digamma(model.b0[k])- digamma(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON)
		end
	end
	for mbn in mb.mbnonlinks
		# a = nonlink.src;b=nonlink.dst;
		ϕout=mbn.ϕout;ϕin=mbn.ϕin;
		for k in 1:model.K
			s+=ϕout[k]*ϕin[k]*(digamma(model.b1[k])-digamma(model.b1[k]+model.b0[k])-log1p(-EPSILON))+log1p(-EPSILON)
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
function elogqzl(model::LNMMSB)
	s = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s+=(mbl.ϕout[k]*log(mbl.ϕout[k])+mbl.ϕin[k]*log(mbl.ϕin[k]))
		end
	end
	s
end
function elogqznl(model::LNMMSB)
	s = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s+=(mbn.ϕout[k]*log(mbn.ϕout[k])+mbn.ϕin[k]*log(mbn.ϕin[k]))
		end
	end
	s
end
function computeelbo!(model::LNMMSB)
	model.elbo=model.newelbo
	model.newelbo=0
	return model.elbo
end
##Pay attention to the weightings of them accordint the the train/mb
function computenewelbo!(model::LNMMSB)
	model.newelbo = elogpmu(model)+	elogpLambda(model)+	elogptheta(model)+	elogpzlout(model)+	elogpzlin(model)+elogpznlout(model)+elogpznlin(model)+elogpbeta(model)+elogpnetwork(model)-
	(elogqmu(model)+elogqLambda(model)+	elogqtheta(model)+elogqbeta(model)+elogqzl(model)+elogqznl(model))
end
function updatem!(model::LNMMSB)
	model.m= inv(model.mbsize*model.l*model.L + eye(model.K))*model.l*model.L*view(model.μ_var,model.mbids,:)'*ones(model.mbsize)
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
	inv(model.K*eye(model.K)+model.mbsize*inv(model.M) + inv(s2)+s1)
end
#Newton
function updatemua!(model::LNMMSB, a::Int64)

end
#Newton
function updateLambdaa!(model::LNMMSB, a::Int64)
end
function updateb0!(model::LNMMSB, mb::MiniBatch)
	for k in 1:model.K
		model.ϕlinoutsum[k] = zero(Float64)
		for mbl in mb.mblinks
			model.ϕlinoutsum[k]+=mbl.ϕout[k]*mbl.ϕin[k]
		end
	end
	model.b0[:] = model.ϕlinoutsum[:].+model.η0
end
function updateb1!(model::LNMMSB)
	for k in 1:model.K
		model.ϕnlinoutsum[k] = zero(Float64)
		for mbn in mb.mbnonlinks
			model.ϕnlinoutsum[k]+=mbn.ϕout[k]*mbn.ϕin[k]
		end
	end
	model.b1[:] = model.ϕnlinoutsum[:].+1.0
end

function updatemzetaa!(model::LNMMSB, a::Int64)

end
##The following need update but need to think how it affects the actual phis? after an iteration
## But also local and used among first updates, so to be used by other updates with the same minibatch
## hence we may only need to keep average for init of the thetas
function updatephiout!(model::LNMMSB, a::Int64, b::Int64)
end
function updatephiin!(model::LNMMSB, a::Int64, b::Int64)
end

function train!(model::LNMMSB, iter::Int64, etol::Float64, niter::Int64, ntol::Float64, viter::Int64, vtol::Float64, elboevery::Int64, mb::MiniBatch)##only initiated MiniBatch
	preparedata(model)
	##the following deepcopy is very important
	mb=deepcopy(mb_zeroer)
	mbsampling!(mb,model)


end