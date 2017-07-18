using Optim
#negative cross entropies
function elogpmu(model::LNMMSB)
	-.5*(model.K*log(2*pi)+trace(model.m*model.m') + trace(inv(model.M)))
end
function elogpLambda(model::LNMMSB)
	-.5*(-(model.K^2.0)*log(model.K) + logdet(model.L) + model.l*model.K*trace(model.L)+
	.5*model.K.*(model.K-1.0)*log(pi) + 2.0*lgamma(.5*model.K, model.K) + digamma(.5*model.l, model.K)+
	model.K*(model.K)*log(2.0))
end
##MB dependent
function elogptheta(model::LNMMSB, mb::MiniBatch)
	s = zero(Float64)
	for a in mb.mballnodes
		s += model.K*log(2pi) - digamma(.5*model.l, model.K) -model.K*log(2.0) - logdet(model.L)+
		model.l*trace(model.L*(inv(model.M) + inv(model.Λ_var[a,:,:])+(model.μ_var[a,:] - model.m)*(model.μ_var[a,:] - model.m)'))
	end
	s*-.5
end
#think about how to keep track of phis
##MB dependent
function elogpzlout(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s1+=mbl.ϕout[k]*model.μ_var[mbl.src,k]
		end
		for l in 1:model.K
			s2-= (1.0/(model.ζ[mbl.src])*exp(model.μ_var[mbl.src,l]+1.0/model.Λ_var[mbl.src,l,l]) + log(model.ζ[mbl.src])-1)
		end
	end
	s1+s2
end
##MB dependent
function elogpzlin(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s1+=mbl.ϕin[k]*model.μ_var[mbl.dst,k]
		end
		for l in 1:model.K
			s2-= (1.0/(model.ζ[mbl.dst])*exp(model.μ_var[mbl.dst,l]+1.0/model.Λ_var[mbl.dst,l,l]) + log(model.ζ[mbl.dst])-1)
		end
	end
	s1+s2
end
##MB dependent
function elogpznlout(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s1+=mbn.ϕout[k]*model.μ_var[mbn.src,k]
		end
		for l in 1:model.K
			s2-= (1.0/(model.ζ[mbn.src])*exp(model.μ_var[mbn.src,l]+1.0/model.Λ_var[mbn.src,l,l]) + log(model.ζ[mbn.src])-1)
		end
	end
	s1+s2
end
##MB dependent
function elogpznlin(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s1+=mbn.ϕin[k]*model.μ_var[mbn.dst,k]
		end
		for l in 1:model.K
			s2-= (1.0/(model.ζ[mbn.dst])*exp(model.μ_var[mbn.dst,l]+1.0/model.Λ_var[mbn.dst,l,l]) + log(model.ζ[mbn.dst])-1)
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
##MB dependent
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
##MB Dependent
function updatem!(model::LNMMSB, mb::MiniBatch)
	model.m= inv(model.mbsize*(model.l*model.L + (1.0/model.mbsize)*eye(model.K)))*
	model.l*model.L*view(model.μ_var,collect(mb.mballnodes),:)'*ones(model.mbsize)
end
##MB dependent
function updateM!(model::LNMMSB,mb::MiniBatch)
	##Only to make it MB dependent
	model.M = model.mbsize*((1.0/model.mbsize)*eye(model.K) + length(mb.mballnodes)*model.l*model.L)
end
##MB Dependent
function updatel!(model::LNMMSB, mb::MiniBatch)
	model.l = model.K+length(mb.mballnodes)
end
#MB Dependent
function updateL!(model::LNMMSB, mb::MiniBatch)
	s1=view((model.μ_var .- model.m')', :,collect(mb.mballnodes))*view((model.μ_var .- model.m'), collect(mb.mballnodes),:)
	s2 = reshape(sum(view(model.Λ_var, collect(mb.mballnodes), :, :),1),model.K, model.K)
	inv(model.K*eye(model.K)+model.mbsize*inv(model.M) + inv(s2)+s1)
end

#MB Dependent
function updateb0!(model::LNMMSB, mb::MiniBatch)
	for k in 1:model.K
		model.ϕlinoutsum[k] = zero(Float64)
		for mbl in mb.mblinks
			model.ϕlinoutsum[k]+=mbl.ϕout[k]*mbl.ϕin[k]
		end
	end
	model.b0[:] = model.ϕlinoutsum[:].+model.η0;
end
#MB Dependent
function updateb1!(model::LNMMSB, mb::MiniBatch)
	for k in 1:model.K
		model.ϕnlinoutsum[k] = zero(Float64)
		for mbn in mb.mbnonlinks
			model.ϕnlinoutsum[k]+=mbn.ϕout[k]*mbn.ϕin[k]
		end
	end
	model.b1[:] = model.ϕnlinoutsum[:].+1.0;
end



##The following need update but need to think how it affects the actual phis? after an iteration
## But also local and used among first updates, so to be used by other updates with the same minibatch
## hence we may only need to keep average for init of the thetas
function updatephilout!(model::LNMMSB,  mb::MiniBatch)
	for l in mb.mblinks
		for k in 1:model.K
			l.ϕout[k]=exp(model.μ_var[l.src,k] + l.ϕin[k]*(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-10))
		end
	end
	for l in mb.mblinks
		expnormalize!(l.ϕout)
	end
end
function updatephilin!(model::LNMMSB,mb::MiniBatch)
	for l in mb.mblinks
		for k in 1:model.K
			l.ϕin[k]=exp(model.μ_var[l.dst,k] + l.ϕout[k]*(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-10))
		end
	end
	for l in mb.mblinks
		expnormalize!(l.ϕin)
	end

end
function updatephinlout!(model::LNMMSB, mb::MiniBatch)
	for nl in mb.mbnonlinks
		for k in 1:model.K
			nl.ϕout[k]=exp(model.μ_var[nl.src,k] + nl.ϕin[k]*(digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log(1.0-EPSILON)))
		end
	end
	for nl in mb.mbnonlinks
		expnormalize!(nl.ϕout)
	end
end
function updatephinlin!(model::LNMMSB,mb::MiniBatch)
	for nl in mb.mbnonlinks
		for k in 1:model.K
			nl.ϕin[k]=exp(model.μ_var[nl.dst,k] + nl.ϕout[k]*(digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log(1.0-EPSILON)))
		end
	end
	for nl in mb.mbnonlinks
		expnormalize!(nl.ϕin)
	end
end
#MB dependent
function updatezetaa!(model::LNMMSB, mb::MiniBatch, a::Int64)
	model.ζ[a]=exp(logsumexp(model.μ_var[a,:]+.5*diag(model.Λ_var[a,:,:])))
end
#Newton\
#MB dependent
##here we need to think better about the accessing of phis or ways of recording them.
function gmu!(storage,x)

end
function hmu!(storage,x)
end
function gLambda!(storage,x)
end
function hLambda!(storage,x)
end
function updatemua!(model::LNMMSB, a::Int64, niter::Int64, ntol::Float64,mb::MiniBatch)
	for i in 1:niter
		μ_grad=-model.L[k,k]*(model.μ_var[a,k]-model.m[k])+sum()-sum(softmax(model.μ_var[a,:],k))
		μ_invH=
		model.μ_var[a,k] -= μ_invH * μ_grad
		if norm(μ_grad) < ntol
			break
		end
	end
end
#Newton
#MB dependent
function updateLambdaa!(model::LNMMSB, a::Int64, niter::Int64, ntol::Float64,mb::MiniBatch)
	Λ_grad=
	Λ_invH	=
	model.Λ_var[a,k,k] -= Λ_invH * Λ_grad
	if norm(Λ_grad) < ntol
		break
	end
end
##only initiated MiniBatch
function train!(model::LNMMSB; iter::Int64=150, etol::Float64=1, niter::Int64=1000, ntol::Float64=1.0/(model.K^2), viter::Int64=10, vtol::Float64=1.0/(model.K^2), elboevery::Int64=10, mb::MiniBatch,lr::Float64)
	preparedata(model)
	##the following deepcopy is very important

	for i in 1:iter
		checkelbo = (i % elboevery == 0)
		lr = 1.0/((1.0+Float64(i))^.9)
		mb=deepcopy(mb_zeroer)
		mbsampling!(mb,model)
		updatem!(model, mb)
		updateM!(model, mb)
		updatel!(model, mb)
		updateL!(model, mb)
		updatephilout!(model, mb)
		updatephilin!(model, mb)
		updatephinlout!(model, mb)
		updatephinlin!(model, mb)
		updateb0!(model, mb)
		updateb1!(model, mb)
		for a in mb.mballnodes
			updatezetaa!(model,mb, a)
			updatemua!(model,mb, a)
			updateLambdaa!(model,mb, a)
		end


	end

end
