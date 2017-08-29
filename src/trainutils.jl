using SpecialFunctions

#negative cross entropies
# ELOGS NEED SERIOUS REVISIONS
function elogpmu(model::LNMMSB)
	-.5*(model.K*logdet(model.M0)+trace(model.M0*(model.m-model.m0)*(model.m-model.m0)')+trace(model.M0*inv(model.M)))
end
# elogpmu(model)
#
function elogpLambda(model::LNMMSB)
	+.5*(-model.K*(model.K+1)*log(2.0)-.5*model.K*(model.K-1)*log(pi)+
	(model.l0-model.K-1)*digamma(.5*model.l,model.K)-2.0*lgamma(.5*model.l0,model.K)-
	model.l*trace(inv(model.L0)*model.L)- (model.K+1)*(logdet(model.L))+model.l0*logdet(inv(model.L0)*(model.L)))
end
# elogpLambda(model)
#
##Very large in magnitude
function elogptheta(model::LNMMSB, mb::MiniBatch)
	s = zero(Float64)
	for a in collect(mb.mballnodes)
		s += model.K*log(2.0*pi)-model.K*log(2.0) - digamma(.5*model.l, model.K)  - logdet(model.L)+
		model.l*trace(model.L*(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+inv(model.M)+diagm(1.0./model.Λ_var[a,:]))
	end
	s*-.5
end
# elogptheta(model, mb)
# #think about how to keep track of phis
function elogpzlout(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s1+=mbl.ϕout[k]*model.μ_var[mbl.src,k]
		end
		for l in 1:model.K
			s2+= exp(model.μ_var[mbl.src,l]+.5/model.Λ_var[mbl.src,l])
		end
	end
	s1-log(s2)
end
# elogpzlout(model, mb)
function elogpzlin(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s1+=mbl.ϕin[k]*model.μ_var[mbl.dst,k]
		end
		for l in 1:model.K
			s2+= exp(model.μ_var[mbl.dst,l]+.5/model.Λ_var[mbl.dst,l])
		end
	end
	s1-log(s2)
end
# elogpzlin(model, mb)
function elogpznlout(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s1+=mbn.ϕout[k]*model.μ_var[mbn.src,k]
		end
		for l in 1:model.K
			s2+= exp(model.μ_var[mbn.src,l]+.5/model.Λ_var[mbn.src,l])
		end
	end
	s1-log(s2)
end
# elogpznlout(model, mb)
function elogpznlin(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s1+=mbn.ϕin[k]*model.μ_var[mbn.dst,k]
		end
		for l in 1:model.K
			s2+= exp(model.μ_var[mbn.dst,l]+.5/model.Λ_var[mbn.dst,l])
		end
	end
	s1-log(s2)
end
# elogpznlin(model, mb)
function elogpbeta(model::LNMMSB)
	s = zero(Float64)
	for k in 1:model.K
		s+=lgamma(model.η0+model.η1)-lgamma(model.η0)-lgamma(model.η1)+(model.η0-1)*digamma(model.b0[k])+
		(model.η1-1)*digamma(model.b1[k]) - (model.η0+model.η1-2)*digamma(model.b0[k]+model.b1[k])
	end
	s
end
elogpbeta(model)
##check the effect of epsilon on the size of the change, so that in computations maybe we can skip it.
function elogpnetwork(model::LNMMSB, mb::MiniBatch)
	s = zero(Float64)
	for mbl in mb.mblinks
		# a = link.src;b=link.dst;
		ϕout=mbl.ϕout;ϕin=mbl.ϕin;
		for k in 1:model.K
			s+=ϕout[k]*ϕin[k]*(digamma(model.b0[k])- digamma(model.b1[k]+model.b0[k])-log1p(-1.0+EPSILON))+log1p(-1.0+EPSILON)
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
#
####The negative entropies
function elogqmu(model::LNMMSB)
	-.5*(model.K*log(2.0*pi) + model.K - logdet(model.M))
end
# elogqmu(model)
function elogqLambda(model::LNMMSB)
	-.5*((model.K+1)*logdet(model.L) + model.K*(model.K+1)*log(2)+model.K*model.l+
	.5*model.K*(model.K-1)*log(pi) + 2*lgamma(.5*model.l, model.K) - (model.l-model.K-1)*digamma(.5*model.l, model.K))
end
# elogqLambda(model)
function elogqtheta(model::LNMMSB)
	s = zero(Float64)
	for a in collect(mb.mballnodes)
		s += model.K*log(2.0*pi)-logdet(diagm(model.Λ_var[a,:]))+model.K
	end
	-.5*s
end
# elogqtheta(model)
function elogqbeta(model::LNMMSB)
	s = zero(Float64)

	for k in 1:model.K
		s+=lgamma(model.b0[k])+lgamma(model.b1[k])-lgamma(model.b0[k]+model.b1[k]) - (model.b0[k]-1)*digamma(model.b0[k]) -(model.b1[k]-1)*digamma(model.b1[k]) +(model.b0[k]+model.b1[k]-2)*digamma(model.b0[k]+model.b1[k])
	end
	-s
end
# elogqbeta(model)
function elogqzl(model::LNMMSB)
	s = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s+=(mbl.ϕout[k]*log(mbl.ϕout[k])+mbl.ϕin[k]*log(mbl.ϕin[k]))
		end
	end
	s
end
# elogqzl(model)
function elogqznl(model::LNMMSB)
	s = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s+=(mbn.ϕout[k]*log(mbn.ϕout[k])+mbn.ϕin[k]*log(mbn.ϕin[k]))
		end
	end
	s
end
# elogqznl(model)
function computeelbo!(model::LNMMSB, mb::MiniBatch)
	model.elbo=elogpmu(model)+elogpLambda(model)+elogptheta(model,mb)+elogpzlout(model,mb)+elogpzlin(model,mb)+
	elogpznlout(model,mb)+elogpznlin(model,mb)+elogpbeta(model)+elogpnetwork(model,mb)-
	(elogqmu(model)+elogqLambda(model)+elogqtheta(model)+elogqbeta(model)+elogqzl(model)+elogqznl(model))
	return model.elbo
end
# computeelbo!(model, mb)

function updateM!(model::LNMMSB,mb::MiniBatch)
	##Only to make it MB dependent
	model.M_old = deepcopy(model.M)
	model.M = model.l.*model.N.*model.L + model.M0
end
#updateM!(model,mb)
function updatem!(model::LNMMSB, mb::MiniBatch)
	model.m_old = deepcopy(model.m)
	model.m= inv(model.M)*(model.M0*model.m0+model.l*model.L*(model.N/model.mbsize)*sum(model.μ_var[a,:] for a in collect(mb.mballnodes)))
end
#updatem!(model, mb)
function updatel!(model::LNMMSB, mb::MiniBatch)
	##should be set in advance, not needed in the loop
	model.l = model.l0+model.N
end
#updatel!(model,mb)
function updateL!(model::LNMMSB, mb::MiniBatch)
	x=model.μ_var[collect(mb.mballnodes),:]'.-model.m
	s1= x*x'
	s2 = diagm(sum(1.0./model.Λ_var[a,:] for a in collect(mb.mballnodes)))
	model.L_old = deepcopy(model.L)
	model.L = inv(inv(model.L0) + model.N*inv(model.M) + (model.N/model.mbsize)*(s1+s2))
end
#updateL!(model, mb)
function updateb0!(model::LNMMSB, mb::MiniBatch)
	for k in 1:model.K
		model.ϕlinoutsum[k] = zero(Float64)
		for mbl in mb.mblinks
			model.ϕlinoutsum[k]+=mbl.ϕout[k]*mbl.ϕin[k]
		end
	end
	model.b0_old = deepcopy(model.b0)
	train_links_num=nnz(model.network)-length(model.ho_linkdict)
	model.b0[:] = (train_links_num)/length(mb.mblinks)*(model.ϕlinoutsum[:]).+model.η0;
end
#updateb0!(model, mb)
function updateb1!(model::LNMMSB, mb::MiniBatch)
	for k in 1:model.K
		model.ϕnlinoutsum[k] = zero(Float64)
		for mbn in mb.mbnonlinks
			model.ϕnlinoutsum[k]+=mbn.ϕout[k]*mbn.ϕin[k]
		end
	end
	model.b1_old = deepcopy(model.b1)
	train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyaddict) -length(mb.mblinks)
	model.b1[:] = (train_nlinks_num*1.0/length(mb.mbnonlinks))*(model.ϕnlinoutsum[:]).+model.η1
end
#updateb1!(model, mb)

##The following need update but need to think how it affects the actual phis? after an iteration
## But also local and used among first updates, so to be used by other updates with the same minibatch
## hence we may only need to keep average for init of the thetas
#MB Dependent
##need switching rounds between out and in
function updatephil!(model::LNMMSB,  mb::MiniBatch, early::Bool, switchrounds::Bool)
	for l in mb.mblinks
		if switchrounds
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕout[k]=(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			l.ϕout[:]=expnormalize(l.ϕout)
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
			end
			l.ϕin[:]=expnormalize(l.ϕin)
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕout[k]=(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			l.ϕout[:]=expnormalize(l.ϕout)

		else
			#recv

			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
			end
			l.ϕin[:]=expnormalize(l.ϕin)
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕout[k]=(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			l.ϕout[:]=expnormalize(l.ϕout)
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
			end
			l.ϕin[:]=expnormalize(l.ϕin)
		end
	end
	model.ϕloutsum=zeros(Float64,(model.N,model.K))
	model.ϕlinsum=zeros(Float64,(model.N,model.K))
	for l in mb.mblinks
		for k in 1:model.K
			model.ϕloutsum[l.src,k]+=l.ϕout[k]
			model.ϕlinsum[l.dst,k]+=l.ϕin[k]
		end
	end
end

#updatephilin!(model, mb)
##need switching rounds between out and in
function updatephinl!(model::LNMMSB, mb::MiniBatch,early::Bool, dep2::Float64,switchrounds::Bool)
	for nl in mb.mbnonlinks
		if switchrounds
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
			end
			nl.ϕout[:]=expnormalize(nl.ϕout)
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
			end
			nl.ϕin[:]=expnormalize(nl.ϕin)
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
			end
			nl.ϕout[:]=expnormalize(nl.ϕout)
		else
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
			end
			nl.ϕin[:]=expnormalize(nl.ϕin)
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
			end
			nl.ϕout[:]=expnormalize(nl.ϕout)
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
			end
			nl.ϕin[:]=expnormalize(nl.ϕin)
		end
	end
	model.ϕnloutsum=zeros(Float64,(model.N,model.K))
	model.ϕnlinsum=zeros(Float64,(model.N,model.K))
	for nl in mb.mbnonlinks
		for k in 1:model.K
			model.ϕnloutsum[nl.src,k]+=nl.ϕout[k]
			model.ϕnlinsum[nl.dst,k]+=nl.ϕin[k]
		end
	end
end

function mu_grad(model::LNMMSB, mb::MiniBatch, a::Int64)
	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
	sfx_vec = [softmax(model.μ_var[a,:]+.5./model.Λ_var[a,:],k) for k in 1:model.K]
	s = -model.l*model.L*(model.μ_var[a,:] - model.m) +
	model.ϕloutsum[a] + model.ϕnloutsum[a]+model.ϕlinsum[a] + model.ϕnlinsum[a]-
	sumb*sfx_vec

	return s
end
function mu_hess(model::LNMMSB, mb::MiniBatch, a::Int64)
	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
	sfx_vec = [softmax(model.μ_var[a,:]+.5./model.Λ_var[a,:],k) for k in 1:model.K]
	s = -model.l*model.L - 	sumb*(diagm(sfx_vec)-sfx_vec*sfx_vec')

	return s

end
#
# function updatemua!(model::LNMMSB, a::Int64, niter::Int64, ntol::Float64,mb::MiniBatch)
# 	for k in 1:model.K
# 		for i in 1:niter
# 			μ_grad=gmu(model, mb, a,k)
# 			μ_invH=inv(hmu(model, mb, a, k))
# 			model.μ_var[a,k] -= μ_invH * μ_grad
# 			if norm(μ_grad) < ntol
# 				break
# 			end
# 		end
# 	end
# end
# #Newton
function Lambdainv_grad(model::LNMMSB, mb::MiniBatch, a::Int64)
	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
	sfx_vec = [softmax(model.μ_var[a,:]+.5./model.Λ_var[a,:],k) for k in 1:model.K]
	##check the diagm if this is correct
	s = -.5*model.l*model.L + .5*diagm(model.Λ_var[a,:])-.5*sumb*diagm(sfx_vec)

	return s
end
function Lambdainv_hess(model::LNMMSB, mb::MiniBatch, a::Int64)#####CHECK
	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
	sfx_vec = [softmax(model.μ_var[a,:]+.5./model.Λ_var[a,:],k) for k in 1:model.K]
	s =  .5*ones(Float64,(K,K))*(model.Λ_var[a,:])*(model.Λ_var[a,:])'-.25*sumb*(diagm(sfx_vec)-sfx_vec*sfx_vec')

	return s

end
print("");
# function updateLambdaa!(model::LNMMSB, a::Int64, niter::Int64, ntol::Float64,mb::MiniBatch)
# 	temp = deepcopy(inv(model.Λ_var[a,:,:]))
# 	for k in 1:model.K
# 		for i in 1:niter
# 			Λinv_grad=gLambdainv(model, mb, a,k)
# 			Λinv_invH=inv(hLambdainv(model, mb, a, k))
# 			temp[k,k] -= Λinv_invH * Λinv_grad
# 			if norm(Λinv_grad) < ntol || i == niter
# 				model.Λ_var[a,k,k] = inv(temp[k,k])
# 				break
# 			end
# 		end
# 	end
# end
