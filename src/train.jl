using SpecialFunctions

#negative cross entropies
# ELOGS NEED SERIOUS REVISIONS
function elogpmu(model::LNMMSB)
	-.5*(model.K*logdet(model.M0)+trace(model.M0)*(model.m-model.m0)*(model.m-model.m0)'+trace(model.M0*inv(M)))
end
#
# function elogpLambda(model::LNMMSB)
# 	-.5*(-(model.K^2.0)*log(model.K) + logdet(model.L) + model.l*model.K*trace(model.L)+
# 	.5*model.K.*(model.K-1.0)*log(pi) + 2.0*lgamma(.5*model.K, model.K) + digamma(.5*model.l, model.K)+
# 	model.K*(model.K)*log(2.0))
# end
#
# ##?MB dependent
# ##needs a better subset of nodes for computing the train ELBO
# #so maybe changing it to the appropriate train
# # For now I skip the elbo on the full in sample data
# function elogptheta(model::LNMMSB, mb::MiniBatch)
# 	s = zero(Float64)
# 	for a in mb.mballnodes
# 		s += model.K*log(2pi) - digamma(.5*model.l, model.K) -model.K*log(2.0) - logdet(model.L)+
# 		model.l*(
# 		sum(model.L.*(1.0./model.Λ_var[a,:])) + sum(model.L.*(1.0./(model.M))) + sum(model.L.*(model.μ_var[a,:]-model.m).^2)
# 		)
# 	end
# 	s*-.5
# end
# elogptheta(model, mb)
# #think about how to keep track of phis
# ##MB dependent
# function elogpzlout(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	s2 = zero(Float64)
# 	for mbl in mb.mblinks
# 		for k in 1:model.K
# 			s1+=mbl.ϕout[k]*model.μ_var[mbl.src,k]
# 		end
# 		for l in 1:model.K
# 			s2-= (1.0/(model.ζ[mbl.src])*exp(model.μ_var[mbl.src,l]+1.0/model.Λ_var[mbl.src,l]) + log(model.ζ[mbl.src])-1)
# 		end
# 	end
# 	s1+s2
# end
#
# ##MB dependent
# function elogpzlin(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	s2 = zero(Float64)
# 	for mbl in mb.mblinks
# 		for k in 1:model.K
# 			s1+=mbl.ϕin[k]*model.μ_var[mbl.dst,k]
# 		end
# 		for l in 1:model.K
# 			s2-= (1.0/(model.ζ[mbl.dst])*exp(model.μ_var[mbl.dst,l]+1.0/model.Λ_var[mbl.dst,l]) + log(model.ζ[mbl.dst])-1)
# 		end
# 	end
# 	s1+s2
# end
# elogpzlin(model, mb)
# ##MB dependent
# function elogpznlout(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	s2 = zero(Float64)
# 	for mbn in mb.mbnonlinks
# 		for k in 1:model.K
# 			s1+=mbn.ϕout[k]*model.μ_var[mbn.src,k]
# 		end
# 		for l in 1:model.K
# 			s2-= (1.0/(model.ζ[mbn.src])*exp(model.μ_var[mbn.src,l]+1.0/model.Λ_var[mbn.src,l]) + log(model.ζ[mbn.src])-1)
# 		end
# 	end
# 	s1+s2
# end
# ##MB dependent
# function elogpznlin(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	s2 = zero(Float64)
# 	for mbn in mb.mbnonlinks
# 		for k in 1:model.K
# 			s1+=mbn.ϕin[k]*model.μ_var[mbn.dst,k]
# 		end
# 		for l in 1:model.K
# 			s2-= (1.0/(model.ζ[mbn.dst])*exp(model.μ_var[mbn.dst,l]+1.0/model.Λ_var[mbn.dst,l]) + log(model.ζ[mbn.dst])-1)
# 		end
# 	end
# 	s1+s2
# end
#
# function elogpbeta(model::LNMMSB)
# 	s = zero(Float64)
# 	for k in 1:model.K
# 		(model.η0-1)*digamma(model.b0[k]) - (model.η0-2)*digamma(model.b0[k]+model.b1[k])
# 	end
# 	s
# end
#
# ##MB dependent
# function elogpnetwork(model::LNMMSB, mb::MiniBatch)
# 	s = zero(Float64)
# 	for mbl in mb.mblinks
# 		# a = link.src;b=link.dst;
# 		ϕout=mbl.ϕout;ϕin=mbl.ϕin;
# 		for k in 1:model.K
# 			s+=ϕout[k]*ϕin[k]*(digamma(model.b0[k])- digamma(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON)
# 		end
# 	end
# 	for mbn in mb.mbnonlinks
# 		# a = nonlink.src;b=nonlink.dst;
# 		ϕout=mbn.ϕout;ϕin=mbn.ϕin;
# 		for k in 1:model.K
# 			s+=ϕout[k]*ϕin[k]*(digamma(model.b1[k])-digamma(model.b1[k]+model.b0[k])-log1p(-EPSILON))+log1p(-EPSILON)
# 		end
# 	end
# 	s
# end
#
# function elogqmu(model::LNMMSB)
# 	-.5*(model.K*log(2pi) + model.K - sum(log.(model.M)))
# end
# function elogqLambda(model::LNMMSB)
# 	-.5*((model.K+1)*sum(log.(model.L)) + model.K*(model.K+1)*log(2)+model.K*model.l+
# 	.5*model.K*(model.K-1)*log(pi) + 2*lgamma(.5*model.l, model.K) - (model.l-model.K-1)*digamma(.5*model.l, model.K))
# end
#
# function elogqtheta(model::LNMMSB)
# 	s = zero(Float64)
# 	for a in model.mbids
# 		s += model.K*log(2pi)-sum(log.(model.Λ_var))+model.K
# 	end
# 	-.5*s
# end
#
# function elogqbeta(model::LNMMSB)
# 	s = zero(Float64)
# 	for k in 1:model.K
# 		s+=lbeta(model.b0[k],model.b1[k]) - (model.b0[k]-1)*digamma(model.b0[k]) -(model.b1[k]-1)*digamma(model.b1[k]) +(model.b0[k]+model.b1[k]-2)*digamma(model.b0[k]+model.b1[k])
# 	end
# 	s
# end
#
# function elogqzl(model::LNMMSB)
# 	s = zero(Float64)
# 	for mbl in mb.mblinks
# 		for k in 1:model.K
# 			s+=(mbl.ϕout[k]*log(mbl.ϕout[k])+mbl.ϕin[k]*log(mbl.ϕin[k]))
# 		end
# 	end
# 	s
# end
# function elogqznl(model::LNMMSB)
# 	s = zero(Float64)
# 	for mbn in mb.mbnonlinks
# 		for k in 1:model.K
# 			s+=(mbn.ϕout[k]*log(mbn.ϕout[k])+mbn.ϕin[k]*log(mbn.ϕin[k]))
# 		end
# 	end
# 	s
# end
# function computeelbo!(model::LNMMSB)
# 	model.elbo=model.newelbo
# 	model.newelbo=0
# 	return model.elbo
# end
# ##Pay attention to the weightings of them accordint the the train/mb
# function computenewelbo!(model::LNMMSB)
# 	model.newelbo = elogpmu(model)+	elogpLambda(model)+	elogptheta(model)+	elogpzlout(model)+	elogpzlin(model)+elogpznlout(model)+elogpznlin(model)+elogpbeta(model)+elogpnetwork(model)-
# 	(elogqmu(model)+elogqLambda(model)+	elogqtheta(model)+elogqbeta(model)+elogqzl(model)+elogqznl(model))
# end
##MB dependent
function updateM!(model::LNMMSB,mb::MiniBatch)
	##Only to make it MB dependent
	model.M_old = deepcopy(model.M)
	model.M = model.l.*model.N.*model.L + model.M0
end
#updateM!(model,mb)
##MB Dependent
function updatem!(model::LNMMSB, mb::MiniBatch)
	model.m_old = deepcopy(model.m)
	model.m= inv(model.M)*(model.M0*model.m0+model.l*model.L*(model.N/model.mbsize)*sum(model.μ_var[a,:] for a in collect(mb.mballnodes)))
end
#updatem!(model, mb)
##MB Dependent
function updatel!(model::LNMMSB, mb::MiniBatch)
	##should be set in advance, not needed in the loop
	model.l = model.l0+model.N
end
#updatel!(model,mb)
#MB Dependent
function updateL!(model::LNMMSB, mb::MiniBatch)
	x=model.μ_var[collect(mb.mballnodes),:]'.-model.m
	s1= x*x'
	s2 = diagm(sum(1.0./model.Λ_var[a,:] for a in collect(mb.mballnodes)))
	model.L_old = deepcopy(model.L)
	model.L = inv(inv(model.L0) + model.N*inv(model.M) + (model.N/model.mbsize)*(s1+s2))
end
#updateL!(model, mb)
#MB Dependent
# sometihng wrong with b updates and weightings of ϕnlinoutsum
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
#MB Dependent
function updateb1!(model::LNMMSB, mb::MiniBatch)
	for k in 1:model.K
		model.ϕnlinoutsum[k] = zero(Float64)
		for mbn in mb.mbnonlinks
			model.ϕnlinoutsum[k]+=mbn.ϕout[k]*mbn.ϕin[k]
		end
	end
	train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyaddict) -length(mb.mblinks)
	model.b1_old = deepcopy(model.b1)
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
				l.ϕout[k]=exp(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			l.ϕout[:]=expnormalize(l.ϕout)
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕin[k]=exp(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
			end
			l.ϕin[:]=expnormalize(l.ϕin)
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕout[k]=exp(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			l.ϕout[:]=expnormalize(l.ϕout)

		else
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕin[k]=exp(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
			end
			l.ϕin[:]=expnormalize(l.ϕin)
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕout[k]=exp(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			l.ϕout[:]=expnormalize(l.ϕout)
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?3.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log1p(-1.0+EPSILON))
				l.ϕin[k]=exp(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
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
#MB Dependent
##need switching rounds between out and in
function updatephinl!(model::LNMMSB, mb::MiniBatch,early::Bool, dep2::Float64,switchrounds::Bool)
	for nl in mb.mbnonlinks
		if switchrounds
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕout[k]=exp(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
			end
			nl.ϕout[:]=expnormalize(nl.ϕout)
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕin[k]=exp(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
			end
			nl.ϕin[:]=expnormalize(nl.ϕin)
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕout[k]=exp(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
			end
			nl.ϕout[:]=expnormalize(nl.ϕout)
		else
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕin[k]=exp(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
			end
			nl.ϕin[:]=expnormalize(nl.ϕin)
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕout[k]=exp(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
			end
			nl.ϕout[:]=expnormalize(nl.ϕout)
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log1p(-EPSILON)
				nl.ϕin[k]=exp(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
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


#implement the newton step
#Newton\
#MB dependent
##here we need to think better about the accessing of phis or ways of recording them.
#question is whether I need to use softmax or model.ζ, this will be clear when I resolve the orderin of the updates
#Do I need to reweight the sumb as well?
# function gmu(model::LNMMSB, mb::MiniBatch, a::Int64, k::Int64)
# 	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
# 	sfx = softmax(model.μ_var[a,:] + .5*inv(diag(model.Λ_var[a,:,:])),k)
# 	#or sfx = exp(model.μ_var[a,k] + .5*inv(diag(model.Λ_var[a,k,k])))*inv(model.ζ[a])
# 	-model.l*model.L[k,k]*(model.μ_var[a,k] - model.m[k]) - sumb*(sfx)*(1-sfx))+(ϕloutsum+ϕnloutsum+ϕlinsum+ϕnlinsum)
#
# end
# function hmu(model::LNMMSB, mb::MiniBatch, a::Int64, k::Int64)
# 	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
# 	sfx = softmax(model.μ_var[a,:] + .5*inv(diag(model.Λ_var[a,:,:])),k)
# 	#or sfx = exp(model.μ_var[a,k] + .5*inv(diag(model.Λ_var[a,k,k])))*inv(model.ζ[a])
# 	-model.l*model.L[k,k] - sumb*(sfx-3*sfx^2+2*sfx^3)
# end
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
# #MB dependent
# function gLambdainv(model::LNMMSB, mb::MiniBatch, a::Int64, k::Int64)
# 	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
# 	sfx = softmax(model.μ_var[a,:] + .5*inv(diag(model.Λ_var[a,:,:])),k)
# 	#or sfx = exp(model.μ_var[a,k] + .5*inv(diag(model.Λ_var[a,k,k])))*inv(model.ζ[a])
# 	-.5*model.l*model.L[k,k] + .5*inv(model.Λ_var[a,k,k])-.5*sumb*(sfx)*(1-sfx)
# end
# function hLambdainv(model::LNMMSB, mb::MiniBatch, a::Int64, k::Int64)
# 	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
# 	sfx = softmax(model.μ_var[a,:] + .5*inv(diag(model.Λ_var[a,:,:])),k)
# 	#or sfx = exp(model.μ_var[a,k] + .5*inv(diag(model.Λ_var[a,k,k])))*inv(model.ζ[a])
# 	.5-.25*(sumb)*(sfx-3*sfx^2+2*sfx^3)
#
# end
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

##only initiated MiniBatch
##initialization for the check is very important, havent yet figured it out.
function train!(model::LNMMSB; iter::Int64=150, etol::Float64=1, niter::Int64=1000, ntol::Float64=1.0/(model.K^2), viter::Int64=10, vtol::Float64=1.0/(model.K^2), elboevery::Int64=10, mb::MiniBatch,lr::Float64)
	preparedata(model)
	mu_curr=ones(model.N)
	Lambda_curr=ones(model.N)
	lr_mu = zeros(Float64, model.N)
	lr_Lambda = zeros(Float64, model.N)
	early=true
	switchrounds=true
	#let's say for now:

	for a in 1:model.N
		model.μ_var[a,:]=[-0.16645342111013306,  -0.6374507015168214,  0.052970559974790075,  0.031601875772308]
		model.Λ_var[a,:]=[0.1963771801404379,  0.18922306769147973,  0.19457540225231443,  0.19951992669472524]
	end
	model.m = [-0.16,  -0.63,  0.05,  0.03]
	model.l = model.K+1
	model.L=diagm(1.0./[0.1963771801404379,  0.18922306769147973,  0.19457540225231443,  0.19951992669472524])./model.l
	i=1
	for i in 1:iter
		#Minibatch sampling/new sample
		##the following deepcopy is very important
		mb=deepcopy(mb_zeroer)
		mbsampling!(mb,model)
		# communities
		#global update-- can be done outside
		updatel!(model, mb)

		#checkelbo = (i % elboevery == 0)
		#Learning rates

		lr_M = 1.0/((1.0+Float64(i))^.5)
		lr_m = 1.0/((1.0+Float64(i))^.7)
		lr_L = 1.0/((1.0+Float64(i))^.9)
		lr_b = 1.0/((1.0+Float64(i))^.5)


		#locals:phis
		#local update
		ExpectedAllSeen=(model.N/model.mbsize)*1.5#round(Int64,nv(network)*sum([1.0/i for i in 1:nv(network)]))
    	if i == round(Int64,ExpectedAllSeen)
        	early = false
    	end
		train_links_num=nnz(model.network)-length(model.ho_linkdict)
		train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyaddict) -length(mb.mblinks)
		dep2 = .1*(train_links_num)/(train_links_num+train_nlinks_num)

		updatephil!(model, mb, early,switchrounds)
		updatephinl!(model, mb,early,dep2,switchrounds)


		#global update
		#globals:m,M,L,mu, Lambda, b
		updateM!(model, mb)

		model.M = model.M_old*(1.0-lr_M)+lr_M*model.M
		updatem!(model, mb)
		model.m = model.m_old*(1.0-lr_m)+lr_m*model.m

		updateL!(model, mb)
		model.L = model.L_old*(1.0-lr_L)+lr_L*model.L
		updateb0!(model, mb)
		updateb1!(model, mb)
		model.b0 = model.b0_old*(1.0-lr_b)+lr_b*model.b0
		model.b1 = model.b1_old*(1.0-lr_b)+lr_b*model.b1


		mb.mblinks[1].ϕout
		mb.mblinks[1].ϕin
		mb.mbnonlinks[1].ϕout
		mb.mbnonlinks[1].ϕin
		model.M
		model.m
		model.L
		model.b0
		model.b1
		println(i)
		println(model.b0./(model.b0.+model.b1))
		## KEEP AN AVERAGE FOR MUS AND LAMBDAS TO INITIATE THE PHIS EACH TIME

		# for a in collect(mb.mballnodes)
		# 	updatemua!(model, a, niter, ntol,mb)
		# 	lr_mu[a] = 1.0/((1.0+Float64(mu_curr[a]))^.9)##could be  a macro
		# 	mu_curr[a] += 1
		# 	model.ζ[a] = model.μ_var_old[a]*(1.0-lr_mu[a])+lr_mu[a]*model.μ_var[a]
		# 	updateLambdaa!(model, a, niter, ntol,mb)
		# 	lr_Lambda[a] = 1.0/((1.0+Float64(Lambda_curr[a]))^.9)##could be  a macro
		# 	Lambda_curr[a] += 1
		# 	model.Λ_var[a] = model.Λ_var_old[a]*(1.0-lr_Lambda[a])+lr_Lambda[a]*model.Λ_var[a]
		# end
		switchrounds = !switchrounds
		i=i+1
	end
end
