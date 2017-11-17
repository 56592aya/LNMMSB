# using SpecialFunctions

using GradDescent
using ForwardDiff

#negative cross entropies
# function elogpmu(model::LNMMSB)
# 	-.5*(
# 	model.K*log(2.0*pi)-
# 	logdet(model.M0)+
# 	trace(model.M0*(model.m-model.m0)*(model.m-model.m0)')+
# 	trace(model.M0*inv(model.M))
# 	)
# end
# #
# # # elogpmu(model)
# # #
# function elogpLambda(model::LNMMSB)
# 	+.5*(
# 	-model.K*(model.K+1)*log(2.0)+
# 	(model.l0-model.K-1)*digamma_(.5*model.l,model.K)-
# 	2.0*lgamma_(.5*model.l0,model.K)-
# 	model.l*trace(inv(model.L0)*model.L)-
# 	(model.K+1)*logdet(model.L)+
# 	model.l0*logdet(inv(model.L0)*(model.L))
# 	)
# end
# # # elogpLambda(model)
# # #
# # ##Very large in magnitude
# function elogptheta(model::LNMMSB, mb::MiniBatch)
# 	s = zero(Float64)
# 	for a in collect(mb.mballnodes)
# 		s +=
# 		(
# 		model.K*log(2.0*pi)-
# 		digamma_(.5*model.l, model.K)-
# 		model.K*log(2.0) -
# 		logdet(model.L)+
# 		model.l*trace(model.L*((model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+inv(model.M)+diagm(1.0./model.Λ_var[a,:])
# 		)
# 		)
# 		)
# 	end
# 	s*-.5
# end
# # # elogptheta(model, mb)
# # # #think about how to keep track of phis
# function elogpzlout(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	s2 = zero(Float64)
# 	for mbl in mb.mblinks
# 		for k in 1:model.K
# 			s1+=mbl.ϕout[k]*model.μ_var[mbl.src,k]
# 		end
# 		s2+= logsumexp(model.μ_var[mbl.src,:]+.5./model.Λ_var[mbl.src,:])
# 	end
# 	s1-s2
# end
# # # elogpzlout(model, mb)
# function elogpzlin(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	s2 = zero(Float64)
# 	for mbl in mb.mblinks
# 		for k in 1:model.K
# 			s1+=mbl.ϕin[k]*model.μ_var[mbl.dst,k]
# 		end
# 		s2+= logsumexp(model.μ_var[mbl.dst,:]+.5./model.Λ_var[mbl.dst,:])
# 	end
# 	s1-s2
# end
# # # elogpzlin(model, mb)
# function elogpznlout(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	s2 = zero(Float64)
# 	for mbn in mb.mbnonlinks
# 		for k in 1:model.K
# 			s1+=mbn.ϕout[k]*model.μ_var[mbn.src,k]
# 		end
# 		s2+= logsumexp(model.μ_var[mbn.src,:]+.5./model.Λ_var[mbn.src,:])
# 	end
# 	s1-s2
# end
# # # elogpznlout(model, mb)
# function elogpznlin(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	s2 = zero(Float64)
# 	for mbn in mb.mbnonlinks
# 		for k in 1:model.K
# 			s1+=mbn.ϕin[k]*model.μ_var[mbn.dst,k]
# 		end
# 		s2+= logsumexp(model.μ_var[mbn.dst,:]+.5./model.Λ_var[mbn.dst,:])
# 	end
# 	s1-s2
# end
# # # elogpznlin(model, mb)
# function elogpbeta(model::LNMMSB)
# 	s = zero(Float64)
# 	for k in 1:model.K
# 		s+=(
# 		-lgamma_(model.η0)-
# 		lgamma_(model.η1)+
# 		lgamma_(model.η0+model.η1) +
# 		(model.η0-1.0)*digamma_(model.b0[k]) +
# 		(model.η1-1.0)*digamma_(model.b1[k]) -
# 		(model.η0+model.η1-2.0)*digamma_(model.b0[k]+model.b1[k])
# 		)
# 	end
# 	s
# end
# # # elogpbeta(model)
# # ##check the effect of epsilon on the size of the change, so that in computations maybe we can skip it.
# function elogpnetwork(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	# s = zero(Float64)
# 	for mbl in mb.mblinks
#
# 		# a = link.src;b=link.dst;
# 		ϕout=mbl.ϕout;ϕin=mbl.ϕin;
# 		for k in 1:model.K
# 			s1+=(ϕout[k]*ϕin[k]*(digamma_(model.b0[k])- digamma_(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON))#as is constant for
# 			# s+=(ϕout[k]*ϕin[k]*(digamma_(model.b0[k])- digamma_(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON))#as is constant for numerical stability for now
# 		end
# 	end
# 	s2 = zero(Float64)
# 	for mbn in mb.mbnonlinks
# 		# a = nonlink.src;b=nonlink.dst;
# 		ϕout=mbn.ϕout;ϕin=mbn.ϕin;
# 		for k in 1:model.K
# 			s2+=(ϕout[k]*ϕin[k]*(digamma_(model.b1[k])-digamma_(model.b1[k]+model.b0[k])-log(1.0-EPSILON))+log(1.0-EPSILON))#as is
# 			# s+=(ϕout[k]*ϕin[k]*(digamma_(model.b1[k])-digamma_(model.b1[k]+model.b0[k])-log(1.0-EPSILON))+log(1.0-EPSILON))#as is constant for numerical stability for now
# 		end
# 	end
# 	# s
# 	s1+s2
# 	# s1+s2 == s
# end
# function elogpnetwork1(model::LNMMSB, mb::MiniBatch)
# 	s1 = zero(Float64)
# 	# s = zero(Float64)
# 	for mbl in mb.mblinks
#
# 		# a = link.src;b=link.dst;
# 		ϕout=mbl.ϕout;ϕin=mbl.ϕin;
# 		for k in 1:model.K
# 			s1+=(ϕout[k]*ϕin[k]*(digamma_(model.b0[k])- digamma_(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON))#as is constant for
# 			# s+=(ϕout[k]*ϕin[k]*(digamma_(model.b0[k])- digamma_(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON))#as is constant for numerical stability for now
# 		end
# 	end
# 	s1
# end
# function elogpnetwork0(model::LNMMSB, mb::MiniBatch)
# 	s2 = zero(Float64)
# 	for mbn in mb.mbnonlinks
# 		# a = nonlink.src;b=nonlink.dst;
# 		ϕout=mbn.ϕout;ϕin=mbn.ϕin;
# 		for k in 1:model.K
# 			s2+=(ϕout[k]*ϕin[k]*(digamma_(model.b1[k])-digamma_(model.b1[k]+model.b0[k])-log(1.0-EPSILON))+log(1.0-EPSILON))#as is
# 			# s+=(ϕout[k]*ϕin[k]*(digamma_(model.b1[k])-digamma_(model.b1[k]+model.b0[k])-log(1.0-EPSILON))+log(1.0-EPSILON))#as is constant for numerical stability for now
# 		end
# 	end
# 	s2
# end
# # #
# # ####The negative entropies
# function elogqmu(model::LNMMSB)
# 	-.5*(model.K*log(2.0*pi) + model.K - logdet(model.M))
# end
# # # elogqmu(model)
# function elogqLambda(model::LNMMSB)
# 	-.5*(
# 	(model.K+1)*logdet(model.L) +
# 	model.K*(model.K+1)*log(2.0)+
# 	model.K*model.l+
# 	.5*model.K*(model.K-1)*log(pi) +
# 	2.0*lgamma_(.5*model.l, model.K) -
# 	(model.l-model.K-1)*digamma_(.5*model.l, model.K)
# 	)
# end
# # # elogqLambda(model)
# function elogqtheta(model::LNMMSB)
# 	s = zero(Float64)
# 	for a in collect(mb.mballnodes)
# 		s += (model.K*log(2.0*pi)-logdet(diagm(model.Λ_var[a,:]))+model.K)
# 	end
# 	-.5*s
# end
#
# # # elogqtheta(model)
# function elogqbeta(model::LNMMSB)
# 	s = zero(Float64)
#
# 	for k in 1:model.K
# 		s-=(
# 		lgamma_(model.b0[k])+
# 		lgamma_(model.b1[k])-
# 		lgamma_(model.b0[k]+model.b1[k]) -
# 		(model.b0[k]-1.0)*digamma_(model.b0[k]) -
# 		(model.b1[k]-1.0)*digamma_(model.b1[k]) +
# 		(model.b0[k]+model.b1[k]-2.0)*digamma_(model.b0[k]+model.b1[k])
# 		)
# 	end
# 	s
# end
# # # elogqbeta(model)
# function elogqzl(model::LNMMSB, mb::MiniBatch)
# 	s = zero(Float64)
# 	for mbl in mb.mblinks
# 		for k in 1:model.K
# 			s+=((mbl.ϕout[k]*log(mbl.ϕout[k]))+(mbl.ϕin[k]*log(mbl.ϕin[k])))
# 		end
# 	end
# 	s
# end
# # elogqzl(model)
# function elogqznl(model::LNMMSB,mb::MiniBatch)
# 	s = zero(Float64)
# 	for mbn in mb.mbnonlinks
# 		for k in 1:model.K
# 			s+=((mbn.ϕout[k]*log(mbn.ϕout[k]))+(mbn.ϕin[k]*log(mbn.ϕin[k])))
# 		end
# 	end
# 	s
# end
# # # elogqznl(model)
# function computeelbo!(model::LNMMSB, mb::MiniBatch)
# 	model.elbo=elogpmu(model)+elogpLambda(model)+elogptheta(model,mb)+elogpzlout(model,mb)+elogpzlin(model,mb)+
# 	elogpznlout(model,mb)+elogpznlin(model,mb)+elogpbeta(model)+elogpnetwork(model,mb)-
# 	(elogqmu(model)+elogqLambda(model)+elogqtheta(model)+elogqbeta(model)+elogqzl(model, mb)+elogqznl(model,mb))
# 	return model.elbo
# end
# computeelbo!(model, mb)
function updatephi!(model.LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Int64)

end
function updatephilout!(model::LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Int64, link::Link)
	for k in 1:model.K
		link.ϕout[k] = model.μ_var[link.src,k] + link.ϕin[k]*(model.Elogβ0[k]-log(EPSILON))
	end
	r = logsumexp(link.ϕout)
	link.ϕout[:] = exp(link.ϕout[:] .- r)[:]
	# for k in 1:model.K
	# 	model.ϕloutsum[link.src,k] += link.ϕout[k]##remember to zero this when needed
	# end
end
# link = Link(1,2, _init_ϕ,_init_ϕ)
function updatephilin!(model::LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Int64, link::Link)
	for k in 1:model.K
		link.ϕin[k] = model.μ_var[link.dst,k] + link.ϕout[k]*(model.Elogβ0[k]-log(EPSILON))
	end
	r=logsumexp(link.ϕin)
	link.ϕin[:] = exp.(link.ϕin[:] .- r)[:]
	# for k in 1:model.K
	# 	model.ϕlinsum[link.src,k] += link.ϕin[k]##remember to zero this when needed
	# end
end
function updatephinlout!(model::LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Int64, nlink::NonLink)
	for k in 1:model.K
		nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*(model.Elogβ1[k]-log1p(-EPSILON))
	end
	r = logsumexp(nlink.ϕout)
	nlink.ϕout[:] = exp(nlink.ϕout[:] .- r)[:]
	# for k in 1:model.K
	# 	model.ϕnloutsum[nlink.src,k] += nlink.ϕout[k]##remember to zero this when needed
	# end
end
# link = Link(1,2, _init_ϕ,_init_ϕ)
function updatephinlin!(model::LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Int64, nlink::NonLink)
	for k in 1:model.K
		nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*(model.Elogβ1[k]-log1p(-EPSILON))
	end
	r=logsumexp(nlink.ϕin)
	nlink.ϕin[:] = exp.(nlink.ϕin[:] .- r)[:]
	# for k in 1:model.K
	# 	model.ϕnlinsum[nlink.src,k] += nlink.ϕin[k]##remember to zero this when needed
	# end
end

## Need to repeat updatephilout!,updatephilin!,,updatephinlout!,updatephinlin! until
## convergence before the until convergence loop these variables need to be reinitialized
## all phis pertaining to the minibatch links and nonlinks and
## model.ϕloutsum,model.ϕlinsum,model.ϕnlinsum, model.ϕnloutsum


# function updatephil!(model::LNMMSB,  mb::MiniBatch, early::Bool, switchrounds::Bool)
#
# 	for l in mb.mblinks
# 		temp_send = zeros(Float64, model.K)
# 		s_send = zero(eltype(EPSILON))
# 		temp_recv = zeros(Float64, model.K)
# 		s_recv = zero(eltype(EPSILON))
# 		depdom = zeros(Float64, model.K)
# 		for k in 1:model.K
# 			depdom[k] = early?4.0:(model.Elogβ0[k]-log(EPSILON))
# 		end
# 		if switchrounds
# 			#send
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early?4.0:(model.Elogβ0[k]-log(EPSILON))
# 			# 	temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom)
# 			# 	s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
# 			# end
#  		   #  l.ϕout = exp.(temp_send .- s_send)
# 			for k in 1:model.K
# 				temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom[k])
# 			end
# 			l.ϕout[:] = softmax(temp_send)[:]
#
# 			#recv
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early?4.0:(model.Elogβ0[k]-log(EPSILON))
# 			# 	temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
# 			# 	s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
# 			# 	# l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
# 			# end
# 			for k in 1:model.K
# 				temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom[k])
# 			end
# 			l.ϕin[:] = softmax(temp_recv)[:]
#       # 		l.ϕin = exp.(temp_recv .- s_recv)
#
# 			# l.ϕin[:]=softmax!(l.ϕin[:])
# 			#send
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early?4.0:(model.Elogβ0[k]-log(EPSILON))
# 			# 	temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom)
# 			# 	s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
# 			# 	# l.ϕout[k]=(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
# 			# end
# 			# l.ϕout = exp.(temp_send .- s_send)
# 			for k in 1:model.K
# 				temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom[k])
# 			end
# 			l.ϕout[:] = softmax(temp_send)[:]
# 			# l.ϕout[:]=softmax!(l.ϕout[:])
# 		else
# 			#recv
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early?4.0:(model.Elogβ0[k]-log(EPSILON))
# 			# 	temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
# 			# 	s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
# 			# 	# l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
# 			# end
# 			# l.ϕin = exp.(temp_recv .- s_recv)
#
# 			for k in 1:model.K
# 				temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom[k])
# 			end
# 			l.ϕin[:] = softmax(temp_recv)[:]
# 			# l.ϕin[:]=softmax!(l.ϕin[:])
# 			#send
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early?4.0:(model.Elogβ0[k]-log(EPSILON))
# 			# 	temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom)
# 			# 	s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
# 			# 	# l.ϕout[k]=(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
# 			# end
# 			# l.ϕout = exp.(temp_send .- s_send)
# 			for k in 1:model.K
# 				temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom[k])
# 			end
# 			l.ϕout[:] = softmax(temp_send)[:]
# 			# l.ϕout[:]=softmax!(l.ϕout[:])
# 			#recv
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early?4.0:(model.Elogβ0[k]-log(EPSILON))
# 			# 	temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
# 			# 	s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
# 			# 	# l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
# 			# end
# 			# l.ϕin = exp.(temp_recv .- s_recv)
#
# 			for k in 1:model.K
# 				temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom[k])
# 			end
# 			l.ϕin[:] = softmax(temp_recv)[:]
# 			# l.ϕin[:]=softmax!(l.ϕin[:])
# 		end
# 	end
# 	model.ϕloutsum=zeros(Float64,(model.N,model.K))
# 	model.ϕlinsum=zeros(Float64,(model.N,model.K))
# 	model.ϕlinoutsum = zeros(Float64, model.K)
# 	for k in 1:model.K
# 		for l in mb.mblinks
# 			# l.ϕout[k] = l.ϕout[k] < 1e-10?1e-10:l.ϕout[k]
# 			# l.ϕin[k] = l.ϕin[k] < 1e-10 ?1e-10:l.ϕin[k]
# 			model.ϕloutsum[l.src,k]+=l.ϕout[k]
# 			model.ϕlinsum[l.dst,k]+=l.ϕin[k]
# 			model.ϕlinoutsum[k]+= l.ϕin[k]*l.ϕout[k]
# 		end
# 	end
# end

#updatephilin!(model, mb)
##need switching rounds between out and in
# function updatephinl!(model::LNMMSB, mb::MiniBatch,early::Bool, dep2::Float64,switchrounds::Bool)
# 	for nl in mb.mbnonlinks
# 		temp_send = zeros(Float64, model.K)
# 		s_send = zero(eltype(EPSILON))
# 		temp_recv = zeros(Float64, model.K)
# 		s_recv = zero(eltype(EPSILON))
# 		depdom = zeros(Float64, model.K)
# 		for k in 1:model.K
# 			depdom[k] = early ? log(dep2) : (model.Elogβ1[k]-log(1.0-EPSILON))
# 		end
# 		if switchrounds
# 			#send
# 			# for k in 1:model.K
# 			#
# 			# 	#not using extreme epsilon and instead a fixed amount
# 	        #   	depdom = early ? log(dep2) : (model.Elogβ1[k]-log(1.0-EPSILON))
# 			# 	# nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
# 			# 	temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
# 			# 	s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
# 			# end
# 			for k in 1:model.K
# 				temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom[k])
# 			end
# 			nl.ϕout[:] = softmax(temp_send)[:]
# 			# nl.ϕout = exp.(temp_send .- s_send)
# 			# nl.ϕout[:]=softmax!(nl.ϕout[:])
# 			#recv
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early ? log(dep2) : (model.Elogβ1[k]-log(1.0-EPSILON))
# 			# 	# nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
# 			# 	temp_recv[k] =(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
# 			# 	s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
# 			#
# 			# end
# 			#
# 			# nl.ϕin = exp.(temp_recv .- s_recv)
# 			for k in 1:model.K
# 				temp_recv[k] = (model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom[k])
# 			end
# 			nl.ϕin[:] = softmax(temp_recv)[:]
#
#
# 			# nl.ϕin[:]=softmax!(nl.ϕin[:])
# 			#send
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 	        #   	depdom = early ? log(dep2) : (model.Elogβ1[k]-log(1.0-EPSILON))
# 			# 	temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
# 			# 	s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
# 			# 	# nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
# 			# end
# 			# nl.ϕout = exp.(temp_send .- s_send)
# 			for k in 1:model.K
# 				temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom[k])
# 			end
# 			nl.ϕout[:] = softmax(temp_send)[:]
#
# 			# nl.ϕout[:]=softmax!(nl.ϕout[:])
# 		else
# 			#recvcomputeelbo
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early ? log(dep2) : (model.Elogβ1[k]-log(1.0-EPSILON))
# 			# 	temp_recv[k] =(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
# 			# 	s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
# 			# 	# nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
# 			# end
# 			# nl.ϕin = exp.(temp_recv .- s_recv)
#
# 			for k in 1:model.K
# 				temp_recv[k] = (model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom[k])
# 			end
# 			nl.ϕin[:] = softmax(temp_recv)[:]
# 			# nl.ϕin[:]=softmax!(nl.ϕin[:])
# 			#send
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 	        #   	depdom = early ? log(dep2) : (model.Elogβ1[k]-log(1.0-EPSILON))
# 			# 	temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
# 			# 	s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
# 			# 	# nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
# 			# end
# 			# nl.ϕout = exp.(temp_send .- s_send)
# 			for k in 1:model.K
# 				temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom[k])
# 			end
# 			nl.ϕout[:] = softmax(temp_send)[:]
#
# 			# nl.ϕout[:]=softmax!(nl.ϕout[:])
# 			#recv
# 			# for k in 1:model.K
# 			# 	#not using extreme epsilon and instead a fixed amount
# 			# 	depdom = early ? log(dep2) : (model.Elogβ1[k]-log(1.0-EPSILON))
# 			# 	temp_recv[k] =(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
# 			# 	s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
# 			# 	# nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
# 			# end
# 			# nl.ϕin = exp.(temp_recv .- s_recv)
#
# 			for k in 1:model.K
# 				temp_recv[k] = (model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom[k])
# 			end
# 			nl.ϕin[:] = softmax(temp_recv)[:]
# 			# nl.ϕin[:]=softmax!(nl.ϕin[:])
# 		end
# 	end
# 	model.ϕnloutsum=zeros(Float64,(model.N,model.K))
# 	model.ϕnlinsum=zeros(Float64,(model.N,model.K))
# 	model.ϕnlinoutsum = zeros(Float64, model.K)
# 	for k in 1:model.K
# 		for nl in mb.mbnonlinks
#
# 			model.ϕnloutsum[nl.src,k]+=nl.ϕout[k]
# 			model.ϕnlinsum[nl.dst,k]+=nl.ϕin[k]
# 			model.ϕnlinoutsum[k]+= nl.ϕin[k]*nl.ϕout[k]
# 		end
# 	end
# end



function updatesimulμΛ!(model::LNMMSB, a::Int64,mb::MiniBatch)

	model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])
	model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])


	sumb = model.train_outdeg[a]+model.train_indeg[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
	μ_var = deepcopy(model.μ_var[a,:])
	Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
	ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
	# sfx(μ_var)=softmax(μ_var + .5.*exp.(ltemp))
	sfx(μ_var)=softmax(μ_var + .5.*Λ_ivar)


	for i in 1:10
		x = rand(MvNormalCanon(diagm(model.Λ_var[a,:])*model.μ_var[a,:], diagm(model.Λ_var[a,:])))
		if i == 1

			m=model.μ_var[a,:]
			P=diagm(model.Λ_var[a,:])
			g=zeros(Float64, model.K)
			H=zeros(Float64, (model.K, model.K))
			mbar=zeros(Float64, model.K)
			mbar==zeros(Float64, model.K)
			Pbar=zeros(Float64, (model.K, model.K))
		end
		g=-model.l.*model.L*(x-model.m) +
		(model.ϕloutsum[a,:]+model.ϕlinsum[a,:]+(length(train.mbfnadj[a])/length(mb.mbfnadj[a]))*model.ϕnloutsum[a,:]+(length(train.mbbnadj[a])/length(mb.mbbnadj[a]))*model.ϕnlinsum[a,:])-
		sumb.*sfx(x)
		##train.mbfnadj should be a better way, in general we only need the counts from train
		H=-model.l.*model.L - subm.*(diagm(sfx(x)) - sfx(x)*sfx(x)')
		P = (1.0-w).*P - w.*H
		g = (1.0-w).*g + w.*g
		m = (1.0-w).*m + w.*x
		Λ_ivar = P
		μ_var = Λ_ivar*g+m
		if i > 5
			Pbar = Pbar-.2.*H
			gbar = gbar +.2.*g
			mbar = mbar +.2.*x
			Λ_ivar = Pbar
			μ_var = Λ_ivar*gbar+mbar
			Λ_ivar = Pbar
			μ_var = Λ_ivar*gbar+mbar
		end
	end


	sfx(μ_var)=softmax(μ_var + .5.*exp.(ltemp))
	##to think about
	trainfnadj[a] = model.N-1-model.ho_fnadj[a]-mb.mbfnadj[a]
	trainbnadj[a] = model.N-1-model.ho_bnadj[a]-mb.mbbnadj[a]
	dfunc(μ_var) = -model.l.*model.L*(μ_var-model.m) +
	(model.ϕloutsum[a,:]+model.ϕlinsum[a,:]+(length(train.mbfnadj[a])/length(mb.mbfnadj[a]))*model.ϕnloutsum[a,:]+(length(train.mbbnadj[a])/length(mb.mbbnadj[a]))*model.ϕnlinsum[a,:])-
	sumb.*sfx(μ_var)


	func(μ_var,ltemp) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
	(model.ϕloutsum[a,:]'+model.ϕlinsum[a,:]'+(length(train.mbfnadj[a])/length(mb.mbfnadj[a]))*model.ϕnloutsum[a,:]'+(length(train.mbbnadj[a])/length(mb.mbbnadj[a]))*model.ϕnlinsum[a,:]')*μ_var-
	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp

	func1(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
	(model.ϕloutsum[a,:]'+model.ϕlinsum[a,:]'+(length(train.mbfnadj[a])/length(mb.mbfnadj[a]))*model.ϕnloutsum[a,:]'+(length(train.mbbnadj[a])/length(mb.mbbnadj[a]))*model.ϕnlinsum[a,:]')*μ_var-
	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

	func2(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))

	opt1 = Adagrad()
	opt2 = Adagrad()
	oldval = func(μ_var, ltemp)
	g1 = -dfunc(μ_var)
	δ1 = update(opt1,g1)
	g2 = -ForwardDiff.gradient(func2, ltemp)
	δ2 = update(opt2,g2)
	newval = func(μ_var-δ1, ltemp-δ2)
	μ_var-=δ1
	ltemp-=δ2
	##Have to have this newval and oldval outside to not take the first step
	while oldval > newval
		if isapprox(newval, oldval)
			break;
		else
			g1 = -dfunc(μ_var)
			δ1 = update(opt1,g1)
			g2 = -ForwardDiff.gradient(func2, ltemp)
			δ2 = update(opt2,g2)
			μ_var-=δ1
			ltemp-=δ2
			oldval=newval
			newval = func(μ_var,ltemp)
		end
	end
	model.μ_var[a,:]=μ_var
	model.Λ_var[a,:]=1.0./exp.(ltemp)
	print();
#######
end
function updateM!(model::LNMMSB,mb::MiniBatch)
	##Only to make it MB dependent
	model.M_old = deepcopy(model.M)
	model.M = ((model.l*model.N).*model.L)+model.M0
end
#updateM!(model,mb)
function updatem!(model::LNMMSB, mb::MiniBatch)
	s = zeros(Float64, model.K)
	for a in mb.mbnodes
		s.+=model.μ_var[a,:]
	end
	model.m_old = deepcopy(model.m)
	model.m=inv(model.M)*(model.M0*model.m0+((convert(Float64,model.N)/convert(Float64,model.mbsize))*model.l).*model.L*s)
end
#updatem!(model, mb)
function updatel!(model::LNMMSB)
	##should be set in advance, not needed in the loop
	model.l = model.l0+convert(Float64,model.N)
end
#updatel!(model,mb)
function updateL!(model::LNMMSB, mb::MiniBatch)
	s = zero(Float64)
	for a in mb.mbnodes
		s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+diagm(1.0./model.Λ_var[a,:])
	end
	s=(convert(Float64,model.N)/convert(Float64,model.mbsize)).*s
	s+=inv(model.L0)+model.N.*inv(model.M)
	model.L_old = deepcopy(model.L)
	model.L = inv(s)
end
#updateL!(model, mb)
function updateb0!(model::LNMMSB, mb::MiniBatch)
	model.b0_old = deepcopy(model.b0)
	#replace #length(train.mblinks)
	train_links_num=convert(Float64, length(train.mblinks))
	@assert !isequal(model.ϕlinoutsum[:], zeros(Float64, model.K))
	model.b0[:] = model.η0.+(train_links_num/convert(Float64,length(mb.mblinks))).*model.ϕlinoutsum[:]
end
#updateb0!(model, mb)
function updateb1!(model::LNMMSB, mb::MiniBatch)
	model.b1_old = deepcopy(model.b1)
	#replace #length(train.mblinks)
	train_nlinks_num = convert(Float64,length(model.train_nonlinks))
	@assert !isequal(model.ϕnlinoutsum[:], zeros(Float64, model.K))
	model.b1[:] = model.η1.+(train_nlinks_num/convert(Float64,length(mb.mbnonlinks))).*model.ϕnlinoutsum[:]
end
#updateb1!(model, mb)



function mcsampler(model,mean, diagprec, num)
	num=100
	mean=[2.0,1.0,3.0,1.0]
	diagprec=[2.0,.3,4.0, 1.5]
	vec = zeros(Float64, (num, model.K))
	for i in 1:num
		e = rand(MvNormal(eye(Float64, model.K)),1)
		vec[i,:] = mean+(1.0./diagprec).*(e)
	end
	return vec

end
function estimate_βs(model::LNMMSB, mb::MiniBatch)
	model.est_β=model.b0./(model.b0.+model.b1)
end


function estimate_θs!(model::LNMMSB, mb::MiniBatch)
	for a in collect(mb.mballnodes)
		# model.est_θ[a,:]=exp(model.μ_var[a,:])./sum(exp(model.μ_var[a,:]))#
		model.est_θ[a,:]=softmax(model.μ_var[a,:])
	end
	# model.μ_var[1,:]
	# softmax!(model.μ_var[1,:])
	# model.μ_var[1,:]
	# model.est_θ[1,:]=softmax!(model.μ_var[1,:])
	# model.μ_var
	# model.est_θ
	# softmax!()
	# model.μ_var[1,:]
	# exp(model.μ_var[1,1])/sum(exp(model.μ_var[1,:]))
	# exp(model.μ_var[1,4])/sum(exp(model.μ_var[1,:]))
	model.est_θ[collect(mb.mballnodes),:]
end

function estimate_μs(model::LNMMSB, mb::MiniBatch)
	model.est_μ = reshape(reduce(mean, model.μ_var, 1)',model.K)
end

function estimate_Λs(model::LNMMSB, mb::MiniBatch)
end

##How to determine belonging to a community based on the membership vector???
function computeNMI(x::Matrix{Float64}, y::Matrix{Float64}, communities::Dict{Int64, Vector{Int64}}, threshold::Float64)
	open("./file2", "w") do f
	  for k in 1:size(x,2)
	    for i in 1:size(x,1)
	      if x[i,k] > threshold #3.0/size(x,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end
	open("./file1", "w") do f
	  for k in 1:size(y,2)
	    for i in 1:size(y,1)
	      if y[i,k] > threshold #3.0/size(y,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end

	open("./file3", "w") do f
	  for k in 1:length(communities)
	  	if length(communities[k]) <= 10
			continue;
		end
	    for e in communities[k]
        	write(f, "$e ")
	    end
	    write(f, "\n")
	  end
	end
	println("NMI of estimated vs truth")
	run(`src/cpp/NMI/onmi file2 file1`)
	println("NMI of estimated vs init")
	run(`src/cpp/NMI/onmi file2 file3`)
	println("NMI of truth vs init")
	run(`src/cpp/NMI/onmi file1 file3`)

end
function computeNMI_med(x::Matrix{Float64}, y::Matrix{Float64}, communities::Dict{Int64, Vector{Int64}}, threshold::Float64)
	open("./file2", "w") do f
	  for k in 1:size(x,2)
	    for i in 1:size(x,1)
	      if x[i,k] > threshold #3.0/size(x,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end
	open("./file1", "w") do f
	  for k in 1:size(y,2)
	    for i in 1:size(y,1)
	      if y[i,k] > threshold #3.0/size(y,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end
	nmi = read(`src/cpp/NMI/onmi file2 file1`, String)
	nmi=parse(Float64,nmi[6:(end-1)])
	return nmi
end

function computeNMI2(model::LNMMSB, mb::MiniBatch)
	# Px = mean(model.est_θ, 1)
	K = size(true_thetas,2)
	true_thetas = readdlm("data/true_thetas.txt")
	Py = reshape(mean(true_thetas, 1), K)
	Px = reshape(mean(x, 1),K)
	a = Matrix2d{Float64}(K, K)
	b = Matrix2d{Float64}(K, K)
	c = Matrix2d{Float64}(K, K)
	d = Matrix2d{Float64}(K, K)

	for i in 1:K
		for j in 1:K
			a[i,j] = .5*model.N*((1.0-Px[i])+(1.0-Py[j]))
			b[i,j] = .5*model.N*((1.0-Px[i])+Py[j])
			c[i,j] = .5*model.N*(Px[i]+(1.0-Py[j]))
			d[i,j] = .5*model.N*(Px[i]+Py[j])
		end
	end
	HxiCyj = Matrix2d{Float64}(K, K)
	h(w,n) = -w*log(2, w/n)
	for i in 1:K
		for j in 1:K
			HxiCyj[i,j] = (h(a[i,j], model.N)+h(d[i,j], model.N))>(h(b[i,j], model.N)+h(c[i,j], model.N))? h(a[i,j], model.N)+h(b[i,j], model.N)+h(c[i,j], model.N)+h(d[i,j], model.N)-
			h(b[i,j]+d[i,j], model.N)-h(a[i,j]+c[i,j], model.N) :h(c[i,j]+d[i,j], model.N)+h(a[i,j]+b[i,j], model.N)
		end
	end
	HxiCy = Vector{Float64}(K)
	for i in 1:K
		HxiCy[i] = minimum(HxiCyj[i,:])
	end
	HxCy = sum(HxiCy)
end
function edge_likelihood(model::LNMMSB,pair::Dyad, β_est::Vector{Float64})
    s = zero(Float64)
    S = Float64
    prob = zero(Float64)
	src = pair.src
	dst = pair.dst
    for k in 1:model.K
        if isalink(model.network, pair.src, pair.dst)
            prob += (model.μ_var[src,k]/sum(model.μ_var[src,:]))*(model.μ_var[dst,k]/sum(model.μ_var[dst,:]))*(β_est[k])
        else
            prob += (model.μ_var[src,k]/sum(model.μ_var[src,:]))*(model.μ_var[dst,k]/sum(model.μ_var[dst,:]))*(1.0-β_est[k])
        end
        s += (model.μ_var[src,k]/sum(model.μ_var[src,:]))*(model.μ_var[dst,k]/sum(model.μ_var[dst,:]))
    end

    if isalink(model.network, pair.src, pair.dst)
        prob += (1.0-s)*EPSILON
    else
        prob += (1.0-s)*(1.0-EPSILON)
    end
    return log(prob)::Float64
end
print("");
