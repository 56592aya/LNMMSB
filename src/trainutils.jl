# using SpecialFunctions
using ForwardDiff

#negative cross entropies
function elogpmu(model::LNMMSB)
	-.5*(
	model.K*log(2.0*pi)-
	logdet(model.M0)+
	trace(model.M0*(model.m-model.m0)*(model.m-model.m0)')+
	trace(model.M0*inv(model.M))
	)
end

# elogpmu(model)
#
function elogpLambda(model::LNMMSB)
	+.5*(
	-model.K*(model.K+1)*log(2.0)+
	(model.l0-model.K-1)*digamma(.5*model.l,model.K)-
	2.0*lgamma(.5*model.l0,model.K)-
	model.l*trace(inv(model.L0)*model.L)-
	(model.K+1)*logdet(model.L)+
	model.l0*logdet(inv(model.L0)*(model.L))
	)
end
# elogpLambda(model)
#
##Very large in magnitude
function elogptheta(model::LNMMSB, mb::MiniBatch)
	s = zero(Float64)
	for a in collect(mb.mballnodes)
		s +=
		(
		model.K*log(2.0*pi)-
		digamma(.5*model.l, model.K)-
		model.K*log(2.0) -
		logdet(model.L)+
		model.l*trace(model.L*((model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+inv(model.M)+diagm(1.0./model.Λ_var[a,:])
		)
		)
		)
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
		s2+= logsumexp(model.μ_var[mbl.src,:]+.5./model.Λ_var[mbl.src,:])
	end
	s1-s2
end
# elogpzlout(model, mb)
function elogpzlin(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s1+=mbl.ϕin[k]*model.μ_var[mbl.dst,k]
		end
		s2+= logsumexp(model.μ_var[mbl.dst,:]+.5./model.Λ_var[mbl.dst,:])
	end
	s1-s2
end
# elogpzlin(model, mb)
function elogpznlout(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s1+=mbn.ϕout[k]*model.μ_var[mbn.src,k]
		end
		s2+= logsumexp(model.μ_var[mbn.src,:]+.5./model.Λ_var[mbn.src,:])
	end
	s1-s2
end
# elogpznlout(model, mb)
function elogpznlin(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s1+=mbn.ϕin[k]*model.μ_var[mbn.dst,k]
		end
		s2+= logsumexp(model.μ_var[mbn.dst,:]+.5./model.Λ_var[mbn.dst,:])
	end
	s1-s2
end
# elogpznlin(model, mb)
function elogpbeta(model::LNMMSB)
	s = zero(Float64)
	for k in 1:model.K
		s+=(
		-lgamma(model.η0)-
		lgamma(model.η1)+
		lgamma(model.η0+model.η1) +
		(model.η0-1.0)*digamma(model.b0[k]) +
		(model.η1-1.0)*digamma(model.b1[k]) -
		(model.η0+model.η1-2.0)*digamma(model.b0[k]+model.b1[k])
		)
	end
	s
end
# elogpbeta(model)
##check the effect of epsilon on the size of the change, so that in computations maybe we can skip it.
function elogpnetwork(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	# s = zero(Float64)
	for mbl in mb.mblinks

		# a = link.src;b=link.dst;
		ϕout=mbl.ϕout;ϕin=mbl.ϕin;
		for k in 1:model.K
			s1+=(ϕout[k]*ϕin[k]*(digamma(model.b0[k])- digamma(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON))#as is constant for
			# s+=(ϕout[k]*ϕin[k]*(digamma(model.b0[k])- digamma(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON))#as is constant for numerical stability for now
		end
	end
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		# a = nonlink.src;b=nonlink.dst;
		ϕout=mbn.ϕout;ϕin=mbn.ϕin;
		for k in 1:model.K
			s2+=(ϕout[k]*ϕin[k]*(digamma(model.b1[k])-digamma(model.b1[k]+model.b0[k])-log(1.0-EPSILON))+log(1.0-EPSILON))#as is
			# s+=(ϕout[k]*ϕin[k]*(digamma(model.b1[k])-digamma(model.b1[k]+model.b0[k])-log(1.0-EPSILON))+log(1.0-EPSILON))#as is constant for numerical stability for now
		end
	end
	# s
	s1+s2
	# s1+s2 == s
end
function elogpnetwork1(model::LNMMSB, mb::MiniBatch)
	s1 = zero(Float64)
	# s = zero(Float64)
	for mbl in mb.mblinks

		# a = link.src;b=link.dst;
		ϕout=mbl.ϕout;ϕin=mbl.ϕin;
		for k in 1:model.K
			s1+=(ϕout[k]*ϕin[k]*(digamma(model.b0[k])- digamma(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON))#as is constant for
			# s+=(ϕout[k]*ϕin[k]*(digamma(model.b0[k])- digamma(model.b1[k]+model.b0[k])-log(EPSILON))+log(EPSILON))#as is constant for numerical stability for now
		end
	end
	s1
end
function elogpnetwork0(model::LNMMSB, mb::MiniBatch)
	s2 = zero(Float64)
	for mbn in mb.mbnonlinks
		# a = nonlink.src;b=nonlink.dst;
		ϕout=mbn.ϕout;ϕin=mbn.ϕin;
		for k in 1:model.K
			s2+=(ϕout[k]*ϕin[k]*(digamma(model.b1[k])-digamma(model.b1[k]+model.b0[k])-log(1.0-EPSILON))+log(1.0-EPSILON))#as is
			# s+=(ϕout[k]*ϕin[k]*(digamma(model.b1[k])-digamma(model.b1[k]+model.b0[k])-log(1.0-EPSILON))+log(1.0-EPSILON))#as is constant for numerical stability for now
		end
	end
	s2
end
#
####The negative entropies
function elogqmu(model::LNMMSB)
	-.5*(model.K*log(2.0*pi) + model.K - logdet(model.M))
end
# elogqmu(model)
function elogqLambda(model::LNMMSB)
	-.5*(
	(model.K+1)*logdet(model.L) +
	model.K*(model.K+1)*log(2.0)+
	model.K*model.l+
	.5*model.K*(model.K-1)*log(pi) +
	2.0*lgamma(.5*model.l, model.K) -
	(model.l-model.K-1)*digamma(.5*model.l, model.K)
	)
end
# elogqLambda(model)
function elogqtheta(model::LNMMSB)
	s = zero(Float64)
	for a in collect(mb.mballnodes)
		# a=1
		# diagm(model.Λ_var[a,:])
		s += (model.K*log(2.0*pi)-logdet(diagm(model.Λ_var[a,:]))+model.K)
		# det(diagm(model.Λ_var[a,:]))
	end
	-.5*s
end
# elogqtheta(model)
function elogqbeta(model::LNMMSB)
	s = zero(Float64)

	for k in 1:model.K
		s-=(
		lgamma(model.b0[k])+
		lgamma(model.b1[k])-
		lgamma(model.b0[k]+model.b1[k]) -
		(model.b0[k]-1.0)*digamma(model.b0[k]) -
		(model.b1[k]-1.0)*digamma(model.b1[k]) +
		(model.b0[k]+model.b1[k]-2.0)*digamma(model.b0[k]+model.b1[k])
		)
	end
	s
end
# elogqbeta(model)
function elogqzl(model::LNMMSB, mb::MiniBatch)
	s = zero(Float64)
	for mbl in mb.mblinks
		for k in 1:model.K
			s+=((mbl.ϕout[k]*log(mbl.ϕout[k]))+(mbl.ϕin[k]*log(mbl.ϕin[k])))
		end
	end
	s
end
# elogqzl(model)
function elogqznl(model::LNMMSB,mb::MiniBatch)
	s = zero(Float64)
	for mbn in mb.mbnonlinks
		for k in 1:model.K
			s+=((mbn.ϕout[k]*log(mbn.ϕout[k]))+(mbn.ϕin[k]*log(mbn.ϕin[k])))
		end
	end
	s
end
# elogqznl(model)
function computeelbo!(model::LNMMSB, mb::MiniBatch)
	model.elbo=elogpmu(model)+elogpLambda(model)+elogptheta(model,mb)+elogpzlout(model,mb)+elogpzlin(model,mb)+
	elogpznlout(model,mb)+elogpznlin(model,mb)+elogpbeta(model)+elogpnetwork(model,mb)-
	(elogqmu(model)+elogqLambda(model)+elogqtheta(model)+elogqbeta(model)+elogqzl(model, mb)+elogqznl(model,mb))
	return model.elbo
end
# computeelbo!(model, mb)

function updateM!(model::LNMMSB,mb::MiniBatch)
	##Only to make it MB dependent
	model.M_old = deepcopy(model.M)
	model.M = (model.l*model.N).*model.L+model.M0
end
#updateM!(model,mb)
function updatem!(model::LNMMSB, mb::MiniBatch)
	s = zeros(Float64, model.K)
	for a in collect(mb.mballnodes)
		s+=model.μ_var[a,:]
	end
	model.m_old = deepcopy(model.m)
	model.m=inv((model.l*model.N).*model.L+model.M0)*(model.M0*model.m0+((convert(Float64,model.N)/convert(Float64,model.mbsize))*model.l).*model.L*s)
end
#updatem!(model, mb)
function updatel!(model::LNMMSB)
	##should be set in advance, not needed in the loop
	model.l = model.l0+convert(Float64,model.N)
end
#updatel!(model,mb)
function updateL!(model::LNMMSB, mb::MiniBatch)
	s = zero(Float64)
	for a in collect(mb.mballnodes)
		s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+inv(model.M)+diagm(1.0./model.Λ_var[a,:])
	end
	s=(convert(Float64,model.N)/convert(Float64,model.mbsize)).*s
	s+=inv(model.L0)
	model.L_old = deepcopy(model.L)
	model.L = inv(s)
end
#updateL!(model, mb)
function updateb0!(model::LNMMSB, mb::MiniBatch)
	model.b0_old = deepcopy(model.b0)
	train_links_num=nnz(model.network)-length(model.ho_linkdict)
	if isequal(model.ϕlinoutsum[:], zeros(Float64, model.K))
		model.ϕlinoutsum = zeros(Float64, model.K)
		for k in 1:model.K
			for mbl in mb.mblinks
				model.ϕlinoutsum[k]+=mbl.ϕout[k]*mbl.ϕin[k]
			end
		end
	end
	model.b0[:] = model.η0.+((convert(Float64,train_links_num)/convert(Float64,length(mb.mblinks))).*model.ϕlinoutsum[:])
end
#updateb0!(model, mb)
function updateb1!(model::LNMMSB, mb::MiniBatch)
	model.b1_old = deepcopy(model.b1)
	train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyaddict) -length(mb.mblinks)
	if isequal(model.ϕnlinoutsum[:], zeros(Float64, model.K))
		model.ϕnlinoutsum = zeros(Float64, model.K)
		for k in 1:model.K
			for mbn in mb.mbnonlinks
				model.ϕnlinoutsum[k]+=mbn.ϕout[k]*mbn.ϕin[k]
			end
		end
	end
	model.b1[:] = model.η1.+((convert(Float64,train_nlinks_num)/convert(Float64,length(mb.mbnonlinks)))*model.ϕnlinoutsum[:])
end
#updateb1!(model, mb)

function updatephil!(model::LNMMSB,  mb::MiniBatch, early::Bool, switchrounds::Bool)
	for l in mb.mblinks
		temp_send = zeros(Float64, model.K)
		s_send = zero(eltype(EPSILON))
		temp_recv = zeros(Float64, model.K)
		s_recv = zero(eltype(EPSILON))
		if switchrounds
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?4.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log(EPSILON))
				temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom)
				s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
				# l.ϕout[k]=(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			# l.ϕout[:]=softmax!(l.ϕout[:])
			for k in 1:model.K
      			@inbounds l.ϕout[k] = exp(temp_send[k] - s_send)
    		end
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?4.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log(EPSILON))
				temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
				s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
				# l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
			end
			for k in 1:model.K
      			@inbounds l.ϕin[k] = exp(temp_recv[k] - s_recv)
    		end
			# l.ϕin[:]=softmax!(l.ϕin[:])
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?4.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log(EPSILON))
				temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom)
				s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
				# l.ϕout[k]=(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			for k in 1:model.K
      			@inbounds l.ϕout[k] = exp(temp_send[k] - s_send)
    		end
			# l.ϕout[:]=softmax!(l.ϕout[:])
		else
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?4.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log(EPSILON))
				temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
				s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
				# l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
			end
			for k in 1:model.K
      			@inbounds l.ϕin[k] = exp(temp_recv[k] - s_recv)
    		end
			# l.ϕin[:]=softmax!(l.ϕin[:])
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?4.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log(EPSILON))
				temp_send[k] = (model.μ_var[l.src,k] + l.ϕin[k]*depdom)
				s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
				# l.ϕout[k]=(model.μ_var[l.src,k] + l.ϕin[k]*depdom)
			end
			for k in 1:model.K
      			@inbounds l.ϕout[k] = exp(temp_send[k] - s_send)
    		end
			# l.ϕout[:]=softmax!(l.ϕout[:])
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early?4.0:(digamma(model.b0[k])-digamma(model.b0[k]+model.b1[k])-log(EPSILON))
				temp_recv[k] =(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
				s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
				# l.ϕin[k]=(model.μ_var[l.dst,k] + l.ϕout[k]*depdom)
			end
			for k in 1:model.K
				@inbounds l.ϕin[k] = exp(temp_recv[k] - s_recv)
			end
			# l.ϕin[:]=softmax!(l.ϕin[:])
		end
	end
	model.ϕloutsum=zeros(Float64,(model.N,model.K))
	model.ϕlinsum=zeros(Float64,(model.N,model.K))
	model.ϕlinoutsum = zeros(Float64, model.K)
	for k in 1:model.K
		for l in mb.mblinks
			# l.ϕout[k] = l.ϕout[k] < 1e-10?1e-10:l.ϕout[k]
			# l.ϕin[k] = l.ϕin[k] < 1e-10 ?1e-10:l.ϕin[k]
			model.ϕloutsum[l.src,k]+=l.ϕout[k]
			model.ϕlinsum[l.dst,k]+=l.ϕin[k]
			model.ϕlinoutsum[k]+= l.ϕin[k]*l.ϕout[k]
		end
	end
end

#updatephilin!(model, mb)
##need switching rounds between out and in
function updatephinl!(model::LNMMSB, mb::MiniBatch,early::Bool, dep2::Float64,switchrounds::Bool)
	for nl in mb.mbnonlinks
		temp_send = zeros(Float64, model.K)
		s_send = zero(eltype(EPSILON))
		temp_recv = zeros(Float64, model.K)
		s_recv = zero(eltype(EPSILON))
		if switchrounds
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : (digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log(1.0-EPSILON))
				# nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
				temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
				s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
			end
			for k in 1:model.K
				@inbounds nl.ϕout[k] = exp(temp_send[k] - s_send)
			end
			# nl.ϕout[:]=softmax!(nl.ϕout[:])
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : (digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log(1.0-EPSILON))
				# nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
				temp_recv[k] =(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
				s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]

			end
			for k in 1:model.K
				@inbounds nl.ϕin[k] = exp(temp_recv[k] - s_recv)
			end
			# nl.ϕin[:]=softmax!(nl.ϕin[:])
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : (digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log(1.0-EPSILON))
				temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
				s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
				# nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
			end
			for k in 1:model.K
				@inbounds nl.ϕout[k] = exp(temp_send[k] - s_send)
			end
			# nl.ϕout[:]=softmax!(nl.ϕout[:])
		else
			#recvcomputeelbo
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : (digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log(1.0-EPSILON))
				temp_recv[k] =(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
				s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
				# nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
			end
			for k in 1:model.K
				@inbounds nl.ϕin[k] = exp(temp_recv[k] - s_recv)
			end
			# nl.ϕin[:]=softmax!(nl.ϕin[:])
			#send
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
	          	depdom = early ? log(dep2) : (digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log(1.0-EPSILON))
				temp_send[k] = (model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
				s_send = k > 1 ? logsumexp(s_send,temp_send[k]) : temp_send[k]
				# nl.ϕout[k]=(model.μ_var[nl.src,k] + nl.ϕin[k]*depdom)
			end
			for k in 1:model.K
				@inbounds nl.ϕout[k] = exp(temp_send[k] - s_send)
			end
			# nl.ϕout[:]=softmax!(nl.ϕout[:])
			#recv
			for k in 1:model.K
				#not using extreme epsilon and instead a fixed amount
				depdom = early ? log(dep2) : (digamma(model.b1[k])-digamma(model.b0[k]+model.b1[k])-log(1.0-EPSILON))
				temp_recv[k] =(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
				s_recv = k > 1 ? logsumexp(s_recv,temp_recv[k]) : temp_recv[k]
				# nl.ϕin[k]=(model.μ_var[nl.dst,k] + nl.ϕout[k]*depdom)
			end
			for k in 1:model.K
				@inbounds nl.ϕin[k] = exp(temp_recv[k] - s_recv)
			end
			# nl.ϕin[:]=softmax!(nl.ϕin[:])
		end
	end
	model.ϕnloutsum=zeros(Float64,(model.N,model.K))
	model.ϕnlinsum=zeros(Float64,(model.N,model.K))
	model.ϕnlinoutsum = zeros(Float64, model.K)
	for k in 1:model.K
		for nl in mb.mbnonlinks

			model.ϕnloutsum[nl.src,k]+=nl.ϕout[k]
			model.ϕnlinsum[nl.dst,k]+=nl.ϕin[k]
			model.ϕnlinoutsum[k]+= nl.ϕin[k]*nl.ϕout[k]
		end
	end
end

#Newton
function updatemua!(model::LNMMSB, a::Int64, niter::Int64, ntol::Float64,mb::MiniBatch)
	model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])
	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
	μ_var = model.μ_var[a,:]
	rho=1.0
	for i in 1:niter
		# f(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		# (model.ϕloutsum[a,:]'+model.ϕlinsum[a,:]'+model.ϕnloutsum[a,:]'+model.ϕnlinsum[a,:]')*μ_var-
		# sumb*(log(ones(model.K)'*exp.(μ_var+.5./model.Λ_var[a,:])))
		rho=1.0/((1.0+Float64(i))^.9)

		# S = f(μ_var)
		# sfx=softmax(model.μ_var[a,:]+.5./model.Λ_var[a,:])
		sfx=softmax(μ_var + .5./model.Λ_var[a,:])
		mu_grad = -model.l.*model.L*(μ_var-model.m) +
		(model.ϕloutsum[a,:]+model.ϕlinsum[a,:]+model.ϕnloutsum[a,:]+model.ϕnlinsum[a,:])-
		sumb.*(sfx)
		mu_hess = -model.l.*model.L - sumb.*(diagm(sfx) - sfx*sfx')
		μ_var -=rho.*(inv(mu_hess)*mu_grad)
		i % 100 == 0 && println("niter ",string(i),", a ", string(a),", rho ", rho, ", norm_mu ", norm(mu_grad))
		# println(μ_var)
		# println(norm(mu_grad))
		if norm(mu_grad) < ntol || isnan(norm(mu_grad))
			# println(norm(mu_grad))
			# println(norm(ntol))
			# println(i)
			break
		end
	end
	model.μ_var[a,:]=μ_var
	print();
end
#Newton
###Needs fixings
function updateLambdaa!(model::LNMMSB, a::Int64, niter::Int64, ntol::Float64,mb::MiniBatch)

	model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])
	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
	Λ_var = model.Λ_var[a,:]
	inv_temp = 1.0./Λ_var
	for i in 1:niter
		rho=1.0
		# rho=1.0/((1.0+Float64(i))^.9)
		sfx=softmax(model.μ_var[a,:]+.5./Λ_var)
		Lam_grad=-.5*model.l.*diag(model.L)+.5.*Λ_var- .5.*sfx
		# ForwardDiff.hessian(g, Λ_var)
		Lam_hess=-.5*diagm(Λ_var.*Λ_var)-.25.*(diagm(sfx)-sfx*sfx')

		p =inv(Lam_hess)*Lam_grad
		while minimum((1.0./Λ_var) - rho * p) <= 0
			rho *= 0.5
		end
		inv_temp -= rho * p
		i % 100 == 0 && println("niter ",string(i),", a ", string(a),", rho ", rho, ", norm_Lam ", norm(Lam_grad))

		if norm(Lam_grad) < ntol || isnan(norm(Lam_grad))
			break
		end
	end
	model.Λ_var[a,:]=1.0./inv_temp
	print();
end

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

function estimate_θs(model::LNMMSB, mb::MiniBatch)
	for a in collect(mb.mballnodes)
		model.est_θ[a,:]=softmax!(model.μ_var[a,:])
	end
	model.est_θ
end

function estimate_μs(model::LNMMSB, mb::MiniBatch)
	model.est_μ = reshape(reduce(mean, model.μ_var, 1)',model.K)
end

function estimate_Λs(model::LNMMSB, mb::MiniBatch)
end

print("");
