# using SpecialFunctions

using GradDescent
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
		s += (model.K*log(2.0*pi)-logdet(diagm(model.Λ_var[a,:]))+model.K)
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
	model.M = ((model.l*model.N).*model.L)+model.M0
end
#updateM!(model,mb)
function updatem!(model::LNMMSB, mb::MiniBatch)
	s = zeros(Float64, model.K)
	for a in collect(mb.mballnodes)
		s.+=model.μ_var[a,:]
	end
	model.m_old = deepcopy(model.m)
	model.m=inv(((model.l*model.N).*model.L)+model.M0)*(model.M0*model.m0+((convert(Float64,model.N)/convert(Float64,model.mbsize))*model.l).*model.L*s)
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
	train_links_num=nnz(model.network)-length(model.ho_links)
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
	train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyads) -length(mb.mblinks)
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



function updatesimulμΛ!(model::LNMMSB, a::Int64, niter::Int64, ntol::Float64,mb::MiniBatch)

	model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])

	model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])

	sumb = model.train_out[a]+model.train_in[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])

	μ_var = model.μ_var[a,:]

	Λ_ivar = 1.0./model.Λ_var[a,:]

	ltemp = [log(Λ_ivar[k]) for k in 1:model.K]

	sfx(μ_var)=softmax(μ_var + .5.*exp.(ltemp))

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
		# model.est_θ[a,:]=exp(model.μ_var[a,:])./sum(exp(model.μ_var[a,:]))#
		model.est_θ[a,:]=softmax!(model.μ_var[a,:])
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
function computeNMI(x::Matrix{Float64}, y::Matrix{Float64}, communities::Dict{Int64, Vector{Int64}})
	open("./file2", "w") do f
	  for k in 1:size(x,2)
	    for i in 1:size(x,1)
	      if x[i,k] > .3 #3.0/size(x,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end
	open("./file1", "w") do f
	  for k in 1:size(y,2)
	    for i in 1:size(y,1)
	      if y[i,k] > .3 #3.0/size(y,2)
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

# function computeNMI2(model::LNMMSB, mb::MiniBatch)
# 	Px = mean(model.est_θ, 1)
# 	Hx_vec = -Px.*log.(2,Px)
# 	Hx = sum(Hx_vec)
# 	true_thetas = readdlm("data/true_thetas.txt")
# 	Py = mean(true_thetas, 1)
# 	Hy_vec = -Py.*log.(2,Py)
# 	Hy = sum(Hy_vec)
# 	Hxcy_vec =
# 	Hxcy
# 	Hycx_vec
# 	Hycx
# end
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
