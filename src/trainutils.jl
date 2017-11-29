# using SpecialFunctions
using GradDescent
using ForwardDiff

function updatephibar!(model::LNMMSB, mb::MiniBatch, a::Int64)
	model.ϕbar[a,:]= (model.ϕloutsum[a,:]+model.ϕlinsum[a,:])/(model.train_outdeg[a]+model.train_indeg[a])
end
function updatephilout!(model::LNMMSB, mb::MiniBatch, early::Bool, link::Link, tuner::Float64)
	for k in 1:model.K
		if early
			link.ϕout[k] = model.μ_var[link.src,k] + link.ϕin[k]*tuner
		else
			link.ϕout[k] = model.μ_var[link.src,k] + link.ϕin[k]*(model.Elogβ0[k]-log(EPSILON))
			# link.ϕout[k] = model.μ_var[link.src,k] + link.ϕin[k]*(model.Elogβ0[k])#+sum(link.ϕin[1:end .!=k].*log(EPSILON))
		end
	end
	x=[1,2,3,4]

	r = logsumexp(link.ϕout)
	link.ϕout[:] = exp.(link.ϕout[:] .- r)[:]

end
function updatephilin!(model::LNMMSB, mb::MiniBatch, early::Bool, link::Link,tuner::Float64)
	for k in 1:model.K
		if early
			link.ϕin[k] = model.μ_var[link.dst,k] + link.ϕout[k]*tuner
		else
			link.ϕin[k] = model.μ_var[link.dst,k] + link.ϕout[k]*(model.Elogβ0[k]-log(EPSILON))
			# link.ϕin[k] = model.μ_var[link.dst,k] + link.ϕout[k]*(model.Elogβ0[k])#+sum(link.ϕout[1:end .!=k].*log(EPSILON))
		end
	end
	r=logsumexp(link.ϕin)
	link.ϕin[:] = exp.(link.ϕin[:] .- r)[:]
end
function updatephinlout!(model::LNMMSB, mb::MiniBatch, early::Bool, nlink::NonLink, tuner::Float64)

	for k in 1:model.K
		if early
			nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*tuner
		else
			nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*(model.Elogβ1[k]-log(1-EPSILON))
			# nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*(model.Elogβ1[k])#+sum(nlink.ϕin[1:end .!=k].*log1p(-EPSILON))
		end
	end
	r = logsumexp(nlink.ϕout)
	nlink.ϕout[:] = exp.(nlink.ϕout[:] .- r)[:]
end
mb.mblinks
function updatephinlin!(model::LNMMSB, mb::MiniBatch, early::Bool, nlink::NonLink,tuner::Float64)
	for k in 1:model.K
		if early
			nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*tuner
		else
			nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*(model.Elogβ1[k]-log(1-EPSILON))
			# nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*(model.Elogβ1[k])#+sum(nlink.ϕout[1:end .!=k].*log1p(-EPSILON))
		end
	end
	r=logsumexp(nlink.ϕin)
	nlink.ϕin[:] = exp.(nlink.ϕin[:] .- r)[:]
end

## Need to repeat updatephilout!,updatephilin!,,updatephinlout!,updatephinlin! until
## convergence before the until convergence loop these variables need to be reinitialized
## all phis pertaining to the minibatch links and nonlinks and
## model.ϕloutsum,model.ϕlinsum,model.ϕnlinsum, model.ϕnloutsum

function sfx(μ_var::Vector{Float64},ltemp::Vector{Float64})
	return softmax(μ_var .+.5.*exp.(ltemp))
end
function updatesimulμΛ!(model::LNMMSB, a::Int64,mb::MiniBatch,meth::String)
		model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])
		model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])
		μ_var = deepcopy(model.μ_var[a,:])
		Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
		ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
		x = sfx(μ_var,ltemp)
		s1 = haskey(mb.mbfnadj,a)?(N-1-model.train_outdeg[a]):0
		c1 = haskey(mb.mbfnadj,a)?length(mb.mbfnadj[a]):1
		s2 = haskey(mb.mbbnadj,a)?(N-1-model.train_indeg[a]):0
		c2 = haskey(mb.mbbnadj,a)?length(mb.mbbnadj[a]):1
		sumb =2*(model.N-1)
		X=model.ϕloutsum[a,:]+model.ϕlinsum[a,:]+(s1/c1).*(model.ϕnloutsum[a,:])+(s2/c2).*(model.ϕnlinsum[a,:])
		# X=2*model.N.*(model.ϕloutsum[a,:]+model.ϕlinsum[a,:])+(2*10*model.N).*((model.ϕnloutsum[a,:]).+(model.ϕnlinsum[a,:]))

		dfunci(μ_var) = -model.l.*model.L*(μ_var-model.m) +X - sumb.*x

		funci(μ_var,ltemp) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp

		func1i(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

		func2i(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))

		# opt1 = RMSprop()
		# opt2 = RMSprop()
		opt1 = Adagrad()
		opt2 = Adagrad()
		for i in 1:100
			x  = sfx(μ_var,ltemp)
			g1 = dfunci(μ_var)
			δ1 = update(opt1,g1)
			g2 = ForwardDiff.gradient(func2i, ltemp)
			δ2 = update(opt2,g2)
			μ_var+=δ1
			ltemp+=δ2
		end
		model.μ_var[a,:]=μ_var
		model.Λ_var[a,:]=1.0./exp.(ltemp)
	# end
	print();
end
function updatesimulμΛlink!(model::LNMMSB, a::Int64,mb::MiniBatch,meth::String)
		model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])
		model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])
		μ_var = deepcopy(model.μ_var[a,:])
		Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
		ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
		x = sfx(μ_var,ltemp)
		sumb =2*(model.N-1)
		X=model.ϕloutsum[a,:]+model.ϕlinsum[a,:]*((2*model.N-2)/(model.train_outdeg[a]+model.train_indeg[a]))
		# X=2*model.N.*(model.ϕloutsum[a,:]+model.ϕlinsum[a,:])+(2*10*model.N).*((model.ϕnloutsum[a,:]).+(model.ϕnlinsum[a,:]))

		dfunci(μ_var) = -model.l.*model.L*(μ_var.-model.m) +X - sumb.*x

		funci(μ_var,ltemp) = -.5*model.l*((μ_var.-model.m)'*model.L*(μ_var.-model.m))+
		(X)'*μ_var-sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp

		func1i(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

		func2i(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))

		opt1 = Adagrad()
		opt2 = Adagrad()
		for i in 1:5
			x  = sfx(μ_var,ltemp)
			g1 = dfunci(μ_var)
			δ1 = update(opt1,g1)
			g2 = ForwardDiff.gradient(func2i, ltemp)
			δ2 = update(opt2,g2)
			μ_var+=δ1
			ltemp+=δ2
		end
		model.μ_var[a,:]=μ_var
		model.Λ_var[a,:]=1.0./exp.(ltemp)
	# end
	print();
end
function updatesimulμΛfull!(model::LNMMSB, a::Int64,mb::MiniBatch,meth::String)
		model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])
		model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])
		μ_var = deepcopy(model.μ_var[a,:])
		Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
		ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
		x = sfx(μ_var,ltemp)
		sumb =2*(model.N-1)
		X=(model.N/model.mbsize).*(model.ϕloutsum[a,:].+model.ϕlinsum[a,:]).+(model.N*10/model.mbsize).*(model.ϕnloutsum[a,:].+model.ϕnlinsum[a,:])
		dfunci(μ_var) = -model.l.*model.L*(μ_var-model.m) + X - sumb.*x

		funci(μ_var,ltemp) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp

		func1i(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

		func2i(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))
		opt1 = Adagrad(η=1.0)
		opt2 = Adagrad(η=1.0)
		for i in 1:2
			x  = sfx(μ_var,ltemp)
			g1 = dfunci(μ_var)
			δ1 = update(opt1,g1)
			g2 = ForwardDiff.gradient(func2i, ltemp)
			δ2 = update(opt2,g2)
			μ_var+=δ1
			ltemp+=δ2
		end
		model.μ_var[a,:]=μ_var
		model.Λ_var[a,:]=1.0./exp.(ltemp)
	# end
	print();
end
function updatesimulμΛ2!(model::LNMMSB, a::Int64,mb::MiniBatch,scale::Int64)
		model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])
		model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])
		μ_var = deepcopy(model.μ_var[a,:])
		Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
		ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
		x = scale.*sfx(μ_var,ltemp)

		sumb =2*(model.N-1)
		X=scale.*(model.ϕloutsum[a,:]+model.ϕlinsum[a,:]+(model.ϕnloutsum[a,:])+(model.ϕnlinsum[a,:]))

		dfunci(μ_var) = -model.l.*model.L*(μ_var-model.m) +X - sumb.*x

		funci(μ_var,ltemp) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-scale*sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp

		func1i(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-	scale*su#updateb1!(model, mb)mb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

		func2i(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-scale*sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))

		# opt1 = RMSprop()
		# opt2 = RMSprop()
		opt1 = Adagrad(η=1.0)
		opt2 = Adagrad(η=1.0)
		for i in 1:10
			x  = scale.*sfx(μ_var,ltemp)
			g1 = dfunci(μ_var)
			δ1 = update(opt1,g1)
			g2 = ForwardDiff.gradient(func2i, ltemp)
			δ2 = update(opt2,g2)
			μ_var+=δ1
			ltemp+=δ2
		end
		model.μ_var[a,:]=μ_var
		model.Λ_var[a,:]=1.0./exp.(ltemp)
	# end
	print();
end
function updateM!(model::LNMMSB,mb::MiniBatch)
	##Only to make it MB dependent
	model.M_old = deepcopy(model.M)
	model.M = ((model.l*model.N).*model.L)+model.M0
end
#updateM!(model,mb)
function updatem!(model::LNMMSB, mb::MiniBatch)
	model.m_old = deepcopy(model.m)
	s = zeros(Float64, model.K)
	for a in mb.mbnodes
		s.+=model.μ_var[a,:]
	end
	# scaler=(convert(Float64,model.N)/convert(Float64,model.mbsize))
	scaler = model.N/model.mbsize
	model.m=inv(model.M)*(model.M0*model.m0+(scaler*model.l).*model.L*s)
end
function updatemfull!(model::LNMMSB, mb::MiniBatch)
	model.m_old = deepcopy(model.m)
	s = zeros(Float64, model.K)
	for a in mb.mbnodes
		s.+=model.μ_var[a,:]
	end
	model.m=inv(model.M)*(model.M0*model.m0+model.l).*model.L*s
end
function updatemtemp!(model::LNMMSB, mb::MiniBatch, scale)
	model.m_old = deepcopy(model.m)
	s = zeros(Float64, model.K)
	for a in mb.mbnodes
		s.+=model.μ_var[a,:]
	end
	# scaler=(convert(Float64,model.N)/convert(Float64,model.mbsize))
	model.m=inv(model.M)*(model.M0*model.m0+scale*model.l).*model.L*s
end
#updatem!(model, mb)
function updatel!(model::LNMMSB)
	##should be set in advance, not needed in the loop
	model.l = model.l0+convert(Float64,model.N)
end
#updatel!(model,mb)
function updateL!(model::LNMMSB, mb::MiniBatch)
	model.L_old = deepcopy(model.L)
	s = zero(Float64)
	for a in mb.mbnodes
		s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+diagm(1.0./model.Λ_var[a,:])
	end
	s=(model.N/model.mbsize)*s
	# s=(convert(Float64,model.N)/convert(Float64,model.mbsize)).*s
	s+=inv(model.L0)+model.N.*inv(model.M)
	model.L = inv(s)
end
function updateLfull!(model::LNMMSB, mb::MiniBatch)
	model.L_old = deepcopy(model.L)
	s = zero(Float64)
	for a in mb.mbnodes
		s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+diagm(1.0./model.Λ_var[a,:])
	end
	s+=inv(model.L0)+model.N.*inv(model.M)
	model.L = inv(s)
end
function updateLtemp!(model::LNMMSB, mb::MiniBatch,scale)
	model.L_old = deepcopy(model.L)
	s = zero(Float64)
	for a in mb.mbnodes
		s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+diagm(1.0./model.Λ_var[a,:])
	end
	s=scale.*s
	s+=inv(model.L0)+model.N.*inv(model.M)
	model.L = inv(s)
end
#updateL!(model, mb)
function updateb0!(model::LNMMSB, mb::MiniBatch)
	model.b0_old = deepcopy(model.b0)
	# scaler=2*model.N
	scaler=nnz(model.network)/length(mb.mblinks)
	@assert !isequal(model.ϕlinoutsum[:], zeros(Float64, model.K))
	model.b0[:] = model.η0.+ (scaler.*model.ϕlinoutsum[:])
end
function updateb0full!(model::LNMMSB, mb::MiniBatch)
	model.b0_old = deepcopy(model.b0)
	# @assert !isequal(model.ϕlinoutsum[:], zeros(Float64, model.K))
	model.b0[:] = model.η0.+ model.ϕlinoutsum[:]
end
function updateb0temp!(model::LNMMSB, mb::MiniBatch, scale,ϕlinoutsum_old)
	model.b0_old = deepcopy(model.b0)
	#replace #length(train.mblinks)
	# train_links_num=convert(Float64, train_links_num)
	train_links_num=convert(Float64,nnz(model.network))
	# @assert !isequal(model.ϕlinoutsum[:], zeros(Float64, model.K))
	model.b0[:] = model.η0.+ (scale.*ϕnlinoutsum_old[:])
end
#updateb0!(model, mb)
function updateb1!(model::LNMMSB, mb::MiniBatch, meth::String)
	model.b1_old = deepcopy(model.b1)
	scaler = (model.N^2 - model.N - nnz(model.network))/length(mb.mbnonlinks)
	# scaler = (model.N/model.mbsize)*10
	model.b1[:] = model.η1 .+ (scaler.*model.ϕnlinoutsum[:])
end
function updateb1link!(model::LNMMSB, mb::MiniBatch, meth::String)
	model.b1_old = deepcopy(model.b1)
	scaler = (model.N^2 - model.N - nnz(model.network))/length(mb.mbnonlinks)
	s1 = zeros(Float64, model.K)
	s2 = zeros(Float64, model.K)
	s3 = zeros(Float64, model.K)
	s4 = zeros(Float64, (model.N, model.K))
	for k in 1:model.K
		for a in mb.mbnodes
			s1[k] += model.ϕbar[a,k]
			s2[k] += (model.ϕbar[a,k]).^2
			for b in model.train_sinks[a]
				s4[a,k]+=model.ϕbar[b,k]
			end
		end
	end
	for k in 1:model.K
		for a in mb.mbnodes
			s3[k]+=model.ϕbar[a,k]*s4[a,k]
		end
	end
	s1*=(model.N/model.mbsize)
	s1 = s1.^2
	s2*=(model.N/model.mbsize)
	s3*= (model.N/model.mbsize)
	# scaler = ((model.N^2 - model.N - nnz(model.network))/length(mb.mbnonlinks))

	# model.b1[:] = model.η1 .+ scaler.*(s1  .-s2 .- s3)
	model.b1[:] = model.η1 .+ (s1  .-s2 .- s3)
end
function updateb1full!(model::LNMMSB, mb::MiniBatch, meth::String)
	model.b1_old = deepcopy(model.b1)
	model.b1[:] = model.η1 .+ model.ϕnlinoutsum[:]
end
function updateb1temp!(model::LNMMSB, mb::MiniBatch, scale,ϕnlinoutsum_old)
	model.b1_old = deepcopy(model.b1)
	# train_nlinks_num = convert(Float64,train_nlinks_num)
	train_nlinks_num = model.N^2 - model.N - nnz(model.network)
	# scaler= length(model.train_nonlinks)/(model.N*length(mb.mbnonlinks))

	model.b1[:] = model.η1 .+ (scale.*ϕnlinoutsum_old[:])
	# model.b1[model.b1 .< .0] = model.η1
end

function prune!(model::LNMMSB, mb::MiniBatch)
	estk = zeros(Float64, model.K)
	estk = (sum(model.est_θ,1)./sum(model.est_θ))[1,:]
	orderedindexes = sortperm(estk)
	flagk = [false for k in 1:model.K]
	remove = Int64[]
	for k in orderedindexes
		if length(remove) == 2
			break;
		end
		if estk[k] < log2(model.K)/model.N
			flagk[k] = true
			push!(remove, k)
		end
	end

	if isempty(remove)
		return nothing;
	end
	s = 0.0
	if length(remove) < 3
		for r in remove
			s += sum(model.est_θ[:,r])
		end
		s /= (model.K-length(remove))*model.N
		for k in 1:model.K
			if !(k in remove)
				model.est_θ[:,k] += s
				model.μ_var[:,k] = log.(model.est_θ[:,k])
			end
		end
	end
	kindexes = [j for j in 1:model.K if !(j in remove) ]
	model.K -=length(remove)
	return kindexes
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
function estimate_βs!(model::LNMMSB, mb::MiniBatch)
	model.est_β=model.b0./(model.b0.+model.b1)
end


function estimate_θs!(model::LNMMSB, mb::MiniBatch)
	for a in mb.mbnodes
		# model.est_θ[a,:]=exp(model.μ_var[a,:])./sum(exp(model.μ_var[a,:]))#
		model.est_θ[a,:]=softmax(model.μ_var[a,:])
	end
	model.est_θ[mb.mbnodes,:]
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
	nmi = read(`src/cpp/NMI/onmi file2 file1`, String)
	nmi=parse(Float64,nmi[6:(end-1)])
	println("NMI of estimated vs init")
	run(`src/cpp/NMI/onmi file2 file3`)
	println("NMI of truth vs init")
	run(`src/cpp/NMI/onmi file1 file3`)
	return nmi
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

function log_comm(model::LNMMSB, mb::MiniBatch, link::Link, link_thresh::Float64, min_deg::Int64)
	m_out = indmax(link.ϕout)
	if link.ϕout[m_out] > link_thresh
		model.fmap[link.src,m_out]+=1
		model.fmap[link.dst,m_out]+=1
		if model.fmap[link.src,m_out] > minimum(model.train_outdeg)
			push!(model.comm[m_out], link.src)
		end

		if model.fmap[link.src,m_out] >minimum(model.train_outdeg)
			push!(model.comm[m_out], link.dst)
		end
	end
	m_in = indmax(link.ϕin)
	if link.ϕin[m_in] > link_thresh
		model.fmap[link.src,m_in]+=1
		model.fmap[link.dst,m_in]+=1
		if model.fmap[link.dst,m_in] > minimum(model.train_indeg)
			push!(model.comm[m_in], link.src)
		end

		if model.fmap[link.dst,m_in] >minimum(model.train_indeg)
			push!(model.comm[m_in], link.dst)
		end
	end
end
function compute_NMI3(model::LNMMSB)

	open("./data/est_comm.txt", "w") do f
	  for k in 1:length(model.comm)
	    for n in model.comm[k]
	        write(f, "$n ")
	    end
	    write(f, "\n")
	  end
	end

	println("NMI of estimated vs truth")
	# run(`src/cpp/NMI/onmi file2 file1`)
	nmi = read(`src/cpp/NMI/onmi data/truth_comm.txt data/est_comm.txt`, String)
	nmi=parse(Float64,nmi[6:(end-1)])
	return nmi
end
function edge_likelihood(model::LNMMSB,pair::Dyad, β_est::Vector{Float64})
    s = zero(Float64)
    S = Float64
    prob = zero(Float64)
	src = pair.src
	dst = pair.dst
	sfxa=softmax(model.μ_var[pair.src,:])
	sfxb=softmax(model.μ_var[pair.dst,:])
    for k in 1:model.K

        if isalink(model, "network",src, dst)
            prob += (sfxa[k]/sum(sfxa))*(sfxb[k]/sum(sfxb))*(β_est[k])
        else
            prob += (sfxa[k]/sum(sfxa))*(sfxb[k]/sum(sfxb))*(1.0-β_est[k])
        end
        s += (sfxa[k]/sum(sfxa))*(sfxb[k]/sum(sfxb))
    end

    if isalink(model, "network",src, dst)
        prob += (1.0-s)*EPSILON
    else
        prob += (1.0-s)*(1.0-EPSILON)
    end
    return log1p(-1.0+prob)::Float64
end
print("");
