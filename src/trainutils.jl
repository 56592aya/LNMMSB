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
			link.ϕout[k] = model.μ_var[link.src,k] + link.ϕin[k]*(model.Elogβ0[k]-log1p(-1.0+EPSILON))
		end
	end
	r = logsumexp(link.ϕout)
	link.ϕout[:] = exp.(link.ϕout[:] .- r)[:]
end
function updatephilin!(model::LNMMSB, mb::MiniBatch, early::Bool, link::Link,tuner::Float64)

	for k in 1:model.K
		if early
			link.ϕin[k] = model.μ_var[link.dst,k] + link.ϕout[k]*tuner
		else
			link.ϕin[k] = model.μ_var[link.dst,k] + link.ϕout[k]*(model.Elogβ0[k]-log1p(-1.0+EPSILON))
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
			nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*(model.Elogβ1[k]-log1p(-EPSILON))
		end
	end
	r = logsumexp(nlink.ϕout)
	nlink.ϕout[:] = exp.(nlink.ϕout[:] .- r)[:]
end
function updatephinlin!(model::LNMMSB, mb::MiniBatch, early::Bool, nlink::NonLink,tuner::Float64)
	for k in 1:model.K
		if early
			nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*tuner
		else
			nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*(model.Elogβ1[k]-log1p(-EPSILON))
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
	# if meth == "link"
	# 	μ_var = deepcopy(model.μ_var[a,:])
	# 	Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
	# 	ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
	#
	# 	sumb = 2.0*(model.N-1.0)
	# 	ca = model.train_outdeg[a]+model.train_indeg[a]
	# 	x = sfx(μ_var,ltemp)
	# 	X=model.ϕloutsum[a,:]+model.ϕlinsum[a,:]
	# 	scaler = sumb/ca
	# 	dfunc(μ_var) = -model.l.*model.L*(μ_var-model.m) +scaler.*X - sumb.*x
	#
	# 	func(μ_var,ltemp) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
	# 	(scaler.*X)'*μ_var-sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp
	#
	# 	func1(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
	# 	(scaler.*X)'*μ_var-	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))
	#
	# 	func2(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))
	#
	# 	opt1 = Adagrad()
	# 	opt2 = Adagrad()
	# 	oldval = func(μ_var, ltemp)
	# 	g1 = -dfunc(μ_var)
	# 	δ1 = update(opt1,g1)
	# 	g2 = -ForwardDiff.gradient(func2, ltemp)
	# 	δ2 = update(opt2,g2)
	# 	newval = func(μ_var-δ1, ltemp-δ2)
	# 	μ_var-=δ1
	# 	ltemp-=δ2
	#
	# 	g1 = -dfunc(μ_var)
	# 	δ1 = update(opt1,g1)
	# 	g2 = -ForwardDiff.gradient(func2, ltemp)
	# 	δ2 = update(opt2,g2)
	# 	μ_var-=δ1
	# 	ltemp-=δ2
	#
	# 	model.μ_var[a,:]=μ_var
	# 	model.Λ_var[a,:]=1.0./exp.(ltemp)
	# elseif meth == "isns2"

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

		dfunci(μ_var) = -model.l.*model.L*(μ_var-model.m) +X - sumb.*x

		funci(μ_var,ltemp) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp

		func1i(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

		func2i(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))

		# opt1 = RMSprop()
		# opt2 = RMSprop()
		opt1 = Adagrad(η=1.0)
		opt2 = Adagrad(η=1.0)
		for i in 1:10
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
	scaler=(convert(Float64,model.N)/convert(Float64,model.mbsize))
	model.m=inv(model.M)*(model.M0*model.m0+scaler*model.l).*model.L*s
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
	# train_links_num=convert(Float64, train_links_num)
	train_links_num=convert(Float64,nnz(model.network))
	scaler=(train_links_num/convert(Float64,length(mb.mblinks)))
	@assert !isequal(model.ϕlinoutsum[:], zeros(Float64, model.K))
	model.b0[:] = model.η0.+ (scaler.*model.ϕlinoutsum[:])
end
#updateb0!(model, mb)
function updateb1!(model::LNMMSB, mb::MiniBatch, meth::String)
	model.b1_old = deepcopy(model.b1)
	# train_nlinks_num = convert(Float64,train_nlinks_num)
	train_nlinks_num = convert(Float64,model.N*model.N-model.N-sum(model.train_outdeg))

	if meth == "link"
		scaler = model.N/model.mbsize
		# @assert !isequal(model.ϕnlinoutsum[:], zeros(Float64, model.K))
		# model.b1[:] = model.η1.+scaler.*model.ϕnlinoutsum[:]
		r = zeros(Float64, model.K)
		s1 = zeros(Float64, model.K)
		s2 = zeros(Float64, model.K)
		s3 = zeros(Float64, (model.N,model.K))
		s4 = zeros(Float64, model.K)
		for a in mb.mbnodes
			s1[:] += model.ϕbar[a,:]
			s2[:] += (model.ϕbar[a,:]).^2
			for b in model.train_sinks[a]
				if (Dyad(a,b) in model.ho_links)
					continue;
				else
					s3[a,:]+=model.ϕbar[b,:]
				end
			end
		end

		s1 = (scaler.*s1).^2
		for a in mb.mbnodes
			s4[:]+=model.ϕbar[a,:].*s3[a,:]
		end


		r = s1.-scaler.*s2.-scaler.*s4
		model.b1[:] = model.η1.+r
	elseif meth == "isns2"
		scaler= length(model.train_nonlinks)/(length(mb.mbnonlinks))
		model.b1[:] = model.η1 .+ (scaler.*model.ϕnlinoutsum[:])
	end
	# model.b1[model.b1 .< .0] = model.η1
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
