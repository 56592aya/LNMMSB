# using SpecialFunctions
using GradDescent
using ForwardDiff

# function updatephibar!(model::LNMMSB, mb::MiniBatch, a::Int64)
# 	model.ϕbar[a,:]= (model.ϕloutsum[a,:]+model.ϕlinsum[a,:])/(model.train_outdeg[a]+model.train_indeg[a])
# end
function updatephil!(model::LNMMSB, mb::MiniBatch, link::Link)
	for k in 1:model.K
		link.ϕ[k] = model.μ_var[link.src,k] + model.μ_var[link.dst,k]+model.Elogβ0[k]
	end
	r = logsumexp(link.ϕ)
	link.ϕ[:] = exp.(link.ϕ[:] .- r)[:]
end

function updatephil!(model::LNMMSB, mb::MiniBatch, link::Link, check::String)
	# activecandidateidxsrc = union(model.Active[link.src], model.Candidate[link.src])
	# activecandidateidxdst = union(model.Active[link.dst], model.Candidate[link.dst])
	activecandidateidx = union(model.Active[link.src], model.Candidate[link.src],model.Active[link.dst], model.Candidate[link.dst])

	for k in activecandidateidx
		link.ϕ[k] = model.μ_var[link.src,k] + model.μ_var[link.dst,k]+model.Elogβ0[k]
	end
	rest = setdiff(1:model.K,activecandidateidx)
	if !isempty(rest)
		link.ϕ[rest] .= model.μ_var[link.src,rest[1]] + model.μ_var[link.dst,rest[1]]+model.Elogβ0[rest[1]]
	end

	r = logsumexp(link.ϕ)
	link.ϕ[:] = exp.(link.ϕ[:] .- r)[:]
end
function updatephinl!(model::LNMMSB, mb::MiniBatch, nlink::NonLink)

	for k in 1:model.K
		nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*(model.Elogβ1[k]-log1p(-EPSILON))
		nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*(model.Elogβ1[k]-log1p(-EPSILON))
	end
	r = logsumexp(nlink.ϕout)
	nlink.ϕout[:] = exp.(nlink.ϕout[:] .- r)[:]
	r=logsumexp(nlink.ϕin)
	nlink.ϕin[:] = exp.(nlink.ϕin[:] .- r)[:]
end
function updatephinl!(model::LNMMSB, mb::MiniBatch, nlink::NonLink, check::String)
	activecandidateidx = union(model.Active[nlink.src], model.Active[nlink.dst], model.Candidate[nlink.src], model.Candidate[nlink.dst])
	for k in activecandidateidx
		nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*(model.Elogβ1[k]-log1p(-EPSILON))
		nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*(model.Elogβ1[k]-log1p(-EPSILON))

	end
	rest = setdiff(1:model.K,activecandidateidx)
	if !isempty(rest)
		nlink.ϕout[rest] .= model.μ_var[nlink.src,rest[1]] + nlink.ϕin[rest[1]]*(model.Elogβ1[rest[1]]-log1p(-EPSILON))
		nlink.ϕin[rest] .= model.μ_var[nlink.dst,rest[1]] + nlink.ϕout[rest[1]]*(model.Elogβ1[rest[1]]-log1p(-EPSILON))
	end
	r = logsumexp(nlink.ϕout)
	nlink.ϕout[:] = exp.(nlink.ϕout[:] .- r)[:]
	r=logsumexp(nlink.ϕin)
	nlink.ϕin[:] = exp.(nlink.ϕin[:] .- r)[:]
end

function sfx(μ_var::Vector{Float64},ltemp::Vector{Float64})
	return softmax(μ_var .+.5.*exp.(ltemp))
end
function updatesimulμΛ!(model::LNMMSB, a::Int64,mb::MiniBatch)
		model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])
		model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])
		μ_var = deepcopy(model.μ_var[a,:])
		Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
		ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
		x = sfx(μ_var,ltemp)
		s1 = haskey(mb.mbnot,a)?(N-1-model.train_deg[a]):0
		c1 = haskey(mb.mbnot,a)?length(mb.mbnot[a]):1

		sumb =(model.N-1)
		X=model.ϕlsum[a,:]+(s1/c1).*.5.*(model.ϕnloutsum[a,:]+model.ϕnlinsum[a,:])

		dfunci(μ_var) = -model.l.*model.L*(μ_var-model.m) +X - sumb.*x

		func1i(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

		func2i(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))

		# opt1 = RMSprop()
		# opt2 = RMSprop()
		opt1 = Adagrad()
		opt2 = Adagrad()

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
function updatesimulμΛ!(model::LNMMSB, a::Int64,mb::MiniBatch, check::String)
		activecandidateidx = union(model.Active[a], model.Candidate[a])
		rest = setdiff(1:model.K,activecandidateidx)
		model.μ_var_old[a,activecandidateidx]=deepcopy(model.μ_var[a,activecandidateidx])
		model.Λ_var_old[a,activecandidateidx]=deepcopy(model.Λ_var[a,activecandidateidx])
		model.μ_var_old[a,rest].=deepcopy(model.μ_var[a,rest[1]])
		model.Λ_var_old[a,rest].=deepcopy(model.Λ_var[a,rest[1]])
		μ_var = deepcopy(model.μ_var[a,:])
		Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
		ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
		x = sfx(μ_var,ltemp)
		s1 = haskey(mb.mbnot,a)?(N-1-model.train_deg[a]):0
		c1 = haskey(mb.mbnot,a)?length(mb.mbnot[a]):1

		sumb =(model.N-1)
		X=model.ϕlsum[a,:]+(s1/c1).*.5.*(model.ϕnloutsum[a,:]+model.ϕnlinsum[a,:])

		dfunci(μ_var) = -model.l.*model.L*(μ_var-model.m) +X - sumb.*x

		func1i(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
		(X)'*μ_var-	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

		func2i(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))

		# opt1 = RMSprop()
		# opt2 = RMSprop()
		opt1 = Adagrad()
		opt2 = Adagrad()

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

	scaler = model.N/model.mbsize
	model.m=inv(model.M)*(model.M0*model.m0+(scaler*model.l).*model.L*s)
end
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
	s+=inv(model.L0)+model.N.*inv(model.M)
	model.L = inv(s)
end

function updateb0!(model::LNMMSB, mb::MiniBatch)
	model.b0_old = deepcopy(model.b0)
	scaler=nnz(model.network)/(2*length(mb.mblinks))
	# @assert !isequal(model.ϕlinoutsum[:], zeros(Float64, model.K))
	s = zeros(Float64, model.K)
	for a in model.mbids
		s[:] += .5*model.ϕlsum[a,:]
	end
	model.b0[:] = model.η0.+ (scaler.*s)

end


#updateb0!(model, mb)
function updateb1!(model::LNMMSB, mb::MiniBatch)
	model.b1_old = deepcopy(model.b1)
	scaler = (model.N^2 - model.N - nnz(model.network))/(2*length(mb.mbnonlinks))
	# scaler = (model.N/model.mbsize)*10
	for k in 1:model.K
		model.b1[k] = model.η1 + (.5*scaler*model.ϕnlinoutsum[k])
	end
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
function estimate_θs!(model::LNMMSB)
	for a in 1:model.N
		# model.est_θ[a,:]=exp(model.μ_var[a,:])./sum(exp(model.μ_var[a,:]))#
		model.est_θ[a,:]=softmax(model.μ_var[a,:])
	end
	model.est_θ[1:model.N,:]
end

function estimate_μs(model::LNMMSB, mb::MiniBatch)
	model.est_μ = reshape(reduce(mean, model.μ_var, 1)',model.K)
end

function estimate_Λs(model::LNMMSB, mb::MiniBatch)
end

##How to determine belonging to a community based on the membership vector???
# function computeNMI(x::Matrix{Float64}, y::Matrix{Float64}, communities::Dict{Int64, Vector{Int64}}, threshold::Float64)
# 	open("./file2", "w") do f
# 	  for k in 1:size(x,2)
# 	    for i in 1:size(x,1)
# 	      if x[i,k] > threshold #3.0/size(x,2)
# 	        write(f, "$i ")
# 	      end
# 	    end
# 	    write(f, "\n")
# 	  end
# 	end
# 	open("./file1", "w") do f
# 	  for k in 1:size(y,2)
# 	    for i in 1:size(y,1)
# 	      if y[i,k] > threshold #3.0/size(y,2)
# 	        write(f, "$i ")
# 	      end
# 	    end
# 	    write(f, "\n")
# 	  end
# 	end
#
# 	open("./file3", "w") do f
# 	  for k in 1:length(communities)
# 	  	if length(communities[k]) <= 10
# 			continue;
# 		end
# 	    for e in communities[k]
#         	write(f, "$e ")
# 	    end
# 	    write(f, "\n")
# 	  end
# 	end
# 	println("NMI of estimated vs truth")
# 	run(`src/cpp/NMI/onmi file2 file1`)
# 	nmi = read(`src/cpp/NMI/onmi file2 file1`, String)
# 	nmi=parse(Float64,nmi[6:(end-1)])
# 	println("NMI of estimated vs init")
# 	run(`src/cpp/NMI/onmi file2 file3`)
# 	println("NMI of truth vs init")
# 	run(`src/cpp/NMI/onmi file1 file3`)
# 	return nmi
# end
# function computeNMI_med(x::Matrix{Float64}, y::Matrix{Float64}, communities::Dict{Int64, Vector{Int64}}, threshold::Float64)
# 	open("./file2", "w") do f
# 	  for k in 1:size(x,2)
# 	    for i in 1:size(x,1)
# 	      if x[i,k] > threshold #3.0/size(x,2)
# 	        write(f, "$i ")
# 	      end
# 	    end
# 	    write(f, "\n")
# 	  end
# 	end
# 	open("./file1", "w") do f
# 	  for k in 1:size(y,2)
# 	    for i in 1:size(y,1)
# 	      if y[i,k] > threshold #3.0/size(y,2)
# 	        write(f, "$i ")
# 	      end
# 	    end
# 	    write(f, "\n")
# 	  end
# 	end
# 	nmi = read(`src/cpp/NMI/onmi file2 file1`, String)
# 	nmi=parse(Float64,nmi[6:(end-1)])
# 	return nmi
# end
#
# function log_comm(model::LNMMSB, mb::MiniBatch, link::Link, link_thresh::Float64, min_deg::Int64)
# 	m_out = indmax(link.ϕout)
# 	if link.ϕout[m_out] > link_thresh
# 		model.fmap[link.src,m_out]+=1
# 		model.fmap[link.dst,m_out]+=1
# 		if model.fmap[link.src,m_out] > minimum(model.train_outdeg)
# 			push!(model.comm[m_out], link.src)
# 		end
#
# 		if model.fmap[link.src,m_out] >minimum(model.train_outdeg)
# 			push!(model.comm[m_out], link.dst)
# 		end
# 	end
# 	m_in = indmax(link.ϕin)
# 	if link.ϕin[m_in] > link_thresh
# 		model.fmap[link.src,m_in]+=1
# 		model.fmap[link.dst,m_in]+=1
# 		if model.fmap[link.dst,m_in] > minimum(model.train_indeg)
# 			push!(model.comm[m_in], link.src)
# 		end
#
# 		if model.fmap[link.dst,m_in] >minimum(model.train_indeg)
# 			push!(model.comm[m_in], link.dst)
# 		end
# 	end
# end
# function compute_NMI3(model::LNMMSB)
#
# 	open("./data/est_comm.txt", "w") do f
# 	  for k in 1:length(model.comm)
# 	    for n in model.comm[k]
# 	        write(f, "$n ")
# 	    end
# 	    write(f, "\n")
# 	  end
# 	end
#
# 	println("NMI of estimated vs truth")
# 	# run(`src/cpp/NMI/onmi file2 file1`)
# 	nmi = read(`src/cpp/NMI/onmi data/truth_comm.txt data/est_comm.txt`, String)
# 	nmi=parse(Float64,nmi[6:(end-1)])
# 	return nmi
# end
# function edge_likelihood(model::LNMMSB,pair::Dyad, β_est::Vector{Float64})
#     s = zero(Float64)
#     S = Float64
#     prob = zero(Float64)
# 	src = pair.src
# 	dst = pair.dst
# 	sfxa=softmax(model.μ_var[pair.src,:])
# 	sfxb=softmax(model.μ_var[pair.dst,:])
#     for k in 1:model.K
#
#         if isalink(model, "network",src, dst)
#             prob += (sfxa[k]/sum(sfxa))*(sfxb[k]/sum(sfxb))*(β_est[k])
#         else
#             prob += (sfxa[k]/sum(sfxa))*(sfxb[k]/sum(sfxb))*(1.0-β_est[k])
#         end
#         s += (sfxa[k]/sum(sfxa))*(sfxb[k]/sum(sfxb))
#     end
#
#     if isalink(model, "network",src, dst)
#         prob += (1.0-s)*EPSILON
#     else
#         prob += (1.0-s)*(1.0-EPSILON)
#     end
#     return log1p(-1.0+prob)::Float64
# end
print("");
