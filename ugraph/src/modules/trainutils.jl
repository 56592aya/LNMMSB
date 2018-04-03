__precompile__()
module TrainUtils

export updatephil!,updatephinl!,updatephinlout!,updatephinlin!,sfx,dfunci,updateμ!,updateM!,updatem!,updatel!
export updateL!,updateb0!,updateb1!,estimate_βs!,estimate_θs!,estimate_θs!,estimate_μs,update_Elogβ!
import Model: LNMMSB
import Utils: MiniBatch, Link, NonLink, logsumexp, softmax,digamma_, EPSILON
using GradDescent
using ForwardDiff


function updatephil!(model::LNMMSB, mb::MiniBatch, link::Link, check::String)
	a = link.src
	b = link.dst
	union_a = vcat(model.A[a], model.C[a])
	union_b = vcat(model.A[b], model.C[b])
	vala = !isempty(model.B[a])? model.μ_var[a, model.B[a][1]]: 0.0
	valb = !isempty(model.B[b])? model.μ_var[b, model.B[b][1]]: 0.0
	link.ϕ[:] = model.Elogβ0[:] + vala + valb
	for k in union_a
		link.ϕ[k] += model.μ_var[a,k] - vala
	end
	for k in union_b
		link.ϕ[k] += model.μ_var[b,k] - valb
	end
	r = logsumexp(link.ϕ)
	link.ϕ[:] = exp.(link.ϕ[:] .- r)[:]
end
# function updatephinl!(model::LNMMSB, mb::MiniBatch, nlink::NonLink)
#
# 	for k in 1:model.K
# 		nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*(model.Elogβ1[k]-log1p(-EPSILON))
# 		nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*(model.Elogβ1[k]-log1p(-EPSILON))
# 	end
# 	r = logsumexp(nlink.ϕout)
# 	nlink.ϕout[:] = exp.(nlink.ϕout[:] .- r)[:]
# 	r=logsumexp(nlink.ϕin)
# 	nlink.ϕin[:] = exp.(nlink.ϕin[:] .- r)[:]
# end
#I think this requires the two updates to be separated
function updatephinl!(model::LNMMSB, mb::MiniBatch, nlink::NonLink, check::String)
	if rand(Bool, 1)[1]
		updatephinlout!(model, mb, nlink, check)
		updatephinlin!(model, mb, nlink, check)
	else
		updatephinlin!(model, mb, nlink, check)
		updatephinlout!(model, mb, nlink, check)
	end
end


##
function updatephinlout!(model::LNMMSB, mb::MiniBatch, nlink::NonLink, check::String)
	a = nlink.src
	b = nlink.dst
	constant = (model.Elogβ1 .-log1p(-EPSILON))
	nlink.ϕout[:] = model.μ_var[a,:]
	union_b = vcat(model.A[b], model.C[b])
	for k in union_b
		nlink.ϕout[k] += nlink.ϕin[k]*constant[k]
	end
	r = logsumexp(nlink.ϕout)
	nlink.ϕout[:] = exp.(nlink.ϕout[:] .- r)[:]

end
function updatephinlin!(model::LNMMSB, mb::MiniBatch, nlink::NonLink, check::String)
	a = nlink.src
	b = nlink.dst
	constant = (model.Elogβ1 .-log1p(-EPSILON))
	nlink.ϕin[:] = model.μ_var[b,:]
	union_a = vcat(model.A[a], model.C[a])
	for k in union_a
		nlink.ϕin[k] += nlink.ϕout[k]*constant[k]
	end
	r=logsumexp(nlink.ϕin)
	nlink.ϕin[:] = exp.(nlink.ϕin[:] .- r)[:]
end
##

function sfx(μ_var::Vector{Float64},ltemp::Vector{Float64})
	return softmax(μ_var .+.5.*exp.(ltemp))
end
function sfx(μ_var::Vector{Float64})
	return softmax(μ_var)
end

function my_adagrad(f_grad, x0, data, args, stepsize=1e-2, fudge_factor=1e-6, max_it=1000, minibatchsize=nothing, minibatch_ratio=0.01)

end



function dfunci(μ_var::Vector{Float64}, model::LNMMSB, X::Vector{Float64}, x::Vector{Float64}, sumb::Float64)
	-model.l.*model.L*(μ_var-model.m) +X - sumb.*x
end
function updateμ!(model::LNMMSB, a::Int64,mb::MiniBatch, check::String, N::Int64)

	model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])

	μ_var = deepcopy(model.μ_var[a,:])
	x = sfx(μ_var)
	s1 = haskey(mb.mbnot,a)?convert(Float64,(N-1-model.train_deg[a])):0.0
	c1 = haskey(mb.mbnot,a)?convert(Float64, length(mb.mbnot[a])):1.0

	sumb =(convert(Float64,model.N)*-1.0)
	X=model.ϕlsum[a,:]+(s1/c1).*.5.*(model.ϕnloutsum[a,:]+model.ϕnlinsum[a,:])


	opt1 = Adagrad()

	for i in 1:10
		x  = sfx(μ_var)
		g1 = dfunci(μ_var, model, X, x, sumb)
		δ1 = update(opt1,g1)
		# @code_warntype update(opt1,g1)
		μ_var+=δ1
	end
	model.μ_var[a,:]=μ_var
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
function updateL!(model::LNMMSB, mb::MiniBatch, i::Int64)
	model.L_old = deepcopy(model.L)
	s = zero(Float64)
	# for a in mb.mbnodes
	# 	s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+diagm(1.0./model.Λ_var[a,:])
	# end
	#added this for instead
	for a in mb.mbnodes
		s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'
	end
	s=(model.N/model.mbsize)*s
	s+=inv(model.L0)+model.N.*inv(model.M)
	model.L = try
		inv(s)
	catch y
		if isa(y, Base.LinAlg.SingularException)
			println("i is ", i)
			error("hey hey hey singular")
		end
	end

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
function update_Elogβ!(model::LNMMSB)
	for k in 1:model.K
		model.Elogβ0[k] = digamma_(model.b0[k]) - (digamma_(model.b0[k])+digamma_(model.b1[k]))
		model.Elogβ1[k] = digamma_(model.b1[k]) - (digamma_(model.b0[k])+digamma_(model.b1[k]))
	end
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
end
