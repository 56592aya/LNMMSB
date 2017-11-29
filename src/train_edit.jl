using Plots
using GraphPlot
##initialization for the check is very important, havent yet figured it out.
# function train!(model::LNMMSB; iter::Int64=150, etol::Float64=1, niter::Int64=1000, ntol::Float64=1.0/(model.K^2), viter::Int64=10, vtol::Float64=1.0/(model.K^2), elboevery::Int64=10, mb::MiniBatch,lr::Float64)
	# preparedata(model)
	# nmitemp = Float64[]
	model=LNMMSB(network, onlyK)
	model.nho=0
	mb=deepcopy(model.mb_zeroer)
	include("modelutils.jl")
	preparedata2!(model)
	minibatch_set_srns(model)
	include("trainutils.jl")
	iter=5000
	# train_links_num=0
	# train_nlinks_num = 0
	# link_ratio = 0.0
	# prev_ll = -1000000
	# store_ll = Array{Float64, 1}()
	# store_nmi = Array{Float64, 1}()
	# first_converge = false
	early=true
	switchrounds=true
	elboevery=20
	# model.elborecord = Vector{Float64}()
	# model.elbo = 0.0
	# model.oldelbo= -Inf
	links_num=nnz(model.network)
	nonlinks_num = N^2 - N - links_num
	link_ratio = links_num/(nonlinks_num+links_num)
	# train_nlinks_num=model.N*model.N-model.N- length(model.ho_dyads)-train_links_num
	link_tuner = log(link_ratio)
	nlink_tuner = log(1-link_ratio)
	# link_ratio = convert(Float64, train_links_num)/convert(Float64,train_nlinks_num)
	# dep2 = .1*(train_links_num)/(train_links_num+train_nlinks_num)
	_init_ϕ = deepcopy(ones(Float64, model.K).*1.0/model.K)
	# link_thresh=.9
	# min_deg = minimum(model.train_outdeg)
	true_θs = (readdlm("data/true_thetas.txt"))
	init_mu(model,communities,model.K)##from Gopalan
	_init_μ = deepcopy(model.μ_var)
	model.Λ_var = 100.0*ones(Float64, (model.N, model.K))
	_init_Λ = deepcopy(model.Λ_var)
	function update_Elogβ!(model::LNMMSB)
		for k in 1:model.K
			model.Elogβ0[k] = digamma_(model.b0[k]) - (digamma_(model.b0[k])+digamma_(model.b1[k]))
			model.Elogβ1[k] = digamma_(model.b1[k]) - (digamma_(model.b0[k])+digamma_(model.b1[k]))
		end
	end
	update_Elogβ!(model)
	updatel!(model)
	lr_M = 1.0
	lr_m = 1.0
	lr_L = 1.0
	lr_b = 1.0
	true_μ = readdlm("data/true_mu.txt")
	model.m = zeros(Float64,model.K)
	model.M = (100.0).*eye(Float64,model.K)
	model.l = model.K+2
	Ltemp = zeros(Float64, model.K, model.K)
  	for i in 1:model.N
    	Ltemp .+= rand(Wishart(model.K+1,0.001*diagm(ones(Float64, model.K))))
  	end
	Ltemp ./= model.N
	model.L=100.0.*(Ltemp)./model.l
	isfullsample=false
	if model.mbsize == model.N
		isfullsample = true
	end
	lr_μ=ones(Float64, model.N)
	lr_Λ=ones(Float64, model.N)
	count_μ = zeros(Int64, model.N)
	count_Λ = zeros(Int64, model.N)
	count_a = zeros(Int64, model.N)
	# model.fmap = zeros(Float64, (model.N,model.K))
	# model.comm = [Int64[] for i in 1:model.K]
	# if isfile("./data/est_comm.txt")
	# 	rm("./data/est_comm.txt")
	# end

	model.μ_var=deepcopy(_init_μ)
	ϕlinoutsum_old = zeros(Float64, model.K)
	ϕnlinoutsum_old = zeros(Float64, model.K)
	for i in 1:iter

		mb=deepcopy(model.mb_zeroer)
		# mbsampling!(mb, model, meth, model.mbsize)
		mbtype="nonlink"
		mbset, scale =minibatch_set_srns(model)
		d = collect(mbset)[1]
		if isalink(model, "network", d.src, d.dst) ##is a link minibatch
			mbtype = "link"
		end
		if mbtype=="link"
			for d in collect(mbset)
				push!(mb.mblinks, Link(d.src, d.dst, _init_ϕ, _init_ϕ))
				if !(d.src in mb.mbnodes)
					push!(mb.mbnodes, d.src)
				end
				if !(d.dst in mb.mbnodes)
					push!(mb.mbnodes, d.dst)
				end
			end
		else
			for d in collect(mbset)
				push!(mb.mbnonlinks, NonLink(d.src, d.dst, _init_ϕ, _init_ϕ))
				if !haskey(mb.mbfnadj,d.src)
					mb.mbfnadj[d.src] = get(mb.mbfnadj,d.src, Vector{Int64}())
				end
				push!(mb.mbfnadj[d.src], d.dst)
				if !haskey(mb.mbbnadj,d.dst)
					mb.mbbnadj[d.dst] = get(mb.mbbnadj,d.dst, Vector{Int64}())
				end
				push!(mb.mbbnadj[d.dst], d.src)
				if !(d.src in mb.mbnodes)
					push!(mb.mbnodes, d.src)
				end
				if !(d.dst in mb.mbnodes)
					push!(mb.mbnodes, d.dst)
				end
			end
		end



		# model.fmap = zeros(Float64, (model.N,model.K))
		lr_M = 1.0
		lr_m = 1.0
		lr_L = 1.0
		lr_b = 1.0
		if !isfullsample
			lr_M = (1024+(i-1))^-.9
			lr_m = (1024+(i-1))^-.9
			lr_L = (1024+(i-1))^-.9
			lr_b = (1024+(i-1))^-.9
		end
		if i > model.N/model.mbsize
			early = false
		end


		switchrounds = bitrand(1)[1]
		model.ϕlinoutsum[:] = zeros(Float64, model.K)
		model.ϕloutsum = zeros(Float64, (model.N,model.K))
		model.ϕlinsum = zeros(Float64, (model.N,model.K))

		model.μ_var_old[mb.mbnodes,:]=deepcopy(model.μ_var[mb.mbnodes,:])
	#	model.μ_var[model.mbids,:]=zeros(Float64,(model.mbsize,model.K))
		for l in mb.mblinks
			for j in 1:15
				# while !_converged##decide what you mean
				if switchrounds
					updatephilout!(model, mb, early,l,link_tuner)
					updatephilin!(model, mb, early,l,link_tuner)
					updatephilout!(model, mb, early,l,link_tuner)
					switchrounds = !switchrounds
				else
					updatephilin!(model, mb, early,l,link_tuner)
					updatephilout!(model, mb, early,l,link_tuner)
					updatephilin!(model, mb, early,l,link_tuner)
					switchrounds = !switchrounds
				end
				# end
			end

			for k in 1:model.K
				model.ϕloutsum[l.src,k] += l.ϕout[k]
				model.ϕlinsum[l.dst,k] += l.ϕin[k]
				model.ϕlinoutsum[k] += l.ϕout[k]*l.ϕin[k]
				ϕlinoutsum_old[k] = deepcopy(model.ϕlinoutsum[k])
			end
			# if i>900
			# log_comm(model, mb, l, link_thresh, min_deg)
			# end
		end
		# if i>900
		# for k in 1:model.K
		# 	model.comm[k] = unique(model.comm[k])
		# end
		# end

		switchrounds = bitrand(1)[1]
		model.ϕnlinoutsum[:] = zeros(Float64, model.K)
		model.ϕnloutsum = zeros(Float64, (model.N,model.K))
		model.ϕnlinsum = zeros(Float64, (model.N,model.K))
		for nl in mb.mbnonlinks
			for j in 1:15
				# while !_converged##decide what you mean
				if switchrounds
					updatephinlout!(model, mb, early,nl,nlink_tuner)
					updatephinlin!(model, mb, early,nl,nlink_tuner)
					updatephinlout!(model, mb, early,nl,nlink_tuner)
					switchrounds = !switchrounds
				else
					updatephinlin!(model, mb, early,nl,nlink_tuner)
					updatephinlout!(model, mb, early,nl,nlink_tuner)
					updatephinlin!(model, mb, early,nl,nlink_tuner)
					switchrounds = !switchrounds
				end
			end
			for k in 1:model.K

				model.ϕnloutsum[nl.src,k] += nl.ϕout[k]
				model.ϕnlinsum[nl.dst,k] += nl.ϕin[k]
				model.ϕnlinoutsum[k] += nl.ϕout[k]*nl.ϕin[k]
				ϕnlinoutsum_old[k] = deepcopy(model.ϕnlinoutsum[k])
				# nlout = zeros(Float64, (length(mb.mbnonlinks), model.K))
				# nlin = zeros(Float64, (length(mb.mbnonlinks), model.K))
				# lout = zeros(Float64, (length(mb.mblinks), model.K))
				# lin = zeros(Float64, (length(mb.mblinks), model.K))
				# for (i,nl) in enumerate(mb.mbnonlinks)
				# 	nlout[i,:] = nl.ϕout[:]
				# 	nlin[i,:] = nl.ϕin[:]
				# end
				# for (i,l) in enumerate(mb.mblinks)
				# 	lout[i,:] = l.ϕout[:]
				# 	lin[i,:] = l.ϕin[:]
				# end
			end

		end
		num_node_sample = div(model.N,50)
		neighborset = Set{Int64}()
		for node in mb.mbnodes
			neighborset = sample_neighbors(neighborset, num_node_sample, node)
			for n in collect(neighborset)
				if !(n in mb.mbnodes)
					push!(mb.mbnodes, n)
				end
			end
		end
		model.μ_var[mb.mbnodes,:]=deepcopy(model.μ_var_old[mb.mbnodes,:])## I commented
		model.μ_var[mb.mbnodes,:]=deepcopy(_init_μ[mb.mbnodes,:])##I added instead of the above)
		for a in mb.mbnodes
			if !isfullsample
				# count_μ[a]+=1
				# count_Λ[a]+=1
				count_a[a] += 1
				updatesimulμΛ2!(model, a, mb,scale*num_node_sample)
				# lr_μ[a] = (1024.0+Float64(count_μ[a]-1.0))^(-.5)
				# lr_Λ[a] = (1024.0+Float64(count_Λ[a]-1.0))^(-.5)
				# model.μ_var[a,:] = model.μ_var_old[a,:].*(1.0.-lr_μ[a])+lr_μ[a].*model.μ_var[a,:]
				# model.Λ_var[a,:] = model.Λ_var_old[a,:].*(1.0.-lr_Λ[a])+lr_Λ[a].*model.Λ_var[a,:]
			else
				updatesimulμΛ!(model, a, mb,meth)
			end
		end

		updatemtemp!(model, mb,scale)
		model.m = model.m_old.*(1.0-lr_m)+lr_m.*model.m

		updateM!(model, mb)
		model.M = model.M_old.*(1.0-lr_M)+lr_M.*model.M

		updateLtemp!(model, mb,scale)
		model.L = model.L_old.*(1.0-lr_L)+lr_L*model.L

		updateb0temp!(model, mb,scale, ϕlinoutsum_old)
		model.b0 = (1.0-lr_b).*model.b0_old + lr_b.*((model.b0))
		updateb1temp!(model,mb,scale,ϕnlinoutsum_old)
		model.b1 = (1.0-lr_b).*model.b1_old+lr_b.*((model.b1))
		update_Elogβ!(model)


		estimate_θs!(model, mb)

		checkelbo = (i % elboevery == 0)
		if checkelbo || i == 1
			print(i);print(": ")
			estimate_βs!(model,mb)
			println(model.est_β)
		end

		# if (i % 100 == 0)
		# 	println(compute_NMI3(model))
		# 	push!(nmitemp,compute_NMI3(model))
		# end
		x = deepcopy(model.est_θ)
		sort_by_argmax!(x)
		table=[sortperm(x[i,:]) for i in 1:model.N]
		count = zeros(Int64, model.K)
		diffK = model.K-inputtomodelgen[2]
		for k in 1:model.K
			for i in 1:model.N
				for j in 1:diffK
					if table[i][j] == k
						count[k] +=1
					end
				end
			end
		end
		idx=sortperm(count,rev=true)[(diffK+1):end]
		x = x[:,sort(idx)]
	end
	x = deepcopy(model.est_θ)
	sort_by_argmax!(x)
	table=[sortperm(x[i,:]) for i in 1:model.N]
	count = zeros(Int64, model.K)
	diffK = model.K-inputtomodelgen[2]
	for k in 1:model.K
		for i in 1:model.N
			for j in 1:diffK
				if table[i][j] == k
					count[k] +=1
				end
			end
		end
	end
	idx=sortperm(count,rev=true)[(diffK+1):end]
	x = x[:,sort(idx)]
	p2=Plots.heatmap(x, yflip=true)
	p3=Plots.heatmap(true_θs, yflip=true)
	Plots.plot(p2,p3, layout=(2,1))
	est = deepcopy(model.est_θ)
	sort_by_argmax!(est)
	# Plots.plot(1:length(nmitemp), nmitemp)
	# println(maximum(nmitemp))
	Plots.plot(1:length(vec(true_θs)),sort(vec(true_θs)))
	Plots.plot(1:length(vec(est)),sort(vec(est)))
	Plots.heatmap(est, yflip=true)
# end
