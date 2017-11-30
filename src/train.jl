using Plots
using GraphPlot
##initialization for the check is very important, havent yet figured it out.
# function train!(model::LNMMSB; iter::Int64=150, etol::Float64=1, niter::Int64=1000, ntol::Float64=1.0/(model.K^2), viter::Int64=10, vtol::Float64=1.0/(model.K^2), elboevery::Int64=10, mb::MiniBatch,lr::Float64)
	# preparedata(model)
	nmitemp = Float64[]
	iter=50000
	train_links_num=0
	train_nlinks_num = 0
	link_ratio = 0.0
	prev_ll = -1000000
	store_ll = Array{Float64, 1}()
	store_nmi = Array{Float64, 1}()
	first_converge = false
	early=true
	switchrounds=true
	elboevery=20
	model.elborecord = Vector{Float64}()
	model.elbo = 0.0
	model.oldelbo= -Inf
	train_links_num=sum(model.train_outdeg)
	train_nlinks_num=model.N*model.N-model.N- length(model.ho_dyads)-train_links_num
	link_tuner = log((train_nlinks_num+train_links_num)/train_nlinks_num)
	nlink_tuner = log(train_links_num/(train_links_num+train_nlinks_num))
	link_ratio = convert(Float64, train_links_num)/convert(Float64,train_nlinks_num)
	dep2 = .1*(train_links_num)/(train_links_num+train_nlinks_num)
	_init_ϕ = deepcopy(ones(Float64, model.K).*1.0/model.K)
	link_thresh=.9
	min_deg = minimum(model.train_outdeg)
	true_θs = (readdlm("data/true_thetas.txt"))

	init_mu(model,communities,model.K)##from Gopalan
	_init_μ = deepcopy(model.μ_var)

	model.μ_var = deepcopy(_init_μ)
	# _init_μ = log.((2*model.N/model.K).*ones(Float64, (model.N, model.K)))
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
	model.m = deepcopy(zeros(Float64,model.K))
	model.M = deepcopy((100.0).*eye(Float64,model.K))
	model.l = model.K+2
	Ltemp = zeros(Float64, model.K, model.K)
  	for i in 1:model.N
    	Ltemp .+= rand(Wishart(model.K+1,0.001*diagm(ones(Float64, model.K))))
  	end
	Ltemp ./= model.N
	model.L=deepcopy(100.0.*(Ltemp)./model.l)
	isfullsample=false
	if model.mbsize == model.N
		isfullsample = true
	end
	lr_μ=ones(Float64, model.N)
	lr_Λ=ones(Float64, model.N)
	count_μ = zeros(Int64, model.N)
	count_Λ = zeros(Int64, model.N)
	count_a = zeros(Int64, model.N)
	model.fmap = zeros(Float64, (model.N,model.K))
	model.comm = [Int64[] for i in 1:model.K]
	if isfile("./data/est_comm.txt")
		rm("./data/est_comm.txt")
	end

	model.μ_var=deepcopy(_init_μ)
	# model.μ_var = log.(true_θs)
	do_linked_edges!(model)
	rprogL = zeros(Float64, (iter,model.K))
	rprogm = zeros(Float64, (iter,model.K))
	rprogb0 = zeros(Float64, (iter,model.K))
	rprogb1 = zeros(Float64, (iter,model.K))
	num_past_iter = 20
	alpha = .4
	kappa=1.2

	flag=false
	for i in 1:iter
		mb=deepcopy(model.mb_zeroer)
		dyads = deepcopy(mbsampling!(mb, model, meth, model.mbsize))
		shuffle!(dyads)
		# mbsamplinglink!(mb, model, meth, model.mbsize)
		model.fmap = deepcopy(zeros(Float64, (model.N,model.K)))
		lr_M = 1.0
		lr_m = 1.0
		lr_L = 1.0
		lr_b = 1.0
		if !isfullsample
			# lr_M = (1024+((i/1)-1))^-.9

			# Plots.plot(1:iter, [(1024./(1024.0+Float64(i-1.0)))^(.9) for i in 1:iter])
			# Plots.plot(1:300, [(1024+((i/1)-1))^-.5 for i in 1:300])
			lr_m = ((model.N/(model.mbsize))/((model.N/(model.mbsize))+Float64(i-1.0)))^(.5)
			lr_L = ((model.N/(model.mbsize))/((model.N/(model.mbsize))+Float64(i-1.0)))^(.5)
			lr_b = ((model.N/(model.mbsize))/((model.N/(model.mbsize))+Float64(i-1.0)))^(.5)
		end
		if i > 2*model.N/model.mbsize
			early = false
		end



		switchrounds = bitrand(1)[1]
		model.ϕlinoutsum = deepcopy(zeros(Float64, model.K))
		model.ϕloutsum = deepcopy(zeros(Float64, (model.N,model.K)))
		model.ϕlinsum = deepcopy(zeros(Float64, (model.N,model.K)))
		model.μ_var_old = deepcopy(model.μ_var)
		model.μ_var_old[model.mbids,:]=deepcopy(model.μ_var[model.mbids,:])
		model.ϕnlinoutsum = deepcopy(zeros(Float64, model.K))
		model.ϕnloutsum = deepcopy(zeros(Float64, (model.N,model.K)))
		model.ϕnlinsum = deepcopy(zeros(Float64, (model.N,model.K)))
		one_over_K = deepcopy(ones(Float64,model.K)./model.K)
		for d in dyads
			if d in mb.mblinks
				l=deepcopy(mb.mblinks[mb.mblinks .== d][1])
				# l.ϕout = deepcopy(model.est_θ[l.src,:])
				l.ϕout = deepcopy(one_over_K)
				# l.ϕin = model.est_θ[l.dst,:]
				l.ϕin = deepcopy(one_over_K)
				for j in 1:20
					updatephilout!(model, mb, early,l,link_tuner)
					updatephilin!(model, mb, early,l,link_tuner)
				end
				for k in 1:model.K
					model.ϕloutsum[l.src,k] += l.ϕout[k]
					model.ϕlinsum[l.dst,k] += l.ϕin[k]
					model.ϕlinoutsum[k] += l.ϕout[k]*l.ϕin[k]
				end
				# if i>900

				log_comm(model, mb, l, link_thresh, min_deg)
			else
				nl=deepcopy(mb.mbnonlinks[mb.mbnonlinks .== d][1])
				# nl.ϕout = model.est_θ[nl.src,:]
				nl.ϕout = deepcopy(one_over_K)
				# nl.ϕin = model.est_θ[nl.dst,:]
				nl.ϕin = deepcopy(one_over_K)
				for j in 1:20
					updatephinlout!(model, mb, early,nl,nlink_tuner)
					updatephinlin!(model, mb, early,nl,nlink_tuner)
				end
				for k in 1:model.K
					model.ϕnloutsum[nl.src,k] += nl.ϕout[k]
					model.ϕnlinsum[nl.dst,k] += nl.ϕin[k]
					model.ϕnlinoutsum[k] += nl.ϕout[k]*nl.ϕin[k]
				end
			end
		end
		for k in 1:model.K
			model.comm[k] = unique(model.comm[k])
		end
		# model.ϕbar = zeros(Float64, (model.N, model.K))
		# for a in mb.mbnodes
		# 	updatephibar!(model, mb, a)
		# end
		# model.μ_var[model.mbids,:]=deepcopy(model.μ_var_old[model.mbids,:])## I commented
		model.μ_var[model.mbids,:]=deepcopy(_init_μ[model.mbids,:])##I added instead of the above)
		for a in mb.mbnodes
			if !isfullsample
				# count_μ[a]+=1
				# count_Λ[a]+=1
				count_a[a] += 1
				updatesimulμΛ!(model, a, mb,meth)
				# updatesimulμΛlink!(model, a, mb,meth)
				# if early
				# 	model.μ_var[a,:] = log.((length(mb.mblinks)./model.ϕlinoutsum).*softmax(model.μ_var[a,:]))
				# end
				lr_μ[a] = ((iter/(model.N/model.mbsize))/((iter/(model.N/model.mbsize))+count_a[a]-1))^.5
				lr_Λ[a] = ((iter/(model.N/model.mbsize))/((iter/(model.N/model.mbsize))+count_a[a]-1))^.5
				# lr_Λ[a] = (1024./(1024.0+Float64(count_Λ[a]-1.0)))^(-.5)
				model.μ_var[a,:] = model.μ_var_old[a,:].*(1.0.-lr_μ[a])+lr_μ[a].*model.μ_var[a,:]
				model.Λ_var[a,:] = model.Λ_var_old[a,:].*(1.0.-lr_Λ[a])+lr_Λ[a].*model.Λ_var[a,:]
			else
				updatesimulμΛ!(model, a, mb,meth)
			end
		end

		# if i % 100 == 0

		updatem!(model, mb)
		model.m = model.m_old.*(1.0-lr_m)+lr_m.*model.m
		# model.m = model.m_old.*(1.0-alpha)+alpha.*model.m
		# if isempty(model.m_hist) || length(model.m_hist) < num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 	push!(model.m_hist, model.m)
		# 	model.m_hist = circshift(model.m_hist,1)
		# elseif length(model.m_hist) == num_past_iter &&(length(count_a[count_a.>3]) == model.N)
		# 	pop!(model.m_hist)
		# 	push!(model.m_hist, model.m)
		# 	model.m_hist = circshift(model.m_hist,1)
		# end
		updateM!(model, mb)
		model.M = model.M_old.*(1.0-lr_M)+lr_M.*model.M

		updateL!(model, mb)
		model.L = model.L_old.*(1.0-lr_L)+lr_L*model.L
		# model.L = model.L_old.*(1.0-alpha)+alpha*model.L
		# if isempty(model.L_hist) || length(model.L_hist) < num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 	push!(model.L_hist, model.L)
		# 	model.L_hist = circshift(model.L_hist,1)
		# elseif length(model.L_hist) == num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 	pop!(model.L_hist)
		# 	push!(model.L_hist, model.L)
		# 	model.L_hist = circshift(model.L_hist,1)
		# end
		updateb0!(model, mb)
		model.b0 = (1.0-lr_b).*model.b0_old + lr_b.*((model.b0))
		# model.b0 = (1.0-alpha).*model.b0_old + alpha.*((model.b0))
		# if isempty(model.b0_hist) || length(model.b0_hist) < num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 	push!(model.b0_hist, model.b0)
		# 	model.b0_hist = circshift(model.b0_hist,1)
		# elseif length(model.b0_hist) == num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 	pop!(model.b0_hist)
		# 	push!(model.b0_hist, model.b0)
		# 	model.b0_hist = circshift(model.b0_hist,1)
		# end
		updateb1!(model,mb,meth)
		# updateb1link!(model,mb,meth)
		model.b1 = (1.0-lr_b).*model.b1_old+lr_b.*((model.b1))
		# model.b1 = (1.0-alpha).*model.b1_old+alpha.*((model.b1))
		# if isempty(model.b1_hist) || length(model.b1_hist) < num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 	push!(model.b1_hist, model.b1)
		# 	model.b1_hist = circshift(model.b1_hist,1)
		# elseif length(model.b1_hist) == num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 	pop!(model.b1_hist)
		# 	push!(model.b1_hist, model.b1)
		# 	model.b1_hist = circshift(model.b1_hist,1)
		# end
		update_Elogβ!(model)

		# end
		# for k in 1:model.K
		# 	if i >= 5 && i < num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 		rprogL[i,k] = abs(model.L_hist[1][k,k] - model.L_hist[end][k,k])/sum([abs(model.L_hist[j][k,k] - model.L_hist[j+1][k,k]) for j in 1:(length(model.L_hist)-1)])
		# 		rprogm[i,k] = abs(model.m_hist[1][k] - model.m_hist[end][k])/sum([abs(model.m_hist[j][k] - model.m_hist[j+1][k]) for j in 1:(length(model.m_hist)-1)])
		# 		rprogb0[i,k] = abs(model.b0_hist[1][k] - model.b0_hist[end][k])/sum([abs(model.b0_hist[j][k] - model.b0_hist[j+1][k]) for j in 1:(length(model.b0_hist)-1)])
		# 		rprogb1[i,k] =abs(model.b1_hist[1][k] - model.b1_hist[end][k])/sum([abs(model.b1_hist[j][k] - model.b1_hist[j+1][k]) for j in 1:(length(model.b1_hist)-1)])
		# 	elseif  i >= num_past_iter && (length(count_a[count_a.>3]) == model.N)
		# 		rprogL[i,k] = abs(model.L_hist[1][k,k] - model.L_hist[end][k,k])/sum([abs(model.L_hist[j][k,k] - model.L_hist[j+1][k,k]) for j in 1:(length(model.L_hist)-1)])
		# 		rprogm[i,k] = abs(model.m_hist[1][k] - model.m_hist[end][k])/sum([abs(model.m_hist[j][k] - model.m_hist[j+1][k]) for j in 1:(length(model.m_hist)-1)])
		# 		rprogb0[i,k] = abs(model.b0_hist[1][k] - model.b0_hist[end][k])/sum([abs(model.b0_hist[j][k] - model.b0_hist[j+1][k]) for j in 1:(length(model.b0_hist)-1)])
		# 		rprogb1[i,k] =abs(model.b1_hist[1][k] - model.b1_hist[end][k])/sum([abs(model.b1_hist[j][k] - model.b1_hist[j+1][k]) for j in 1:(length(model.b1_hist)-1)])
		# 	end
		# end
		# if (minimum(vcat(rprogL[i,:],rprogm[i,:],rprogb0[i,:],rprogb1[i,:]))) < alpha && (length(count_a[count_a.>3]) == model.N) &&
		# 	!isapprox(minimum(vcat(rprogL[i,:],rprogm[i,:],rprogb0[i,:],rprogb1[i,:])), 0.0)
		# 	model.mbsize = minimum([ceil(Int64, kappa*model.mbsize), round(Int64,1.0*model.N)])
		# 	if model.mbsize == round(Int64,1.0*model.N) && !flag
		# 		flag=true
		# 		i=1
		# 		count_a .= 0
		# 	end
		# 	if model.mbsize != round(Int64,1.0*model.N) && !flag
		# 		i = 1
		# 		count_a .= 0
		# 	end
		# end
		# alpha = .4 + .6*((model.mbsize - minibatchsize)/(model.N-minibatchsize))
		estimate_θs!(model, mb)

		kindexes = Int64[]
		if i % (model.N/2) ==0
			kindexes = prune!(model, mb)
			if kindexes == nothing
				continue;
			else
				_init_ϕ = deepcopy(ones(Float64, model.K).*1.0/model.K)
				_init_μ = _init_μ[:,kindexes]
				model.Λ_var = 100.0*ones(Float64, (model.N, model.K))
				_init_Λ = deepcopy(model.Λ_var)
				model.m = zeros(Float64,model.K)
				model.M = (100.0).*eye(Float64,model.K)
				model.l = model.K+2
				Ltemp = zeros(Float64, model.K, model.K)
			  	for o in 1:model.N
			    	Ltemp .+= rand(Wishart(model.K+1,0.001*diagm(ones(Float64, model.K))))
			  	end
				Ltemp ./= model.N
				model.L=100.0.*(Ltemp)./model.l
				# model.fmap = zeros(Float64, (model.N,model.K))
				# model.comm = [Int64[] for i in 1:model.K]
				# if isfile("./data/est_comm.txt")
				# 	rm("./data/est_comm.txt")
				# end
				model.b0 = model.b0[kindexes]
				model.b1 = model.b1[kindexes]
				model.μ_var=model.μ_var[:,kindexes]
				model.est_θ         = model.est_θ[:,kindexes]
			  	model.est_β         = model.est_β[kindexes]
			  	model.est_μ         = model.est_μ[kindexes]
			  	model.est_Λ         = model.est_Λ[kindexes,kindexes]
				model.Elogβ0        = model.Elogβ0[kindexes]
			  	model.Elogβ1        = model.Elogβ1[kindexes]
				model.μ_var_old     = model.μ_var_old[:,kindexes]
				model.m_old         = model.m_old[kindexes]
				model.M_old         = model.M_old[kindexes,kindexes]
				model.Λ_var_old     = model.Λ_var_old[:,kindexes]
				model.L_old         = model.L_old[kindexes,kindexes]
				model.b0_old        = model.b0_old[kindexes]
				model.b1_old        = model.b1_old[kindexes]
				model.m0            = model.m0[kindexes]
				model.M0            =model.M0[kindexes,kindexes]
			  	model.L0            =model.L0[kindexes,kindexes]
			end
		end

		checkelbo = (i % elboevery == 0)
		if checkelbo || i == 1
			print(i);print(": ")
			estimate_βs!(model,mb)
			# rho = 1-(nnz(model.network)/(model.N*model.N))
			println(model.est_β)
			# println(model.est_β./(1.0-rho))
		end


		if (i % 200 == 0)
			println(compute_NMI3(model))
			push!(nmitemp,compute_NMI3(model))
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
	end
	# Plots.plot(1:count_a[2], [(500/(500+i))^.5 for i in 1:count_a[2]])
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
	# sort_by_argmax!(est)
	println(maximum(nmitemp))
	Plots.plot(1:length(vec(true_θs)),sort(vec(true_θs)))
	Plots.plot(1:length(nmitemp), nmitemp)
	est = deepcopy(model.est_θ)
	Plots.plot(1:length(vec(est)),sort(vec(est)))
	p4=Plots.heatmap(est, yflip=true)
	Plots.plot(p3,p4, layout=(2,1))

# end
