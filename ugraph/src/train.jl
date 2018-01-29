	using Plots
	using GraphPlot

	##initialization for the check is very important, havent yet figured it out.
	# function train!(model::LNMMSB; iter::Int64=150, etol::Float64=1, niter::Int64=1000, ntol::Float64=1.0/(model.K^2), viter::Int64=10, vtol::Float64=1.0/(model.K^2), elboevery::Int64=10, mb::MiniBatch,lr::Float64)
		# preparedata(model)
		for a in 1:model.N
			model.train_deg[a] = sum(model.network[a,:])
		end
		iter=10000
		train_links_num=0
		train_nlinks_num = 0
		link_ratio = 0.0
		prev_ll = -1000000
		store_ll = Array{Float64, 1}()
		store_nmi = Array{Float64, 1}()
		first_converge = false
		switchrounds=true
		elboevery=200
		model.elborecord = Vector{Float64}()
		model.elbo = 0.0
		model.oldelbo= -Inf
		train_links_num=sum(model.train_deg)/2
		train_nlinks_num=(model.N*model.N-model.N)/2- length(model.ho_dyads)-train_links_num
		link_ratio = convert(Float64, train_links_num)/convert(Float64,train_nlinks_num)
		_init_ϕ = deepcopy(ones(Float64, model.K).*1.0/model.K)
		link_thresh=.9
		true_θs = (readdlm("../data/true_thetas.txt"))

		init_mu(model,communities,model.K)##from Gopalan
		_init_μ = deepcopy(model.μ_var)
		model.μ_var = deepcopy(_init_μ)
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
		true_μ = readdlm("../data/true_mu.txt")
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
		model.μ_var=deepcopy(_init_μ)

		# function getCommSets(model::LNMMSB, mb::MiniBatch) ## uses model.est_θ and returns or inplace updates?
		# 	est=deepcopy(model.est_θ)
		#
		# 	for a in mb.mbnodes
		# 		model.sortedK[a] = sortperm(est[a,:],rev=true)
		# 		s = 0.0
		# 		for (i,k) in enumerate(model.sortedK[a])
		# 			if s > .9
		# 				model.stopAt[a] = i-1
		# 				break;
		# 			else
		# 				s += est[a,k]
		# 			end
		# 		end
		# 	end
		# 	for a in mb.mbnodes
		# 		for j in model.sortedK[a][1:model.stopAt[a]]
		# 			push!(model.Active[a], j)
		# 		end
		# 		model.Active[a] = unique(model.Active[a])
		# 		for b in neighbors(lg,a)
		#
		# 			for k in model.sortedK[a]
		# 				if k in model.Active[a]
		# 					continue;
		# 				else
		# 					if k in model.Active[b]
		# 						push!(model.Candidate[a], k)
		# 					else
		# 						push!(model.Bulk[a], k)
		# 					end
		# 				end
		# 			end
		# 		end
		# 		model.Candidate[a] = unique(model.Candidate[a])
		# 		model.Bulk[a] = unique(model.Bulk[a])
		# 	end
		# end
		# function getCommSets(model::LNMMSB) ## uses model.est_θ and returns or inplace updates?
		# 	est=deepcopy(model.est_θ)
		#
		# 	for a in model.N
		# 		model.sortedK[a] = sortperm(est[a,:],rev=true)
		# 		s = 0.0
		# 		for (i,k) in enumerate(model.sortedK[a])
		# 			if s > .9
		# 				model.stopAt[a] = i-1
		# 				break;
		# 			else
		# 				s += est[a,k]
		# 			end
		# 		end
		# 	end
		# 	for a in 1:model.N
		# 		for j in model.sortedK[a][1:model.stopAt[a]]
		# 			push!(model.Active[a], j)
		# 		end
		# 		model.Active[a] = unique(model.Active[a])
		# 		for b in neighbors(lg,a)
		#
		# 			for k in model.sortedK[a]
		# 				if k in model.Active[a]
		# 					continue;
		# 				else
		# 					if k in model.Active[b]
		# 						push!(model.Candidate[a], k)
		# 					else
		# 						push!(model.Bulk[a], k)
		# 					end
		# 				end
		# 			end
		# 		end
		# 		model.Candidate[a] = unique(model.Candidate[a])
		# 		model.Bulk[a] = unique(model.Bulk[a])
		# 	end
		# end


		estimate_θs!(model)

		# getCommSets(model)
		# for a in 1:model.N
		# 	model.est_θ[a,model.Bulk[a]]=(1.0 - sum(model.est_θ[a,union(model.Active[a], model.Candidate[a])]))/(model.K-length(union(model.Active[a], model.Candidate[a])))
		# 	model.μ_var[a,:] = log.(model.est_θ[a,:])
		# end


# @btime updatesimulμΛ!(model, mb.mbnodes[1],mb)
# using ProfileView
# Profile.clear()
# @profile updatesimulμΛ!(model, mb.mbnodes[1],mb)
# ProfileView.view()
		#####

		#####
		for i in 1:iter

			mb=deepcopy(model.mb_zeroer)
			minibatch_set_srns(model)
			model.mbids = deepcopy(mb.mbnodes)
			shuffled = shuffle!(collect(model.minibatch_set))


			for d in shuffled
				if isalink(model, "network", d.src, d.dst)
					l = Link(d.src, d.dst,_init_ϕ)
					if l in mb.mblinks
						continue;
					else
						push!(mb.mblinks, l)
					end
				else
					nl = NonLink(d.src, d.dst, _init_ϕ,_init_ϕ)
					if nl in mb.mbnonlinks
						continue;
					else
						push!(mb.mbnonlinks, nl)
						if !haskey(mb.mbnot, nl.src)
							mb.mbnot[nl.src] = get(mb.mbnot, nl.src, Vector{Int64}())
						end
						if nl.dst in mb.mbnot[nl.src]
							continue;
						else
							push!(mb.mbnot[nl.src], nl.dst)
						end
						if !haskey(mb.mbnot, nl.dst)
							mb.mbnot[nl.dst] = get(mb.mbnot, nl.dst, Vector{Int64}())
						end
						if nl.src in mb.mbnot[nl.dst]
							continue;
						else
							push!(mb.mbnot[nl.dst], nl.src)
						end
					end
				end
			end
			# shuffle!(dyads)

			# mbsamplinglink!(mb, model, meth, model.mbsize)
			# model.fmap = deepcopy(zeros(Float64, (model.N,model.K)))
			lr_M = 1.0
			lr_m = 1.0
			lr_L = 1.0
			lr_b = 1.0
			if !isfullsample
				# lr_m = (1024./(1024.0+Float64(i/(model.N/model.mbsize)-1.0)))^(.9)
				# lr_L = (1024./(1024.0+Float64(i/(model.N/model.mbsize)-1.0)))^(.9)
				# lr_b = (1024./(1024.0+Float64(i/(model.N/model.mbsize)-1.0)))^(.9)
				lr_m = (100./(101.0+Float64(20000/(model.N/model.mbsize)-1.0)))^(.9)
				lr_L = (100./(101.0+Float64(20000/(model.N/model.mbsize)-1.0)))^(.9)
				lr_b = (100./(101.0+Float64(20000/(model.N/model.mbsize)-1.0)))^(.9)


			end



			switchrounds = bitrand(1)[1]

			# model.μ_var_old = deepcopy(model.μ_var)
			# model.μ_var_old[model.mbids,:]=deepcopy(model.μ_var[model.mbids,:])
			model.ϕlsum = deepcopy(zeros(Float64, (model.N,model.K)))
			model.μ_var_old[model.mbids,:]=deepcopy(_init_μ[model.mbids,:])
			model.ϕnlinoutsum = deepcopy(zeros(Float64, model.K))
			model.ϕnloutsum = deepcopy(zeros(Float64, (model.N,model.K)))
			model.ϕnlinsum = deepcopy(zeros(Float64, (model.N,model.K)))
			one_over_K = deepcopy(ones(Float64,model.K)./model.K)

			for d in shuffled
				if d in mb.mblinks
					l=deepcopy(mb.mblinks[mb.mblinks .== d][1])
					l.ϕ = deepcopy(one_over_K)
					for j in 1:10
						updatephil!(model, mb, l)
						# updatephil!(model, mb, l,"check")
					end
					for k in 1:model.K
						model.ϕlsum[l.src,k] += l.ϕ[k]
						model.ϕlsum[l.dst,k] += l.ϕ[k]
					end


				else
					nl=deepcopy(mb.mbnonlinks[mb.mbnonlinks .== d][1])
					nl.ϕout = deepcopy(one_over_K)
					nl.ϕin = deepcopy(one_over_K)
					for j in 1:10
						updatephinl!(model, mb, nl)
						# updatephinl!(model, mb, nl, "check")
					end
					for k in 1:model.K
						model.ϕnloutsum[nl.src,k] += nl.ϕout[k]
						model.ϕnlinsum[nl.src,k] += nl.ϕout[k]
						model.ϕnlinsum[nl.dst,k] += nl.ϕin[k]
						model.ϕnloutsum[nl.dst,k] += nl.ϕin[k]
						model.ϕnlinoutsum[k] += nl.ϕout[k]*nl.ϕin[k]
					end
				end
			end

			model.μ_var[model.mbids,:]=deepcopy(_init_μ[model.mbids,:])
			# model.μ_var[model.mbids,:]=deepcopy(model.μ_var_old[model.mbids,:])


			for a in mb.mbnodes
				if !isfullsample

					count_a[a] += 1
					expectedvisits = (iter/(4*model.N/model.mbsize))
					updatesimulμΛ!(model, a, mb)
					lr_μ[a] = (expectedvisits/(expectedvisits+Float64(count_a[a]-1.0)))^(.5)
					lr_Λ[a] = (expectedvisits/(expectedvisits+Float64(count_a[a]-1.0)))^(.5)
					model.μ_var[a,:] = model.μ_var_old[a,:].*(1.0.-lr_μ[a])+lr_μ[a].*model.μ_var[a,:]
					if i  < 1500
						model.μ_var[a,:] .+= log(nnz(model.network)./(2.*sum(model.ϕlsum, 1)[:]))
					end
					model.Λ_var[a,:] = model.Λ_var_old[a,:].*(1.0.-lr_Λ[a])+lr_Λ[a].*model.Λ_var[a,:]
				else
					updatesimulμΛ!(model, a, mb,meth)
				end
			end
			estimate_θs!(model, mb)
			# getCommSets(model, mb)
			# for a in mb.mbnodes
			# 	model.est_θ[a,model.Bulk[a]]=(1.0 - sum(model.est_θ[a,union(model.Active[a], model.Candidate[a])]))/(model.K-length(union(model.Active[a], model.Candidate[a])))
			# 	model.μ_var[a,:] = log.(model.est_θ[a,:])
			# end

## define a function that returns the A, B, C, ...
## also have a promote/demote function if necessary



			# if i % 100 == 0

			updatem!(model, mb)
			model.m = model.m_old.*(1.0-lr_m)+lr_m.*model.m

			updateM!(model, mb)
			model.M = model.M_old.*(1.0-lr_M)+lr_M.*model.M

			updateL!(model, mb)
			model.L = model.L_old.*(1.0-lr_L)+lr_L*model.L

			updateb0!(model, mb)
			model.b0 = (1.0-lr_b).*model.b0_old + lr_b.*((model.b0))


			updateb1!(model,mb)

			model.b1 = (1.0-lr_b).*model.b1_old+lr_b.*((model.b1))


			update_Elogβ!(model)





			checkelbo = (i % model.N/model.mbsize == 0)
			if checkelbo || i == 1
				print(i);print(": ")
				estimate_βs!(model,mb)
				println(model.est_β)
			end


			# x = deepcopy(model.est_θ)
			# sort_by_argmax!(x)
			# table=[sortperm(x[i,:]) for i in 1:model.N]
			# count = zeros(Int64, model.K)
			# diffK = model.K-inputomodelgen[2]
			# for k in 1:model.K
			# 	for i in 1:model.N
			# 		for j in 1:diffK
			# 			if table[i][j] == k
			# 				count[k] +=1
			# 			end
			# 		end
			# 	end
			# end
			# idx=sortperm(count,rev=true)[(diffK+1):end]
			# x = x[:,sort(idx)]
		end
		# Plots.plot(1:count_a[2], [(500/(500+i))^.5 for i in 1:count_a[2]])
		x = deepcopy(model.est_θ)
		sort_by_argmax!(x)
		table=[sortperm(x[i,:]) for i in 1:model.N]
		# count = zeros(Int64, model.K)
		# diffK = model.K-inputtomodelgen[2]
		# for k in 1:model.K
		# 	for i in 1:model.N
		# 		for j in 1:diffK
		# 			if table[i][j] == k
		# 				count[k] +=1
		# 			end
		# 		end
		# 	end
		# end
		# idx=sortperm(count,rev=true)[(diffK+1):end]
		# x = x[:,sort(idx)]
		est=deepcopy(model.est_θ)
		p2=Plots.heatmap(x, yflip=true)
		p3=Plots.heatmap(true_θs, yflip=true)
		Plots.plot(p2,p3, layout=(2,1))
		sort_by_argmax!(est)
		# println(maximum(nmitemp))
		Plots.plot(1:length(vec(true_θs)),sort(vec(true_θs)))
		# Plots.plot(1:length(nmitemp), nmitemp)
		est = deepcopy(model.est_θ)
		Plots.plot(1:length(vec(est)),sort(vec(est)))
		pyplot()

		p3=Plots.heatmap(true_θs, yflip=true)
		p4=Plots.heatmap(est, yflip=true)
		Plots.plot(p4,p3, layout=(2,1))

	# end
