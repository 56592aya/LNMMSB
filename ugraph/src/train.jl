using Plots
using GraphPlot
#initializing some parameters outside of the variational loop
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



estimate_θs!(model)
threshold=.9
# Here we need do determine the A, C, B for the first round before getting into the variational loop
"""
	getSets(model::LNMMSB, threshold::Float64)
	Function that returns the A, C, B of all nodes and keeps an ordering for communities
	Input: estimated_θ's
	Output: None, but updates A, C, B, Ordering, and mu's in the bulk set
"""
function getSets!(model::LNMMSB, threshold::Float64)
	est_θ = deepcopy(model.est_θ)

	for a in 1:model.N
		model.Korder[a] = sortperm(est_θ[a,:], rev=true)
		F = 0.0
		counter = 1
		while (F < threshold && counter < model.K)
			k = model.Korder[a][counter]
			F += est_θ[a,k]
			counter += 1
			push!(model.A[a], k)
		end
	end
	for a in 1:model.N
		neighbors = neighbors_(model, a)
		for b in neighbors
			for k in model.A[b]
				if !(k in model.A[a])
					push!(model.C[a], k)
				end
			end
		end
		model.C[a] = unique(model.C[a])
		model.B[a] = setdiff(model.Korder[a], union(model.A[a], model.C[a]))
		if !(isempty(model.B[a]))
			bulk_θs  = sum(est_θ[a,model.B[a]])/length(model.B[a])
			model.est_θ[a,model.B[a]] = bulk_θs
			model.μ_var[a,model.B[a]] = log.(bulk_θs)
		end
		_init_μ[a,:] = model.μ_var[a,:]
	end
end
getSets!(model, threshold)
#Starting the variational loop
for i in 1:iter
	#MB sampling, eveyr time we create an empty minibatch object
	mb=deepcopy(model.mb_zeroer)
	#fill in the mb with the nodes and links sampled
	minibatch_set_srns(model)
	model.mbids = deepcopy(mb.mbnodes)
	#shuffle them, so that the order is random
	shuffled = shuffle!(collect(model.minibatch_set))


	for d in shuffled
		#creating link objects and initializing their phis
		if isalink(model, "network", d.src, d.dst)
			l = Link(d.src, d.dst,_init_ϕ)
			if l in mb.mblinks
				continue;
			else
				push!(mb.mblinks, l)
			end
		else
			#creating nonlink objects and initializing their phis
			nl = NonLink(d.src, d.dst, _init_ϕ,_init_ϕ)

			if nl in mb.mbnonlinks
				continue;
			else
				#also adding their nonsources and nonsinks
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
	##Place to construct the C and B in the next iteration where the minibatch is set up
	for a in mb.mbnodes
		neighbors = neighbors_(model, a)
		#we can speed up here if we have visited the neighbor before but for later
		# idea is that if have not visited we can skip it, also should be reset somewhere
		for b in neighbors
			for k in model.A[b]
				if !(k in model.A[a])
					push!(model.C[a], k)
				end
			end
		end
		model.C[a] = unique(model.C[a])
		model.B[a] = setdiff(model.Korder[a], union(model.A[a], model.C[a]))
		if !(isempty(model.B[a]))
			bulk_θs  = sum(model.est_θ[a,model.B[a]])/length(model.B[a])
			model.est_θ[a,model.B[a]] = bulk_θs
			model.μ_var[a,model.B[a]] = log.(bulk_θs)
		end
	end
	##
	# mbsamplinglink!(mb, model, meth, model.mbsize)
	# model.fmap = deepcopy(zeros(Float64, (model.N,model.K)))
	#Setting the learning rates for full sample
	lr_M = 1.0
	lr_m = 1.0
	lr_L = 1.0
	lr_b = 1.0
	#Setting the learning rates for minibatch, very dubious for now
	if !isfullsample
		lr_m = (100./(101.0+Float64(20000/(model.N/model.mbsize)-1.0)))^(.9)
		lr_L = (100./(101.0+Float64(20000/(model.N/model.mbsize)-1.0)))^(.9)
		lr_b = (100./(101.0+Float64(20000/(model.N/model.mbsize)-1.0)))^(.9)
	end



	switchrounds = bitrand(1)[1]
	#getSets!(model, mb, threshold)
	# model.μ_var_old = deepcopy(model.μ_var)
	# model.μ_var_old[model.mbids,:]=deepcopy(model.μ_var[model.mbids,:])
	#setting the old value of mu to its init value
	# model.μ_var_old[model.mbids,:]=deepcopy(_init_μ[model.mbids,:])
	#creating some storage for variables to be used later
	model.ϕlsum = deepcopy(zeros(Float64, (model.N,model.K)))
	model.ϕnlinoutsum = deepcopy(zeros(Float64, model.K))
	model.ϕnloutsum = deepcopy(zeros(Float64, (model.N,model.K)))
	model.ϕnlinsum = deepcopy(zeros(Float64, (model.N,model.K)))
	one_over_K = deepcopy(ones(Float64,model.K)./model.K)

	for d in shuffled
		## updating phis for links
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
			## updating phis for nonlinks
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
	# Each time setting mu to its init value, and not the previous mu or mu_old
	# or maybe to some random start(not done yet)
	model.μ_var[model.mbids,:]=deepcopy(_init_μ[model.mbids,:])
	#updating the mus for minibatch
	for a in mb.mbnodes
		if !isfullsample
			#we also ignore the Lambda assuming its very large or cov is 0
			count_a[a] += 1
			expectedvisits = (iter/(4*model.N/model.mbsize))
			updatesimulμΛ!(model, a, mb)
			lr_μ[a] = (expectedvisits/(expectedvisits+Float64(count_a[a]-1.0)))^(.5)
			#lr_Λ[a] = (expectedvisits/(expectedvisits+Float64(count_a[a]-1.0)))^(.5)
			model.μ_var[a,:] = model.μ_var_old[a,:].*(1.0.-lr_μ[a])+lr_μ[a].*model.μ_var[a,:]
			# if i  < 1500
			# 	model.μ_var[a,:] .+= log.(nnz(model.network)./(2.*sum(model.ϕlsum, 1)[:]))
			# end
		else
			updatesimulμΛ!(model, a, mb,meth)
		end
	end
	estimate_θs!(model, mb)
	#
	###Finger example
	### old ordering where F(k ∈ A U C | est_θ[a,k] > est_θ[a,κ]) = .8 < .9
	### condition est_θ[a,k] > est_θ[a,κ] may leave out some of the ones in C(or even A potentially)
	### Vanilla version where condition est_θ[a,k] > est_θ[a,κ] always holds for any k in A or C
		### |A U C|=8, and |B|=20, so est_θ[a,κ] = .01
		### we have to sample (.9-.8)/0.01 = 10 from B
		### Now we have |A U C U sampled|=18 where F(A U C U sampled)=.9
		### and |B_rest| = 10
		### now since they meet .9 threshold I put (A U C U sampled) into active set of A
		### For that I first sort (A U C U sampled)
		### Should see which ones in B can be promoted to C, and leave rest as B
			### this could be done in the next rounds of iteration given already modified A's
	### General version where we need to check for the condition est_θ[a,k] > est_θ[a,κ]
		### We know who is in A, C, or B, we have the indices
		### We sort the new est_θ and and keep the indices in the original {K}
		### we check whether est_θ[a,k] > est_θ[a,κ] for k in A U C we call this A* U C*
			### for that in the new sorted list
				### as we want to construct the F, we check whether the k is in A or C and
				### if est_θ[a,k] > est_θ[a,κ]
		### if the F we constructed hits the .9 that is the new A
		### Else
			### if this F is less than the .9 threshold, we have to sample (threshold - F)/est_θ[a,κ] from B
		### Now we have new A as (A* U C* U sampled)
		### in the next iteration we construct the C
			### from Actives of neighbors which could potentially be from the B
			### the rest will remain as B

	est_θ = deepcopy(model.est_θ)
	for a in mb.mbnodes

		model.Korder[a] = sortperm(est_θ[a,:],rev=true)
		F = 0.0
		count = 1
		newA=Int64[]
		while (F < threshold && count < model.K)
			k = model.Korder[a][count]

			if (k in model.B[a])
				count+=1
			else
				if !isempty(model.B[a])
					if est_θ[a,k] > est_θ[a,model.B[a][1]]
						#println("I got here!")
						F += est_θ[a,k]
						push!(newA, k)
						count += 1
					end
				else
					F += est_θ[a,k]
					push!(newA, k)
					count += 1
				end
			end
		end
		toAdd = Int64[]
		if (threshold-F > 0.0)
			toSample = (threshold-F)/est_θ[a,model.B[a][1]]
			toAdd = sample(model.B[a], toSample, replace=false)
		end
		if !isempty(toAdd)
			for el in toAdd
				model.A[a]= push!(newA, el)
			end
		else
			model.A[a] = newA
		end
	end
	#

	updatem!(model, mb)
	model.m = model.m_old.*(1.0-lr_m)+lr_m.*model.m

	updateM!(model, mb)
	model.M = model.M_old.*(1.0-lr_M)+lr_M.*model.M

	updateL!(model, mb, i)
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
