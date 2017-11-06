using Plots
using GraphPlot
##initialization for the check is very important, havent yet figured it out.
# function train!(model::LNMMSB; iter::Int64=150, etol::Float64=1, niter::Int64=1000, ntol::Float64=1.0/(model.K^2), viter::Int64=10, vtol::Float64=1.0/(model.K^2), elboevery::Int64=10, mb::MiniBatch,lr::Float64)
	####TESTING ELBO INCREASE:#########
	###################################
	# if isfile("data/messages.txt")
	# 	rm("data/messages.txt")
	# end
	# f = open("data/messages.txt","w")


	##############################
	##############################
	# preparedata(model)
	iter=50000
	train_links_num=nnz(model.network)-length(model.ho_links)
	train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyads) -length(mb.mblinks)
	link_ratio = .1*(train_links_num)/(train_links_num+train_nlinks_num)
	prev_ll = -1000000
	store_ll = Array{Float64, 1}()
	first_converge = false
	# mu_curr=ones(model.N)
	# Lambda_curr=ones(model.N)
	# lr_mu = zeros(Float64, model.N)
	# lr_Lambda = zeros(Float64, model.N)
	early=true
	switchrounds=true
	elboevery=20
	model.elborecord = Vector{Float64}()
	model.elbo = 0.0
  	model.oldelbo= -Inf

	init_mu(model,communities,onlyK)##from Gopalan
	# true_θ=readdlm("data/true_thetas.txt")
	# model.μ_var=deepcopy(true_θ);
	# for i in 1:model.N
	# 	model.μ_var[i,:] = (model.μ_var[i,:])
	# end
	model.Λ_var = .01*ones(Float64, (model.N, model.K))
	true_mu = readdlm("data/true_mu.txt")
	model.m = zeros(Float64,model.K)#deepcopy(reshape(true_mu,model.K))
	model.M = (1.0).*eye(Float64,model.K)##to move around more
	model.l = model.K+2
	if length(communities) == inputtomodelgen[2]
		true_Lambda = readdlm("data/true_Lambda.txt")
		Lambda = deepcopy(true_Lambda)
		model.L=1.0.*(Lambda)./model.l
	else
		Ltemp = zeros(Float64, model.K, model.K)
	  	for i in 1:model.N
	    	Ltemp .+= rand(Wishart(model.K+1,0.001*diagm(ones(Float64, model.K))))
	  	end
  		Ltemp ./= model.N
		model.L=1.0.*(Ltemp)./model.l
	end
	# model.L=.5.*(model.L+model.L')
	isfullsample=false
	if model.mbsize == model.N
		isfullsample = true
	end
	updatel!(model)
	ntol=0.05
	niter=1000
	lr_M = 1.0
	lr_m = 1.0
	lr_L = 1.0
	lr_b = 1.0
	lr_μ=ones(Float64, model.N)
	lr_Λ=ones(Float64, model.N)
	count_μ = zeros(Int64, model.N)
	count_Λ = zeros(Int64, model.N)



	for i in 1:iter

		#Minibatch sampling/new sample
		##the following deepcopy is very important
		if isfullsample && i==1
			#for full sample only once
			mb=deepcopy(mb_zeroer)
			mbsampling!(mb,model, isfullsample)
		elseif !isfullsample
			mb=deepcopy(mb_zeroer)
			# mbsampling!(mb,model, isfullsample)
			mbsampling_partition!(mb,model,isfullsample)
		end
		#Learning rates
		# mb.mballnodes
		# model.mbids
		# lr_M = 1.0
		# lr_m = 1.0
		# lr_L = 1.0
		# lr_b = 1.0

		lr_M = (1024.0+Float64(i))^(-.9)
		lr_m = (1024.0+Float64(i))^(-.9)
		lr_L = (1024.0+Float64(i))^(-.9)
		lr_b = (1024.0+Float64(i))^(-.9)


		# ExpectedAllSeen=(model.N/model.mbsize)*1.5#round(Int64,nv(network)*sum([1.0/i for i in 1:nv(network)]))
		if sum(model.visit_count .>= 1) == model.N
			early = false
		end

		# early = false
    	# if i == round(Int64,ExpectedAllSeen)+1
        # 	early = false
    	# end


		updatephil!(model, mb,early, switchrounds)
		train_links_num=nnz(model.network)-length(model.ho_links)
		train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyads) -length(mb.mblinks)
		dep2 = .1*(train_links_num)/(train_links_num+train_nlinks_num)
		updatephinl!(model, mb,early, dep2,switchrounds)

		for a in collect(mb.mballnodes)

			count_μ[a]+=1
			count_Λ[a]+=1
			updatesimulμΛ!(model, a, niter, ntol,mb)
			lr_μ[a] = (1024.0+Float64(count_μ[a]))^(-.6)
			lr_Λ[a] = (1024.0+Float64(count_Λ[a]))^(-.6)
			model.μ_var[a,:] = model.μ_var_old[a,:].*(1.0.-lr_μ[a])+lr_μ[a].*model.μ_var[a,:]
			model.Λ_var[a,:] = model.Λ_var_old[a,:].*(1.0.-lr_Λ[a])+lr_Λ[a].*model.Λ_var[a,:]
		end



		# rate1=(convert(Float64,train_nlinks_num)/convert(Float64,length(mb.mbnonlinks)))
		# rate0=(convert(Float64,train_links_num)/convert(Float64,length(mb.mblinks)))

		updateb0!(model, mb)
		model.b0 = (1.0-lr_b).*model.b0_old + lr_b.*((model.b0))
		updateb1!(model,mb)
		model.b1 = (1.0-lr_b).*model.b1_old+lr_b.*((model.b1))

		updateL!(model, mb)
		model.L = model.L_old.*(1.0-lr_L)+lr_L*model.L

		updatem!(model, mb)
		model.m = model.m_old.*(1.0-lr_m)+lr_m.*model.m

		updateM!(model, mb)
		model.M = model.M_old.*(1.0-lr_M)+lr_M.*model.M


		model.est_θ[model.mbids,:] = estimate_θs(model, mb)


		checkelbo = (i % elboevery == 0)
		if checkelbo || i == 1
			print(i);print(": ")
			println(model.b0./(model.b0.+model.b1))
			if isfullsample
				computeelbo!(model, mb)
				increase=isinf(model.oldelbo)?65535.0:(model.elbo-model.oldelbo)/model.oldelbo;
				model.oldelbo=deepcopy(model.elbo)
				println(model.elbo)
				push!(model.elborecord, model.elbo)
			end
			# print(i);print("-ElBO:");println(model.elbo)
			# print("elbo improvement:");
			# println(increase);
			# if increase < 0 && i > 10
			#  	break;
			# end
		end
		switchrounds = !switchrounds
		if ((i == 1) || (i == iter) || (iter % elboevery == 0))
			β_est = zeros(Float64, model.K)
		    for k in 1:model.K
		        β_est[k]=model.b0[k]/(model.b0[k]+model.b1[k])
		    end
		        ####
		    link_lik = 0.0
		    nonlink_lik = 0.0
		    edge_lik = 0.0
		    link_count = 0; nonlink_count = 0

		    for pair in model.ho_dyads
				src = pair.src
				dst = pair.dst
		        edge_lik = edge_likelihood(model,pair, β_est)
		        if isalink(model.network, pair.src, pair.dst)
		            link_count +=1
		            link_lik += edge_lik
		        else
		            nonlink_count +=1
		            nonlink_lik += edge_lik
		        end
		    end

		    avg_lik = (link_ratio*(link_lik/link_count))+((1-link_ratio)*(nonlink_lik/nonlink_count))
		    # println()
		    # println("")
		    # println("")
		    # println("===================================================")
		    # print("Perplexity score is : ")
		    perp_score = exp(-avg_lik)
		    # println(perp_score)
		    # println("===================================================")
		    push!(store_ll, avg_lik)
		    # println(abs((prev_ll-avg_lik)/prev_ll))
		    # if !early
		    #     println("EARLY OFF")
		    # end
		    # if ((abs((prev_ll-avg_lik)/prev_ll) <= (1e-3)))
		    #     first_converge = true
		    #     #early = false
		    # end
		    # if ((abs((prev_ll-avg_lik)/prev_ll) <= (1e-10)))
		    #     # break;
		    #     #early = false
		    # end
		    prev_ll = avg_lik
		    # println("===================================================")
		    # print("loglikelihood: ")
		    # println(avg_lik)
		    # println("===================================================")
		end
	end

# isfullsample=true
# model.mbsize=model.N
	# mb=deepcopy(mb_zeroer)
	# mbsampling!(mb,model, isfullsample)
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
	# p1=Plots.plot(2:length(model.elborecord),model.elborecord[2:end])
	p2=Plots.heatmap(x, yflip=true)
	y = (readdlm("data/true_thetas.txt"))
	p3=Plots.heatmap(y, yflip=true)

	# decrease=0
	# for (i,v) in enumerate(model.elborecord)
	# 	if i < length(model.elborecord)
	# 		if model.elborecord[i+1] < model.elborecord[i]
	# 			##Decrease can happen
	# 			if !isapprox(model.elborecord[i+1], model.elborecord[i])
	# 				decrease+=1
	# 			end
	# 		end
	# 	end
	# end
	# println(decrease)

	computeNMI(x,y,communities)

	# Plots.plot(p1,p2,p3, layout=(3,1))
	# Plots.plot(p1)


	Plots.plot(p2,p3, layout=(2,1))
	# Plots.savefig(p1, "thetaest0.png")
	Plots.plot(1:length(store_ll), store_ll)


#
# end
