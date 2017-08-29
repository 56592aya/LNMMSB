##initialization for the check is very important, havent yet figured it out.
function train!(model::LNMMSB; iter::Int64=150, etol::Float64=1, niter::Int64=1000, ntol::Float64=1.0/(model.K^2), viter::Int64=10, vtol::Float64=1.0/(model.K^2), elboevery::Int64=10, mb::MiniBatch,lr::Float64)

	preparedata(model)
	mu_curr=ones(model.N)
	Lambda_curr=ones(model.N)
	lr_mu = zeros(Float64, model.N)
	lr_Lambda = zeros(Float64, model.N)
	early=true
	switchrounds=true
	#let's say for now:
	elboevery=10
	true_θ=readdlm("data/true_theta.txt")
	model.μ_var=deepcopy(true_θ);

	for i in 1:model.N
		model.μ_var[i,:] = log.(model.μ_var[i,:])
	end
	model.Λ_var = 10.0*ones(Float64, (model.N, model.K))
	true_mu = readdlm("data/true_mu.txt")
	model.m = deepcopy(reshape(true_mu,model.K))
	model.M = 10.0*eye(Float64, K)
	model.l = model.K
	true_Lambda = readdlm("data/true_Lambda.txt")
	Lambda = deepcopy(true_Lambda)
	model.L=inv(Lambda)./model.l
	i=1

	isfullsample=false
	if model.mbsize == model.N
		isfullsample = true
	end

	for i in 1:iter
		#Minibatch sampling/new sample
		##the following deepcopy is very important
		if isfullsample && i==1
			#for full sample only once
			mb=deepcopy(mb_zeroer)
			mbsampling!(mb,model, isfullsample)
		elseif !isfullsample
			mb=deepcopy(mb_zeroer)
			mbsampling!(mb,model, isfullsample)
		end
		#global update-- can be done outside
		updatel!(model, mb)

		#Learning rates

		lr_M = 1.0
		lr_m = 1.0
		lr_L = 1.0
		lr_b = 1.0
		# lr_M = 1.0/((1.0+Float64(i))^.5)
		# lr_m = 1.0/((1.0+Float64(i))^.7)
		# lr_L = 1.0/((1.0+Float64(i))^.9)
		# lr_b = 1.0/((1.0+Float64(i))^.5)


		#locals:phis
		#local update
		ExpectedAllSeen=(model.N/model.mbsize)*1.5#round(Int64,nv(network)*sum([1.0/i for i in 1:nv(network)]))
    	if i == round(Int64,ExpectedAllSeen)+1
        	early = false
    	end
		train_links_num=nnz(model.network)-length(model.ho_linkdict)
		train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyaddict) -length(mb.mblinks)
		dep2 = .1*(train_links_num)/(train_links_num+train_nlinks_num)

		updatephil!(model, mb, early,switchrounds)
		updatephinl!(model, mb,early,dep2,switchrounds)


		#global update
		#globals:m,M,L,mu, Lambda, b
		updateM!(model, mb)

		model.M = model.M_old*(1.0-lr_M)+lr_M*model.M
		updatem!(model, mb)
		model.m = model.m_old*(1.0-lr_m)+lr_m*model.m

		updateL!(model, mb)
		model.L = model.L_old*(1.0-lr_L)+lr_L*model.L
		updateb0!(model, mb)
		model.b0 = model.b0_old*(1.0-lr_b)+lr_b*model.b0
		println(model.b0)
		println(model.b0_old)
		updateb1!(model, mb)
		model.b1 = model.b1_old*(1.0-lr_b)+lr_b*model.b1
		println(model.b1)
		println(model.b1_old)


		mb.mblinks[1].ϕout
		mb.mblinks[1].ϕin
		mb.mbnonlinks[1].ϕout
		mb.mbnonlinks[1].ϕin
		model.M
		model.m
		model.L
		model.b0
		model.b1
		print(i);print(": ")
		println(model.b0./(model.b0.+model.b1))


		# for a in collect(mb.mballnodes)
		# 	updatemua!(model, a, niter, ntol,mb)
		# 	lr_mu[a] = 1.0/((1.0+Float64(mu_curr[a]))^.9)##could be  a macro
		# 	mu_curr[a] += 1
		# 	model.ζ[a] = model.μ_var_old[a]*(1.0-lr_mu[a])+lr_mu[a]*model.μ_var[a]
		# 	updateLambdaa!(model, a, niter, ntol,mb)
		# 	lr_Lambda[a] = 1.0/((1.0+Float64(Lambda_curr[a]))^.9)##could be  a macro
		# 	Lambda_curr[a] += 1
		# 	model.Λ_var[a] = model.Λ_var_old[a]*(1.0-lr_Lambda[a])+lr_Lambda[a]*model.Λ_var[a]
		# end
		checkelbo = (i % elboevery == 0)
		if checkelbo || i == 1
			computeelbo!(model, mb)
			print(i);print("-ElBO:");println(model.elbo)
			model.oldelbo=model.elbo
			push!(model.elborecord, model.elbo)
			abs(model.oldelbo-model.elbo)/model.oldelbo
		end
		switchrounds = !switchrounds

		i=i+1
	end
end
Plots.plot(1:length(model.elborecord),model.elborecord)
