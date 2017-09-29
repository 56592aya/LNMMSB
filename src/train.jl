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
	preparedata(model)
	iter=1000
	# mu_curr=ones(model.N)
	# Lambda_curr=ones(model.N)
	# lr_mu = zeros(Float64, model.N)
	# lr_Lambda = zeros(Float64, model.N)
	early=true
	switchrounds=true
	elboevery=5
	model.elborecord = Vector{Float64}()
	model.elbo = 0.0
  	model.oldelbo= -Inf

	init_mu(model,communities)##from Gopalan
	# true_θ=readdlm("data/true_thetas.txt")
	# model.μ_var=deepcopy(true_θ);
	# for i in 1:model.N
	# 	model.μ_var[i,:] = (model.μ_var[i,:])
	# end
	model.Λ_var = .01*ones(Float64, (model.N, model.K))
	true_mu = readdlm("data/true_mu.txt")
	model.m = zeros(Float64,model.K)#deepcopy(reshape(true_mu,model.K))
	model.M = (1.0).*eye(Float64,K)##to move around more
	model.l = model.K+2
	true_Lambda = readdlm("data/true_Lambda.txt")
	Lambda = deepcopy(true_Lambda)
	model.L=1.0.*(Lambda)./model.l
	# model.L=.5.*(model.L+model.L')
	isfullsample=false
	if model.mbsize == model.N
		isfullsample = true
	end
	updatel!(model)
	ntol=0.05
	niter=1000
	lr_μ=ones(Float64, model.mbsize)

	for i in 1:iter
		#Minibatch sampling/new sample
		##the following deepcopy is very important
		if isfullsample && i==1
			#for full sample only once
			mb=deepcopy(mb_zeroer)
			mbsampling!(mb,model, isfullsample)
			# for l in mb.mblinks
			# 	l.ϕin[:] = (1.0/model.K)*ones(Float64, model.K)
			# 	l.ϕout[:] = (1.0/model.K)*ones(Float64, model.K)
			# end
			# for nl in mb.mbnonlinks
			# 	nl.ϕin[:] = (1.0/model.K)*ones(Float64, model.K)
			# 	nl.ϕout[:] = (1.0/model.K)*ones(Float64, model.K)
			# end
			# model.b0 = 5.0*ones(Float64, model.K)
			# model.b1 = 5.0*ones(Float64, model.K)
		elseif !isfullsample
			mb=deepcopy(mb_zeroer)
			mbsampling!(mb,model, isfullsample)
		end
		#Learning rates

		lr_M = 1.0
		lr_m = 1.0
		lr_L = 1.0
		lr_b = 1.0

		# lr_M = 1.0/((1.0+Float64(i))^.5)
		# lr_m = 1.0/((1.0+Float64(i))^.7)
		# lr_L = 1.0/((1.0+Float64(i))^.9)
		# lr_b = 1.0/((1.0+Float64(i))^.5)


		ExpectedAllSeen=(model.N/model.mbsize)*1.5#round(Int64,nv(network)*sum([1.0/i for i in 1:nv(network)]))
		early = false
    	# if i == round(Int64,ExpectedAllSeen)+1
        # 	early = false
    	# end

		# ELBO_pre= elogpzlout(model,mb)+elogpzlin(model,mb)+elogpnetwork1(model, mb)-elogqzl(model,mb)
		updatephil!(model, mb,early, switchrounds)
		# ELBO_post=elogpzlout(model,mb)+elogpzlin(model,mb)+elogpnetwork1(model, mb)-elogqzl(model,mb)
		# if (ELBO_post<ELBO_pre)
		# 	println("have decrease in ELBO in updatephilink")
		# 	println(ELBO_pre)
		# 	println(ELBO_post)
		# 	if !isapprox(ELBO_pre,ELBO_post)
		# 		break
		# 	end
		# end

		# train_links_num=nnz(model.network)-length(model.ho_linkdict)
		train_links_num=nnz(model.network)-length(model.ho_linkdict)
		train_nlinks_num = model.N*(model.N-1) - length(model.ho_dyaddict) -length(mb.mblinks)
		dep2 = .1*(train_links_num)/(train_links_num+train_nlinks_num)
		# ELBO_pre= elogpznlout(model,mb)+elogpznlin(model,mb)+elogpnetwork0(model, mb)-elogqznl(model, mb)
		updatephinl!(model, mb,early, dep2,switchrounds)
		# ELBO_post= elogpznlout(model,mb)+elogpznlin(model,mb)+elogpnetwork0(model, mb)-elogqznl(model, mb)
		# if (ELBO_post<ELBO_pre)
		# 	println("have decrease in ELBO in updatephinonlink")
		# 	println(ELBO_pre)
		# 	println(ELBO_post)
		# 	if !isapprox(ELBO_pre,ELBO_post)
		# 		break
		# 	end
		# end


		# init_mu(model,communities)
		# ELBO_pre=  elogpzlout(model,mb)+elogpzlin(model,mb)+elogpznlout(model,mb)+elogpznlin(model,mb) + elogptheta(model,mb)#-elogqtheta(model)
		# model.Λ_var = .1*ones(Float64, (model.N, model.K))
		for a in collect(mb.mballnodes)
			# updatelzeta!(model, mb, a)
			# updatemua!(model, a, niter, ntol,mb)
			# updateLambdaa!(model, a, niter, ntol,mb)
			# updatemua2!(model, a, niter, ntol,mb)
			# updateLambdaa2!(model, a, niter, ntol,mb)

			updatesimulμΛ!(model, a, niter, ntol,mb)
			# model.μ_var[a,:] = model.μ_var_old[a,:].*(1.0.-lr_μ[a])+lr_μ[a].*model.μ_var[a,:]
		end
		# model.μ_var[1,:]
		# ELBO_post=  elogpzlout(model,mb)+elogpzlin(model,mb)+elogpznlout(model,mb)+elogpznlin(model,mb) + elogptheta(model,mb)#-elogqtheta(model)
		# if (ELBO_post<ELBO_pre)
		# 	println("have decrease in ELBO in updatemuLam")
		# 	println(ELBO_pre)
		# 	println(ELBO_post)
		# 	if !isapprox(ELBO_pre,ELBO_post) && i > 10
		# 		break
		# 	end
		# end



		rate1=(convert(Float64,train_nlinks_num)/convert(Float64,length(mb.mbnonlinks)))
		rate0=(convert(Float64,train_links_num)/convert(Float64,length(mb.mblinks)))
		# ELBO_pre=elogpbeta(model)+elogpnetwork(model,mb)-elogqbeta(model)

		updateb0!(model, mb)
		model.b0 = (1.0-lr_b).*model.b0_old + lr_b.*((model.b0))
		updateb1!(model,mb)
		model.b1 = (1.0-lr_b).*model.b1_old+lr_b.*((model.b1))

		# if (ELBO_post<ELBO_pre)
		# 	println("have decrease in ELBO in updateb")
		# 	println(ELBO_pre)
		# 	println(ELBO_post)
		# 	if !isapprox(ELBO_pre,ELBO_post)
		# 		break
		# 	end
		# end
		#####
		# ELBO_pre = (elogpLambda(model) + elogptheta(model,mb) - (elogqLambda(model)))
		updateL!(model, mb)
		model.L = model.L_old.*(1.0-lr_L)+lr_L*model.L
		# model.L=.5.*(model.L+model.L')
		# ELBO_post = (elogpLambda(model) + elogptheta(model,mb) - (elogqLambda(model)))

		# if (ELBO_post<ELBO_pre)
		# 	println("have decrease in ELBO in updateL")
		# 	println(ELBO_pre)
		# 	println(ELBO_post)
		# 	if !isapprox(ELBO_pre,ELBO_post)
		# 		break
		# 	end
		# end
		#

		# ELBO_pre = (elogpmu(model) + elogptheta(model,mb))
		updatem!(model, mb)
		model.m = model.m_old.*(1.0-lr_m)+lr_m.*model.m
		# ELBO_post = (elogpmu(model) + elogptheta(model,mb))
		# if (ELBO_post<ELBO_pre)
		# 	println("have decrease in ELBO in updatem")
		# 	println(ELBO_pre)
		# 	println(ELBO_post)
		# 	if !isapprox(ELBO_pre,ELBO_post)
		# 		break
		# 	end
		#
		# end
		# ELBO_pre=  (elogpmu(model) + elogptheta(model,mb) - (elogqmu(model)))
		updateM!(model, mb)
		model.M = model.M_old.*(1.0-lr_M)+lr_M.*model.M
		# model.M = .5.*(model.M+model.M')
		# ELBO_post=  (elogpmu(model) + elogptheta(model,mb) - (elogqmu(model)))
		# if (ELBO_post<ELBO_pre)
		# 	println("have decrease in ELBO in updateM")
		# 	println(ELBO_pre)
		# 	println(ELBO_post)
		# 	if !isapprox(ELBO_pre,ELBO_post)
		# 		break
		# 	end
		# end





		print(i);print(": ")
		println(model.b0./(model.b0.+model.b1))

		checkelbo = (i % elboevery == 0)
		if checkelbo || i == 1
			computeelbo!(model, mb)
			# print(i);print("-ElBO:");println(model.elbo)
			# print("elbo improvement:");
			increase=isinf(model.oldelbo)?65535.0:(model.elbo-model.oldelbo)/model.oldelbo;
			# println(increase);
			# if increase < 0 && i > 10
			#  	break;
			# end
			model.oldelbo=deepcopy(model.elbo)
			println(model.elbo)
			push!(model.elborecord, model.elbo)
		end
		switchrounds = !switchrounds
	end
	p1=Plots.plot(2:length(model.elborecord),model.elborecord[2:end])
	p2=Plots.heatmap(estimate_θs(model, mb), yflip=true)
	plot(p1,p2, layout=(2,1))
	for (i,v) in enumerate(model.elborecord)
		if i < length(model.elborecord)
			if model.elborecord[i+1] < model.elborecord[i]
				println("decrease")
			end
		end
	end

#
# end
