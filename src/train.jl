##initialization for the check is very important, havent yet figured it out.
function train!(model::LNMMSB; iter::Int64=150, etol::Float64=1, niter::Int64=1000, ntol::Float64=1.0/(model.K^2), viter::Int64=10, vtol::Float64=1.0/(model.K^2), elboevery::Int64=10, mb::MiniBatch,lr::Float64)
	####TESTING ELBO INCREASE:#########
	###################################
	if isfile("data/messages.txt")
		rm("data/messages.txt")
	end
	f = open("data/messages.txt","w")


	##############################
	##############################
	preparedata(model)
	iter=50
	mu_curr=ones(model.N)
	Lambda_curr=ones(model.N)
	lr_mu = zeros(Float64, model.N)
	lr_Lambda = zeros(Float64, model.N)
	early=true
	switchrounds=true
	#let's say for now:
	elboevery=10
	model.elborecord = Vector{Float64}()
	model.elbo = 0.0
  	model.oldelbo= -Inf

	init_mu(model,communities)##from Gopalan
	# true_θ=readdlm("data/true_thetas.txt")
	# model.μ_var=deepcopy(true_θ);
	# for i in 1:model.N
	# 	model.μ_var[i,:] = (model.μ_var[i,:])
	# end
	model.Λ_var = 1.0*ones(Float64, (model.N, model.K))
	true_mu = readdlm("data/true_mu.txt")
	model.m = deepcopy(reshape(true_mu,model.K))
	# model.M = (10.0).*eye(Float64,K)
	model.M = (0.1).*eye(Float64,K)##to move around more
	model.l = model.K+3
	true_Lambda = readdlm("data/true_Lambda.txt")
	Lambda = deepcopy(true_Lambda)
	model.L=0.1.*(Lambda)./model.l
	# i=1

	isfullsample=false
	if model.mbsize == model.N
		isfullsample = true
	end
	updatel!(model)
	ELBO_Lrecord = Vector{Float64}()
	ELBO_mrecord = Vector{Float64}()
	ELBO_Mrecord = Vector{Float64}()
	ELBO_μarecord = Vector{Float64}()
	ELBO_Λarecord = Vector{Float64}()
	ELBO_b0record = Vector{Float64}()
	ELBO_b1record = Vector{Float64}()
	ELBO_ϕloutrecord = Vector{Float64}()
	ELBO_ϕlinrecord = Vector{Float64}()
	ELBO_ϕnloutrecord = Vector{Float64}()
	ELBO_ϕnlinrecord = Vector{Float64}()


	# ELBO_L = elogpLambda(model) + elogptheta(model,mb) - (elogqLambda(model))
	# ELBO_m = elogpmu(model) + elogptheta(model,mb)
	# ELBO_M =  elogpmu(model) + elogptheta(model,mb) - (elogqmu(model))
	# ELBO_μa = elogptheta(model,mb)+elogpzlout(model,mb)+elogpzlin(model,mb)+elogpznlout(model,mb)+	elogpznlin(model,mb)
	# ELBO_Λa = elogptheta(model,mb)+elogpzlout(model,mb)+elogpzlin(model,mb)+elogpznlout(model,mb)+	elogpznlin(model,mb)-(elogqtheta(model))
	# ELBO_b0 =elogpbeta(model)+elogpnetwork(model,mb)-(elogqbeta(model))
	# ELBO_b1 =elogpbeta(model)+elogpnetwork(model,mb)-(elogqbeta(model))
	# ELBO_ϕlout =elogpzlout(model,mb)+elogpnetwork(model,mb)-(elogqzl(model))
	# ELBO_ϕlin = elogpzlin(model,mb)+elogpnetwork(model,mb)-(elogqzl(model))
	# ELBO_ϕnlout =elogpznlout(model,mb)+elogpnetwork(model,mb)-(elogqznl(model))
	# ELBO_ϕnlin =elogpznlin(model,mb)+elogpnetwork(model,mb)-(elogqznl(model))


	ELBO_Lold = -Inf
	ELBO_mold = -Inf
	ELBO_Mold = -Inf
	ELBO_μaold = -Inf
	ELBO_Λaold = -Inf
	ELBO_b0old = -Inf
	ELBO_b1old = -Inf
	ELBO_ϕloutold = -Inf
	ELBO_ϕlinold = -Inf
	ELBO_ϕnloutold = -Inf
	ELBO_ϕnlinold = -Inf

	# push!(ELBO_Lrecord, ELBO_L)
	# push!(ELBO_mrecord, ELBO_m)
	# push!(ELBO_Mrecord, ELBO_M)
	# push!(ELBO_μarecord, ELBO_μa)
	# push!(ELBO_Λarecord, ELBO_Λa)
	# push!(ELBO_b0record, ELBO_b0)
	# push!(ELBO_b1record, ELBO_b1)
	# push!(ELBO_ϕloutrecord, ELBO_ϕlout)
	# push!(ELBO_ϕlinrecord, ELBO_ϕlin)
	# push!(ELBO_ϕnloutrecord, ELBO_ϕnlout)
	# push!(ELBO_ϕnlinrecord, ELBO_ϕnlin)

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

		# updatephil!(model, mb, early,switchrounds)
		# updatephinl!(model, mb,early,dep2,switchrounds)


		#global update
		#globals:m,M,L,mu, Lambda, b
		ELBO_pre=  elogpmu(model) + elogptheta(model,mb) - (elogqmu(model))
		updateM!(model, mb)
		model.M = model.M_old.*(1.0-lr_M)+lr_M.*model.M
		ELBO_post=  elogpmu(model) + elogptheta(model,mb) - (elogqmu(model))
		if (ELBO_post<ELBO_pre)
			println("have decrease in ELBO in updateM")
			break
		end


		ELBO_pre = elogpmu(model) + elogptheta(model,mb)
		updatem!(model, mb)
		model.m = model.m_old.*(1.0-lr_m)+lr_m.*model.m
		ELBO_post = elogpmu(model) + elogptheta(model,mb)
		if (ELBO_post<ELBO_pre)
			println("have decrease in ELBO in updatem")
			break
		end

		#
		ELBO_pre = elogpLambda(model) + elogptheta(model,mb) - (elogqLambda(model))
		println(ELBO_pre)
		updateL!(model, mb)
		model.L = model.L_old.*(1.0-lr_L)+lr_L*model.L
		ELBO_post = elogpLambda(model) + elogptheta(model,mb) - (elogqLambda(model))
		println(model.L)
		println(model.L_old)
		println(ELBO_post)
		if (ELBO_post<ELBO_pre)
			println("have decrease in ELBO in updateL")
			println(ELBO_post-ELBO_pre)
			break
		end


		ELBO_pre=elogpbeta(model)+elogpnetwork(model,mb)-(elogqbeta(model))
		updateb0!(model, mb)
		model.b0 = model.b0_old.*(1.0-lr_b)+lr_b.*model.b0
		println(model.b0)
		println(model.b0_old)
		ELBO_post=elogpbeta(model)+elogpnetwork(model,mb)-(elogqbeta(model))
		if (ELBO_post<ELBO_pre)
			println("have decrease in ELBO in updateb0")
			println(ELBO_post-ELBO_pre)
			break
		end
		ELBO_pre=elogpbeta(model)+elogpnetwork(model,mb)-(elogqbeta(model))
		updateb1!(model, mb)
		model.b1 = model.b1_old.*(1.0-lr_b)+lr_b.*model.b1
		println(model.b1)
		println(model.b1_old)
		ELBO_post=elogpbeta(model)+elogpnetwork(model,mb)-(elogqbeta(model))
		if (ELBO_post<ELBO_pre)
			println("have decrease in ELBO in updateb1")
			println(ELBO_post-ELBO_pre)
			break
		end


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
		# niter=500; ntol=1e-5;
		# for a in collect(mb.mballnodes)
		#
		# 	updatemua!(model, a, niter, ntol,mb)
		# 	lr_mu[a] = 1.0/((1.0+Float64(mu_curr[a]))^.9)##could be  a macro
		# 	mu_curr[a] += 1
		#   model.μ_var[a,:] = model.μ_var_old[a,:]*(1.0-lr_mu[a])+lr_mu[a]*model.μ_var[a,:]
		# endf = open("data/messages.txt","w")
		###############SEPARATE ELBO TESTING###########
		###############################################
		ELBO_L = elogpLambda(model) + elogptheta(model,mb) - (elogqLambda(model))
		ELBO_m = elogpmu(model) + elogptheta(model,mb)
		ELBO_M =  elogpmu(model) + elogptheta(model,mb) - (elogqmu(model))
		ELBO_μa = elogptheta(model,mb)+elogpzlout(model,mb)+elogpzlin(model,mb)+elogpznlout(model,mb)+	elogpznlin(model,mb)
		ELBO_Λa = elogptheta(model,mb)+elogpzlout(model,mb)+elogpzlin(model,mb)+elogpznlout(model,mb)+	elogpznlin(model,mb)-(elogqtheta(model))
		ELBO_b0 =elogpbeta(model)+elogpnetwork(model,mb)-(elogqbeta(model))
		ELBO_b1 =elogpbeta(model)+elogpnetwork(model,mb)-(elogqbeta(model))
		ELBO_ϕlout =elogpzlout(model,mb)+elogpnetwork(model,mb)-(elogqzl(model))
		ELBO_ϕlin = elogpzlin(model,mb)+elogpnetwork(model,mb)-(elogqzl(model))
		ELBO_ϕnlout =elogpznlout(model,mb)+elogpnetwork(model,mb)-(elogqznl(model))
		ELBO_ϕnlin =elogpznlin(model,mb)+elogpnetwork(model,mb)-(elogqznl(model))

		push!(ELBO_Lrecord, ELBO_L)
		push!(ELBO_mrecord, ELBO_m)
		push!(ELBO_Mrecord, ELBO_M)
		push!(ELBO_μarecord, ELBO_μa)
		push!(ELBO_Λarecord, ELBO_Λa)
		push!(ELBO_b0record, ELBO_b0)
		push!(ELBO_b1record, ELBO_b1)
		push!(ELBO_ϕloutrecord, ELBO_ϕlout)
		push!(ELBO_ϕlinrecord, ELBO_ϕlin)
		push!(ELBO_ϕnloutrecord, ELBO_ϕnlout)
		push!(ELBO_ϕnlinrecord, ELBO_ϕnlin)



		if (ELBO_L-ELBO_Lold) < 0
			write(f, "ELBO_L decreased\n")
			# println("Whaaaat L")
			# break;
		end
		if (ELBO_m-ELBO_mold) < 0
			write(f, "ELBO_m decreased\n")
			# println("Whaaaat m")
			# break;
		end
		if (ELBO_M-ELBO_Mold) < 0
			write(f, "ELBO_M decreased\n")
			# println("Whaaaat M")
			# break
		end
		if (ELBO_μa-ELBO_μaold) < 0
			write(f, "ELBO_μa decreased\n")
			# println("Whaaaat μa")
			# break;
		end
		if (ELBO_Λa-ELBO_Λaold) < 0
			write(f, "ELBO_Λa decreased\n")
			# println("Whaaaat Λa")
			# break
		end
		if (ELBO_b0-ELBO_b0old) < 0
			write(f, "ELBO_b0 decreased\n")
			# println("Whaaaat b0")
			# break;
		end
		if (ELBO_b1-ELBO_b1old) < 0
			write(f, "ELBO_b1 decreased\n")
			# println("Whaaaat b1")
			# break;
		end
		if (ELBO_ϕlout-ELBO_ϕloutold) < 0
			write(f, "ELBO_ϕlout decreased\n")
			# println("Whaaaat ϕlout")
			# break;
		end
		if (ELBO_ϕlin-ELBO_ϕlinold) < 0
			write(f, "ELBO_ϕlin decreased\n")
			# println("Whaaaat ϕlin")
			# break;
		end
		if (ELBO_ϕnlout-ELBO_ϕnloutold) < 0
			write(f, "ELBO_ϕnlout decreased\n")
			# println("Whaaaat ϕnlout")
			# break;
		end
		if (ELBO_ϕnlin-ELBO_ϕnlinold) < 0
			write(f, "ELBO_ϕnlin decreased\n")
			# println("Whaaaat ϕnlin")
			# break;
		end
		ELBO_Lold = deepcopy(ELBO_L)
		ELBO_mold = deepcopy(ELBO_m)
		ELBO_Mold = deepcopy(ELBO_M)
		ELBO_μaold = deepcopy(ELBO_μa)
		ELBO_Λaold = deepcopy(ELBO_Λa)
		ELBO_b0old = deepcopy(ELBO_b0)
		ELBO_b1old = deepcopy(ELBO_b1)
		ELBO_ϕloutold = deepcopy(ELBO_ϕlout)
		ELBO_ϕlinold = deepcopy(ELBO_ϕlin)
		ELBO_ϕnloutold = deepcopy(ELBO_ϕnlout)
		ELBO_ϕnlinold = deepcopy(ELBO_ϕnlin)
		###############SEPARATE ELBO TESTING###########
		###############################################
		########################################################
		########################################################
		###############
		checkelbo = (i % elboevery == 0)
		if checkelbo || i == 1
			computeelbo!(model, mb)
			print(i);print("-ElBO:");println(model.elbo)
			print("elbo improvement:");
			increase=isinf(model.oldelbo)?65535.0:(model.elbo-model.oldelbo)/model.oldelbo;
			println(increase);
			# if increase < 0 && i > 10
			#  	break;
			# end



			model.oldelbo=deepcopy(model.elbo)
			push!(model.elborecord, model.elbo)
		end
		switchrounds = !switchrounds
		# i=i+1
	end
	close(f)
	Plots.plot(1:length(model.elborecord),model.elborecord)
end
isposdef(.5.*(model.L+model.L'))
###DO ONE BY ONE
#FOR PHIS ALL ELBOS CAN DECREASE, THE REASON IS NOT NECESSARILY FOR EARLY STAGES WE DO CRAP
##ALSO IT COULD BE BECAUSE OF THE BACK AND FORTH ORDER OF UPDATES OF PHI OUT AND IN AND THE FACT THAT
## WE UPDATE PHI LINKS AND PHI NONLINK AS COMPARED TO PHI OUT/IN LINK OR NONLINK
## WE ALSO USE ANOTHER JENSEN INEQUALITY FOR THE E[LSE].
