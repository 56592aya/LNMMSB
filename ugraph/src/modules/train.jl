__precompile__()
module Train
export train
import Utils: MiniBatch, Link, NonLink, logsumexp, softmax
import TrainUtils: update_Elogβ!,updatel!,estimate_θs!
import TrainUtils: updatephil!,updatephinl!,updateμ!,updatem!,updateM!,updateL!
import TrainUtils: updateb0!,updateb1!,estimate_βs!
import ModelUtils: getSets!,test_sets,minibatch_set_srns,setup_mblnl!,update_sets!,init_mu,update_A!,preparedata2!
import Model:LNMMSB
import Distributions:Wishart
using Plots
using GraphPlot
function train(model::LNMMSB, mb::MiniBatch,iter::Int64, elboevery::Int64, communities::Dict{Int64,Array{Int64,1}})
	preparedata2!(model)
	for a in 1:model.N
		model.train_deg[a] = sum(model.network[a,:])
	end
	train_links_num=0
	train_nlinks_num = 0
	link_ratio = 0.0
	prev_ll = -1000000
	store_ll = Array{Float64, 1}()
	store_nmi = Array{Float64, 1}()
	first_converge = false
	model.elborecord = Vector{Float64}()
	model.elbo = 0.0
	model.oldelbo= -Inf
	train_links_num=sum(model.train_deg)/2
	train_nlinks_num=(model.N*model.N-model.N)/2- length(model.ho_dyads)-train_links_num
	link_ratio = convert(Float64, train_links_num)/convert(Float64,train_nlinks_num)
	_init_ϕ = deepcopy(ones(Float64, model.K).*1.0/model.K)
	link_thresh=.9
	init_mu(model,communities,model.K, model.N)##from Gopalan
	_init_μ = deepcopy(model.μ_var)
	model.μ_var = deepcopy(_init_μ)
	model.Λ_var = 100.0*ones(Float64, (model.N, model.K))
	_init_Λ = deepcopy(model.Λ_var)

	update_Elogβ!(model)
	updatel!(model)
	lr_M = 1.0
	lr_m = 1.0
	lr_L = 1.0
	lr_b = 1.0
	true_μ = readdlm("../../data/true_mu.txt")
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
	count_a = zeros(Float64, model.N)
	model.μ_var=deepcopy(_init_μ)
	estimate_θs!(model)
	threshold=.9
	getSets!(model, threshold,_init_μ)
	println()
	test_sets(model)
	for i in 1:iter
		#MB sampling, eveyr time we create an empty minibatch object
		mb=deepcopy(model.mb_zeroer)
		minibatch_set_srns(model, mb, model.N)
		model.mbids = deepcopy(mb.mbnodes)
		shuffled = shuffle!(collect(model.minibatch_set))
		setup_mblnl!(model, mb, shuffled,_init_ϕ)
		update_sets!(model, mb)
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
					updatephil!(model, mb, l,"check")
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
					updatephinl!(model, mb, nl, "check")

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
		#updating the mus for minibatch
		for a in mb.mbnodes
			count_a[a] += 1.0
			expectedvisits = (Float64(iter)/(4.0*Float64(model.N)/Float64(model.mbsize)))
			updateμ!(model, a, mb, "check", model.N)
			lr_μ[a] = (expectedvisits/(expectedvisits+(count_a[a]-1.0)))^(.5)
			model.μ_var[a,:] = view(model.μ_var_old, a, :).*(1.0.-lr_μ[a])+lr_μ[a].*view(model.μ_var, a, :)
		end
		estimate_θs!(model, mb)
		est_θ = deepcopy(model.est_θ)
		update_A!(model, mb, est_θ,threshold)
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
	end
	return (model, mb)
end
end
