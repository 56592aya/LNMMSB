# using SpecialFunctions
using GradDescent
using ForwardDiff

function updatephilout!(model::LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Bool, link::Link)
	for k in 1:model.K
		link.ϕout[k] = model.μ_var[link.src,k] + link.ϕin[k]*(model.Elogβ0[k]-log(EPSILON))
	end
	r = logsumexp(link.ϕout)
	link.ϕout[:] = exp.(link.ϕout[:] .- r)[:]
	# for k in 1:model.K
	# 	model.ϕloutsum[link.src,k] += link.ϕout[k]##remember to zero this when needed
	# end
end
# link = Link(1,2, _init_ϕ,_init_ϕ)
function updatephilin!(model::LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Bool, link::Link)
	for k in 1:model.K
		link.ϕin[k] = model.μ_var[link.dst,k] + link.ϕout[k]*(model.Elogβ0[k]-log(EPSILON))
	end
	r=logsumexp(link.ϕin)
	link.ϕin[:] = exp.(link.ϕin[:] .- r)[:]
	# for k in 1:model.K
	# 	model.ϕlinsum[link.src,k] += link.ϕin[k]##remember to zero this when needed
	# end
end
function updatephinlout!(model::LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Bool, nlink::NonLink)
	for k in 1:model.K
		nlink.ϕout[k] = model.μ_var[nlink.src,k] + nlink.ϕin[k]*(model.Elogβ1[k]-log1p(-EPSILON))
	end
	r = logsumexp(nlink.ϕout)
	nlink.ϕout[:] = exp.(nlink.ϕout[:] .- r)[:]
	# for k in 1:model.K
	# 	model.ϕnloutsum[nlink.src,k] += nlink.ϕout[k]##remember to zero this when needed
	# end
end
# link = Link(1,2, _init_ϕ,_init_ϕ)
function updatephinlin!(model::LNMMSB, mb::MiniBatch, early::Bool, switchrounds::Bool, nlink::NonLink)
	for k in 1:model.K
		nlink.ϕin[k] = model.μ_var[nlink.dst,k] + nlink.ϕout[k]*(model.Elogβ1[k]-log1p(-EPSILON))
	end
	r=logsumexp(nlink.ϕin)
	nlink.ϕin[:] = exp.(nlink.ϕin[:] .- r)[:]
	# for k in 1:model.K
	# 	model.ϕnlinsum[nlink.src,k] += nlink.ϕin[k]##remember to zero this when needed
	# end
end

## Need to repeat updatephilout!,updatephilin!,,updatephinlout!,updatephinlin! until
## convergence before the until convergence loop these variables need to be reinitialized
## all phis pertaining to the minibatch links and nonlinks and
## model.ϕloutsum,model.ϕlinsum,model.ϕnlinsum, model.ϕnloutsum


function updatesimulμΛ!(model::LNMMSB, a::Int64,mb::MiniBatch)

	model.μ_var_old[a,:]=deepcopy(model.μ_var[a,:])
	model.Λ_var_old[a,:]=deepcopy(model.Λ_var[a,:])

	sumb = model.train_outdeg[a]+model.train_indeg[a]+length(mb.mbfnadj[a])+length(mb.mbbnadj[a])
	μ_var = deepcopy(model.μ_var[a,:])
	Λ_ivar = deepcopy(1.0./model.Λ_var[a,:])
	ltemp = [log(Λ_ivar[k]) for k in 1:model.K]
	sfx(μ_var)=softmax(μ_var .+.5.*exp.(ltemp))
	x = sfx(μ_var)
	scaler1=(length(model.trainfnadj[a])/length(mb.mbfnadj[a]))
	scaler2 = (length(model.trainbnadj[a])/length(mb.mbbnadj[a]))
	dfunc(μ_var) = -model.l.*model.L*(μ_var-model.m) +
	(model.ϕloutsum[a,:]+model.ϕlinsum[a,:]+scaler1*model.ϕnloutsum[a,:]+scaler2*model.ϕnlinsum[a,:])-
	sumb.*x


	func(μ_var,ltemp) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
	(model.ϕloutsum[a,:]'+model.ϕlinsum[a,:]'+scaler1*model.ϕnloutsum[a,:]'+scaler2*model.ϕnlinsum[a,:]')*μ_var-
	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp

	func1(μ_var) = -.5*model.l*((μ_var-model.m)'*model.L*(μ_var-model.m))+
	(model.ϕloutsum[a,:]'+model.ϕlinsum[a,:]'+scaler1*model.ϕnloutsum[a,:]'+scaler2*model.ϕnlinsum[a,:]')*μ_var-
	sumb*(log(ones(model.K)'*exp.(μ_var+.5.*exp.(ltemp))))

	func2(ltemp) =-.5*model.l*(diag(model.L)'*exp.(ltemp))+.5*ones(Float64, model.K)'*ltemp-sumb*(log(ones(model.K)'*exp.(model.μ_var[a,:]+.5.*exp.(ltemp))))

	opt1 = Adagrad()
	opt2 = Adagrad()
	oldval = func(μ_var, ltemp)
	g1 = -dfunc(μ_var)
	δ1 = update(opt1,g1)
	g2 = -ForwardDiff.gradient(func2, ltemp)
	δ2 = update(opt2,g2)
	newval = func(μ_var-δ1, ltemp-δ2)
	μ_var-=δ1
	ltemp-=δ2
	##Have to have this newval and oldval outside to not take the first step
	for j in 1:10

		g1 = -dfunc(μ_var)
		δ1 = update(opt1,g1)
		g2 = -ForwardDiff.gradient(func2, ltemp)
		δ2 = update(opt2,g2)
		μ_var-=δ1
		ltemp-=δ2
		# oldval=newval
		# newval = func(μ_var,ltemp)
		# end
	end
	# while oldval > newval
	# 	if isapprox(newval, oldval)
	# 		break;
	# 	else
	# 		g1 = -dfunc(μ_var)
	# 		δ1 = update(opt1,g1)
	# 		g2 = -ForwardDiff.gradient(func2, ltemp)
	# 		δ2 = update(opt2,g2)
	# 		μ_var-=δ1
	# 		ltemp-=δ2
	# 		oldval=newval
	# 		newval = func(μ_var,ltemp)
	# 	end
	# end
	model.μ_var[a,:]=μ_var
	model.Λ_var[a,:]=1.0./exp.(ltemp)
	print();
#######
end
function updateM!(model::LNMMSB,mb::MiniBatch)
	##Only to make it MB dependent
	model.M_old = deepcopy(model.M)
	model.M = ((model.l*model.N).*model.L)+model.M0
end
#updateM!(model,mb)
function updatem!(model::LNMMSB, mb::MiniBatch)
	s = zeros(Float64, model.K)
	for a in mb.mbnodes
		s.+=model.μ_var[a,:]
	end
	model.m_old = deepcopy(model.m)
	scaler=(convert(Float64,model.N)/convert(Float64,model.mbsize))
	model.m=inv(model.M)*(model.M0*model.m0+scaler*model.l).*model.L*s
end
#updatem!(model, mb)
function updatel!(model::LNMMSB)
	##should be set in advance, not needed in the loop
	model.l = model.l0+convert(Float64,model.N)
end
#updatel!(model,mb)
function updateL!(model::LNMMSB, mb::MiniBatch)
	s = zero(Float64)
	for a in mb.mbnodes
		s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+diagm(1.0./model.Λ_var[a,:])
	end
	s=(convert(Float64,model.N)/convert(Float64,model.mbsize)).*s
	s+=inv(model.L0)+model.N.*inv(model.M)
	model.L_old = deepcopy(model.L)
	model.L = inv(s)
end
#updateL!(model, mb)
function updateb0!(model::LNMMSB, mb::MiniBatch,train_links_num::Int64)
	model.b0_old = deepcopy(model.b0)
	#replace #length(train.mblinks)
	train_links_num=convert(Float64, train_links_num)
	scaler=(train_links_num/convert(Float64,length(mb.mblinks)))
	@assert !isequal(model.ϕlinoutsum[:], zeros(Float64, model.K))
	model.b0[:] = model.η0.+scaler.*model.ϕlinoutsum[:]
end
#updateb0!(model, mb)
function updateb1!(model::LNMMSB, mb::MiniBatch,train_nlinks_num::Int64)
	model.b1_old = deepcopy(model.b1)
	#replace #length(train.mblinks)
	train_nlinks_num = convert(Float64,train_nlinks_num)
	scaler = (train_nlinks_num/convert(Float64,length(mb.mbnonlinks)))
	@assert !isequal(model.ϕnlinoutsum[:], zeros(Float64, model.K))
	model.b1[:] = model.η1.+scaler.*model.ϕnlinoutsum[:]
end

#updateb1!(model, mb)



function mcsampler(model,mean, diagprec, num)
	num=100
	mean=[2.0,1.0,3.0,1.0]
	diagprec=[2.0,.3,4.0, 1.5]
	vec = zeros(Float64, (num, model.K))
	for i in 1:num
		e = rand(MvNormal(eye(Float64, model.K)),1)
		vec[i,:] = mean+(1.0./diagprec).*(e)
	end
	return vec

end
function estimate_βs(model::LNMMSB, mb::MiniBatch)
	model.est_β=model.b0./(model.b0.+model.b1)
end


function estimate_θs!(model::LNMMSB, mb::MiniBatch)
	for a in mb.mbnodes
		# model.est_θ[a,:]=exp(model.μ_var[a,:])./sum(exp(model.μ_var[a,:]))#
		model.est_θ[a,:]=softmax(model.μ_var[a,:])
	end
	# model.μ_var[1,:]
	# softmax!(model.μ_var[1,:])
	# model.μ_var[1,:]
	# model.est_θ[1,:]=softmax!(model.μ_var[1,:])
	# model.μ_var
	# model.est_θ
	# softmax!()
	# model.μ_var[1,:]
	# exp(model.μ_var[1,1])/sum(exp(model.μ_var[1,:]))
	# exp(model.μ_var[1,4])/sum(exp(model.μ_var[1,:]))
	model.est_θ[mb.mbnodes,:]
end

function estimate_μs(model::LNMMSB, mb::MiniBatch)
	model.est_μ = reshape(reduce(mean, model.μ_var, 1)',model.K)
end

function estimate_Λs(model::LNMMSB, mb::MiniBatch)
end

##How to determine belonging to a community based on the membership vector???
function computeNMI(x::Matrix{Float64}, y::Matrix{Float64}, communities::Dict{Int64, Vector{Int64}}, threshold::Float64)
	open("./file2", "w") do f
	  for k in 1:size(x,2)
	    for i in 1:size(x,1)
	      if x[i,k] > threshold #3.0/size(x,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end
	open("./file1", "w") do f
	  for k in 1:size(y,2)
	    for i in 1:size(y,1)
	      if y[i,k] > threshold #3.0/size(y,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end

	open("./file3", "w") do f
	  for k in 1:length(communities)
	  	if length(communities[k]) <= 10
			continue;
		end
	    for e in communities[k]
        	write(f, "$e ")
	    end
	    write(f, "\n")
	  end
	end
	println("NMI of estimated vs truth")
	run(`src/cpp/NMI/onmi file2 file1`)
	println("NMI of estimated vs init")
	run(`src/cpp/NMI/onmi file2 file3`)
	println("NMI of truth vs init")
	run(`src/cpp/NMI/onmi file1 file3`)

end
function computeNMI_med(x::Matrix{Float64}, y::Matrix{Float64}, communities::Dict{Int64, Vector{Int64}}, threshold::Float64)
	open("./file2", "w") do f
	  for k in 1:size(x,2)
	    for i in 1:size(x,1)
	      if x[i,k] > threshold #3.0/size(x,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end
	open("./file1", "w") do f
	  for k in 1:size(y,2)
	    for i in 1:size(y,1)
	      if y[i,k] > threshold #3.0/size(y,2)
	        write(f, "$i ")
	      end
	    end
	    write(f, "\n")
	  end
	end
	nmi = read(`src/cpp/NMI/onmi file2 file1`, String)
	nmi=parse(Float64,nmi[6:(end-1)])
	return nmi
end

function computeNMI2(model::LNMMSB, mb::MiniBatch)
	# Px = mean(model.est_θ, 1)
	K = size(true_thetas,2)
	true_thetas = readdlm("data/true_thetas.txt")
	Py = reshape(mean(true_thetas, 1), K)
	Px = reshape(mean(x, 1),K)
	a = Matrix2d{Float64}(K, K)
	b = Matrix2d{Float64}(K, K)
	c = Matrix2d{Float64}(K, K)
	d = Matrix2d{Float64}(K, K)

	for i in 1:K
		for j in 1:K
			a[i,j] = .5*model.N*((1.0-Px[i])+(1.0-Py[j]))
			b[i,j] = .5*model.N*((1.0-Px[i])+Py[j])
			c[i,j] = .5*model.N*(Px[i]+(1.0-Py[j]))
			d[i,j] = .5*model.N*(Px[i]+Py[j])
		end
	end
	HxiCyj = Matrix2d{Float64}(K, K)
	h(w,n) = -w*log(2, w/n)
	for i in 1:K
		for j in 1:K
			HxiCyj[i,j] = (h(a[i,j], model.N)+h(d[i,j], model.N))>(h(b[i,j], model.N)+h(c[i,j], model.N))? h(a[i,j], model.N)+h(b[i,j], model.N)+h(c[i,j], model.N)+h(d[i,j], model.N)-
			h(b[i,j]+d[i,j], model.N)-h(a[i,j]+c[i,j], model.N) :h(c[i,j]+d[i,j], model.N)+h(a[i,j]+b[i,j], model.N)
		end
	end
	HxiCy = Vector{Float64}(K)
	for i in 1:K
		HxiCy[i] = minimum(HxiCyj[i,:])
	end
	HxCy = sum(HxiCy)
end
function edge_likelihood(model::LNMMSB,pair::Dyad, β_est::Vector{Float64})
    s = zero(Float64)
    S = Float64
    prob = zero(Float64)
	src = pair.src
	dst = pair.dst
    for k in 1:model.K
        if isalink(model, "network",src, dst)
            prob += (model.μ_var[src,k]/sum(model.μ_var[src,:]))*(model.μ_var[dst,k]/sum(model.μ_var[dst,:]))*(β_est[k])
        else
            prob += (model.μ_var[src,k]/sum(model.μ_var[src,:]))*(model.μ_var[dst,k]/sum(model.μ_var[dst,:]))*(1.0-β_est[k])
        end
        s += (model.μ_var[src,k]/sum(model.μ_var[src,:]))*(model.μ_var[dst,k]/sum(model.μ_var[dst,:]))
    end

    if isalink(model, "network",src, dst)
        prob += (1.0-s)*EPSILON
    else
        prob += (1.0-s)*(1.0-EPSILON)
    end
    return log1p(-1.0+prob)::Float64
end
print("");
