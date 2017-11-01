
function setholdout(model::LNMMSB)
	# sample model.nho from nonzeros of model.network
	countlink = 0
	countnonlink = 0
	num_nonzeros=nnz(model.network)
	A,B,Val=findnz(model.network)
	while countlink <= model.nho
		spidx = 1+floor(Int64,num_nonzeros*rand())
		a,b = A[spidx], B[spidx]
		d = Dyad(a,b)
		if !haskey(model.ho_dyaddict, d)
			model.ho_dyaddict[d] = get(model.ho_dyaddict, d, true)
		end
		l = Link(a,b,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
		if !haskey(model.ho_linkdict, l)
			model.ho_linkdict[l] = get(model.ho_linkdict, l, true)
			countlink+=1
		end
	end
	while countnonlink <= model.nho
		a = 1+floor(Int64,model.N*rand())
		b = 1+floor(Int64,model.N*rand())
		if !isalink(model.network,a, b)
			d = a!=b ? Dyad(a,b) : continue
			if !haskey(model.ho_dyaddict, d)
				model.ho_dyaddict[d] = get(model.ho_dyaddict, d, true)
			end
			nl = NonLink(a,b,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
			if !haskey(model.ho_nlinkdict, nl)
			model.ho_nlinkdict[nl] = get(model.ho_nlinkdict, nl, true)
				countnonlink  += 1
			end
		end
	end
	println("holdout maps created")
end
function train_ss!(model::LNMMSB)
	for a in 1:model.N
		Bsink=sinks(model.network, a, model.N)#length is fadj
		Bsrc=sources(model.network, a, model.N)#length is badj
		xsink=Int64[]
		xsrc=Int64[]

		for b1 in Bsink
			if (Dyad(a,b1) in collect(keys(model.ho_linkdict)))
				push!(xsink,b1)
			end
		end
		for b2 in Bsrc
			if (Dyad(b2,a) in collect(keys(model.ho_linkdict)))
				push!(xsrc,b2)
			end
		end
		model.train_sinks[a] = setdiff(Bsink, xsink)
		model.train_sources[a] = setdiff(Bsrc, xsrc)
	end
	println("training and minibatch sink and sources figured")
end


function train_degree!(model::LNMMSB)
	for a in 1:model.N
		Bsink=sinks(model.network, a, model.N)#length is fadj
		Bsrc=sources(model.network, a, model.N)#length is badj
		model.train_out[a] = length(Bsink)
		model.train_in[a] =  length(Bsrc)
		for b1 in Bsink
			if (Dyad(a,b1) in collect(keys(model.ho_linkdict)))
				model.train_out[a]-=1
			end
		end
		for b2 in Bsrc
			if (Dyad(b2,a) in collect(keys(model.ho_linkdict)))
				model.train_in[a]-=1
			end
		end

	end
	println("outdeg and indeg of train and mb are figured")
end
##think about speeding this up-1ms not good
function mbsampling!(mb::MiniBatch,model::LNMMSB)
	mbcount  = 0
	lcount = 0
	while mbcount < model.mbsize
		lcount = 0
		a = 1+floor(Int64,model.N*rand())
		while a in mb.mballnodes
			a = 1+floor(Int64,model.N*rand())
		end

		push!(mb.mballnodes, a)
		Bsink=sinks(model.network, a, model.N)#length is fadj
		Bsrc=sources(model.network, a, model.N)#length is badj

		for b1 in Bsink
			if !(Dyad(a,b1) in collect(keys(model.ho_dyaddict)))
				l = Link(a,b1,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
				if !(l in mb.mblinks)
					push!(mb.mblinks, l)
					lcount +=1
				end
			end
		end
		for b2 in Bsrc
			if !(Dyad(b2,a) in collect(keys(model.ho_dyaddict)))
				l = Link(b2,a,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
				if !(l in mb.mblinks)
					push!(mb.mblinks, l)
					lcount +=1
				end
			end
		end

		nlcount = 0
		while nlcount < 2*lcount
			b=1+floor(Int64,model.N*rand())
			r = rand()
			if r  < .5
				if !(Dyad(a,b) in collect(keys(model.ho_dyaddict))) && !(isalink(model.network, a, b))
					nl = NonLink(a,b,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
					if !(nl in mb.mbnonlinks)
						push!(mb.mbnonlinks, nl)
						if !haskey(mb.mbfnadj, a)
							mb.mbfnadj[a] = get(mb.mbfnadj, a, Vector{Int64}())
						end
						push!(mb.mbfnadj[a],b)
						if !haskey(mb.mbbnadj, b)
							mb.mbbnadj[b] = get(mb.mbbnadj, b, Vector{Int64}())
						end
						push!(mb.mbbnadj[b],a)
						nlcount+=1
					end
				end
			else
				if !(Dyad(b,a) in collect(keys(model.ho_dyaddict))) && !(isalink(model.network, b, a))
					nl = NonLink(b,a,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
					if !(nl in mb.mbnonlinks)
						push!(mb.mbnonlinks, nl)
						if !haskey(mb.mbfnadj, b)
							mb.mbfnadj[b] = get(mb.mbfnadj, b, Vector{Int64}())
						end
						push!(mb.mbfnadj[b],a)
						if !haskey(mb.mbbnadj, a)
							mb.mbbnadj[a] = get(mb.mbbnadj, a, Vector{Int64}())
						end
						push!(mb.mbbnadj[a],b)
						nlcount+=1
					end
				end
			end
		end
		mbcount +=1
	end
	model.mbids[:] = collect(mb.mballnodes)[:]
end

#better to call this all the time
function mbsampling!(mb::MiniBatch,model::LNMMSB, isfullsample::Bool)
	if !isfullsample
		mbsampling!(mb::MiniBatch,model::LNMMSB)
	else
		mb.mballnodes = Set(collect(1:model.mbsize))
		model.mbids = collect(mb.mballnodes)
		for a in 1:model.mbsize
			Bsink=sinks(model.network, a, model.N)#length is fadj
			Bsrc=sources(model.network, a, model.N)#length is badj

			for b1 in Bsink
				if !(Dyad(a,b1) in collect(keys(model.ho_dyaddict)))
					# l = Link(a,b1,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))

					l = Link(a,b1,softmax(model.μ_var[a,:]),softmax(model.μ_var[b1,:]))
					if !(l in mb.mblinks)
						push!(mb.mblinks, l)
					end
				end
			end
			for b2 in Bsrc
				if !(Dyad(b2,a) in collect(keys(model.ho_dyaddict)))

					l = Link(b2,a,softmax(model.μ_var[b2,:]),softmax(model.μ_var[a,:]))
					if !(l in mb.mblinks)
						push!(mb.mblinks, l)
					end
				end
			end
		end
		for a in 1:model.mbsize
			for b in 1:model.mbsize
				if a != b
					if !(Dyad(a,b) in collect(keys(model.ho_dyaddict))) && !(isalink(model.network, a, b))
						nl = NonLink(a,b,softmax(model.μ_var[a,:]),softmax(model.μ_var[b,:]))
						if !(nl in mb.mbnonlinks)
							push!(mb.mbnonlinks, nl)
							if !haskey(mb.mbfnadj, a)
								mb.mbfnadj[a] = get(mb.mbfnadj, a, Vector{Int64}())
							end
							push!(mb.mbfnadj[a],b)
							if !haskey(mb.mbbnadj, b)
								mb.mbbnadj[b] = get(mb.mbbnadj, b, Vector{Int64}())
							end
							push!(mb.mbbnadj[b],a)
						end
					end
					if !(Dyad(b,a) in collect(keys(model.ho_dyaddict))) && !(isalink(model.network, b, a))
						nl = NonLink(b,a,softmax(model.μ_var[b,:]),softmax(model.μ_var[a,:]))
						if !(nl in mb.mbnonlinks)
							push!(mb.mbnonlinks, nl)
							if !haskey(mb.mbfnadj, b)
								mb.mbfnadj[b] = get(mb.mbfnadj, b, Vector{Int64}())
							end
							push!(mb.mbfnadj[b],a)
							if !haskey(mb.mbbnadj, a)
								mb.mbbnadj[a] = get(mb.mbbnadj, a, Vector{Int64}())
							end
							push!(mb.mbbnadj[a],b)
						end
					end
				end
			end
		end
	end
end


function train_sample!(train::MiniBatch, model::LNMMSB)


	train.mballnodes = Set(1:model.N)
	for a in 1:model.N
		Bsink=sinks(model.network, a, model.N)#length is fadj
		Bsrc=sources(model.network, a, model.N)#length is badj

		for b1 in Bsink
			if !(Dyad(a,b1) in collect(keys(model.ho_dyaddict)))
				# l = Link(a,b1,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))

				l = Link(a,b1,softmax(model.μ_var[a,:]),softmax(model.μ_var[b1,:]))
				if !(l in train.mblinks)
					push!(train.mblinks, l)
				end
			end
		end
		for b2 in Bsrc
			if !(Dyad(b2,a) in collect(keys(model.ho_dyaddict)))

				l = Link(b2,a,softmax(model.μ_var[b2,:]),softmax(model.μ_var[a,:]))
				if !(l in train.mblinks)
					push!(train.mblinks, l)
				end
			end
		end
	end
	for a in 1:model.N
		for b in 1:model.N
			if a != b
				if !(Dyad(a,b) in collect(keys(model.ho_dyaddict))) && !(isalink(model.network, a, b))
					nl = NonLink(a,b,softmax(model.μ_var[a,:]),softmax(model.μ_var[b,:]))
					if !(nl in train.mbnonlinks)
						push!(train.mbnonlinks, nl)
						if !haskey(train.mbfnadj, a)
							train.mbfnadj[a] = get(train.mbfnadj, a, Vector{Int64}())
						end
						push!(train.mbfnadj[a],b)
						if !haskey(train.mbbnadj, b)
							train.mbbnadj[b] = get(train.mbbnadj, b, Vector{Int64}())
						end
						push!(train.mbbnadj[b],a)
					end
				end
				if !(Dyad(b,a) in collect(keys(model.ho_dyaddict))) && !(isalink(model.network, b, a))
					nl = NonLink(b,a,softmax(model.μ_var[b,:]),softmax(model.μ_var[a,:]))
					if !(nl in train.mbnonlinks)
						push!(train.mbnonlinks, nl)
						if !haskey(train.mbfnadj, b)
							train.mbfnadj[b] = get(train.mbfnadj, b, Vector{Int64}())
						end
						push!(train.mbfnadj[b],a)
						if !haskey(train.mbbnadj, a)
							train.mbbnadj[a] = get(train.mbbnadj, a, Vector{Int64}())
						end
						push!(train.mbbnadj[a],b)
					end
				end
			end
		end
	end

end




function preparedata(model::LNMMSB)
	setholdout(model)
	train_degree!(model)
	train_ss!(model)
end
##Better set model.K either true K or number of communities length(communities)
function init_mu(model::LNMMSB, communities::Dict{Int64, Vector{Int64}}, onlyK::Int64)
  Belong = Dict{Int64, Vector{Int64}}()
  model.μ_var = 1e-10*ones(Float64, (N, onlyK))
  for i in 1:N
    if !haskey(Belong, i)
      Belong[i] = get(Belong, i, Int64[])
    end
    for k in 1:length(communities)
      if i in communities[k]
        push!(Belong[i],k)
      end
    end
    if length(Belong[i]) == 0
      push!(Belong[i], sample(1:length(communities)))
      model.μ_var[i,Belong[i]] = .9
    elseif length(Belong[i]) == 1
      model.μ_var[i,Belong[i]] = .9
    else
      val = .9/length(Belong[i])
      for z in Belong[i]
        model.μ_var[i,z] = val
      end
    end
    s = zero(Float64)
    for k in 1:length(communities)
      s+= model.μ_var[i,k]
    end
    for k in 1:length(communities)
      model.μ_var[i,k] = model.μ_var[i,k]/s
    end
  end
  for i in 1:N
    model.μ_var[i,:] = log.(model.μ_var[i,:])
  end
end



print();
