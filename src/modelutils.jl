
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
function mbsampling!(mb::MiniBatch,model::LNMMSB )
	mbcount  = 0
	lcount = 0
	while mbcount < model.mbsize
		lcount = 0
		a = 1+floor(Int64,model.N*rand())
		push!(mb.mballnodes, a)
		Bsink=sinks(model.network, a, model.N)#length is fadj
		Bsrc=sources(model.network, a, model.N)#length is badj

		for b1 in Bsink
			if !(Dyad(a,b1) in collect(keys(model.ho_dyaddict)))
				l = Link(a,b1,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
				push!(mb.mblinks, l)
				# push!(mb.mballnodes, a)
				# push!(mb.mballnodes, b1)
				lcount +=1
			end
		end
		for b2 in Bsrc
			if !(Dyad(b2,a) in collect(keys(model.ho_dyaddict)))
				l = Link(b2,a,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
				push!(mb.mblinks, l)
				# push!(mb.mballnodes, a)
				# push!(mb.mballnodes, b2)
				lcount +=1
			end
		end

		nlcount = 0
		while nlcount < lcount
			b=1+floor(Int64,model.N*rand())
			r = rand()
			if r  < .5
				if !(Dyad(a,b) in collect(keys(model.ho_dyaddict))) && !(isalink(model.network, a, b))
					nl = NonLink(a,b,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
					push!(mb.mbnonlinks, nl)
					if !haskey(mb.mbfnadj, a)
						mb.mbfnadj[a] = get(mb.mbfnadj, a, Vector{Int64}())
					end
					push!(mb.mbfnadj[a],b)
					nlcount+=1
				end
			else
				if !(Dyad(b,a) in collect(keys(model.ho_dyaddict))) && !(isalink(model.network, b, a))
					nl = NonLink(b,a,(1.0/model.K)*ones(Float64, model.K),(1.0/model.K)*ones(Float64, model.K))
					push!(mb.mbnonlinks, nl)
					if !haskey(mb.mbbnadj, a)
						mb.mbbnadj[a] = get(mb.mbbnadj, a, Vector{Int64}())
					end
					push!(mb.mbbnadj[a],b)
					# push!(mb.mballnodes, b)
					nlcount+=1
				end
			end
		end
		mbcount +=1
	end
	model.mbids = collect(mb.mballnodes)[:]
end

function preparedata(model::LNMMSB)
	setholdout(model)
	train_degree!(model)
	train_ss!(model)
end
println()
