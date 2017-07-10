
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
		if !haskey(ho_dyaddict, d)
			ho_dyaddict[d] = get(ho_dyaddict, d, true)
		end
		l = Link(a,b,zeros(Float64, model.K),zeros(Float64, model.K))
		if !haskey(ho_linkdict, l)
			ho_linkdict[l] = get(ho_linkdict, l, true)
			countlink+=1
		end
	end
	while countnonlink <= model.nho
		a = 1+floor(Int64,model.N*rand())
		b = 1+floor(Int64,model.N*rand())
		if !isalink(model.network,a, b)
			d = a!=b ? Dyad(a,b) : continue
			if !haskey(ho_dyaddict, d)
				ho_dyaddict[d] = get(ho_dyaddict, d, true)
			end
			nl = NonLink(a,b,zeros(Float64, model.K),zeros(Float64, model.K))
			if !haskey(ho_nlinkdict, nl)
			ho_nlinkdict[nl] = get(ho_nlinkdict, nl, true)
				countnonlink  += 1
			end
		end
	end
end
train_out = Dict{Int64,Int64}
train_in = Dict{Int64,Int64}
function train_degree!(train_out, train_in, model::LNMMSB)
	for a in 1:model.N
		Bsink=sinks(model.network, a, model.N)#length is fadj
		Bsrc=sources(model.network, a, model.N)#length is bad
		if !haskey(train_out, a)
			train_out[a] = get(train_out, a, length(Bsink))
		end
		if !haskey(train_in, a)
			train_in[a] = get(train_in, a, length(Bsrc))
		end
		for b1 in Bsink
			if (Dyad(a,b1) in collect(keys(ho_linkdict)))
				train_out[a]-=1
			end
		end
		for b2 in Bsrc
			if (Dyad(b2,a) in collect(keys(ho_linkdict)))
				train_in[b]-=1
			end
		end

	end
	train_out, train_in
end
function mbsampling(model::LNMMSB, mb::MiniBatch)
	mbcount  = 0
	lcount = 0
	while mbcount < model.mbsize
		lcount = 0
		a = 1+floor(Int64,model.N*rand())
		push!(mb.mballnodes, a)
		Bsink=sinks(model.network, a, model.N)#length is fadj
		Bsrc=sources(model.network, a, model.N)#length is badj

		for b1 in Bsink
			if !(Dyad(a,b1) in collect(keys(ho_dyaddict)))
				l = Link(a,b1,zeros(Float64, model.K),zeros(Float64, model.K))
				push!(mb.mblinks, l)
				# push!(mb.mballnodes, a)
				# push!(mb.mballnodes, b1)
				lcount +=1
			end
		end
		for b2 in Bsrc
			if !(Dyad(b2,a) in collect(keys(ho_dyaddict)))
				l = Link(b2,a,zeros(Float64, model.K),zeros(Float64, model.K))
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
				if !(Dyad(a,b) in collect(keys(ho_dyaddict))) && !(isalink(model.network, a, b))
					nl = NonLink(a,b,zeros(Float64, model.K),zeros(Float64, model.K))
					push!(mb.mbnonlinks, nl)
					if !haskey(mb.mbfnadj, a)
						mb.mbfnadj[a] = get(mb.mbfnadj, a, Vector{Int64}())
					end
					if !haskey(mb.mbbnadj, b)
						mb.mbbnadj[b] = get(mb.mbbnadj, b, Vector{Int64}())
					end
					push!(mb.mbfnadj[a],b)
					push!(mb.mbbnadj[b],a)
					nlcount+=1
				end
			else
				if !(Dyad(b,a) in collect(keys(ho_dyaddict))) && !(isalink(model.network, b, a))
					nl = NonLink(b,a,zeros(Float64, model.K),zeros(Float64, model.K))
					push!(mb.mbnonlinks, nl)
					if !haskey(mb.mbfnadj, b)
						mb.mbfnadj[b] = get(mb.mbfnadj, b, Vector{Int64}())
					end
					if !haskey(mb.mbbnadj, a)
						mb.mbbnadj[a] = get(mb.mbbnadj, a, Vector{Int64}())
					end
					push!(mb.mbfnadj[b],a)
					push!(mb.mbbnadj[a],b)
					# push!(mb.mballnodes, b)
					nlcount+=1
				end
			end
		end
		mbcount +=1
	end
	model.mbids = collect(mb.mballnodes)[:]
	println()
end
