
function setholdout!(model::LNMMSB,meth::String)
	if meth == "isns"
		_init_ϕ=(1.0/model.K)*ones(Float64, model.K)
		model.ho_dyads = Vector{Dyad}()
		model.ho_links = Vector{Link}()
		model.ho_nlinks= Vector{NonLink}()
		countlink = zero(Int64)
		countnonlink = zero(Int64)
		# sample model.nho from nonzeros of model.network
		num_nonzeros=nnz(model.network)
		A,B,Val=findnz(model.network)
		while countlink <= model.nho
			spidx = 1+floor(Int64,num_nonzeros*rand())
			a,b = A[spidx], B[spidx]
			d = Dyad(a,b)
			if d in model.ho_dyads
				continue;
			else
				push!(model.ho_dyads, d)
			end

			l = Link(a,b,_init_ϕ,_init_ϕ)
			if l in model.ho_links
				continue;
			else
				push!(model.ho_links, l)
				model.ho_fadj[l.src]+=1
				model.ho_badj[l.dst]+=1
				countlink+=1
			end
		end
		while countnonlink <= model.nho
			a = 1+floor(Int64,model.N*rand())
			b = 1+floor(Int64,model.N*rand())
			if model.network[a,b] == 0
				d = a!=b ? Dyad(a,b) : continue
				if d in model.ho_dyads
					continue;
				else
					push!(model.ho_dyads, d)
				end
				nl = NonLink(a,b,_init_ϕ,_init_ϕ)
				if nl in model.ho_nlinks
					continue;
				else
					push!(model.ho_nlinks, nl)
					model.ho_fnadj[nl.src]+=1
					model.ho_bnadj[nl.dst]+=1
					countnonlink  += 1
				end
			end
		end
		println("holdout maps created")
	end
end
function isalink(model::LNMMSB, place::String,a::Int64, b::Int64)
	ret = false
  	if place in ["network", "Network", "net", "Net"]
    	ret = model.network[a,b] == 1
	elseif  place in ["train", "Train"]
	 	ret = model.network[a,b] == 1 && !(Dyad(a,b) in model.ho_dyads)
  	elseif place in ["holdout", "ho", "Holdout"]
	  	ret = Dyad(a,b) in ho_links
  	else
    	error("don't know where you mean!")
  	end
  	ret
end
function isalink(model::LNMMSB, place::String,x...)
	x = x[1]
	a=x[1];b=x[2];
	ret = false
  	if place in ["network", "Network", "net", "Net"]
    	ret = model.network[a,b] == 1
	elseif  place in ["train", "Train"]
	 	ret = model.network[a,b] == 1 && !(Dyad(a,b) in model.ho_dyads)
  	elseif place in ["holdout", "ho", "Holdout"]
	  	ret = Dyad(a,b) in ho_links
  	else
    	error("don't know where you mean!")
  	end
  	ret
end

function issink(model::LNMMSB,place::String,curr::Int64, q::Int64)
  ret = false
  if place in ["network", "Network", "net", "Net"]
	  ret = model.network[curr,q] == 1
  elseif  place in ["train", "Train"]
	  ret = model.network[curr,q] == 1 && !(Dyad(curr,q) in model.ho_dyads)
  elseif place in ["holdout", "ho", "Holdout"]
	  ret = Dyad(curr,q) in ho_links
  else
	  error("don't know where you mean!")
  end
  ret
end
function issource(model::LNMMSB,place::String,curr::Int64, q::Int64)
	ret = false
    if place in ["network", "Network", "net", "Net"]
  	  ret = model.network[q,curr] == 1
    elseif  place in ["train", "Train"]
  	  ret = model.network[q,curr] == 1 && !(Dyad(q,curr) in model.ho_dyads)
    elseif place in ["holdout", "ho", "Holdout"]
  	  ret = Dyad(q,curr) in ho_links
    else
  	  error("don't know where you mean!")
    end
    ret
end
function sinks(model::LNMMSB,place::String,curr::Int64)
  [b for b in 1:model.N if isalink(model,place, curr, b)]
end
function sources(model::LNMMSB,place::String,curr::Int64)
  [b for b in 1:model.N if isalink(model, place,b, curr)]
end
function neighbors_(model::LNMMSB,place::String,curr::Int64)
  vcat(sinks(model, place,curr), sources(model,place, curr))
end

function train_sinksrcs!(model::LNMMSB,meth::String)
	if meth == "isns"
		@assert !isempty(model.ho_links)
		for a in 1:model.N
			model.train_sinks[a] = sinks(model,"train",a)#length is fadj
			model.train_outdeg[a] = length(model.train_sinks[a])
			model.train_sources[a]=sources(model,"train",a)#length is badj
			model.train_indeg[a] = length(model.train_sources[a])
		end
	end
	if meth == "isns2" || meth == "link"
		for a in 1:model.N
			model.train_sinks[a] = sinks(model,"network",a)#length is fadj
			model.train_outdeg[a] = length(model.train_sinks[a])
			model.train_sources[a]=sources(model,"network",a)#length is badj
			model.train_indeg[a] = length(model.train_sources[a])
		end
	end
	println("training sink and sources figured")
end
##think about speeding this up-1ms not good
function train_nonlinks!(model::LNMMSB,ignoreho::Bool)
	_init_ϕ = (1.0/model.K)*ones(Float64, model.K)
	@assert isempty(model.train_nonlinks)
	x = Vector{Dyad}()
	for i in 1:model.N
		for j in 1:model.N
			if i ==j
				continue;
			else
				push!(x,Dyad(i,j))
			end
		end
	end
	if !ignoreho
		x = setdiff(x, model.ho_dyads)

		for xx in x
			if isalink(model, "train", xx.src, xx.dst)
				continue;
			else
				push!(model.train_nonlinks, NonLink(xx.src, xx.dst,_init_ϕ,_init_ϕ))
			end
		end
	else
		for xx in x
			if isalink(model, "network", xx.src, xx.dst)
				continue;
			else
				push!(model.train_nonlinks, NonLink(xx.src, xx.dst,_init_ϕ,_init_ϕ))
			end
		end
	end
	model.train_nonlinks = shuffle!(model.train_nonlinks)
	println("train nonlinks figured")
end
function set_partitions!(model::LNMMSB)
	_init_ϕ = (1.0/model.K)*ones(Float64, model.K)
	model.link_set = [Vector{Link}() for i in 1:model.N] ##all of its links
	for a in 1:model.N
		llist = Vector{Link}()
		for s in model.train_sinks[a]
			push!(llist, Link(a, s, _init_ϕ,_init_ϕ))
		end
		for s in model.train_sources[a]
			push!(llist, Link(s, a, _init_ϕ,_init_ϕ))
		end
		# push!(link_set[a], llist)
		model.link_set[a] = llist
	end

	model.nonlink_setmap=[VectorList{Int64}() for i in 1:model.N]
	model.node_tnmap = [Vector{Int64}() for i in 1:model.N]

	for (ii,tt) in enumerate(model.train_nonlinks)
		model.node_tnmap[tt.src]=vcat(model.node_tnmap[tt.src], ii)
		model.node_tnmap[tt.dst]=vcat(model.node_tnmap[tt.dst],ii)
	end
	# model.train_nonlinks[model.node_tnmap[1]]
	nonlinksetsize = round.(Int64, 1.0*(model.train_indeg .+ model.train_outdeg))
	for a in 1:model.N
		i=1
		while i < div(length(model.node_tnmap[a]),nonlinksetsize[a])
			push!(model.nonlink_setmap[a],model.node_tnmap[a][((i-1)*nonlinksetsize[a]+1):((i)*nonlinksetsize[a])])
			i+=1
		end
		push!(model.nonlink_setmap[a],model.node_tnmap[a][((i-1)*nonlinksetsize[a]+1):end])
	end
	# should look up nonlink_set[a][m] in model.train_nonlinks
	# model.train_nonlinks[model.nonlink_setmap[a][sample(1:length(model.nonlink_setmap[a]))]]
end

function mbsampling!(mb::MiniBatch,model::LNMMSB, meth::String,mbsize::Int64)
	# @assert !isempty(model.ho_links)
	# @assert !isdefined(:mb)
	if meth == "isns" ##informative stratified node sampling
		##Choose random nodes
		node_n  = 0
		while node_n < mbsize
			a = ceil(Int64,model.N*rand())
			if a in mb.mbnodes
				continue;
			else
				push!(mb.mbnodes, a)
				node_n +=1
			end
		end
		model.mbids = mb.mbnodes
		####Node selection done
		for a in mb.mbnodes
			for l in model.link_set[a]
				if l in mb.mblinks
					continue;
				else
					push!(mb.mblinks, l)
				end
			end

			picknl = ceil(Int64,length(model.nonlink_setmap[a])*rand())
			for nl in model.train_nonlinks[model.nonlink_setmap[a][picknl]]
				if nl in mb.mbnonlinks
					continue;
				else
					push!(mb.mbnonlinks, nl)
					if !haskey(mb.mbfnadj, nl.src)
						mb.mbfnadj[nl.src] = get(mb.mbfnadj, nl.src, Vector{Int64}())
					end
					if nl.dst in mb.mbfnadj[nl.src]
						continue;
					else
						push!(mb.mbfnadj[nl.src],nl.dst)
					end
					if !haskey(mb.mbbnadj, nl.dst)
						mb.mbbnadj[nl.dst] = get(mb.mbbnadj, nl.dst, Vector{Int64}())
					end
					if nl.src in mb.mbbnadj[nl.dst]
						continue;
					else
						push!(mb.mbbnadj[nl.dst],nl.src)
					end
				end
			end
		end
		for a in mb.mbnodes
			model.trainfnadj[a] = model.N-1-model.train_outdeg[a]-model.ho_fadj[a]-model.ho_fnadj[a]
			model.trainbnadj[a] = model.N-1-model.train_indeg[a]-model.ho_badj[a]-model.ho_bnadj[a]
		end
	elseif meth == "link"
		node_n  = 0

		# mbsize = model.mbsize

		while node_n < mbsize
			a = ceil(Int64,model.N*rand())
			if a in mb.mbnodes
				continue;
			else
				push!(mb.mbnodes, a)
				node_n +=1
			end

			for l in model.link_set[a]
				if l in mb.mblinks
					continue;
				else
					push!(mb.mblinks, l)
				end
			end
		end
		# for l in mb.mblinks
		# 	if !(l.src in mb.mbnodes)
		# 		push!(mb.mbnodes, l.src)
		# 	elseif !(l.dst in mb.mbnodes)
		# 		push!(mb.mbnodes, l.dst)
		# 	else
		# 		continue;
		# 	end
		# end
		# mb.mbnodes = unique(mb.mbnodes)
		model.mbids = mb.mbnodes
		# model.mbsize = length(mb.mbnodes)
	elseif meth == "isns2"
		node_n  = 0
		while node_n < mbsize
			a = ceil(Int64,model.N*rand())
			if a in mb.mbnodes
				continue;
			else
				push!(mb.mbnodes, a)
				node_n +=1
			end
		end
		model.mbids[:] = mb.mbnodes[:]

		for a in mb.mbnodes
			for l in model.link_set[a]
				if l in mb.mblinks
					continue;
				else
					push!(mb.mblinks, l)
				end
			end
			model.nonlink_setmap[a][1]
			# length(mb.mbfnadj[a])+length(mb.mbbnadj[a])-length(intersect(mb.mbfnadj[a],	mb.mbbnadj[a]))
			# model.train_nonlinks[model.nonlink_setmap[a][1]]
			
			picknl = ceil(Int64,length(model.nonlink_setmap[a])*rand())
			# length(model.nonlink_setmap[a][1])+length(model.nonlink_setmap[a][2])
			# 2N-2-model.train_outdeg[a]-model.train_indeg[a]
			for nl in model.train_nonlinks[model.nonlink_setmap[a][picknl]]
				if nl in mb.mbnonlinks
					continue;
				else
					push!(mb.mbnonlinks, nl)
					if !haskey(mb.mbfnadj, nl.src)
						mb.mbfnadj[nl.src] = get(mb.mbfnadj, nl.src, Vector{Int64}())
					end
					if nl.dst in mb.mbfnadj[nl.src]
						continue;
					else
						push!(mb.mbfnadj[nl.src],nl.dst)
					end
					if !haskey(mb.mbbnadj, nl.dst)
						mb.mbbnadj[nl.dst] = get(mb.mbbnadj, nl.dst, Vector{Int64}())
					end
					if nl.src in mb.mbbnadj[nl.dst]
						continue;
					else
						push!(mb.mbbnadj[nl.dst],nl.src)
					end
				end
			end
		end
		# for a in mb.mbnodes
		# 	model.trainfnadj[a] = model.N-1-model.train_outdeg[a]-model.ho_fadj[a]-model.ho_fnadj[a]
		# 	model.trainbnadj[a] = model.N-1-model.train_indeg[a]-model.ho_badj[a]-model.ho_bnadj[a]
		# end

	end
	print();
end
# mbsampling!(mb,model, "isns")
function preparedata(model::LNMMSB,ignoreho::Bool,meth::String)
	setholdout!(model, meth)
	train_sinksrcs!(model,meth)
	train_nonlinks!(model,ignoreho)
	set_partitions!(model)
	# mb = deepcopy(model.mb_zeroer)
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
