function do_linked_edges!(model::LNMMSB)
	Src, Sink, Val= findnz(model.network)
	model.linked_edges=Set{Dyad}()
	for i in 1:length(Val)
		if Src[i] < Sink[i]
			push!(model.linked_edges, Dyad(Src[i],Sink[i]))
		end
	end
end

function minibatch_set_srns(model::LNMMSB)
	model.minibatch_set = Set{Dyad}()
	node_count = 0
	while node_count < model.mbsize
		a = ceil(Int64,model.N*rand())
		if a in mb.mbnodes
			continue;
		else
			push!(mb.mbnodes, a)
		end
		minibatch_size = round(Int64, model.N/model.num_peices)
		p = minibatch_size
		while p > 1
			node_list = sample(1:N, minibatch_size*2)
			for neighbor in node_list
				if p < 1
					break;
				end
				if neighbor == a
					continue;
				end
				if !isalink(model, "network", a, neighbor)
					dd = a < neighbor ? Dyad(a, neighbor) : Dyad(neighbor, a)
					if dd in model.linked_edges || haskey(model.ho_map,dd) ||
						haskey(model.test_map,dd) || dd in  model.minibatch_set
						continue;
					else
						push!(model.minibatch_set, dd)
						p-=1
					end
			  	end
			end
		end
		for neighbor in model.train_link_map[a]
			if isfadj(model, "network", a, neighbor)
				dd = a < neighbor ? Dyad(a, neighbor) : Dyad(neighbor, a)
				push!(model.minibatch_set, Dyad(a, neighbor))
			end
		end
		node_count +=1
	end
end

function init_train_link_map!(model::LNMMSB)
	for a in 1:model.N
		if !haskey(model.train_link_map, a)
			model.train_link_map[a] = get(model.train_link_map, a, Set{Int64}())
		end
	end
	for d in model.linked_edges
		push!(model.train_link_map[d.src],d.dst)
		push!(model.train_link_map[d.dst],d.src)
	end
end


function init_ho_map!(model::LNMMSB)
	p= round(Int64,model.nho/2)
	if length(model.linked_edges) < p
		exit("something wrong here!")
	end
	sampled_linked_edges = collect(model.linked_edges)[sample(1:length(model.linked_edges), p, replace=false)]
	for edge in sampled_linked_edges

		if !haskey(model.ho_map, edge)
			model.ho_map[edge] = get(model.ho_map, edge, true)
		end
		model.ho_map[edge]=true
		if haskey(model.train_link_map,edge.src)
			delete!(model.train_link_map[edge.src], edge.dst)
		end
		if haskey(model.train_link_map,edge.dst)
			delete!(model.train_link_map[edge.dst], edge.src)
		end
	end
	while p > 1

		edge = get_random_ho_nonlink(model)
		if !haskey(model.ho_map, edge)
			model.ho_map[edge] = get(model.ho_map, edge, false)
		end
		model.ho_map[edge]=false
		p-=1
	end
end
function init_test_map!(model::LNMMSB)
	p = round(Int64,model.nho/2)
	while p > 1
		sampled_linked_edges = collect(model.linked_edges)[sample(1:length(model.linked_edges), 2p, replace=false)]
		for edge in sampled_linked_edges
			if p < 1
				break;
			end
			if haskey(model.ho_map,edge) || haskey(model.test_map,edge)
				continue;
			else
				if !haskey(model.test_map, edge)
					model.test_map[edge] = get(model.test_map, edge, true)
				end
				model.ho_map[edge]=true
				if haskey(model.train_link_map,edge.src)
					delete!(model.train_link_map[edge.src], edge.dst)
				end
				if haskey(model.train_link_map,edge.dst)
					delete!(model.train_link_map[edge.dst], edge.src)
				end
				p-=1
			end
		end
	end
	p = round(Int64,model.nho/2)
	while p > 1
		edge = get_random_test_nonlink(model)
		if !haskey(model.test_map, edge)
			model.test_map[edge] = get(model.test_map, edge, false)
		end
		model.test_map[edge] = false
		p-=1
	end
end
function get_random_ho_nonlink(model::LNMMSB)
	while true
		a,b = sample(1:model.N, 2)
		if a!=b && a < b
			edge = Dyad(a,b)
			if edge in model.linked_edges || haskey(model.ho_map,edge)
				continue;
			end
			return edge
		end
	end
end
function get_random_test_nonlink(model::LNMMSB)
	while true
		a,b = sample(1:model.N, 2)
		if a!=b && a < b
			edge = Dyad(a,b)
			if edge in model.linked_edges || haskey(model.ho_map,edge) || haskey(model.test_map,edge)
				continue;
			end
			return edge
		end
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

function isfadj(model::LNMMSB,place::String,curr::Int64, q::Int64)
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

function neighbors_(model::LNMMSB,place::String,curr::Int64)
  [b for b in 1:model.N if isalink(model,place, curr, b)]
end

function preparedata2!(model::LNMMSB)
	do_linked_edges!(model)
	init_train_link_map!(model)
	init_ho_map!(model)
	init_test_map!(model)
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
    model.μ_var[i,:] = log.(model.μ_var[i,:]/(sum(model.μ_var[i,:])))
  end
end
print();
