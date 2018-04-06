__precompile__()
module ModelUtils

export negateIndex, do_linked_edges, minibatch_set_srns
export init_train_link_map!,init_ho_map!,init_test_map!,get_random_ho_nonlink,get_random_test_nonlink
export isalink,isfadj,neighbors_,preparedata2!,init_mu,getSets!,update_sets!,update_A!,setup_mblnl!,test_sets

import Utils: Dyad, Link, NonLink, MiniBatch
import Model: LNMMSB
using StatsBase
"""
	To get the complement of the indices, much faster than setdiff
	especially if the not_index is large
"""
function negateIndex(full_indices::Vector{Int64}, negative_index::Vector{Int64})
	filter!(i->!in(i,negative_index), full_indices)
end

function do_linked_edges!(model::LNMMSB)
	Src, Sink, Val= findnz(model.network)
	model.linked_edges=Set{Dyad}()
	for i in 1:length(Val)
		if Src[i] < Sink[i]
			push!(model.linked_edges, Dyad(Src[i],Sink[i]))
		end
	end
end


function minibatch_set_srns(model::LNMMSB, mb::MiniBatch, N::Int64)
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
# function f(model::LNMMSB, mb::MiniBatch)
# 	mb = deepcopy(model.mb_zeroer)
# 	model.minibatch_set = Set{Dyad}()
# 	node_count = 0
# 	while node_count < model.mbsize
# 		a = ceil(Int64,model.N*rand())
# 		if a in mb.mbnodes
# 			continue;
# 		else
# 			push!(mb.mbnodes, a)
# 			node_count +=1
# 		end
# 		neigh = neighbors_(model, a)
# 		nonneigh = setdiff(1:model.N , vcat(a, neigh))
# 		x = collect.(zip(repmat([a], length(neigh)), neigh))
# 		x = Set([convert(Dyad, xx) for xx in x])
# 		union!(model.minibatch_set,x)
# 		minibatch_size = round(Int64, model.N/model.num_peices)
# 		p = minibatch_size
# 		nind =StatsBase.knuths_sample!(1:length(nonneigh), zeros(Int64, p))
# 		nonneigh=nonneigh[nind]
# 		x = collect.(zip(repmat([a], length(nonneigh)), nonneigh))
# 		x = Set([convert(Dyad, xx) for xx in x])
# 		union!(model.minibatch_set,x)
# 	end
# end
# @time minibatch_set_srns(model, mb)
# @time f(model, mb)
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


function isalink(model::LNMMSB, a::Int64, b::Int64)
	return (model.network[a,b] == 1)
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

function neighbors_(model::LNMMSB, curr::Int64)
	model.network[:,curr].nzind
end

# function neighbors_(model::LNMMSB,curr::Int64)
# 	[b for b in 1:model.N if isalink(model,curr, b)]
# end
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
function init_mu(model::LNMMSB, communities::Dict{Int64, Vector{Int64}}, onlyK::Int64, N::Int64)
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
  z = zeros(Float64, 1000, 1000)
  # x = rand(1000, 1000)
  # y = rand(1000, 1000)
  # @btime A_mul_B!(z, x,y)
  # @btime x*y

end

# Here we need do determine the A, C, B for the first round before getting into the variational loop
"""
	getSets(model::LNMMSB, threshold::Float64)
	Function that returns the A, C, B of all nodes and keeps an ordering for communities
	Input: estimated_θ's
	Output: None, but updates A, C, B, Ordering, and mu's in the bulk set
"""


function getSets!(model::LNMMSB, threshold::Float64, _init_μ::Array{Float64,2})
	est_θ = deepcopy(model.est_θ)
	for a in 1:model.N
		model.Korder[a] = sortperm(est_θ[a,:], rev=true)
		F = 0.0
		counter = 1
		while (F < threshold && counter < model.K)
			k = model.Korder[a][counter]
			F += est_θ[a,k]
			counter += 1
			push!(model.A[a], k)
		end
	end
	for a in 1:model.N
		neighbors = neighbors_(model, a)
		for b in neighbors
			for k in model.A[b]
				if (!(k in model.A[a]) && !(k in model.C[a]))
					push!(model.C[a], k)
				end
			end
		end
		# model.C[a] = unique(model.C[a])
		#negatteIndex is wronf for now
		model.B[a] = negateIndex(model.Korder[a],vcat(model.A[a], model.C[a]))
		#model.B[a] = setdiff(model.Korder[a], union(model.A[a], model.C[a]))

		# full_indices = collect(1:10000)
		# not_index = sample(1:10000, 9800, replace=false)
		# @btime negateIndex(full_indices, not_index)
		# @btime setdiff(full_indices, not_index)

		if !(isempty(model.B[a]))
			bulk_θs  = sum(est_θ[a,model.B[a]])/length(model.B[a])
			model.est_θ[a,model.B[a]] = bulk_θs
			model.μ_var[a,model.B[a]] = log.(bulk_θs)
		end
		_init_μ[a,:] = model.μ_var[a,:]
	end
end


function update_sets!(model::LNMMSB, mb::MiniBatch)
	@inbounds for a in mb.mbnodes
		neighbors = neighbors_(model, a)
		#we can speed up here if we have visited the neighbor before but for later
		# idea is that if have not visited we can skip it, also should be reset somewhere
		@inbounds for b in neighbors
			@inbounds for k in model.A[b]
				if k in model.A[a]
					continue
				else
					push!(model.C[a], k)
				end
			end
		end
		model.C[a] = unique(model.C[a])
		# @btime union(model.A[a], model.C[a])
		# @btime vcat(model.A[a], model.C[a])
		model.B[a] = negateIndex(model.Korder[a],vcat(model.A[a], model.C[a]))
		# model.B[a] = setdiff(model.Korder[a], union(model.A[a], model.C[a]))
		if !(isempty(model.B[a]))
			bulk_θs  = sum(model.est_θ[a,model.B[a]])/length(model.B[a])
			model.est_θ[a,model.B[a]] = bulk_θs
			model.μ_var[a,model.B[a]] = log.(bulk_θs)
		end
	end
end
function update_A!(model::LNMMSB, mb::MiniBatch, est_θ::Matrix{Float64},threshold::Float64)
	for a in mb.mbnodes
		model.Korder[a] = sortperm(est_θ[a,:],rev=true)
		F = 0.0
		count = 1
		newA=Int64[]
		#could use enumerate and continue instead
		while (F < threshold && count < model.K)
			k = model.Korder[a][count]

			if (k in model.B[a])
				count+=1
			else
				if !isempty(model.B[a])
					if est_θ[a,k] > est_θ[a,model.B[a][1]]
						#println("I got here!")
						F += est_θ[a,k]
						push!(newA, k)
						count += 1
					end
				else
					F += est_θ[a,k]
					push!(newA, k)
					count += 1
				end
			end
		end
		toAdd = Int64[]
		if (threshold-F > 0.0)
			toSample = (threshold-F)/est_θ[a,model.B[a][1]]
			toAdd = sample(model.B[a], toSample, replace=false)
		end
		if !isempty(toAdd)
			for el in toAdd
				model.A[a]= push!(newA, el)
			end
		else
			model.A[a] = newA
		end
	end
end
function setup_mblnl!(model::LNMMSB, mb::MiniBatch, shuffled::Vector{Dyad}, _init_ϕ::Vector{Float64})
	#shuffle them, so that the order is random

	for d in shuffled
		#creating link objects and initializing their phis
		if isalink(model, "network", d.src, d.dst)
			l = Link(d.src, d.dst,_init_ϕ)
			if l in mb.mblinks
				continue;
			else
				push!(mb.mblinks, l)
			end
		else
			#creating nonlink objects and initializing their phis
			nl = NonLink(d.src, d.dst, _init_ϕ,_init_ϕ)

			if nl in mb.mbnonlinks
				continue;
			else
				#also adding their nonsources and nonsinks
				push!(mb.mbnonlinks, nl)
				if !haskey(mb.mbnot, nl.src)
					mb.mbnot[nl.src] = get(mb.mbnot, nl.src, Vector{Int64}())
				end
				if nl.dst in mb.mbnot[nl.src]
					continue;
				else
					push!(mb.mbnot[nl.src], nl.dst)
				end
				if !haskey(mb.mbnot, nl.dst)
					mb.mbnot[nl.dst] = get(mb.mbnot, nl.dst, Vector{Int64}())
				end
				if nl.src in mb.mbnot[nl.dst]
					continue;
				else
					push!(mb.mbnot[nl.dst], nl.src)
				end
			end
		end
	end
end
function test_sets(model)
	wrongs = Int64[]
	for a in 1:model.N
		cond = isempty(intersect(model.A[a], model.C[a])) &&	isempty(intersect(model.A[a], model.B[a])) &&
				isempty(intersect(model.B[a], model.C[a]))
		if !cond
			push!(wrongs, a)
		end
	end
	if isempty(wrongs)
		println("sets correctly set")
		println("Initialized the sets for all nodes")
	end
	println(wrongs)
end
end
