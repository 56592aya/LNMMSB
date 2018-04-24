using MiniLogging
using GradDescent
using DoWhile
using DataFrames

import Base: ==, hash, isequal, zeros
import Core: ===
Base.zero(::Type{KeyVal}) = KeyVal(0,0.0)
==(x::Dyad, y::Dyad) = (x.src == y.src && x.dst === y.dst) || (x.src === y.dst && x.dst === y.src)
hash(x::Dyad, h::UInt) = hash(minmax(x.src, x.dst), h)
==(x::Link, y::Link) = (x.src == y.src && x.dst == y.dst) || (x.src == y.dst && x.dst == y.dst)
hash(x::Link, h::UInt) = hash(minmax(x.src, x.dst), h)
==(x::NonLink, y::NonLink) = (x.src == y.src && x.dst == y.dst) || (x.src == y.dst && x.dst == y.dst)
hash(x::NonLink, h::UInt) = hash(minmax(x.src, x.dst), h)
==(x::Dyad, y::Link) = (x.src == y.src && x.dst == y.dst) || (x.src == y.dst && x.dst == y.dst)
==(x::Dyad, y::NonLink) = (x.src == y.src && x.dst == y.dst) || (x.src == y.dst && x.dst == y.dst)
==(x::Link, y::Dyad) = (x.src == y.src && x.dst == y.dst) || (x.src == y.dst && x.dst == y.dst)
==(x::NonLink, y::Dyad) = (x.src == y.src && x.dst == y.dst) || (x.src == y.dst && x.dst == y.dst)
Network{T<:Integer}(nrows::T) = SparseMatrixCSC{T,T}(nrows, nrows, ones(T, nrows+1), Vector{T}(0), Vector{T}(0))
function digamma_(x::Float64)
	p=zero(Float64)
  x=x+6.0
  p=1.0/(x*x)
  p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
  p=p+log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
  p
end
function lgamma_(x::Float64)
	z=1.0/(x*x)
 	x=x+6.0
  z=(((-0.000595238095238*z+0.000793650793651)*z-0.002777777777778)*z+0.083333333333333)/x
  z=(x-0.5)*log(x)-x+0.918938533204673+z-log(x-1.0)-log(x-2.0)-log(x-3.0)-log(x-4.0)-log(x-5.0)-log(x-6.0)
  z
end

digamma_{T<:Number,R<:Integer}(x::T, dim::R) = @fastmath @inbounds return sum(digamma_(x+.5*(1-i)) for i in 1:dim)
lgamma_{T<:Number,R<:Integer}(x::T, dim::R)=.25*(dim*dim-1)*log(pi)+sum(lgamma_(x+.5*(1-i)) for i in 1:dim)

function logsumexp(X::Vector{Float64})
    alpha = -Inf;r = 0.0;
    for x = X
        if x <= alpha
            r += exp(x - alpha)
        else
            r *= exp(alpha - x)
            r += 1.0
            alpha = x
        end
    end
    log1p(r-1.0) + alpha
end
function logsumexp(X::Float64, Y::Float64)
    alpha = -Inf;r = 0.0;
    if X <= alpha
        r += exp(X - alpha)
    else
        r *= exp(alpha - X)
        r += 1.0
        alpha = X
    end
    if Y <= alpha
        r += exp(Y - alpha)
    else
        r *= exp(alpha - Y)
        r += 1.0
        alpha = Y
    end
    log1p(r-1.0) + alpha
end


function softmax(X::Vector{Float64})
  exp.(X.-logsumexp(X))
end


function sort_by_argmax!{T<:Real}(X::Matrix2d{T})
	n_row=size(X,1)
	n_col = size(X,2)
	ind_max=zeros(Int64, n_col)
	@simd for a in 1:n_col
    @inbounds ind_max[a] = indmax(view(X,1:n_row,a))
	end
	X_tmp = similar(X)
	count = 1
	for j in 1:maximum(ind_max)
  	for i in 1:n_col
    	if ind_max[i] == j
      	for k in 1:n_row
        	X_tmp[k,count] = X[k,i]
      	end
      	count += 1
    	end
  	end
	end
	# This way of assignment is important in arrays, el by el
	X[:]=X_tmp[:]
	X
end
###################INIT#############################
function sort_by_values(v::Vector{KeyVal})
  d = DataFrame()
  d[:X1] = [i for i in 1:length(v)]
  d[:X2] = [0.0 for i in 1:length(v)]
  x = []
  for (i,val) in enumerate(v)
    #length(v)
    push!(x,i)
    d[i,:X1] = val.first
    d[i,:X2] = val.second
  end
  sort!(d, cols=:X2, rev=true)
  temp = [KeyVal(0,0.0) for z in 1:length(v)]
  for i in 1:length(v)
    temp[i].first = d[i,:X1]
    temp[i].second = d[i,:X2]
  end
  return temp
end
function init_mu(mu, maxmu, N, topK)
  for i in 1:N
      mu[i,1].first=i
      mu[i,1].second = 1.0 + rand()
      maxmu[i] = i
      for j in 2:topK
        mu[i,j].first = (i+j-1)%N
        mu[i,j].second = rand()
      end
  end
end
function estimate_thetas(mu,N,topK)
  theta_est = [KeyVal(0,0.0) for i=1:N,j=1:topK]
  for i in 1:N
    s = 0.0
    for k in 1:topK
      s += mu[i,k].second
    end
    for k in 1:topK
      theta_est[i,k].first  = mu[i,k].first
      theta_est[i,k].second = mu[i,k].second*1.0/s
    end
  end
  return theta_est
end
function log_groups(communities, theta_est, topK, ulinks)
  for link in ulinks
    i = link.first;m=link.second;
    # if i < m
      max_k = 65535
      max = 0.0
      sum = 0.0
      for k1 in 1:topK
        for k2 in 1:topK
          if theta_est[i,k1].first == theta_est[m,k2].first
            u = theta_est[i,k1].second * theta_est[m,k2].second
            sum += u
            if u > max
              max = u
              max_k = theta_est[i,k1].first
            end
          end
        end
      end
      #print("max before is $max and ")
      if sum > 0.0
        max = max/sum
      else
        max = 0.0
      end
      #println(" and max after is $max and sum is $sum")
      if max > .5
        #println(max)
        if max_k != 65535
          i = convert(Int64, i)
          m = convert(Int64, m)
          if haskey(communities, max_k)
            push!(communities[max_k], i)
            push!(communities[max_k], m)
          else
            communities[max_k] = get(communities,max_k,Int64[])
            push!(communities[max_k], i)
            push!(communities[max_k], m)
          end
        end
      end
    # end
  end
  count = 1
  Comm_new = similar(communities)
  for c in collect(keys(communities))
    u = collect(communities[c])
    #println(u)
    uniq = Dict{Int64,Bool}()
    ids = Int64[]
    for p in 1:length(u)
      if !(haskey(uniq, u[p]))
        push!(ids, u[p])
        uniq[u[p]] = true
      end
    end
    vids = Vector{Int64}(length(ids))
    for j in 1:length(ids)
      vids[j] = ids[j]
    end
    vids = sort(vids)
    if !haskey(Comm_new, count)
      Comm_new[count] = get(Comm_new,count, Int64[])
    end
    for j in 1:length(vids)
      push!(Comm_new[count], vids[j])
    end
    count += 1
  end
  return Comm_new
end
###BatchInfer

function batch_infer(network::Network, N::Int64)
  topK = 5
  _α = 1.0/N
  mu = [KeyVal(0,0.0) for i=1:N, j=1:topK]
  munext = [Dict{Int64, Int64}() for i in 1:N]
  maxmu = zeros(Int64,N)
  communities = Dict{Int64, Vector{Int64}}()
  ulinks = Vector{Pair{Int64, Int64}}()
  #undirected
  x,y,z=findnz(network)
  for row in 1:nnz(network)
      push!(ulinks,x[row]=>y[row])
  end
  init_mu(mu, maxmu, N, topK)
  for iter in 1:ceil(Int64, log10(N))
    for link in ulinks
      p = link.first;q = link.second;
      pmap = munext[p]
      qmap = munext[q]
      if !haskey(pmap, maxmu[q])
        pmap[maxmu[q]] = get(pmap, maxmu[q], 0)
      end
      if !haskey(qmap, maxmu[p])
        qmap[maxmu[p]] = get(qmap, maxmu[p], 0)
      end
      pmap[maxmu[q]] +=  1
      qmap[maxmu[p]] +=  1
    end
    #set_gamma(gamma, gammanext, maxgamma)
    ###SETGAMMA begin
    for i in 1:N
      m = munext[i]
      sz = 0
      if length(m) != 0
        if length(m) > topK
          sz = length(m)
        else
          sz = topK
        end
        v = [KeyVal(0,0.0) for z in 1:sz]
        c = 1
        for j in m
          v[c].first = j.first
          v[c].second = j.second
          c += 1
        end
        while c <= topK #assign random communities to rest
          k=0
          @do begin
             k = sample(1:N)
          end :while (k in keys(m))
          v[c].first = k
          v[c].second = _α
          c+=1
        end
        v = sort_by_values(v)
        mu[i,:]
        for k in 1:topK
          mu[i,k].first = v[k].first
          mu[i,k].second = v[k].second + _α
        end
        maxmu[i] = v[1].first
        munext[i] = Dict{Int64, Int64}()
      end
    end
    ###SETGAMMA END
  end
  theta_est = estimate_thetas(mu,N,topK)
  return log_groups(communities, theta_est, topK, ulinks)
end
function findk(network::Network{Int64}, N::Int64, findkb::Bool)
    try
        @assert finkb
    catch e
         isa(e,AssertionError) && @error(root_logger, "  findk was not specified as an argument")
    end
    return batch_infer(network, N)
end


###################Model Utils##########################
function isalink(network::Network{Int64}, a::Int64, b::Int64)
	return (network[a,b] == 1)
end
function neighbors_(network::Network{Int64}, a::Int64)
	network[:,a].nzind
end
function negateIndex(full_indices::Vector{Int64}, negative_index::Vector{Int64})
	filter!(i->!in(i,negative_index), full_indices)
end


# function do_linked_edges!(linked_edges::Vector{Dyad},network::Network{Int64})
# 	Src, Sink, Val= findnz(model.network)
# 	##should be empty outside
#     #linked_edges=Set{Dyad}()
# 	for i in 1:length(Val)
# 		if Src[i] < Sink[i]
# 			push!(linked_edges, Dyad(Src[i],Sink[i]))
# 		end
# 	end
# end

##should look in the args if only K is given do it with random
## if K is not given should do the ginit, and determine the K
# or should have a signal whether ginit has been done or not
# but I want to specify the ginit in the args
function init_μ(N::Int64, network::Network{Int64},findkb::Bool, communities::Dict{Int64, Vector{Int64}})
    if !findkb
        _init_μ = 1e-10*ones(Float64, (K, N))
        for i in 1:N
            _init_μ[:,i] = log.(_init_μ[:,i]/(sum(_init_μ[:,i])))
        end
        info("  mu's are initialized")
        return _init_μ
    else
        Belong = Dict{Int64, Vector{Int64}}()
        K = length(communities)
        _init_μ = 1e-10*ones(Float64, (K, N))
        for i in 1:N
            if !haskey(Belong, i)
                Belong[i] = get(Belong, i, Int64[])
            end
            for k in 1:K
                if i in communities[k]
                    push!(Belong[i],k)
                end
            end
            if length(Belong[i]) == 0
                push!(Belong[i], sample(1:length(communities)))
                _init_μ[Belong[i],i] = .9
            elseif length(Belong[i]) == 1
                _init_μ[Belong[i],i] = .9
            else
                val = .9/length(Belong[i])
                for z in Belong[i]
                    _init_μ[z,i] = val
                end
            end
            s = zero(Float64)
            for k in 1:K
                s+= _init_μ[k,i]
            end
            for k in 1:K
                _init_μ[k,i] = _init_μ[k,i]/s
            end
        end
        for i in 1:N
            _init_μ[:,i] = log.(_init_μ[:,i]/(sum(_init_μ[:,i])))
        end
        @info(root_logger,"  mu's are initialized")
        return _init_μ
    end
end
#comment later

#how to call init_μ(N, network; findk=true)
#remember to treat column wise this time

function setup_mb(network::Network, nnode::Int64, K::Int64)
    ncount = 0
    mbnodes = Vector{Int64}()
    ϕl = Map{Pair{Int64, Int64}, Vector{Float64}}()
    ϕnlout = Map{Pair{Int64, Int64}, Vector{Float64}}()
    ϕnlin = Map{Pair{Int64, Int64}, Vector{Float64}}()
    init_ϕ = 1.0/K.*ones(Float64, K)
    adjlist  = Map{Int64, Vector{Int64}}()

    nadjlist = Map{Int64, Vector{Int64}}()

    while ncount < nnode
        a = ceil(Int64,N*rand())
        if a in mbnodes
            continue;
        else
            push!(mbnodes, a)
            ns = neighbors_(network, a)

            adjlist[a] = getkey(adjlist, a, ns)
            for n in ns
                if a  < n
                    k = Pair(a,n)
                    if !haskey(ϕl, k)
                        ϕl[k] = getkey(ϕl, k, init_ϕ)
                    end
                else
                    k = Pair(n,a)
                    if !haskey(ϕl, k)
                        ϕl[k] = getkey(ϕl, k, init_ϕ)
                    end

                end
            end

            nns = Vector{Int64}()
            nncount = 0
            while nncount < length(ns)
                nn = ceil(Int64,N*rand())
                if !haskey(nadjlist, a)
                    nadjlist[a] = getkey(nadjlist, a, Int64[])
                end
                if nn in nadjlist[a] || nn == a
                    continue;
                else
                    push!(nadjlist[a], nn)
                    if a < nn
                        k = Pair(a,nn)
                        if !haskey(ϕnlout, k)
                            ϕnlout[k] = getkey(ϕnlout, k, init_ϕ)
                        end
                        if !haskey(ϕnlin, k)
                            ϕnlin[k] = getkey(ϕnlin, k, init_ϕ)
                        end
                    else
                        k = Pair(nn,a)
                        if !haskey(ϕnlout, k)
                            ϕnlout[k] = getkey(ϕnlout, k, init_ϕ)
                        end
                        if !haskey(ϕnlin, k)
                            ϕnlin[k] = getkey(ϕnlin, k, init_ϕ)
                        end
                    end
                    nncount +=1
                end
            end
            ncount +=1
        end
    end
    ϕlsum = deepcopy(zeros(Float64, (K,N)))
    ϕnlinoutsum = deepcopy(zeros(Float64, K))
    ϕnloutsum = deepcopy(zeros(Float64, (K,N)))
    ϕnlinsum = deepcopy(zeros(Float64, (K,N)))
    one_over_K = deepcopy(ones(Float64,K)./K)
    @info(root_logger, "   minibatch is set up")
    return mbnodes, adjlist, nadjlist, ϕl, ϕnlout, ϕnlin,ϕlsum,ϕnlinoutsum,ϕnloutsum,ϕnlinsum,one_over_K
end
#comment later

#comment later


function estimate_βs!(b::Matrix2d{Float64})
    return b[1,:]./(b[1,:].+b[2,:])
end
function estimate_θs!(θ::Matrix2d{Float64}, μ::Matrix2d{Float64}, mbnodes::Vector{Int64})
	for a in mbnodes
		θ[:,a]=softmax(μ[:,a])
	end
    return θ
end
#comment later


function update_Elogβ!(Elogβ::Matrix2d{Float64}, b::Matrix2d{Float64}, K::Int64)
	for k in 1:K
		@views Elogβ[1,k] = digamma_(b[1,k]) - (digamma_(b[1,k])+digamma_(b[2,k]))
		@views Elogβ[2,k] = digamma_(b[2,k]) - (digamma_(b[1,k])+digamma_(b[2,k]))
	end
end

#comment later

function getSets!(
    θ::Matrix2d{Float64}, threshold::Float64, _init_μ::Matrix2d{Float64},μ::Matrix2d{Float64},
    network::Network{Int64}, N::Int64, K::Int64,Korder::VectorList{Int64},
    A::VectorList{Int64}, B::VectorList{Int64}, C::VectorList{Int64}
    )
    est_θ = deepcopy(θ)

	for a in 1:N
		Korder[a] = sortperm(est_θ[:,a], rev=true)
		F = 0.0
		counter = 1
		while (F < threshold && counter < K)
			k = Korder[a][counter]
			F += est_θ[k,a]
			counter += 1
			push!(A[a], k)
		end
	end
	for a in 1:N
		neighbors = neighbors_(network, a)
		for b in neighbors
			for k in A[b]
				if (!(k in A[a]) && !(k in C[a]))
					push!(C[a], k)
				end
			end
		end
		B[a] = negateIndex(Korder[a],vcat(A[a], C[a]))
		if !(isempty(B[a]))
			bulk_θs  = sum(est_θ[B[a],a])/length(B[a])
			est_θ[B[a],a] = bulk_θs
			μ[B[a],a] = log.(bulk_θs)
		end
		_init_μ[:,a] = μ[:,a]
	end
end
#comment_later

function update_sets!(θ::Matrix2d{Float64},μ::Matrix2d{Float64},mbnodes::Vector{Int64},
    network::Network{Int64}, A::VectorList{Int64}, B::VectorList{Int64},C::VectorList{Int64},
    Korder::VectorList{Int64}
    )
	@inbounds for a in mbnodes
		neighbors = neighbors_(network, a)
		@inbounds for b in neighbors
			@inbounds for k in A[b]
				if k in A[a]
					continue
				else
					push!(C[a], k)
				end
			end
		end
		C[a] = unique(C[a])
		B[a] = negateIndex(Korder[a],vcat(A[a], C[a]))
		if !(isempty(B[a]))
			bulk_θs  = sum(θ[B[a],a])/length(B[a])
			θ[B[a],a] = bulk_θs
			μ[B[a],a] = log.(bulk_θs)
		end
	end
end

function update_A!(mbnodes::Vector{Int64}, est_θ::Matrix2d{Float64},threshold::Float64,
    Korder::VectorList{Int64},A::VectorList{Int64},B::VectorList{Int64},C::VectorList{Int64},
    K::Int64)
	for a in mbnodes
		Korder[a] = sortperm(est_θ[:,a],rev=true)
		F = 0.0
		count = 1
		newA=Int64[]
		while (F < threshold && count < K)
			k = Korder[a][count]
			if (k in B[a])
				count+=1
			else
				if !isempty(B[a])
					if est_θ[k,a] > est_θ[B[a][1],a]
						F += est_θ[k,a]
						push!(newA, k)
						count += 1
					end
				else
					F += est_θ[k,a]
					push!(newA, k)
					count += 1
				end
			end
		end
		toAdd = Int64[]
		if (threshold-F > 0.0)
			toSample = (threshold-F)/est_θ[B[a][1],a]
			toAdd = sample(B[a], toSample, replace=false)
		end
		if !isempty(toAdd)
			for el in toAdd
				A[a]= push!(newA, el)
			end
		else
			A[a] = newA
		end
	end
end



##################train_functions#######################
#comment later
function update_ϕ!(ϕl::Dict{Pair{Int64, Int64}, Vector{Float64}}, ϕlsum::Matrix2d{Float64},
     ϕnlout::Dict{Pair{Int64, Int64}, Vector{Float64}}, ϕnlin::Dict{Pair{Int64, Int64}, Vector{Float64}},
      ϕnloutsum::Matrix2d{Float64}, ϕnlinsum::Matrix2d{Float64}, ϕnlinoutsum::Vector{Float64},
      A::VectorList{Int64}, C::VectorList{Int64}, B::VectorList{Int64}, μ::Matrix2d{Float64}, Elogβ::Matrix2d{Float64},
       EPSILON::Float64, K::Int64)
    for link in collect(keys(ϕl))
        for _ in 1:10
            updatephil!(ϕl,link,A, C, B, μ, Elogβ)
        end
        for k in 1:K
            ϕlsum[k, link.first] += ϕl[link][k]
            ϕlsum[k, link.second] += ϕl[link][k]
        end
    end
    for nonlink in collect(keys(ϕnlout))
        for _ in 1:10
            updatephinl!(ϕnlout,ϕnlin,nonlink, A, C, B,μ,Elogβ,EPSILON)
        end
        for k in 1:K
            ϕnloutsum[k, nonlink.first] += ϕnlout[nonlink][k]
            ϕnlinsum[k, nonlink.first] += ϕnlout[nonlink][k]
            ϕnlinsum[k, nonlink.second] += ϕnlin[nonlink][k]
            ϕnloutsum[k, nonlink.second] += ϕnlin[nonlink][k]
            ϕnlinoutsum[k] += ϕnlout[nonlink][k]*ϕnlin[nonlink][k]
        end
    end
end
function updatephil!(ϕl::Dict{Pair{Int64,Int64},Array{Float64,1}},link::Pair{Int64, Int64},
    A::VectorList{Int64}, C::VectorList{Int64}, B::VectorList{Int64}, μ::Matrix2d{Float64}, Elogβ::Matrix2d{Float64})

	src = link.first
    dst = link.second
    ϕ = zeros(similar(ϕl[link]))
	union_src = vcat(A[src], C[src])
	union_dst = vcat(A[dst], C[dst])
	vala = !isempty(B[src])? μ[B[src][1],src]: 0.0
	valb = !isempty(B[dst])? μ[B[dst][1],dst]: 0.0
	ϕ = Elogβ[1,:] .+ vala .+ valb
	for k in union_src
		ϕ[k] += μ[k,src] - vala
	end
	for k in union_dst
		ϕ[k] += μ[k,dst] - valb
	end
	r = logsumexp(ϕ)
	ϕl[link] = exp.(ϕ[:] .- r)[:]
end


function updatephinl!(ϕnlout::Dict{Pair{Int64,Int64},Array{Float64,1}},ϕnlin::Dict{Pair{Int64,Int64},Array{Float64,1}},
     nonlink::Pair{Int64, Int64},A::VectorList{Int64}, C::VectorList{Int64}, B::VectorList{Int64},
     μ::Matrix2d{Float64}, Elogβ::Matrix2d{Float64},EPSILON::Float64)
    ###common terms
    src = nonlink.first
	dst = nonlink.second
    ϕout = zeros(similar(ϕnlout[nonlink]))
    ϕin=zeros(similar(ϕnlin[nonlink]))
	constant = (Elogβ[2,:] .-log1p(-EPSILON))
	union_dst = vcat(A[dst], C[dst])
    union_src = vcat(A[src], C[src])
    #####Direction specific
    if rand(Bool, 1)[1]
        ϕout[:] = μ[:,src]
        for k in union_dst
            ϕout[k] += ϕnlin[nonlink][k]*constant[k]
        end
        rout = logsumexp(ϕout)
        ϕnlout[nonlink] = exp.(ϕout[:] .- rout)[:]
        #########
        ϕin[:] = μ[:,dst]
        for k in union_src
            ϕin[k] += ϕnlout[nonlink][k]*constant[k]
        end
        rin=logsumexp(ϕin)
        ϕnlin[nonlink] = exp.(ϕin[:] .- rin)[:]
    else
        ϕin[:] = μ[:,dst]
        for k in union_src
            ϕin[k] += ϕnlout[nonlink][k]*constant[k]
        end
        rin=logsumexp(ϕin)
        ϕnlin[nonlink] = exp.(ϕin[:] .- rin)[:]
        ####################
        ϕout[:] = μ[:,src]
        for k in union_dst
            ϕout[k] += ϕnlin[nonlink][k]*constant[k]
        end
        rout = logsumexp(ϕout)
        ϕnlout[nonlink] = exp.(ϕout[:] .- rout)[:]
    end
end

##comment later

function sfx(x::Vector{Float64})
	return softmax(x)
end

function dfunci(mu::Vector{Float64}, l::Float64, L::Matrix2d{Float64},m::Vector{Float64}, X::Vector{Float64}, x::Vector{Float64}, sumb::Float64)
	-l.*L*(mu-m) +X - sumb.*x
end
function update_μ!(μ::Matrix2d{Float64},μ_old::Matrix2d{Float64},mbnodes::Vector{Int64}, N::Int64, deg::Vector{Int64},ϕlsum::Matrix2d{Float64},ϕnloutsum::Matrix2d{Float64},ϕnlinsum::Matrix2d{Float64},
    l::Float64, L::Matrix2d{Float64}, m::Vector{Float64},ϕnlout::Dict{Pair{Int64, Int64}, Vector{Float64}})
    μ_old[:, mbnodes]=deepcopy(μ[:, mbnodes])
    count = Dict{Int64, Int64}()
    for k in collect(keys(ϕnlout))
        if k.first in mbnodes
            if !haskey(count, k.first)
                count[k.first] = getkey(count, k.first, 1)
            else
                count[k.first] +=1
            end
        else
            if !haskey(count, k.second)
                count[k.second] = getkey(count, k.second, 1)
            else
                count[k.second] +=1
            end
        end
    end

    for a in mbnodes
        mu = deepcopy(μ[:,a])
        x = sfx(mu)
        s1 = haskey(count,a)?convert(Float64,(N-1-deg[a])):0.0
        c1 = haskey(count,a)?convert(Float64, count[a]):1.0
        sumb =(convert(Float64,N)*-1.0)
        X=ϕlsum[:,a]+(s1/c1).*.5.*(ϕnloutsum[:,a]+ϕnlinsum[:,a])
        opt = Adagrad()
        for _ in 1:10
    		x  = sfx(mu)
    		g = dfunci(mu, l, L,m, X, x, sumb)
    		δ = update(opt,g)
    		# @code_warntype update(opt1,g1)
    		mu+=δ
    	end
    	μ[:,a]=mu
    end
end
function update_m!(m::Vector{Float64},m_old::Vector{Float64}, K::Int64, mbnodes::Vector{Int64},
    μ::Matrix2d{Float64}, M::Matrix2d{Float64}, M0::Matrix2d{Float64}, m0::Vector{Float64}, l::Float64, L::Matrix2d{Float64})
    m_old = deepcopy(m)
    s = zeros(Float64, K)
    for a in mbnodes
        s.+=μ[:,a]
    end

    scaler = N/length(mbnodes)
    m=inv(M)*(M0*m0+(scaler*l).*L*s)
end

function update_M!(M::Matrix2d{Float64},M_old::Matrix2d{Float64}, m::Vector{Float64}, l::Float64, L::Matrix2d{Float64}, M0::Matrix2d{Float64}, N::Int64)
	M_old = deepcopy(M)
	M = ((l*N).*L)+M0
end
#updateM!(model,mb)

function update_l!(l::Float64, l0::Float64, N::Int64)
	##should be set in advance, not needed in the loop
	l = l0+convert(Float64,N)
end
#updatel!(model,mb)
function update_L!(L::Matrix2d{Float64}, L_old::Matrix2d{Float64}, mbnodes::Vector{Int64},
     μ::Matrix2d{Float64}, m::Vector{Float64}, N::Int64, L0::Matrix2d{Float64}, M::Matrix2d{Float64}, i::Int64)
	L_old = deepcopy(L)
	s = zero(Float64)
	# for a in mb.mbnodes
	# 	s +=(model.μ_var[a,:]-model.m)*(model.μ_var[a,:]-model.m)'+diagm(1.0./model.Λ_var[a,:])
	# end
	#added this for instead
	for a in mbnodes
		s +=(μ[:,a]-m)*(μ[:,a]-m)'
	end
	s=(N/length(mbnodes))*s
	s+=inv(L0)+N.*inv(M)
	L = try
		inv(s)
	catch y
		if isa(y, Base.LinAlg.SingularException)
			println("i is ", i)
			error("hey hey hey singular")
		end
	end

end


function update_b!(b::Matrix2d{Float64}, b_old::Matrix2d{Float64}, mbnodes::Vector{Int64},
     ϕl::Dict{Pair{Int64, Int64}, Vector{Float64}},ϕnlout::Dict{Pair{Int64, Int64}, Vector{Float64}}, ϕlsum::Matrix2d{Float64},
     ϕnlinoutsum::Vector{Float64}, network::Network{Int64}, η0::Float64, η1::Float64, K::Int64)
	b_old = deepcopy(b)
	scaler0=nnz(network)/(2*length(ϕl))

	s0 = zeros(Float64, K)
	for a in mbnodes
		s0[:] += .5*ϕlsum[:,a]
	end
	b[1,:] = η0.+ (scaler0.*s0)
    scaler1 = (N^2 - N - nnz(network))/(2*length(ϕnlout))
    for k in 1:K
		b[2,k] = η1 + (.5*scaler1*ϕnlinoutsum[k])
	end
end



function init_env(network::Network, N::Int64)
    communities = findk(network, N, true)
    K= length(communities)
    _init_μ =init_μ(N, network,true, communities)
    μ = deepcopy(_init_μ)
    μ_old = deepcopy(_init_μ)
    θ = zeros(Float64, (K, N))
    θ=estimate_θs!(θ, μ, collect(1:N))
    Korder = [sortperm(θ[:,i],rev=true) for  i in 1:N]
    A = [Int64[] for a in 1:N]
    B = [Int64[] for a in 1:N]
    C = [Int64[] for a in 1:N]
    threshold=.9
    m0=zeros(Float64,K)
    m=zeros(Float64,K)
    m_old=zeros(Float64,K)
    M0=eye(Float64,K)
    M=eye(Float64,K)
    M_old=eye(Float64,K)
    l0=K*1.0
    L0=(0.05).*eye(Float64,K)
    l=K*1.0
    L=(1.0/K).*eye(Float64,K)
    L_old=(1.0/K).*eye(Float64,K)
    η0= 9.0
    η1=1.1
    b = ones(Float64, (2,K))
    b[1,:] = η0
    b[2,:] = η1
    b_old = deepcopy(b)
    deg = sum(network, 2)[:,1]
    lr_M = 1.0
    lr_m = 1.0
    lr_L = 1.0
    lr_b = 1.0
    lr_μ=ones(Float64, N)
    iter=10000
    every=500
    EPSILON=1e-10
    Elogβ = zeros(Float64, (2, K))
    update_Elogβ!(Elogβ, b, K)
    vcount = zeros(Float64, N)
    return communities,K,μ,μ_old,θ,Korder,A,B,C,threshold,_init_μ,m0,m,m_old,M0,M,M_old,l0,L0,l,L, L_old,η0,η1,b, b_old, deg,lr_M,lr_m,lr_L,lr_b,lr_μ,iter,every, EPSILON,Elogβ,vcount
end
function update_lr!(lr_μ, lr_m, lr_M, lr_L, lr_b, i, vcount, mbnodes)
    for a in mbnodes
        vcount[a] += 1.0
        expectedvisits = (Float64(i)/(4.0*Float64(N)/Float64(length(mbnodes))))
        lr_μ[a] = (expectedvisits/(expectedvisits+(vcount[a]-1.0)))^(.5)
    end
    lr_M = 1.0
    lr_m = (100./(100.0+Float64(i)+Float64(20000/(N/length(mbnodes))-1.0)))^(.9)
    lr_L = (100./(101.0+Float64(i)+Float64(20000/(N/length(mbnodes))-1.0)))^(.9)
    lr_b = (100./(101.0+Float64(i)+Float64(20000/(N/length(mbnodes))-1.0)))^(.9)
    return lr_μ, lr_m, lr_M, lr_L, lr_b
end

####Actual train
function train(network, N)

    communities,K,μ,μ_old,θ,Korder,A,B,C,threshold,_init_μ,m0,m, m_old,M0,M, M_old,l0,L0,l,L, L_old,η0,η1,b, b_old, deg,lr_M,lr_m,lr_L,lr_b,lr_μ,iter,every,EPSILON,Elogβ, vcount =
    init_env(network, N)

    getSets!(θ,threshold,_init_μ,μ,network,N,K,Korder,A,B,C)
    update_l!(l, l0, N)

    for i in 1:iter
        mbnodes, adjlist, nadjlist, ϕl, ϕnlout, ϕnlin,ϕlsum,ϕnlinoutsum,ϕnloutsum,ϕnlinsum,one_over_K = setup_mb(network, 5, K)

        update_sets!(θ,μ,mbnodes,network,A, B,C,Korder)
        update_ϕ!(ϕl, ϕlsum, ϕnlout, ϕnlin, ϕnloutsum, ϕnlinsum, ϕnlinoutsum,A, C, B, μ, Elogβ, EPSILON, K)
        μ[:, mbnodes]=deepcopy(_init_μ[:, mbnodes])
        update_μ!(μ,μ_old,mbnodes, N, deg,ϕlsum,ϕnloutsum,ϕnlinsum, l, L, m, ϕnlout)
        lr_μ, lr_m, lr_M, lr_L, lr_b = update_lr!(lr_μ, lr_m, lr_M, lr_L, lr_b, i, vcount, mbnodes)
        for a in mbnodes
            μ[:,a] = view(μ_old, :,a).*(1.0.-lr_μ[a])+lr_μ[a].*view(μ, :, a)
        end

        θ = estimate_θs!(θ, μ, mbnodes)
		est_θ = deepcopy(θ)
		update_A!(mbnodes,est_θ,threshold,Korder,A,B,C,K)
		update_m!(m,m_old, K, mbnodes, μ, M, M0, m0, l, L)
		m = m_old.*(1.0-lr_m)+lr_m.*m
		update_M!(M,M_old, m, l, L, M0, N)
		M = M_old.*(1.0-lr_M)+lr_M.*M
		update_L!(L, L_old, mbnodes,μ, m, N, L0, M, i)
		L = L_old.*(1.0-lr_L)+lr_L*L
		update_b!(b, b_old, mbnodes,ϕl,ϕnlout, ϕlsum,ϕnlinoutsum, network, η0, η1, K)
		b = (1.0-lr_b).*b_old + lr_b.*(b)
		update_Elogβ!(Elogβ,b,K)
        checkelbo = (i % N/length(mbnodes) == 0)
		if checkelbo || i == 1
			print(i);print(": ")
			est_β = estimate_βs!(b)
			println(est_β)
		end
    end
    return estimate_θs!(θ, μ, collect(1:N)), estimate_βs!(b)
end
