# using LightGraphs
using Distributions
using DataStructures
using DoWhile
using DataFrames
# using Yeppp
# using FLAGS
###########

type KeyVal
  first::Int64
  second::Float64
end
import Base.zeros
Base.zero(::Type{KeyVal}) = KeyVal(0,0.0)
###this needs to be n from data
#using DGP

topK = 5
_α = 1.0/model.N
mu = [KeyVal(0,0.0) for i=1:model.N, j=1:topK]
munext = [Dict{Int64, Int64}() for i in 1:model.N]
maxmu = zeros(Int64,model.N)
communities = Dict{Int64, Vector{Int64}}()
ulinks = Vector{Pair{Int64, Int64}}()
#undirected
x,y,z=findnz(model.network)
for row in 1:nnz(model.network)
    push!(ulinks,x[row]=>y[row])
end

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
##INIT_mu
function init_mu(mu, maxmu)
  for i in 1:model.N
      mu[i,1].first=i
      mu[i,1].second = 1.0 + rand()
      maxmu[i] = i
      for j in 2:topK
        mu[i,j].first = (i+j-1)%model.N
        mu[i,j].second = rand()
      end
  end
end
#####estimate all thetas
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
function log_groups(communities, theta_est)
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
init_mu(mu, maxmu)
function batch_infer()
  for iter in 1:ceil(Int64, log10(model.N))
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
    for i in 1:model.N
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
             k = sample(1:model.N)
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
  theta_est = estimate_thetas(mu,model.N,topK)
  return log_groups(communities, theta_est)
end
##############

#init_heldout(ratio,heldout_pairs, heldout_map)
communities = batch_infer()

###############
function init_mu(model::LNMMSB, communities::Dict{Int64, Vector{Int64}})
  Belong = Dict{Int64, Vector{Int64}}()
  model.μ_var = 1e-10*ones(Float64, (model.N, model.K))
  for i in 1:model.N
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
  for i in 1:model.N
    model.μ_var[i,:] = log.(model.μ_var[i,:])
  end
end
####
####
####
