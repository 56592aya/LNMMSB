##Only to test new things
using ForwardDiff
x = rand(100)
A = rand(Float64,(100,100))

f(x::Vector) = 2*(transpose(ones(Float64,100))*x) + transpose(x)*A*x
n=150;L=2*eye(4);L0=.5*eye(4); l0=4; l=5; m=rand(4); mu=rand(Float64, (n,4)); M=.2*eye(4);
lambda_a=zeros(Float64,(n,4,4))
for i in 1:n
  lambda_a[i,:,:] = rand(1).*eye(4)
end
f(L::Matrix) = (-l.*trace(inv(L0)*L)+l0*logdet(inv(L0)*L)+n.*logdet(L)-l.*(trace(L*(sum(inv(lambda_a[i]) for i in 1:n))+sum((mu[a,:]-m)*transpose(mu[a,:]-m) for a in 1:n)+n*inv(M))))

f(L)
mydiff=-l*transpose(sum(inv(lambda_a[i]) for i in 1:n)+sum((mu[a,:]-m)*transpose(mu[a,:]-m) for a in 1:n)+n*inv(M))
mydiff
ForwardDiff.gradient(f,L)
ForwardDiff.gradient(f,x)
ForwardDiff.hessian(f,x)
####
using Base.Profile
Profile.init(delay=0.1)
@profile ForwardDiff.hessian(f,x)
Profile.print()
Profile.clear()
using ProfileView
@profile ForwardDiff.hessian(f,x)
ProfileView.view()
@time ForwardDiff.hessian(f,x)
@timev ForwardDiff.hessian(f,x)

using BenchmarkTools
@benchmark ForwardDiff.hessian(f,x)
@code_warntype ForwardDiff.hessian(f,x)
@code_llvm ForwardDiff.hessian(f,x)
@code_typed ForwardDiff.hessian(f,x)

for sym in [:foo, :bar, :baz]
  @eval function $(Symbol(string("func_",sym)))(n::Int64)
          println("hello , ", $(string(sym)))
        end
end
func_baz(2)

ex= :(a~b)
typeof(ex)
dump(ex)
ex.head
ex.args
ex.typ

ex2=Expr(:call,:~,:a,:b)
ex3=:(~(a,b))
ex4 = parse("a~b")
ex==ex2==ex3==ex4
blk = Expr(:block)
push!(blk.args, :(println("Hello!")))
dump(blk)
blk
sym = :foo
f(x,x1,x2) = x^2+x1+x^2-x2
macro x2pxm(ex)
  quote
    local val = $(esc(ex))
    println(val)
    val
  end
end

@x2pxm(f(2,1,3))

function sum_diff(x)
  n = length(x)
  d = 1/(n-1)
  s = zero(eltype(x))
  s = s + (x[2] - x[1]) / d
  for i = 2:length(x)-1
    s = s + (x[i+1] - x[i+1]) / (2*d)
  end
  s = s + (x[n] - x[n-1])/d
end
function sum_diff_fast(x)
  n=length(x)
  d = 1/(n-1)
  s = zero(eltype(x))
  @fastmath s = s + (x[2] - x[1]) / d
  @fastmath for i = 2:n-1
    s = s + (x[i+1] - x[i+1]) / (2*d)
  end
  @fastmath s = s + (x[n] - x[n-1])/d
end
t = rand(10000)
sum_diff(t)
sum_diff_fast(t)
@benchmark sum_diff(t)
@benchmark sum_diff_fast(t)
set_zero_subnormals(true)

fcopy(x) = sum(x[2:end-1])
@views fview(x) = sum(x[2:end-1])
x = rand(10^6);
@time fcopy(x);
@time fview(x);

#
function inner(x, y)
    s = zero(eltype(x))
    for i=1:length(x)
        @inbounds s += x[i]*y[i]
    end
    s
end

function innersimd(x, y)
    s = zero(eltype(x))
    @simd for i=1:length(x)
        @inbounds s += x[i]*y[i]
    end
    s
end
function timeit(n, reps)
    x = rand(Float32,n)
    y = rand(Float32,n)
    s = zero(Float64)
    time = @elapsed for j in 1:reps
        s+=inner(x,y)
    end
    println("GFlop/sec        = ",2.0*n*reps/time*1E-9)
    time = @elapsed for j in 1:reps
        s+=innersimd(x,y)
    end
    println("GFlop/sec (SIMD) = ",2.0*n*reps/time*1E-9)
end
timeit(1000,1000)


M = 4.*eye(4)
l=4
N=10
L=diagm(rand(4))
Minv = inv(M)
f(Minv) = (logdet(Minv)-trace(Minv)-(l)*trace(N.*(L*Minv)))
myexpr(M) = (M-eye(4)-(l*N).*L)
x=ForwardDiff.gradient(f,Minv)
mine=myexpr(M)
diag(mine)./diag(x)
####
mu = reshape(rand(40) ,(N,4))
m = rand(4)

f(m) = trace(m*transpose(m)) + l*sum(transpose(mu[a,:]-m)*L*(mu[a,:]-m) for a in 1:N)
x = ForwardDiff.gradient(f,m)
myexpr(m) = 2*m - (2*l).*L*(sum(mu[a,:] for a in 1:N)) + (2*l*N).*L*m
mine=myexpr(m)
x
mine
###although equal but x - mine != 0####
x - mine .==0## should resolve this


Lambda = zeros(Float64, (N,4,4))
for a in 1:N
  Lambda[a,:,:] = diagm(rand(4))
end
l=5

f(l)=l*4 + 2*sum(lgamma(.5*(l-i+1)) for i in 1:4) - (l-4-1)*sum(digamma(.5*(l-i+1)) for i in 1:4)-4*l*trace(L) - l*trace(L*(sum(inv(Lambda[a,:,:])+(mu[a,:]-m)*transpose(mu[a,:]-m) + inv(M) for a in 1:N )))

f(l)
ForwardDiff.derivative(f,l)
myexpr(l) = 4 -.5*(l-4-1)*sum(trigamma(.5*(l-i+1)) for i in 1:4)-4*trace(L)-trace(L*(sum(inv(Lambda[a,:,:])+(mu[a,:]-m)*transpose(mu[a,:]-m) + inv(M) for a in 1:N )))
myexpr(l)
K=4
g(L) = K*logdet(L)-K*l*trace(L)-l*trace(L*(sum((inv(Lambda[a,:,:])+(mu[a,:]-m)*transpose(mu[a,:]-m)+inv(M)) for a in 1:N)))
g(L)
x = ForwardDiff.gradient(g,L)
myexpr(L) = K*inv(L) - (K*l).*eye(K) - l.*transpose(sum((inv(Lambda[a,:,:])+(mu[a,:]-m)*transpose(mu[a,:]-m)+inv(M)) for a in 1:N))
mine = myexpr(L)

mine ./ x
logdet(2*eye(4))
x = rand(4)
transpose(x)*x == trace(x*transpose(x))==transpose(x)*eye(4)*x

x=K*eye(K)+sum((inv(Lambda[a,:,:])+(mu[a,:]-m)*transpose(mu[a,:]-m)+inv(M)) for a in 1:N)
transpose(x) ==x

using Optim
f(x) = -(transpose(x)*x)-transpose(x)*(x -3)
function g!(storage,x)
  storage = -4.*transpose(x) .+ 3
end
ForwardDiff.gradient(f,[2,2])
optimize(f,g!,[2.0,2.0])

K=4
p = rand(K)
mu = rand(K)
p'*mu
f(mu) = p'*mu
ForwardDiff.gradient(f, mu)
ones(K)'*mu
temp=zeros(Int64,K)
temp[1] = 1
temp
δ(x) = circshift(temp,x-1)
δ(2)
g(mu,l) = exp(δ(l)'*mu)
g(mu) = [g(mu,l) for l in 1:K]
f(mu) = log(ones(K)'*[exp(circshift(temp,l-1)'*mu) for l in 1:K])
res=ForwardDiff.gradient(f, mu)
g(mu)./res
sum(g(mu))
g(mu)./sum(g(mu))

resHess=ForwardDiff.hessian(f, mu)

L0 = 2.*eye(2)
L=reshape([5.0,4.0,4.0,5.0], (2,2))
f(L) = logdet(inv(L0)*L)
ForwardDiff.gradient(f, L)
inv(L)

f(mu) = log(ones(K)'*[exp(mu[i]) for i in 1:K])
ForwardDiff.gradient(f,mu)
temp=[exp(mu[i])/sum(exp.(mu)) for i in 1:K]

ForwardDiff.hessian(f,mu)
diagm(temp)-temp*temp'

Lam = diagm(rand(K))
f(Lam) = log(ones(K)'*[exp(.5*Lam[i,i]) for i in 1:K])
ForwardDiff.gradient(f,Lam)

temp=diagm(.5*[exp(.5*Lam[i,i])/sum(exp.(.5.*(diag(Lam)))) for i in 1:K])

ForwardDiff.hessian(f,Lam)
.25*(diagm(temp)-temp*temp')
g(Lam) = logdet(Lam)
ForwardDiff.hessian(g, Lam)
g(Lam) = inv(Lam)
ForwardDiff.gradient(g, Lam)
inv(eye(4))


l=4
L=diagm(rand(K))
Laminv = diagm(rand(K))
f(Laminv) = -.5*l*trace(L*(Laminv)) + .5*logdet(Laminv)
ForwardDiff.gradient(f, Laminv)
-.5*l*L + .5*inv(Laminv)
ForwardDiff.hessian(f, Laminv)
-.5*kron(inv(Laminv),inv(Laminv'))
Laminv = rand(K)
f(Laminv) = .5*Laminv
ForwardDiff.gradient(f, Laminv)

dat = readcsv("data/Network.csv")
dat = convert(Matrix2d{Int64},dat[2:end,[5,6]])
dat = sortrows(dat, by=x->(x[1],x[2]))
# dat = unique(dat)
using LightGraphs
test=DiGraph(maximum(vcat(dat[:,1],dat[:,2])))
for r in 1:size(dat,1)
  add_edge!(test,Edge(dat[r,1], dat[r,2]))
end
test
x = strongly_connected_components(test)
y=[length(i) for i in x]
x = x[indmax(y)]
test=induced_subgraph(test,x)
test = test[1]
is_strongly_connected(test)
x = adjacency_matrix(test)
dat = convert(Network, x)


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
_α = 1.0/size(dat,1)
mu = [KeyVal(0,0.0) for i=1:size(dat,1), j=1:topK]
munext = [Dict{Int64, Int64}() for i in 1:size(dat,1)]
maxmu = zeros(Int64,size(dat,1))
communities = Dict{Int64, Vector{Int64}}()
ulinks = Vector{Pair{Int64, Int64}}()
#undirected
x,y,z=findnz(dat)
for row in 1:nnz(dat)
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
  for i in 1:size(dat,1)
      mu[i,1].first=i
      mu[i,1].second = 1.0 + rand()
      maxmu[i] = i
      for j in 2:topK
        mu[i,j].first = (i+j-1)%size(dat,1)
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
  for iter in 1:ceil(Int64, log10(size(dat,1)))
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
    for i in 1:size(dat,1)
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
             k = sample(1:size(dat,1))
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
  theta_est = estimate_thetas(mu,size(dat,1),topK)
  return log_groups(communities, theta_est)
end
##############

#init_heldout(ratio,heldout_pairs, heldout_map)
communities = batch_infer()
comm_sizes=sort([length(i) for i in values(communities)],rev=true)
comm_sizes[comm_sizes .> 20]
Plots.histogram(comm_sizes)



vec = [.1, .2, .3, .4]
f(vec) = 1.0./vec
sfx(vec) = exp.(.5*vec)./sum(exp.(.5*vec))
ForwardDiff.jacobian(f, vec)
ForwardDiff.gradient(sfx, vec)
