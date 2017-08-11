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

Lam = (rand(K))
f(Lam) = log(ones(K)'*[exp(.5*Lam[i]) for i in 1:K])
ForwardDiff.gradient(f,Lam)
temp=[exp(.5*Lam[i])/sum(exp.(.5.*(Lam))) for i in 1:K]
.5*(temp)
ForwardDiff.hessian(f,Lam)
.25*(diagm(temp)-temp*temp')
