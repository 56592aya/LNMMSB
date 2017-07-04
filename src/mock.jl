##Only to test new things
using ForwardDiff
x = rand(100)
A = rand(Float64,(100,100))

f(x::Vector) = 2*(transpose(ones(Float64,100))*x) + transpose(x)*A*x

f(x)
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
