addprocs(8)
@everywhere using BenchmarkTools
@everywhere importall DistributedArrays
@everywhere importall DistributedArrays.SPMD

@everywhere mutable struct mytype
  i::Int64
  j::Int64
  vec::Vector{Float64}
end


@everywhere arr = [mytype(i,j, zeros(Float64, 3)) for i in 1:100 for j in 1:100  if i < j]
@everywhere darr = distribute(arr)
@everywhere function f(arr::Vector{mytype})
  for a in arr
    a.vec[:] .+=1.0
  end
  arr
end

@everywhere function g(arr::DArray{mytype})
  for a in arr
    a.vec[:] .+=1.0
  end
  arr
end
@everywhere function f_parallel(darr::DArray{mytype})
  @parallel for d in darr
    d.vec[:] .+=1.0
  end
  darr
end

@everywhere function g_parallel(darr::DArray{mytype})
  @parallel for d in darr
    d.vec[:] .+=1.0
  end
  darr
end

@btime f(arr)
@btime f_parallel(darr)

@everywhere dout=ddata()

spmd(g, darr, dout, pids=workers(), context)
d_in=d=DArray(I->fill(myid(), (map(length,I)...)), (nworkers(), 2), workers(), [nworkers(),1])





t = ccall((:clock,"libc"), Int64, ())
###more on distributed arrays
addprocs(8)
nprocs()

workers()
procs()
a = rand(8,8)
@everywhere using DistributedArrays

j = distribute(a)
localindexes(j)
fieldnames(j)
#only use local indexes in your parallel segment of your code
j.cuts

addprocs(8)
@everywhere using DistributedArrays
@everywhere mutable struct MyType
  i::Int64
  j::Int64
  vec::Vector{Float64}
end
K = 20
x = [MyType(i,j,zeros(Float64, K)) for i=1:1000 for j in 1:1000 if i < j]
@everywhere using Distributions
x = sample(x, 250000, replace=false)
dx = distribute(x)
typeof(dx)
fieldnames(dx)
@everywhere function f(tvec::DistributedArrays.DArray{MyType,1}, a::Float64)
  local_t = localpart(tvec)
  for i in 1:length(local_t)
    local_t[i].vec += a
  end
end

tic()
@sync for p in workers()
  @async remotecall_fetch(f, p, dx, 1.0)
end
toc()
tic()
for i in 1:length(x)
  x[i].vec .+=1.0
end
toc()


#######
