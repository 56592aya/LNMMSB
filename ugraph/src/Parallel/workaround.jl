addprocs(8)

@everywhere using DistributedArrays
@everywhere mutable struct mytype
  i::Int64
  j::Int64
  vec::Vector{Float64}
end
@everywhere arr = [mytype(i,j, zeros(Float64, 3)) for i in 1:10 for j in 1:10  if i < j]
@everywhere darr = distribute(arr)
@everywhere function f(arr::Vector{mytype})
  for a in arr
    a.vec[:] .+=1.0
  end
end
@everywhere function f_parallel(darr::DArray{mytype})
  @parallel for d in darr
    d.vec[:] .+=1.0
  end
end
tic()
f(arr)
toc()
