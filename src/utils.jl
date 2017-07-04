
const EPSILON = eps(1e-14)
typealias VectorList{T} Vector{Vector{T}}
typealias MatrixList{T} Vector{Matrix{T}}
typealias Matrix2d{T} Matrix{T}
typealias Matrix3d{T} Array{T,3}
typealias Network{T} SparseMatrixCSC{T,T}
Base.digamma{T<:Number,R<:Integer}(x::T, dim::R) = @fastmath @inbounds sum(digamma(x+.5*(1-i)) for i in 1:dim)
Base.Math.lgamma{T<:Number,R<:Integer}(x::T, dim::R)=.25*(dim*dim-1)*pi+sum(lgamma(x+.5*(1-i)) for i in 1:dim)
Network{T<:Integer}(nrows::T) = SparseMatrixCSC{T,T}(nrows, 2, ones(T, nrows+1), Vector{T}(0), Vector{T}(0))


isnegative(x::Real) = x < 0
ispositive(x::Real) = x > 0
isnegative{T<:Real}(xs::Array{T}) = Bool[isnegative(x) for x in xs]
ispositive{T<:Real}(xs::Array{T}) = Bool[ispositive(x) for x in xs]

function logsumexp{T <:Real}(xs::Array{T})
  a = maximum(xs)
  s = zero(eltype(xs))
  @simd for x in xs
    @inbounds @fastmath s+=exp(x-a)
  end
  log(s)+a
end

function expnormalize{T<:Real}(xs::Array{T})
  s = zero(eltype(xs))
  s=exp(logsumexp(xs))
  xs.=exp.(xs)./s
end

function sort_by_argmax!{T<:Real}(X::Matrix2d)
  n_row=size(X,1)
  n_col = size(X,2)
  ind_max=zeros(Int64, n_row)
  @simd for a in 1:n_row
      @inbounds ind_max[a] = indmax(view(X,a,1:n_col))
  end
  X_tmp = similar(X)
  count = 1
  for j in 1:maximum(ind_max)
    for i in 1:n_row
      if ind_max[i] == j
        for k in 1:n_col
          X_tmp[count,k] = X[i,k]
        end
        count += 1
      end
    end
  end
  # This way of assignment is important in arrays, el by el
  X[:]=X_tmp[:]
  X
end


println();
