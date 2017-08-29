import Base: ==, hash

const EPSILON = 1e-15
VectorList{T} = Vector{Vector{T}}
MatrixList{T} = Vector{Matrix{T}}
Matrix2d{T}   = Matrix{T}
Matrix3d{T}   = Array{T,3}
Network{T}    = SparseMatrixCSC{T,T}

struct Dyad <: AbstractDyad
 src::Int64
 dst::Int64
 # isholdout::Bool
end

mutable struct Link <: AbstractDyad
  src::Int64
  dst::Int64
  ϕout::Vector{Float64}
  ϕin::Vector{Float64}
  # function Link(src, dst) = new(src, dst, ϕout,ϕin)
  # isholdout::Bool
end
mutable struct NonLink <: AbstractDyad
  src::Int64
  dst::Int64
  ϕout::Vector{Float64}
  ϕin::Vector{Float64}
  # function NonLink(src, dst) = new(src, dst, ϕout,ϕin)
  # isholdout::Bool
end
==(x::Dyad, y::Dyad) = x.src == y.src && x.dst == y.dst
==(x::Link, y::Link) = x.src == y.src && x.dst == y.dst
==(x::NonLink, y::NonLink) = x.src == y.src && x.dst == y.dst
hash(x::Dyad, h::UInt) = hash(x.src, hash(x.dst, hash(0x7d6979235cb005d0, h)))
hash(x::Link, h::UInt) = hash(x.src, hash(x.dst, hash(0x7d6979235cb005d0, h)))
hash(x::NonLink, h::UInt) = hash(x.src, hash(x.dst, hash(0x7d6979235cb005d0, h)))

struct Triad <: AbstractTuple
 head::Int64
 middle::Int64
 tail::Int64
end

mutable struct MiniBatch
	mblinks::Vector{Link}
	mbnonlinks::Vector{NonLink}
	mballnodes::Set{Int64}
  mbfnadj::Dict{Int64,Vector{Int64}}
  mbbnadj::Dict{Int64,Vector{Int64}}
  function MiniBatch()
    mblinks=Vector{Link}()
    mbnonlinks=Vector{NonLink}()
    mballnodes=Set{Int64}()
    mbfnadj=Dict{Int64,Vector{Int64}}()
    mbbnadj=Dict{Int64,Vector{Int64}}()
    new(mblinks,mbnonlinks,mballnodes,mbfnadj,mbbnadj)
  end
end

mutable struct Training
	trainlinks::Vector{Link}
	trainnonlinks::Vector{NonLink}
	trainallnodes::Set{Int64}
  trainfnadj::Dict{Int64,Vector{Int64}}
  trainbnadj::Dict{Int64,Vector{Int64}}
  function Training()
    traininglinks=Vector{Link}()
    trainingnonlinks=Vector{NonLink}()
    trainingallnodes=Set{Int64}()
    trainingfnadj=Dict{Int64,Vector{Int64}}()
    trainingbnadj=Dict{Int64,Vector{Int64}}()
    new(traininglinks,trainingnonlinks,trainingallnodes,trainingfnadj,trainingbnadj)
  end
end

Network{T<:Integer}(nrows::T) = SparseMatrixCSC{T,T}(nrows, nrows, ones(T, nrows+1), Vector{T}(0), Vector{T}(0))
Base.digamma{T<:Number,R<:Integer}(x::T, dim::R) = @fastmath @inbounds return sum(digamma(x+.5*(1-i)) for i in 1:dim)

Base.Math.lgamma{T<:Number,R<:Integer}(x::T, dim::R)=.25*(dim*dim-1)*pi+sum(lgamma(x+.5*(1-i)) for i in 1:dim)





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
  a = maximum(xs)
  for i in 1:size(xs,1)
    xs[i] = exp(xs[i]-a)
    s+=xs[i]##check
  end
  for i in 1:size(xs,1)
    xs[i] = xs[i]/s##check
  end
  xs[:]
end

function softmax{T<:Real}(xs::Array{T}, k::Int64)
  exp(xs[k])/exp(logsumexp(xs))
end

function sort_by_argmax!{T<:Real}(X::Matrix2d{T})
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

##Need to do a lot with the validation , mb sampling and so on, and the functions below depend on these.
function isalink(network::Network{Int64},a::Int64, b::Int64)
  network[a,b] == 1
end
function issink(network::Network{Int64},curr::Int64, q::Int64)
  network[curr,q] == 1
end
function issource(network::Network{Int64},curr::Int64, q::Int64)
  network[q,curr] == 1
end
function sinks(model::Network{Int64},curr::Int64, N::Int64)
  [b for b in 1:N if isalink(network, curr, b)]
end
function sources(network::Network{Int64},curr::Int64, N::Int64)
  [b for b in 1:N if isalink(network, b, curr)]
end
function neighbors(network::Network{Int64},curr::Int64, N::Int64)
  vcat(sinks(network, curr, N), sources(network, curr, N))
end




println();
