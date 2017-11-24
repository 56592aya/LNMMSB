include("AbstractTuple.jl")
include("AbstractDyad.jl")
include("AbstractMMSB.jl")
import Base: ==, hash

const EPSILON = 1e-5
VectorList{T} = Vector{Vector{T}}
MatrixList{T} = Vector{Matrix{T}}
Matrix2d{T}   = Matrix{T}
Matrix3d{T}   = Array{T,3}
Network{T}    = SparseMatrixCSC{T,T}

struct Dyad <: AbstractDyad
 src::Int64
 dst::Int64
end

mutable struct Link <: AbstractDyad
  src::Int64
  dst::Int64
  ϕout::Vector{Float64}
  ϕin::Vector{Float64}

  # ϕoutold::Vector{Float64}
  # ϕinold::Vector{Float64}
end

mutable struct NonLink <: AbstractDyad
  src::Int64
  dst::Int64
  ϕout::Vector{Float64}
  ϕin::Vector{Float64}
  # ϕoutold::Vector{Float64}
  # ϕinold::Vector{Float64}
end
==(x::Dyad, y::Dyad) = x.src == y.src && x.dst == y.dst
==(x::Link, y::Link) = x.src == y.src && x.dst == y.dst
==(x::NonLink, y::NonLink) = x.src == y.src && x.dst == y.dst
==(x::Dyad, y::Link) = x.src == y.src && x.dst == y.dst
==(x::Dyad, y::NonLink) = x.src == y.src && x.dst == y.dst
==(x::Link, y::Dyad) = x.src == y.src && x.dst == y.dst
==(x::NonLink, y::Dyad) = x.src == y.src && x.dst == y.dst

#
# hash(x::Dyad, h::UInt) = hash(x.src, hash(x.dst, hash(0x7d6979235cb005d0, h)))
# hash(x::Link, h::UInt) = hash(x.src, hash(x.dst, hash(0x7d6979235cb005d0, h)))
# hash(x::NonLink, h::UInt) = hash(x.src, hash(x.dst, hash(0x7d6979235cb005d0, h)))

struct Triad <: AbstractTuple
 head::Int64
 middle::Int64
 tail::Int64
end

mutable struct MiniBatch
	mblinks::Vector{Link}
	mbnonlinks::Vector{NonLink}
	mbnodes::Vector{Int64}
  mbfnadj::Dict{Int64,Vector{Int64}}
  mbbnadj::Dict{Int64,Vector{Int64}}
  function MiniBatch()
    mblinks=Vector{Link}()
    mbnonlinks=Vector{NonLink}()
    mbnodes=Vector{Int64}()
    mbfnadj=Dict{Int64,Vector{Int64}}()
    mbbnadj=Dict{Int64,Vector{Int64}}()
    new(mblinks,mbnonlinks,mbnodes,mbfnadj,mbbnadj)
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

Map{R,T} = Dict{R,T}


Network{T<:Integer}(nrows::T) = SparseMatrixCSC{T,T}(nrows, nrows, ones(T, nrows+1), Vector{T}(0), Vector{T}(0))

# Base.digamma{T<:Number,R<:Integer}(x::T, dim::R) = @fastmath @inbounds return sum(digamma(x+.5*(1-i)) for i in 1:dim)
# Base.Math.lgamma{T<:Number,R<:Integer}(x::T, dim::R)=.25*(dim*dim-1)*log(pi)+sum(lgamma(x+.5*(1-i)) for i in 1:dim)

function digamma_(x::Float64)
	p=zero(Float64)
  x=x+6.0
  p=1.0/(x*x)
  p=(((0.004166666666667*p-0.003968253986254)*p+0.008333333333333)*p-0.083333333333333)*p
  p=p+log(x)-0.5/x-1.0/(x-1.0)-1.0/(x-2.0)-1.0/(x-3.0)-1.0/(x-4.0)-1.0/(x-5.0)-1.0/(x-6.0)
  p
end

# import Base.Math.lgamma
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
    alpha = -Inf
    r = 0.0
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
    alpha = -Inf
    r = 0.0

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

##########
print();
