
VectorList{T} = Vector{Vector{T}}
MatrixList{T} = Vector{Matrix{T}}
Matrix2d{T}   = Matrix{T}
Matrix3d{T}   = Array{T,3}
Network{T}    = SparseMatrixCSC{T,T}

Map{R,T} = Dict{R,T}
struct Dyad
 src::Int64
 dst::Int64
end

mutable struct Link
  src::Int64
  dst::Int64
  ϕ::Vector{Float64}
end

mutable struct NonLink
  src::Int64
  dst::Int64
  ϕout::Vector{Float64}
  ϕin::Vector{Float64}
end

type KeyVal
  first::Int64
  second::Float64
end

mutable struct MiniBatch
	mblinks::Vector{Link}
	mbnonlinks::Vector{NonLink}
	mbnodes::Vector{Int64}
  mbnot::Dict{Int64,Vector{Int64}}
  function MiniBatch()
    mblinks=Vector{Link}()
    mbnonlinks=Vector{NonLink}()
    mbnodes=Vector{Int64}()
    mbnot=Dict{Int64,Vector{Int64}}()
    new(mblinks,mbnonlinks,mbnodes,mbnot)
  end
end
