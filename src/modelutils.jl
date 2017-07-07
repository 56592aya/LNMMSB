struct Dyad <: AbstractDyad
 src::Int64
 dst::Int64
 isholdout::Bool
end
mutable struct Link <: AbstractDyad
  src::Int64
  dst::Int64
  ϕout::Vector{Float64}
  ϕin::Vector{Float64}
  isholdout::Bool
end
mutable struct NonLink <: AbstractDyad
  src::Int64
  dst::Int64
  ϕout::Vector{Float64}
  ϕin::Vector{Float64}
  isholdout::Bool
end

struct Triad <: AbstractTuple
 head::Int64
 middle::Int64
 tail::Int64
end

mutable struct MiniBatch
 mblinks::VectorList{Link}
 mbnonlinks::VectorList{NonLink}
 mballnodes::Vector{Int64}
 function MiniBatch(model::LNMMSB)

 end
 mb=new()
end
function mbsampling(model::LNMMSB, mb::MiniBatch)
  neighbors()
end
function setholdout(model::LNMMSB)

end
