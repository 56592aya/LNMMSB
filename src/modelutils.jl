

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
 # sample model.nval from nonzeros of model.network
 # create link and dyad objects and set their dyad and link to isholdout=true
 # sample model.nval from zeros of model.network
 # create nonlink and dyad objects and set their dyad and nonlink to isholdout=true
 # better be a dyad dict where search is easier, and if a key is not found it is for sure not in heldout
end
