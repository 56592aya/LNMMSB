module main
include("utils.jl")
include("AbstractTuple.jl")
include("AbstractDyad.jl")
include("AbstractMMSB.jl")
include("modelgen.jl")
include("LNMMSB.jl")
include("modelutils.jl")
network=load("data/network.jld")["network"]
model=LNMMSB(network,4)

end
