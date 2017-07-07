module main
include("AbstractTuple.jl")
include("AbstractDyad.jl")
include("AbstractMMSB.jl")
include("utils.jl")
include("modelgen.jl")
gennetwork(150, 4)
network=load("data/network.jld")["network"]
include("LNMMSB.jl")
include("modelutils.jl")
model=LNMMSB(network,4)

end
