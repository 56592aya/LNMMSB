module main
include("AbstractTuple.jl")
include("AbstractDyad.jl")
include("AbstractMMSB.jl")
include("utils.jl")
inputtomodelgen=[150,4] ## N, K
include("modelgen.jl")
# using JLD
network=FileIO.load("data/network.jld")["network"]
include("LNMMSB.jl")
##should move to LNMMSB
model=LNMMSB(network, inputtomodelgen[2])
include("modelutils.jl")
mb_zeroer = MiniBatch()





































end
