module main
include("AbstractTuple.jl")
include("AbstractDyad.jl")
include("AbstractMMSB.jl")
include("utils.jl")
inputtomodelgen=[150,4]; ## N, K
include("modelgen.jl")
# using JLD
network=FileIO.load("data/network.jld")["network"]
include("LNMMSB.jl")
##should move to LNMMSB
model=LNMMSB(network, inputtomodelgen[2])
include("modelutils.jl")
include("init.jl")
mb_zeroer = MiniBatch()
train_zeroer = Training()
train = deepcopy(train_zeroer)
train_sampling!(train, model)
train_samplingall!(train, model)




































end
