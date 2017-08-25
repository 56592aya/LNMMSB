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
communities
mb_zeroer = MiniBatch()
mb=deepcopy(mb_zeroer)
include("trainutils.jl")
# train_zeroer = Training()
# train = deepcopy(train_zeroer)
# train_sampling!(train, model)
# train_samplingall!(train, model)




































end
