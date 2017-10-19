# module main
using LightGraphs
using GraphPlot
using Gadfly
include("AbstractTuple.jl")
include("AbstractDyad.jl")
include("AbstractMMSB.jl")
include("utils.jl")
inputtomodelgen=[250,8];; ## N, K
include("modelgen.jl")
true_eta0 = readdlm("data/true_eta0.txt")[1]

# using JLD
network=FileIO.load("data/network.jld")["network"]
include("LNMMSB.jl")
##should move to LNMMSB
include("init.jl")
communities
lg = LightGraphs.DiGraph(network)
draw(PNG("Docs/net250.png", 30cm, 30cm), gplot(lg))

onlyK = length(communities)
model=LNMMSB(network, onlyK)
include("modelutils.jl")
mb_zeroer = MiniBatch()
mb=deepcopy(mb_zeroer)
include("trainutils.jl")
# train_zeroer = Training()
# train = deepcopy(train_zeroer)
# train_sampling!(train, model)
# train_samplingall!(train, model)
# end
