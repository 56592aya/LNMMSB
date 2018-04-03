# module main
using LightGraphs;using GraphPlot;using Gadfly;using PyPlot;
inputtomodelgen=[500,14]; ## N, K
include("utils.jl")
include("modelgen.jl")
true_eta0 = readdlm("../data/true_eta0.txt")[1]
network=FileIO.load("../data/network.jld")["network"]
include("LNMMSB.jl")
include("init.jl");println("# of communities : ",length(communities))
lg = LightGraphs.Graph(network);# draw(PNG("Docs/net$(inputtomodelgen[1]).png", 30cm, 30cm), gplot(lg))
onlyK = length(communities)
minibatchsize = 5
model=LNMMSB(network, onlyK,minibatchsize)
model.η0 = 1.1
model.num_peices = 20
model.nho = .01*model.N*(model.N-1)
model.nho = 0
model.mbsize = minibatchsize
model.K = onlyK

mb=deepcopy(model.mb_zeroer)
include("modelutils.jl")
preparedata2!(model)
include("trainutils.jl")
# using ParallelAccelerator
#think about number of columns getting larger and storing observations column wise# this is a big change
function f()
    include("train.jl")
end
# @acc function f()
#     include("train.jl")
# end
# Pkg.add("StatProfilerHTML")
using StatProfilerHTML
f()
Profile.init(100000000, 0.01)
Profile.clear()
@profile f()
StatProfilerHTML.statprofilehtml()

# Pkg.add("FProfile")
# using ProfileView
#
#
# ProfileView.view()
