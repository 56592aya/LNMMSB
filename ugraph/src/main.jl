# module main
using LightGraphs;using GraphPlot;using Gadfly;using PyPlot;
inputtomodelgen=[250,11]; ## N, K
include("utils.jl")
include("modelgen.jl")
true_eta0 = readdlm("../data/true_eta0.txt")[1]
network=FileIO.load("../data/network.jld")["network"]
include("LNMMSB.jl")
include("init.jl");println("# of communities : ",length(communities))

lg = LightGraphs.Graph(network);# draw(PNG("Docs/net$(inputtomodelgen[1]).png", 30cm, 30cm), gplot(lg))
onlyK = length(communities)
minibatchsize = 1
model=LNMMSB(network, onlyK,minibatchsize)
model.nho = .01*model.N*(model.N-1)
model.nho = 0
model.mbsize = minibatchsize
model.K = onlyK
mb=deepcopy(model.mb_zeroer)
include("modelutils.jl")
preparedata2!(model)
include("trainutils.jl")
