# module main
using LightGraphs;using GraphPlot;using Gadfly;using PyPlot;
inputtomodelgen=[150,4]; ## N, K
include("utils.jl")
include("modelgen.jl")
true_eta0 = readdlm("data/true_eta0.txt")[1]
network=FileIO.load("data/network.jld")["network"]
include("LNMMSB.jl")
include("init.jl");println("# of communities : ",length(communities))

lg = LightGraphs.DiGraph(network);# draw(PNG("Docs/net$(inputtomodelgen[1]).png", 30cm, 30cm), gplot(lg))
onlyK = length(communities)
model=LNMMSB(network, onlyK)
mb=deepcopy(model.mb_zeroer)
include("modelutils.jl")
meth = "isns2"
preparedata(model,true, meth)
# mbsampling!(mb, model, "isns", model.mbsize)
mbsampling!(mb, model, meth, model.mbsize)
include("trainutils.jl")



# g = LightGraphs.DiGraph(network)
# inf_nodes = [Int[] for i in 1:model.N]
# for a in 1:model.N
#     for b in neighbors(g,a)
#         for c in neighbors(g,b)
#             if (network[a,c] == 0) && (network[c,a] == 0) && (c !=a)
#                 push!(inf_nodes[a], c)
#             end
#         end
#     end
#     inf_nodes[a] = unique(inf_nodes[a])
# end
# a=94
# model.train_nonlinks[model.nonlink_setmap[a][1]]













# end
