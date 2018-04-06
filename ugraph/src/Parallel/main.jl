include("./utils.jl")
inputtomodelgen=[1000,100]; ## N, K
include("./modelgen.jl")
include("./init.jl")
import Initt:batch_infer
communities = batch_infer(network)
println("");println(" Number of communities  :  ", length(communities))

comm_len = [length(c) for c in values(communities)]
include("./LNMMSB.jl")
import Model:LNMMSB

# lg = LightGraphs.Graph(network);# draw(PNG("Docs/net$(inputtomodelgen[1]).png", 30cm, 30cm), gplot(lg))
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


include("./modelutils.jl")
include("./trainutils.jl")
iter=10000;elboevery=500;
include("./train.jl")
import Train:train
(model, mb)=train(model, mb, iter, elboevery, communities)
using StatProfilerHTML
Profile.init(100000000, 0.01)
Profile.clear()
@profile train(model, mb, iter, elboevery, communities)
StatProfilerHTML.statprofilehtml()
# true_θs = (readdlm("../../data/true_thetas.txt"))
# est=deepcopy(model.est_θ)
# p3=Plots.heatmap(true_θs, yflip=true)
# p4=Plots.heatmap(est, yflip=true)
# Plots.plot(p4,p3, layout=(2,1))
using BenchmarkTools
println(@code_lowered TrainUtils.updatephinlout!(model, mb, mb.mbnonlinks[1], "check"))
