include("./utils.jl")
# importall Utils
inputtomodelgen=[150,8]; ## N, K
include("./modelgen.jl")
include("./init.jl")
import Initt:batch_infer
communities = batch_infer(network)
println("");println(" Number of communities  :  ", length(communities))

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
iter=10000;elboevery=200;
include("./train.jl")
import Train:train
train(model, mb, iter, elboevery, communities)
true_θs = (readdlm("../../data/true_thetas.txt"))
est=deepcopy(model.est_θ)
p3=Plots.heatmap(true_θs, yflip=true)
p4=Plots.heatmap(est, yflip=true)
Plots.plot(p4,p3, layout=(2,1))
