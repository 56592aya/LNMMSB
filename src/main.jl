# module main
using LightGraphs
using GraphPlot
using Gadfly
include("AbstractTuple.jl")
include("AbstractDyad.jl")
include("AbstractMMSB.jl")
include("utils.jl")
inputtomodelgen=[250,8]; ## N, K

include("modelgen.jl")
true_eta0 = readdlm("data/true_eta0.txt")[1]

# using JLD
network=FileIO.load("data/network.jld")["network"]
include("LNMMSB.jl")
##should move to LNMMSB
include("init.jl")
communities
lg = LightGraphs.DiGraph(network)
# draw(PNG("Docs/net$(inputtomodelgen[1]).png", 30cm, 30cm), gplot(lg))

onlyK = length(communities)
model=LNMMSB(network, onlyK)
include("modelutils.jl")
mb_zeroer = MiniBatch()
mb=deepcopy(mb_zeroer)
train = deepcopy(mb_zeroer)
train_sample!(train, model)
include("trainutils.jl")
# end


N = 1000
a = vec(collect(Iterators.product(1:N,1:N)))
b = vec(collect((zip(1:N, 1:N))))
c = setdiff(a,b)
size(c)
d=[collect(t) for t in c]
d = hcat(d...)'
involve = VectorList{Int64}(N)
for i in 1:N
    x1=find( x -> x == i, d[:,1])
    x2=find( x -> x == i, d[:,2])
    involve[i] = vcat(x1,x2)
end
