# module main
using LightGraphs
using GraphPlot
using Gadfly
using PyPlot
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

# The following makes sense only when we have constructed
# "network" and the holdout samples and have training degrees
a = vec(collect(Iterators.product(1:model.N,1:model.N)))
b = vec(collect((zip(1:model.N, 1:model.N))))
c = setdiff(a,b)
preparedata(model)

d=[collect(t) for t in c if is_trainnonlink(network, model, collect(t))]
model.d = hcat(d...)'
model.d = model.d[shuffle(1:end), :]
involve = VectorList{Int64}(model.N)

for i in 1:N
    x1=find( x -> x == i, model.d[:,1])
    x2=find( x -> x == i, model.d[:,2])
    involve[i] = vcat(x1,x2)
    involve[i] = shuffle(involve[i])
end
##Remember "involve[i]" only tells you the row indices of d that involve the i'th node
model.nl_partition = deepcopy(Dict{Int64, VectorList{Int64}}())
# rounds = ones(Int64, model.N)
for a in 1:model.N

    deg=model.train_in[a]+model.train_out[a]
    # length(involve[1])
    # while rounds[a] <= 200

    partition_size = div(length(involve[a]),deg)
    if !haskey(model.nl_partition, a)
        model.nl_partition[a]=getkey(model.nl_partition, a, VectorList{Int64}())
    end
    while length(model.nl_partition[a]) <= 2000
        i=1;
        while (i <= partition_size)
            # if length(nl_partition[a]) == 2000
            #     break;
            # end
            if i == partition_size
                push!(model.nl_partition[a], involve[a][((partition_size-1)*deg + 1):end])
            else
                push!(model.nl_partition[a], involve[a][((i-1)*deg + 1):i*deg])
            end
            i+=1
        end
        # rounds[a] +=1
        shuffle!(involve[a])
    end
end
