using DataFrames
using CSV
using LightGraphs
# authtab = readtable("../data/Authors/Authors_2010_Final/Authors $i.csv")
edge_df = zeros(Int64, (1,2))
for yr in 1973:1980
    linktab = readtable("../data/Authors/Links_2010_Final/Links $yr.csv")
    edgetab = linktab[:,[1,2]]
    complete_cases!(edgetab)

    temp = zeros(Int64, (nrow(edgetab),2))
    temp[:,1] = edgetab[:,1]
    temp[:,2] = edgetab[:,2]
    for i in 1:size(temp, 1)
        if temp[i,1] < temp[i,2]
            continue;
        else
            tt = temp[i,2]
            temp[i,2] = temp[i,1]
            temp[i,1] = tt
        end
    end
    edgelist = sortrows(temp, by=x->(x[1],x[2]))
    edge_df = (yr == 1973) ? edgelist :vcat(edge_df,edgelist)
    println("year $yr resolved.")
end
edge_df

##############################
include("utils.jl")
describe(unique(edge_df))
seen = Int64[]
nodekeys = Dict{Int64, Int64}()
unqvals =unique(edge_df)
idgen = 1
for val in unqvals
    if !haskey(nodekeys, val)
        nodekeys[val] = get(nodekeys, val, idgen)
        idgen+=1
    end
end
edgelist_old = deepcopy(edge_df)
edge_df[:,1] = [nodekeys[edge_df[r,1]] for r in 1:size(edge_df,1)]
edge_df[:,2] = [nodekeys[edge_df[r,2]] for r in 1:size(edge_df,1)]
edge_df = sortrows(edge_df, by=x->(x[1],x[2]))
edge_df = convert(DataFrame, edge_df)
keeps = find(!nonunique(edge_df))
edge_df=convert(Array{Int64, 2}, edge_df)
edge_df = edge_df[keeps, :]


N = length(unique(edge_df))
network = Network(N)


for i in 1:size(edge_df,1)
    network[edge_df[i,1], edge_df[i,2]] = network[edge_df[i,2], edge_df[i,1]] = 1
end



lg = LightGraphs.Graph(network);
vold = connected_components(lg)[3]
lg, vnew = induced_subgraph(lg, vold)
N = length(vnew)
network = Network(N)
for i in 1:N
    for j in 1:N
        if j > i
            if has_edge(lg, Edge(i,j))
                network[i,j] = network[j,i] = 1
            end
        end
    end
    ##println(i)
end
include("init.jl")
communities
using GraphPlot;using Gadfly;using PyPlot;

lg = LightGraphs.Graph(network);
GraphPlot.draw(PNG("../data/Authors/net.png", 30cm, 30cm), gplot(lg))


include("LNMMSB.jl")
onlyK = length(communities)
minibatchsize = 1
true_eta0 = 9
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
