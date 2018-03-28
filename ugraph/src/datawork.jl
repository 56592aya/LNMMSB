using DataFrames
using CSV
using LightGraphs
# authtab = readtable("../data/Authors/Authors_2010_Final/Authors $i.csv")
edge_df = zeros(Int64, (1,2))
for yr in 1973:2009
    linktab = readtable("../data/Authors/Links_2010_Final/Links $yr.csv")
    edgetab = linktab[:,[1,2]]
    completecases!(edgetab)

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
edge_df = convert(DataFrame,edge_df)

edge_df = unique(edge_df)
orig_nodes = unique(vcat(edge_df[:x1], edge_df[:x2]))
N = length(orig_nodes)
# include("utils.jl")
network = Network(N)
##############################
##Change labels
node_dict = Dict{Int64,Int64}()

for (i,v) in enumerate(orig_nodes)
    if !haskey(node_dict, v)
        node_dict[v] = get(node_dict, v, i)
    else
        continue
    end
end
for r in 1:nrow(edge_df)
    edge_df[r,1] = node_dict[edge_df[r,1]]
    edge_df[r,2] = node_dict[edge_df[r,2]]

end
sort!(edge_df, cols=[:x1, :x2])
for r in 1:nrow(edge_df)
    network[edge_df[r,1], edge_df[r,2]] = network[edge_df[r,2], edge_df[r,1]] = 1
end



lg = LightGraphs.Graph(network);
# GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netauthorsfull.png", 30cm, 30cm), gplot(lg))
which_connected_component = indmax([length(c) for c in connected_components(lg)])
vold = connected_components(lg)[which_connected_component]

lg, vnew = induced_subgraph(lg, vold)
vertices(lg)

vnew_dict = Dict{Int64,Int64}()

for (i,v) in enumerate(vnew)
    if !haskey(vnew_dict, v)
        vnew_dict[v] = get(vnew_dict, v, i)
    else
        continue
    end
end
#vnew_dict tells us who is in the new 1:N

N = length(vertices(lg))
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
###Some info on degree distribution
sorted_degree_hist = SortedDict(degree_histogram(lg))
using Plots
Plots.plot(collect(keys(sorted_degree_hist)),collect(values(sorted_degree_hist)))
##Maybe another round to get rid of those with degree 1 or 2
remove_nodes = Int[]
for i in 1:N
    if degree(lg, i) < 3
        push!(remove_nodes, i)
    end
end

noremove_nodes = setdiff(1:length(vertices(lg)), remove_nodes)
###Some info on degree distribution

using DataArrays
using DataStructures
sorted_degree_hist = SortedDict(degree_histogram(lg))
# Do I have any empty nodes?
describe(collect(keys(sorted_degree_hist)))
Plots.plot(collect(keys(sorted_degree_hist)),collect(values(sorted_degree_hist)))

lg, vnew = induced_subgraph(lg, noremove_nodes)
N = length(vertices(lg))
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
comm_len = [length(c) for c in values(communities)]


describe(comm_len)
# GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netrevised.png", 30cm, 30cm), gplot(lg))


# include("LNMMSB.jl")
onlyK = length(communities)
minibatchsize = 10
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
