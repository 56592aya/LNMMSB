using LightGraphs
using GraphIO

graph = loadgraph("data/SNAP_COMMS/YouTube/com-youtube.ungraph.csv","graph_key", EdgeListFormat())
graph = Graph(graph)
deg_centrality=degree_centrality(graph)
# between_centrality=betweenness_centrality(graph)
describe(deg_centrality)
topcommstemp = readdlm("data/SNAP_COMMS/YouTube/com-youtube.top5000.cmty.txt")
topcomms = VectorList{Any}()
for c in 1:size(topcommstemp,1)
    push!(topcomms, topcommstemp[c,:])
    topcomms[c] = unique(topcomms[c])
    topcomms[c] = topcomms[c][1:(end-1)]
end

topcomms = convert(VectorList{Int64},topcomms)
topcommlen = map(length, topcomms)
describe(map(length, topcomms))
sorted_topcomm = sortperm(topcommlen, rev=true)

g0,v0=induced_subgraph(graph, topcomms[sorted_topcomm[1]])
g1,v1=induced_subgraph(graph, topcomms[sorted_topcomm[2]])
g2,v2=induced_subgraph(graph, topcomms[sorted_topcomm[3]])
g3,v3=induced_subgraph(graph, topcomms[sorted_topcomm[4]])
g4,v4=induced_subgraph(graph, topcomms[sorted_topcomm[5]])
g5,v5=induced_subgraph(graph, topcomms[sorted_topcomm[6]])
g6,v6=induced_subgraph(graph, topcomms[sorted_topcomm[7]])
g7,v7=induced_subgraph(graph, topcomms[sorted_topcomm[8]])
g8,v8=induced_subgraph(graph, topcomms[sorted_topcomm[9]])
g9,v9=induced_subgraph(graph, topcomms[sorted_topcomm[10]])
g10,v10=induced_subgraph(graph, topcomms[sorted_topcomm[11]])
g11,v11=induced_subgraph(graph, topcomms[sorted_topcomm[12]])
g12,v12=induced_subgraph(graph, topcomms[sorted_topcomm[13]])
g13,v13=induced_subgraph(graph, topcomms[sorted_topcomm[14]])
g14,v14=induced_subgraph(graph, topcomms[sorted_topcomm[15]])
g15,v15=induced_subgraph(graph, topcomms[sorted_topcomm[16]])
g16,v16=induced_subgraph(graph, topcomms[sorted_topcomm[17]])
g17,v17=induced_subgraph(graph, topcomms[sorted_topcomm[18]])
collect(keys(degree_histogram(g1[1])))
collect(values(degree_histogram(g1[1])))
degree_histogram(graph)

args = [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17]
res = Matrix2d{Int64}(length(args), length(args))
for i in 1:length(args)
    for j in 1:length(args)
        if i !=j
            res[i,j] = length(intersect(args[i], args[j]))
        else
            res[i,j] =0
        end
    end
end
res
writedlm("data/SNAP_COMMS/YouTube/chert.txt",res)
UpperTriangular(res)
subnodes = intersect(v0, v1,v2)
v0_ind = findin(v0, subnodes)
v1_ind = findin(v1, subnodes)
v2_ind = findin(v2, subnodes)

degree(graph, subnodes)
mainsortidx = sortperm(degree(graph, subnodes))
Plots.plot()
degree(graph,subnodes)
degree(g0, v0_ind)+degree(g1, v1_ind)+degree(g2, v2_ind)
