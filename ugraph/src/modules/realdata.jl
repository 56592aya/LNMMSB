using DataFrames
using CSV
using LightGraphs
using MetaGraphs
import JSON
using GraphPlot, Compose

#####
linktab1 = readtable("../../data/Authors/Links_2010_Final/Links 1973.csv", header=true)
linktab2 = readtable("../../data/Authors/Links_2010_Final/Links 1974.csv", header=true)
linktab3 = readtable("../../data/Authors/Links_2010_Final/Links 1975.csv", header=true)

linktab = vcat(linktab1, linktab2, linktab3)

completecases!(linktab)

    N = length(unique(vcat(linktab[:,1], linktab[:,2])))
    g = SimpleGraph(N)
    mg = MetaGraph(g)

    #maybe instead of using authtab
    id_dict = Dict{Int64, Int64}()
    vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
    vcat_authorids = unique(vcat_authorids)
    count = 1
    for i in 1:size(vcat_authorids,1)
        auth_id = vcat_authorids[i]
        if !(auth_id in collect(keys(id_dict)))
            id_dict[auth_id]=getkey(id_dict, auth_id,count)
            count +=1
        end
    end
    for r in 1:size(linktab,1)
        id1 = id_dict[linktab[r,:researcher_id_1]]
        id2 = id_dict[linktab[r,:researcher_id_2]]
        add_edge!(g, id1, id2)
    end
    nv(g)
    ne(g)
    #get rid of those that have no connections at all
    #remember you have to keep the original ids as well
    #but also if taken only the lgc, this is very sparse, so maybe see if you can do better
    connected_components(g)


    # GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netauthorsfull.png", 30cm, 30cm), gplot(lg))
    which_connected_component = indmax([length(c) for c in connected_components(g)])
    vold = connected_components(g)[which_connected_component]

    lg, vnew = induced_subgraph(g, vold)
    vertices(lg)

    vnew_dict = Dict{Int64,Int64}()
    MetaGraph(lg)
    for (i,v) in enumerate(vnew)
        if !haskey(vnew_dict, v)
            vnew_dict[v] = get(vnew_dict, v, i)
        else
            continue
        end
    end
    #vnew_dict tells us who is in the new 1:N

    N = length(vertices(lg))
    include("utils.jl")
    using Utils
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
    iter=10000;elboevery=500;
    include("./train.jl")
    import Train:train
    (model, mb)=train(model, mb, iter, elboevery, communities)
    est=deepcopy(model.est_θ)
    # sort_by_argmax!(est)
    #Plots.heatmap(est, yflip=true)
    est_comms = Dict{Int64, Vector{Int64}}()
    for i in 1:N
        which_c = [j for j in 1:model.K if est[i,j] > (.2)]
        if !haskey(est_comms, vnew[i])
            est_comms[vnew[i]] = getkey(est_comms, vnew[i], which_c)
        end
    end

    est_comms
    #organize communities:
    comm_inv = Dict{Int64, Vector{Int64}}()
    for i in collect(keys(est_comms))
        for c in est_comms[i]
            if !haskey(comm_inv, c)
                comm_inv[c] = getkey(comm_inv, c, Int64[i])
            else
                push!(comm_inv[c], i)
            end
        end
    end
    comm_inv
    #now match names with actual authors
    id_dict_inv = Dict{Int64, Int64}()
    for v in collect(keys(id_dict))
        id_dict_inv[id_dict[v]] = get(id_dict_inv, id_dict[v], v)
    end


    real_comms = Dict{Int64, Vector{Int64}}()
    for c in collect(keys(comm_inv))
        real_comms[c] = getkey(real_comms, c, [id_dict_inv[i] for i in comm_inv[c]])
    end


    stringdata1 = JSON.json(real_comms)
    stringdata2 = JSON.json(id_dict_inv)
    vnew_dict = Dict{Int64, Int64}()
    for i in 1:length(vnew)
        vnew_dict[i] = getkey(vnew_dict, i, vnew[i])
    end
    stringdata3 = JSON.json(vnew_dict)



    # write the file with the stringdata variable information
    open("./Results/Together/real_comm73-75.json", "w") do f
            write(f, stringdata1)
    end
    open("./Results/Separate/id_dict_inv73-75.json", "w") do f
            write(f, stringdata2)
    end
    open("./Results/Separate/vnew73-75.json", "w") do f
            write(f, stringdata3)
    end
    writedlm("./Results/Separate/est_theta_73-75.csv",est)
    #filter out the largest component from the csv
    lg = LightGraphs.Graph(network)



    nodelabel = [vnew[i] for i in 1:N]
    nodelabel = [id_dict_inv[i] for i in nodelabel]
    draw(PNG("./Results/Separate/net73-75.png" ,30cm, 30cm), gplot(lg,nodelabel =nodelabel))
    savegraph("./Results/Separate/graph73-75.lgz", lg)
    println("done with 73-75")
    #####
    linktab1 = readtable("../../data/Authors/Links_2010_Final/Links 1976.csv", header=true)
    linktab2 = readtable("../../data/Authors/Links_2010_Final/Links 1977.csv", header=true)
    linktab3 = readtable("../../data/Authors/Links_2010_Final/Links 1978.csv", header=true)

    linktab = vcat(linktab1, linktab2, linktab3)

    completecases!(linktab)

        N = length(unique(vcat(linktab[:,1], linktab[:,2])))
        g = SimpleGraph(N)
        mg = MetaGraph(g)

        #maybe instead of using authtab
        id_dict = Dict{Int64, Int64}()
        vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
        vcat_authorids = unique(vcat_authorids)
        count = 1
        for i in 1:size(vcat_authorids,1)
            auth_id = vcat_authorids[i]
            if !(auth_id in collect(keys(id_dict)))
                id_dict[auth_id]=getkey(id_dict, auth_id,count)
                count +=1
            end
        end
        for r in 1:size(linktab,1)
            id1 = id_dict[linktab[r,:researcher_id_1]]
            id2 = id_dict[linktab[r,:researcher_id_2]]
            add_edge!(g, id1, id2)
        end
        nv(g)
        ne(g)
        #get rid of those that have no connections at all
        #remember you have to keep the original ids as well
        #but also if taken only the lgc, this is very sparse, so maybe see if you can do better
        connected_components(g)


        # GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netauthorsfull.png", 30cm, 30cm), gplot(lg))
        which_connected_component = indmax([length(c) for c in connected_components(g)])
        vold = connected_components(g)[which_connected_component]

        lg, vnew = induced_subgraph(g, vold)
        vertices(lg)

        vnew_dict = Dict{Int64,Int64}()
        MetaGraph(lg)
        for (i,v) in enumerate(vnew)
            if !haskey(vnew_dict, v)
                vnew_dict[v] = get(vnew_dict, v, i)
            else
                continue
            end
        end
        #vnew_dict tells us who is in the new 1:N

        N = length(vertices(lg))
        include("utils.jl")
        using Utils
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
        iter=10000;elboevery=500;
        include("./train.jl")
        import Train:train
        (model, mb)=train(model, mb, iter, elboevery, communities)
        est=deepcopy(model.est_θ)
        # sort_by_argmax!(est)
        #Plots.heatmap(est, yflip=true)
        est_comms = Dict{Int64, Vector{Int64}}()
        for i in 1:N
            which_c = [j for j in 1:model.K if est[i,j] > (.2)]
            if !haskey(est_comms, vnew[i])
                est_comms[vnew[i]] = getkey(est_comms, vnew[i], which_c)
            end
        end

        est_comms
        #organize communities:
        comm_inv = Dict{Int64, Vector{Int64}}()
        for i in collect(keys(est_comms))
            for c in est_comms[i]
                if !haskey(comm_inv, c)
                    comm_inv[c] = getkey(comm_inv, c, Int64[i])
                else
                    push!(comm_inv[c], i)
                end
            end
        end
        comm_inv
        #now match names with actual authors
        id_dict_inv = Dict{Int64, Int64}()
        for v in collect(keys(id_dict))
            id_dict_inv[id_dict[v]] = get(id_dict_inv, id_dict[v], v)
        end


        real_comms = Dict{Int64, Vector{Int64}}()
        for c in collect(keys(comm_inv))
            real_comms[c] = getkey(real_comms, c, [id_dict_inv[i] for i in comm_inv[c]])
        end


        stringdata1 = JSON.json(real_comms)
        stringdata2 = JSON.json(id_dict_inv)
        vnew_dict = Dict{Int64, Int64}()
        for i in 1:length(vnew)
            vnew_dict[i] = getkey(vnew_dict, i, vnew[i])
        end
        stringdata3 = JSON.json(vnew_dict)



        # write the file with the stringdata variable information
        open("./Results/Together/real_comm76-78.json", "w") do f
                write(f, stringdata1)
        end
        open("./Results/Separate/id_dict_inv76-78.json", "w") do f
                write(f, stringdata2)
        end
        open("./Results/Separate/vnew76-78.json", "w") do f
                write(f, stringdata3)
        end
        writedlm("./Results/Separate/est_theta_76-78.csv",est)
        #filter out the largest component from the csv
        lg = LightGraphs.Graph(network)



        nodelabel = [vnew[i] for i in 1:N]
        nodelabel = [id_dict_inv[i] for i in nodelabel]
        draw(PNG("./Results/Separate/net76-78.png" ,30cm, 30cm), gplot(lg,nodelabel =nodelabel))
        savegraph("./Results/Separate/graph76-78.lgz", lg)
        println("done with 76-78")

#########################
linktab1 = readtable("../../data/Authors/Links_2010_Final/Links 1973.csv", header=true)
linktab2 = readtable("../../data/Authors/Links_2010_Final/Links 1974.csv", header=true)
linktab3 = readtable("../../data/Authors/Links_2010_Final/Links 1975.csv", header=true)
linktab4 = readtable("../../data/Authors/Links_2010_Final/Links 1976.csv", header=true)
linktab5 = readtable("../../data/Authors/Links_2010_Final/Links 1977.csv", header=true)
linktab6 = readtable("../../data/Authors/Links_2010_Final/Links 1978.csv", header=true)

linktab = vcat(linktab1, linktab2, linktab3,linktab4, linktab5, linktab6)

completecases!(linktab)

    N = length(unique(vcat(linktab[:,1], linktab[:,2])))
    g = SimpleGraph(N)
    mg = MetaGraph(g)

    #maybe instead of using authtab
    id_dict = Dict{Int64, Int64}()
    vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
    vcat_authorids = unique(vcat_authorids)
    count = 1
    for i in 1:size(vcat_authorids,1)
        auth_id = vcat_authorids[i]
        if !(auth_id in collect(keys(id_dict)))
            id_dict[auth_id]=getkey(id_dict, auth_id,count)
            count +=1
        end
    end
    for r in 1:size(linktab,1)
        id1 = id_dict[linktab[r,:researcher_id_1]]
        id2 = id_dict[linktab[r,:researcher_id_2]]
        add_edge!(g, id1, id2)
    end
    nv(g)
    ne(g)
    #get rid of those that have no connections at all
    #remember you have to keep the original ids as well
    #but also if taken only the lgc, this is very sparse, so maybe see if you can do better
    connected_components(g)


    # GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netauthorsfull.png", 30cm, 30cm), gplot(lg))
    which_connected_component = indmax([length(c) for c in connected_components(g)])
    vold = connected_components(g)[which_connected_component]

    lg, vnew = induced_subgraph(g, vold)
    vertices(lg)

    vnew_dict = Dict{Int64,Int64}()
    MetaGraph(lg)
    for (i,v) in enumerate(vnew)
        if !haskey(vnew_dict, v)
            vnew_dict[v] = get(vnew_dict, v, i)
        else
            continue
        end
    end
    #vnew_dict tells us who is in the new 1:N

    N = length(vertices(lg))
    include("utils.jl")
    using Utils
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
    iter=10000;elboevery=500;
    include("./train.jl")
    import Train:train
    (model, mb)=train(model, mb, iter, elboevery, communities)
    est=deepcopy(model.est_θ)
    # sort_by_argmax!(est)
    #Plots.heatmap(est, yflip=true)
    est_comms = Dict{Int64, Vector{Int64}}()
    for i in 1:N
        which_c = [j for j in 1:model.K if est[i,j] > (.2)]
        if !haskey(est_comms, vnew[i])
            est_comms[vnew[i]] = getkey(est_comms, vnew[i], which_c)
        end
    end

    est_comms
    #organize communities:
    comm_inv = Dict{Int64, Vector{Int64}}()
    for i in collect(keys(est_comms))
        for c in est_comms[i]
            if !haskey(comm_inv, c)
                comm_inv[c] = getkey(comm_inv, c, Int64[i])
            else
                push!(comm_inv[c], i)
            end
        end
    end
    comm_inv
    #now match names with actual authors
    id_dict_inv = Dict{Int64, Int64}()
    for v in collect(keys(id_dict))
        id_dict_inv[id_dict[v]] = get(id_dict_inv, id_dict[v], v)
    end


    real_comms = Dict{Int64, Vector{Int64}}()
    for c in collect(keys(comm_inv))
        real_comms[c] = getkey(real_comms, c, [id_dict_inv[i] for i in comm_inv[c]])
    end


    stringdata1 = JSON.json(real_comms)
    stringdata2 = JSON.json(id_dict_inv)
    vnew_dict = Dict{Int64, Int64}()
    for i in 1:length(vnew)
        vnew_dict[i] = getkey(vnew_dict, i, vnew[i])
    end
    stringdata3 = JSON.json(vnew_dict)



    # write the file with the stringdata variable information
    open("./Results/Together/real_comm73-78.json", "w") do f
            write(f, stringdata1)
    end
    open("./Results/Separate/id_dict_inv73-78.json", "w") do f
            write(f, stringdata2)
    end
    open("./Results/Separate/vnew73-78.json", "w") do f
            write(f, stringdata3)
    end
    writedlm("./Results/Separate/est_theta_73-78.csv",est)
    #filter out the largest component from the csv
    lg = LightGraphs.Graph(network)



    nodelabel = [vnew[i] for i in 1:N]
    nodelabel = [id_dict_inv[i] for i in nodelabel]
    draw(PNG("./Results/Separate/net73-78.png" ,30cm, 30cm), gplot(lg,nodelabel =nodelabel))
    savegraph("./Results/Separate/graph73-78.lgz", lg)
    println("done with 73-78")
##############TOGETHER##########################
##############TOGETHER##########################
##############TOGETHER##########################
##############TOGETHER##########################
linktab73 = readtable("../../data/Authors/Links_2010_Final/Links 1973.csv", header=true)
linktab74 = readtable("../../data/Authors/Links_2010_Final/Links 1974.csv", header=true)
linktab = vcat(linktab73, linktab74)
completecases!(linktab)
N = length(unique(vcat(linktab[:,1], linktab[:,2])))
g = SimpleGraph(N)
id_dict = Dict{Int64, Int64}()
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
count = 1
for i in 1:size(vcat_authorids,1)
    auth_id = vcat_authorids[i]
    if !(auth_id in collect(keys(id_dict)))
        id_dict[auth_id]=getkey(id_dict, auth_id,count)
        count +=1
    end
end
for r in 1:size(linktab,1)
    id1 = id_dict[linktab[r,:researcher_id_1]]
    id2 = id_dict[linktab[r,:researcher_id_2]]
    if !has_edge(g, Edge(id1, id2)) && !has_edge(g, Edge(id2, id1))
        add_edge!(g, id1, id2)
    end
end
nv(g)
ne(g)
connected_components(g)


# GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netauthorsfull.png", 30cm, 30cm), gplot(lg))
which_connected_component = indmax([length(c) for c in connected_components(g)])
vold = connected_components(g)[which_connected_component]

lg, vnew = induced_subgraph(g, vold)
vertices(lg)

vnew_dict = Dict{Int64,Int64}()
MetaGraph(lg)
for (i,v) in enumerate(vnew)
    if !haskey(vnew_dict, v)
        vnew_dict[v] = get(vnew_dict, v, i)
    else
        continue
    end
end
#vnew_dict tells us who is in the new 1:N

N = length(vertices(lg))
include("utils.jl")
using Utils
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
iter=10000;elboevery=500;
include("./train.jl")
import Train:train
(model, mb)=train(model, mb, iter, elboevery, communities)
est=deepcopy(model.est_θ)
# sort_by_argmax!(est)
#Plots.heatmap(est, yflip=true)
est_comms = Dict{Int64, Vector{Int64}}()
for i in 1:N
    which_c = [j for j in 1:model.K if est[i,j] > (.2)]
    if !haskey(est_comms, vnew[i])
        est_comms[vnew[i]] = getkey(est_comms, vnew[i], which_c)
    end
end

est_comms
#organize communities:
comm_inv = Dict{Int64, Vector{Int64}}()
for i in collect(keys(est_comms))
    for c in est_comms[i]
        if !haskey(comm_inv, c)
            comm_inv[c] = getkey(comm_inv, c, Int64[i])
        else
            push!(comm_inv[c], i)
        end
    end
end
comm_inv
#now match names with actual authors
id_dict_inv = Dict{Int64, Int64}()
for v in collect(keys(id_dict))
    id_dict_inv[id_dict[v]] = get(id_dict_inv, id_dict[v], v)
end


real_comms = Dict{Int64, Vector{Int64}}()
for c in collect(keys(comm_inv))
    real_comms[c] = getkey(real_comms, c, [id_dict_inv[i] for i in comm_inv[c]])
end
stringdata1 = JSON.json(real_comms)
stringdata2 = JSON.json(id_dict_inv)
vnew_dict = Dict{Int64, Int64}()
for i in 1:length(vnew)
    vnew_dict[i] = getkey(vnew_dict, i, vnew[i])
end
stringdata3 = JSON.json(vnew_dict)



# write the file with the stringdata variable information
open("./Results/Together/real_comm73-74.json", "w") do f
        write(f, stringdata1)
end
open("./Results/Together/id_dict_inv73-74.json", "w") do f
        write(f, stringdata2)
end
open("./Results/Together/vnew73-74.json", "w") do f
        write(f, stringdata3)
end
writedlm("./Results/Together/est_theta_73-74.csv",est)
#filter out the largest component from the csv
lg = LightGraphs.Graph(network)



nodelabel = [vnew[i] for i in 1:N]
nodelabel = [id_dict_inv[i] for i in nodelabel]
draw(PNG("./Results/Together/net73-74.png" ,30cm, 30cm), gplot(lg,nodelabel =nodelabel))
savegraph("./Results/Together/graph73-74.lgz", lg)


######################
#######################





linktab73 = readtable("../../data/Authors/Links_2010_Final/Links 1973.csv", header=true)
linktab74 = readtable("../../data/Authors/Links_2010_Final/Links 1974.csv", header=true)
linktab75 = readtable("../../data/Authors/Links_2010_Final/Links 1975.csv", header=true)
linktab = vcat(linktab73, linktab74, linktab75)
completecases!(linktab)
N = length(unique(vcat(linktab[:,1], linktab[:,2])))
g = SimpleGraph(N)
id_dict = Dict{Int64, Int64}()
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
count = 1
for i in 1:size(vcat_authorids,1)
    auth_id = vcat_authorids[i]
    if !(auth_id in collect(keys(id_dict)))
        id_dict[auth_id]=getkey(id_dict, auth_id,count)
        count +=1
    end
end
for r in 1:size(linktab,1)
    id1 = id_dict[linktab[r,:researcher_id_1]]
    id2 = id_dict[linktab[r,:researcher_id_2]]
    if !has_edge(g, Edge(id1, id2)) && !has_edge(g, Edge(id2, id1))
        add_edge!(g, id1, id2)
    end
end
nv(g)
ne(g)
connected_components(g)


# GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netauthorsfull.png", 30cm, 30cm), gplot(lg))
which_connected_component = indmax([length(c) for c in connected_components(g)])
vold = connected_components(g)[which_connected_component]

lg, vnew = induced_subgraph(g, vold)
vertices(lg)

vnew_dict = Dict{Int64,Int64}()
MetaGraph(lg)
for (i,v) in enumerate(vnew)
    if !haskey(vnew_dict, v)
        vnew_dict[v] = get(vnew_dict, v, i)
    else
        continue
    end
end
#vnew_dict tells us who is in the new 1:N

N = length(vertices(lg))
include("utils.jl")
using Utils
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
iter=10000;elboevery=500;
include("./train.jl")
import Train:train
(model, mb)=train(model, mb, iter, elboevery, communities)
est=deepcopy(model.est_θ)
# sort_by_argmax!(est)
#Plots.heatmap(est, yflip=true)
est_comms = Dict{Int64, Vector{Int64}}()
for i in 1:N
    which_c = [j for j in 1:model.K if est[i,j] > (.2)]
    if !haskey(est_comms, vnew[i])
        est_comms[vnew[i]] = getkey(est_comms, vnew[i], which_c)
    end
end

est_comms
#organize communities:
comm_inv = Dict{Int64, Vector{Int64}}()
for i in collect(keys(est_comms))
    for c in est_comms[i]
        if !haskey(comm_inv, c)
            comm_inv[c] = getkey(comm_inv, c, Int64[i])
        else
            push!(comm_inv[c], i)
        end
    end
end
comm_inv
#now match names with actual authors
id_dict_inv = Dict{Int64, Int64}()
for v in collect(keys(id_dict))
    id_dict_inv[id_dict[v]] = get(id_dict_inv, id_dict[v], v)
end


real_comms = Dict{Int64, Vector{Int64}}()
for c in collect(keys(comm_inv))
    real_comms[c] = getkey(real_comms, c, [id_dict_inv[i] for i in comm_inv[c]])
end
stringdata1 = JSON.json(real_comms)
stringdata2 = JSON.json(id_dict_inv)
vnew_dict = Dict{Int64, Int64}()
for i in 1:length(vnew)
    vnew_dict[i] = getkey(vnew_dict, i, vnew[i])
end
stringdata3 = JSON.json(vnew_dict)



# write the file with the stringdata variable information
open("./Results/Together/real_comm73-75.json", "w") do f
        write(f, stringdata1)
end
open("./Results/Together/id_dict_inv73-75.json", "w") do f
        write(f, stringdata2)
end
open("./Results/Together/vnew73-75.json", "w") do f
        write(f, stringdata3)
end
writedlm("./Results/Together/est_theta_73-75.csv",est)
#filter out the largest component from the csv
lg = LightGraphs.Graph(network)



nodelabel = [vnew[i] for i in 1:N]
nodelabel = [id_dict_inv[i] for i in nodelabel]
draw(PNG("./Results/Together/net73-75.png" ,30cm, 30cm), gplot(lg,nodelabel =nodelabel))
savegraph("./Results/Together/graph73-75.lgz", lg)


######################
#######################





linktab73 = readtable("../../data/Authors/Links_2010_Final/Links 1973.csv", header=true)
linktab74 = readtable("../../data/Authors/Links_2010_Final/Links 1974.csv", header=true)
linktab75 = readtable("../../data/Authors/Links_2010_Final/Links 1975.csv", header=true)
linktab76 = readtable("../../data/Authors/Links_2010_Final/Links 1976.csv", header=true)
linktab = vcat(linktab73, linktab74, linktab75, linktab76)
completecases!(linktab)
N = length(unique(vcat(linktab[:,1], linktab[:,2])))
g = SimpleGraph(N)
id_dict = Dict{Int64, Int64}()
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
count = 1
for i in 1:size(vcat_authorids,1)
    auth_id = vcat_authorids[i]
    if !(auth_id in collect(keys(id_dict)))
        id_dict[auth_id]=getkey(id_dict, auth_id,count)
        count +=1
    end
end
for r in 1:size(linktab,1)
    id1 = id_dict[linktab[r,:researcher_id_1]]
    id2 = id_dict[linktab[r,:researcher_id_2]]
    if !has_edge(g, Edge(id1, id2)) && !has_edge(g, Edge(id2, id1))
        add_edge!(g, id1, id2)
    end
end
nv(g)
ne(g)
connected_components(g)


# GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netauthorsfull.png", 30cm, 30cm), gplot(lg))
which_connected_component = indmax([length(c) for c in connected_components(g)])
vold = connected_components(g)[which_connected_component]

lg, vnew = induced_subgraph(g, vold)
vertices(lg)

vnew_dict = Dict{Int64,Int64}()
MetaGraph(lg)
for (i,v) in enumerate(vnew)
    if !haskey(vnew_dict, v)
        vnew_dict[v] = get(vnew_dict, v, i)
    else
        continue
    end
end
#vnew_dict tells us who is in the new 1:N

N = length(vertices(lg))
include("utils.jl")
using Utils
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
iter=10000;elboevery=500;
include("./train.jl")
import Train:train
(model, mb)=train(model, mb, iter, elboevery, communities)
est=deepcopy(model.est_θ)
# sort_by_argmax!(est)
#Plots.heatmap(est, yflip=true)
est_comms = Dict{Int64, Vector{Int64}}()
for i in 1:N
    which_c = [j for j in 1:model.K if est[i,j] > (.2)]
    if !haskey(est_comms, vnew[i])
        est_comms[vnew[i]] = getkey(est_comms, vnew[i], which_c)
    end
end

est_comms
#organize communities:
comm_inv = Dict{Int64, Vector{Int64}}()
for i in collect(keys(est_comms))
    for c in est_comms[i]
        if !haskey(comm_inv, c)
            comm_inv[c] = getkey(comm_inv, c, Int64[i])
        else
            push!(comm_inv[c], i)
        end
    end
end
comm_inv
#now match names with actual authors
id_dict_inv = Dict{Int64, Int64}()
for v in collect(keys(id_dict))
    id_dict_inv[id_dict[v]] = get(id_dict_inv, id_dict[v], v)
end


real_comms = Dict{Int64, Vector{Int64}}()
for c in collect(keys(comm_inv))
    real_comms[c] = getkey(real_comms, c, [id_dict_inv[i] for i in comm_inv[c]])
end
stringdata1 = JSON.json(real_comms)
stringdata2 = JSON.json(id_dict_inv)
vnew_dict = Dict{Int64, Int64}()
for i in 1:length(vnew)
    vnew_dict[i] = getkey(vnew_dict, i, vnew[i])
end
stringdata3 = JSON.json(vnew_dict)



# write the file with the stringdata variable information
open("./Results/Together/real_comm73-76.json", "w") do f
        write(f, stringdata1)
end
open("./Results/Together/id_dict_inv73-76.json", "w") do f
        write(f, stringdata2)
end
open("./Results/Together/vnew73-76.json", "w") do f
        write(f, stringdata3)
end
writedlm("./Results/Together/est_theta_73-76.csv",est)
#filter out the largest component from the csv
lg = LightGraphs.Graph(network)



nodelabel = [vnew[i] for i in 1:N]
nodelabel = [id_dict_inv[i] for i in nodelabel]
draw(PNG("./Results/Together/net73-76.png" ,30cm, 30cm), gplot(lg,nodelabel =nodelabel))
savegraph("./Results/Together/graph73-76.lgz", lg)



####################
##################
######################
#######################





linktab73 = readtable("../../data/Authors/Links_2010_Final/Links 1973.csv", header=true)
linktab74 = readtable("../../data/Authors/Links_2010_Final/Links 1974.csv", header=true)
linktab75 = readtable("../../data/Authors/Links_2010_Final/Links 1975.csv", header=true)
linktab76 = readtable("../../data/Authors/Links_2010_Final/Links 1976.csv", header=true)
linktab77 = readtable("../../data/Authors/Links_2010_Final/Links 1977.csv", header=true)
linktab = vcat(linktab73, linktab74, linktab75, linktab76,linktab77 )
completecases!(linktab)
N = length(unique(vcat(linktab[:,1], linktab[:,2])))
g = SimpleGraph(N)
id_dict = Dict{Int64, Int64}()
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
count = 1
for i in 1:size(vcat_authorids,1)
    auth_id = vcat_authorids[i]
    if !(auth_id in collect(keys(id_dict)))
        id_dict[auth_id]=getkey(id_dict, auth_id,count)
        count +=1
    end
end
for r in 1:size(linktab,1)
    id1 = id_dict[linktab[r,:researcher_id_1]]
    id2 = id_dict[linktab[r,:researcher_id_2]]
    if !has_edge(g, Edge(id1, id2))
        add_edge!(g, id1, id2)
    end
end
nv(g)
ne(g)
connected_components(g)


# GraphPlot.draw(PNG("/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/data/netauthorsfull.png", 30cm, 30cm), gplot(lg))
which_connected_component = indmax([length(c) for c in connected_components(g)])
vold = connected_components(g)[which_connected_component]

lg, vnew = induced_subgraph(g, vold)
vertices(lg)

vnew_dict = Dict{Int64,Int64}()
MetaGraph(lg)
for (i,v) in enumerate(vnew)
    if !haskey(vnew_dict, v)
        vnew_dict[v] = get(vnew_dict, v, i)
    else
        continue
    end
end
#vnew_dict tells us who is in the new 1:N

N = length(vertices(lg))
include("utils.jl")
using Utils
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
iter=10000;elboevery=500;
include("./train.jl")
import Train:train
(model, mb)=train(model, mb, iter, elboevery, communities)
est=deepcopy(model.est_θ)
# sort_by_argmax!(est)
#Plots.heatmap(est, yflip=true)
est_comms = Dict{Int64, Vector{Int64}}()
for i in 1:N
    which_c = [j for j in 1:model.K if est[i,j] > (.2)]
    if !haskey(est_comms, vnew[i])
        est_comms[vnew[i]] = getkey(est_comms, vnew[i], which_c)
    end
end

est_comms
#organize communities:
comm_inv = Dict{Int64, Vector{Int64}}()
for i in collect(keys(est_comms))
    for c in est_comms[i]
        if !haskey(comm_inv, c)
            comm_inv[c] = getkey(comm_inv, c, Int64[i])
        else
            push!(comm_inv[c], i)
        end
    end
end
comm_inv
#now match names with actual authors
id_dict_inv = Dict{Int64, Int64}()
for v in collect(keys(id_dict))
    id_dict_inv[id_dict[v]] = get(id_dict_inv, id_dict[v], v)
end


real_comms = Dict{Int64, Vector{Int64}}()
for c in collect(keys(comm_inv))
    real_comms[c] = getkey(real_comms, c, [id_dict_inv[i] for i in comm_inv[c]])
end
stringdata1 = JSON.json(real_comms)
stringdata2 = JSON.json(id_dict_inv)
vnew_dict = Dict{Int64, Int64}()
for i in 1:length(vnew)
    vnew_dict[i] = getkey(vnew_dict, i, vnew[i])
end
stringdata3 = JSON.json(vnew_dict)



# write the file with the stringdata variable information
open("./Results/Together/real_comm73-77.json", "w") do f
        write(f, stringdata1)
end
open("./Results/Together/id_dict_inv73-77.json", "w") do f
        write(f, stringdata2)
end
open("./Results/Together/vnew73-77.json", "w") do f
        write(f, stringdata3)
end
writedlm("./Results/Together/est_theta_73-77.csv",est)
#filter out the largest component from the csv
lg = LightGraphs.Graph(network)



nodelabel = [vnew[i] for i in 1:N]
nodelabel = [id_dict_inv[i] for i in nodelabel]
draw(PNG("./Results/Together/net73-77.png" ,30cm, 30cm), gplot(lg,nodelabel =nodelabel))
savegraph("./Results/Together/graph73-77.lgz", lg)


############################

id_name = Dict{Int64, String}()
for id in vcat_authorids
    if !haskey(id_name, id)
        if !isempty(linktab[linktab[:researcher_id_2] .== id,:name_2])
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_2] .== id,:name_2][1])
        else
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_1] .== id,:name_1][1])
        end
    end
end

function givemename(id, name_id::Dict{Int64, String})
    if isinteger(id)
        return name_id[id]
    else
        return name_id[parse(Int64, id)]
    end
end

############



est80 = readdlm("./Results/Separate/est_theta_1980.csv")
graph1980 = loadgraph("./Results/Separate/graph1980.lgz")
graph_ids = collect(vertices(graph1980))
####
address = "./Results/Separate/vnew1980.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end
vnew1980 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    vnew1980[parse(Int64, k)] = getkey(vnew1980, parse(Int64, k), convert(Int64, temp[k]))
end
####
address = "./Results/Separate/real_comm1980.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

real_comm1980 = Dict{Int64, Vector{Int64}}()
for k in collect(keys(temp))
    real_comm1980[parse(Int64, k)] = getkey(real_comm1980, parse(Int64, k), convert(Vector{Int64}, temp[k]))
end
#######
address = "./Results/Separate/id_dict_inv1973.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

id_dict_inv1980 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    id_dict_inv1980[parse(Int64, k)] = getkey(id_dict_inv1980, parse(Int64, k), convert(Int64, temp[k]))
end
#######
function real2graph(id, id_dict_inv, vnew)
    id_dict = Dict{Int64, Int64}()
    for v in collect(keys(id_dict_inv))
        id_dict[id_dict_inv[v]] = getkey(id_dict, id_dict_inv[v], v)
    end
    vnew_inv = Dict{Int64, Int64}()
    for i in 1:length(vnew)
        vnew_inv[vnew[i]] = getkey(vnew_inv, vnew[i], i)
    end
    vnew_inv[id_dict[id]]
end
function graph2real(id, id_dict_inv, vnew)
    id_dict_inv[vnew[id]]
end
real_comm1980
println([length(real_comm1980[c]) for c in collect(keys(real_comm1980))])
id_name = Dict{Int64, String}()

linktab = readtable("../../data/Authors/Links_2010_Final/Links 1989.csv", header=true)
completecases!(linktab)
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
for id in vcat_authorids
    if !haskey(id_name, id)
        if !isempty(linktab[linktab[:researcher_id_2] .== id,:name_2])
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_2] .== id,:name_2][1])
        else
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_1] .== id,:name_1][1])
        end
    end
end

function givemename(id, name_id::Dict{Int64, String})
    if isinteger(id)
        return name_id[id]
    else
        return name_id[parse(Int64, id)]
    end
end
cs = Int64[]
for k in collect(keys(real_comm1980))
    if 40593 in real_comm1980[k]
        push!(cs, k)
    end
end



######################
########################
est73 = readdlm("./Results/Separate/est_theta_1973.csv")
graph1973 = loadgraph("./Results/Separate/graph1973.lgz")
graph_ids = collect(vertices(graph1973))
####
address = "./Results/Separate/vnew1973.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end
vnew1973 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    vnew1973[parse(Int64, k)] = getkey(vnew1973, parse(Int64, k), convert(Int64, temp[k]))
end
####
address = "./Results/Separate/real_comm1973.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

real_comm1973 = Dict{Int64, Vector{Int64}}()
for k in collect(keys(temp))
    real_comm1973[parse(Int64, k)] = getkey(real_comm1973, parse(Int64, k), convert(Vector{Int64}, temp[k]))
end
#######
address = "./Results/Separate/id_dict_inv1973.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

id_dict_inv1973 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    id_dict_inv1973[parse(Int64, k)] = getkey(id_dict_inv1973, parse(Int64, k), convert(Int64, temp[k]))
end
#######
function real2graph(id, id_dict_inv, vnew)
    id_dict = Dict{Int64, Int64}()
    for v in collect(keys(id_dict_inv))
        id_dict[id_dict_inv[v]] = getkey(id_dict, id_dict_inv[v], v)
    end
    vnew_inv = Dict{Int64, Int64}()
    for i in 1:length(vnew)
        vnew_inv[vnew[i]] = getkey(vnew_inv, vnew[i], i)
    end
    vnew_inv[id_dict[id]]
end
function graph2real(id, id_dict_inv, vnew)
    id_dict_inv[vnew[id]]
end
real_comm1973
println([length(real_comm1973[c]) for c in collect(keys(real_comm1973))])
id_name = Dict{Int64, String}()

linktab = readtable("../../data/Authors/Links_2010_Final/Links 1973.csv", header=true)
completecases!(linktab)
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
for id in vcat_authorids
    if !haskey(id_name, id)
        if !isempty(linktab[linktab[:researcher_id_2] .== id,:name_2])
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_2] .== id,:name_2][1])
        else
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_1] .== id,:name_1][1])
        end
    end
end

function givemename(id, name_id::Dict{Int64, String})
    if isinteger(id)
        return name_id[id]
    else
        return name_id[parse(Int64, id)]
    end
end

cs = Int64[]
for k in collect(keys(real_comm1973))
    if 37551 in real_comm1973[k]
        push!(cs, k)
    end
end
#############
#############
# rvertice = sunique(vcat(real_comm1973[2], real_comm1973[2], real_comm1973[12]))
# gvertices = Int64[]
# real2graph(id, id_dict_inv, vnew)
name_dict1973 = Dict{Int64, Vector{String}}()
real_comm1973[2]
names_list = String[]
for i in collect(values(real_comm1973[2]))
    push!(names_list,givemename(i, id_name))
end
sort!(names_list)
name_dict1973[2] = getkey(name_dict1973, 2, names_list)
println(names_list)
println();println();println();println();

real_comm1973[3]
names_list = String[]
for i in collect(values(real_comm1973[3]))
    push!(names_list,givemename(i, id_name))
end
sort!(names_list)
name_dict1973[3] = getkey(name_dict1973, 3, names_list)
println(names_list)
println();println();println();println();


real_comm1973[12]
names_list = String[]
for i in collect(values(real_comm1973[12]))
    push!(names_list,givemename(i, id_name))
end
sort!(names_list)
name_dict1973[12] = getkey(name_dict1973, 12, names_list)
println(names_list)
println();println();println();println();
println(name_dict1973)



######################
########################
est74 = readdlm("./Results/Separate/est_theta_1974.csv")
graph1974 = loadgraph("./Results/Separate/graph1974.lgz")
graph_ids = collect(vertices(graph1974))
####
address = "./Results/Separate/vnew1974.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end
vnew1974 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    vnew1974[parse(Int64, k)] = getkey(vnew1974, parse(Int64, k), convert(Int64, temp[k]))
end
####
address = "./Results/Separate/real_comm1974.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

real_comm1974 = Dict{Int64, Vector{Int64}}()
for k in collect(keys(temp))
    real_comm1974[parse(Int64, k)] = getkey(real_comm1974, parse(Int64, k), convert(Vector{Int64}, temp[k]))
end
#######
address = "./Results/Separate/id_dict_inv1974.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

id_dict_inv1974 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    id_dict_inv1974[parse(Int64, k)] = getkey(id_dict_inv1974, parse(Int64, k), convert(Int64, temp[k]))
end
#######
function real2graph(id, id_dict_inv, vnew)
    id_dict = Dict{Int64, Int64}()
    for v in collect(keys(id_dict_inv))
        id_dict[id_dict_inv[v]] = getkey(id_dict, id_dict_inv[v], v)
    end
    vnew_inv = Dict{Int64, Int64}()
    for i in 1:length(vnew)
        vnew_inv[vnew[i]] = getkey(vnew_inv, vnew[i], i)
    end
    vnew_inv[id_dict[id]]
end
function graph2real(id, id_dict_inv, vnew)
    id_dict_inv[vnew[id]]
end
real_comm1973
println([length(real_comm1974[c]) for c in collect(keys(real_comm1974))])
id_name = Dict{Int64, String}()

linktab = readtable("../../data/Authors/Links_2010_Final/Links 1974.csv", header=true)
completecases!(linktab)
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
for id in vcat_authorids
    if !haskey(id_name, id)
        if !isempty(linktab[linktab[:researcher_id_2] .== id,:name_2])
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_2] .== id,:name_2][1])
        else
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_1] .== id,:name_1][1])
        end
    end
end

function givemename(id, name_id::Dict{Int64, String})
    if isinteger(id)
        return name_id[id]
    else
        return name_id[parse(Int64, id)]
    end
end
cs = Int64[]
for k in collect(keys(real_comm1974))
    if 37551 in real_comm1974[k]
        push!(cs, k)
    end
end
#############
#############
name_dict1974 = Dict{Int64, Vector{String}}()

real_comm1974[1]
names_list = String[]
for i in collect(values(real_comm1974[1]))
    push!(names_list,givemename(i, id_name))
end
sort!(names_list)
name_dict1974[1] = getkey(name_dict1974, 1, names_list)
println(names_list)
println();println();println();println();


real_comm1974[2]
names_list = String[]
for i in collect(values(real_comm1974[2]))
    push!(names_list,givemename(i, id_name))
end
sort!(names_list)
name_dict1974[2] = getkey(name_dict1974, 2, names_list)
println(names_list)
println();println();println();println();

real_comm1974[5]
names_list = String[]
for i in collect(values(real_comm1974[5]))
    push!(names_list,givemename(i, id_name))
end
sort!(names_list)
name_dict1974[5] = getkey(name_dict1974, 5, names_list)
println(names_list)
println();println();println();println();

real_comm1974[11]
names_list = String[]
for i in collect(values(real_comm1974[11]))
    push!(names_list,givemename(i, id_name))
end
sort!(names_list)
name_dict1974[11] = getkey(name_dict1974, 11, names_list)
println(names_list)
println();println();println();println();

println(name_dict1974)
intersect(name_dict1973[2], name_dict1974[1])






######################
########################
est73_75 = readdlm("./Results/Separate/est_theta_73-75.csv")
graph73_75 = loadgraph("./Results/Separate/graph73-75.lgz")
graph_ids = collect(vertices(graph73_75))
####
address = "./Results/Separate/vnew73-75.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end
vnew73_75 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    vnew73_75[parse(Int64, k)] = getkey(vnew73_75, parse(Int64, k), convert(Int64, temp[k]))
end
####
address = "./Results/Separate/real_comm73-75.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

real_comm73_75 = Dict{Int64, Vector{Int64}}()
for k in collect(keys(temp))
    real_comm73_75[parse(Int64, k)] = getkey(real_comm73_75, parse(Int64, k), convert(Vector{Int64}, temp[k]))
end
#######
address = "./Results/Separate/id_dict_inv73-75.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

id_dict_inv73_75 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    id_dict_inv73_75[parse(Int64, k)] = getkey(id_dict_inv73_75, parse(Int64, k), convert(Int64, temp[k]))
end
#######
function real2graph(id, id_dict_inv, vnew)
    id_dict = Dict{Int64, Int64}()
    for v in collect(keys(id_dict_inv))
        id_dict[id_dict_inv[v]] = getkey(id_dict, id_dict_inv[v], v)
    end
    vnew_inv = Dict{Int64, Int64}()
    for i in 1:length(vnew)
        vnew_inv[vnew[i]] = getkey(vnew_inv, vnew[i], i)
    end
    vnew_inv[id_dict[id]]
end
function graph2real(id, id_dict_inv, vnew)
    id_dict_inv[vnew[id]]
end
real_comm73_75
println([length(real_comm73_75[c]) for c in collect(keys(real_comm73_75))])
id_name = Dict{Int64, String}()


completecases!(linktab)
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
for id in vcat_authorids
    if !haskey(id_name, id)
        if !isempty(linktab[linktab[:researcher_id_2] .== id,:name_2])
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_2] .== id,:name_2][1])
        else
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_1] .== id,:name_1][1])
        end
    end
end

function givemename(id, name_id::Dict{Int64, String})
    if isinteger(id)
        return name_id[id]
    else
        return name_id[parse(Int64, id)]
    end
end
cs = Int64[]
for k in collect(keys(real_comm73_75))
    if 37551 in real_comm73_75[k]
        push!(cs, k)
    end
end

#############
#############
name_dict73_75 = Dict{Int64, Vector{String}}()
for c in cs
    names_list = String[]
    for i in collect(values(real_comm73_75[c]))
        push!(names_list,givemename(i, id_name))
    end
    sort!(names_list)
    name_dict73_75[c] = getkey(name_dict73_75, c, names_list)
end
println(names_list)
println();println();println();println();
































est76_78 = readdlm("./Results/Separate/est_theta_76-78.csv")
graph76_78 = loadgraph("./Results/Separate/graph76-78.lgz")
graph_ids = collect(vertices(graph76_78))
####
address = "./Results/Separate/vnew76-78.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end
vnew76_78 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    vnew76_78[parse(Int64, k)] = getkey(vnew76_78, parse(Int64, k), convert(Int64, temp[k]))
end
####
address = "./Results/Separate/real_comm76-78.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

real_comm76_78 = Dict{Int64, Vector{Int64}}()
for k in collect(keys(temp))
    real_comm76_78[parse(Int64, k)] = getkey(real_comm76_78, parse(Int64, k), convert(Vector{Int64}, temp[k]))
end
#######
address = "./Results/Separate/id_dict_inv76-78.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

id_dict_inv76_78 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    id_dict_inv76_78[parse(Int64, k)] = getkey(id_dict_inv76_78, parse(Int64, k), convert(Int64, temp[k]))
end
#######
function real2graph(id, id_dict_inv, vnew)
    id_dict = Dict{Int64, Int64}()
    for v in collect(keys(id_dict_inv))
        id_dict[id_dict_inv[v]] = getkey(id_dict, id_dict_inv[v], v)
    end
    vnew_inv = Dict{Int64, Int64}()
    for i in 1:length(vnew)
        vnew_inv[vnew[i]] = getkey(vnew_inv, vnew[i], i)
    end
    vnew_inv[id_dict[id]]
end
function graph2real(id, id_dict_inv, vnew)
    id_dict_inv[vnew[id]]
end
real_comm76_78
println([length(real_comm76_78[c]) for c in collect(keys(real_comm76_78))])
id_name = Dict{Int64, String}()


completecases!(linktab)
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
for id in vcat_authorids
    if !haskey(id_name, id)
        if !isempty(linktab[linktab[:researcher_id_2] .== id,:name_2])
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_2] .== id,:name_2][1])
        else
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_1] .== id,:name_1][1])
        end
    end
end

function givemename(id, name_id::Dict{Int64, String})
    if isinteger(id)
        return name_id[id]
    else
        return name_id[parse(Int64, id)]
    end
end
cs = Int64[]
for k in collect(keys(real_comm76_78))
    if 37551 in real_comm76_78[k]
        push!(cs, k)
    end
end

#############
#############
name_dict76_78 = Dict{Int64, Vector{String}}()
for c in cs
    names_list = String[]
    for i in collect(values(real_comm76_78[c]))
        push!(names_list,givemename(i, id_name))
    end
    sort!(names_list)
    name_dict76_78[c] = getkey(name_dict76_78, c, names_list)
end
println(names_list)
println();println();println();println();
name_dict73_75
name_dict76_78


##########













est73_78 = readdlm("./Results/Separate/est_theta_73-78.csv")
graph73_78 = loadgraph("./Results/Separate/graph73-78.lgz")
graph_ids = collect(vertices(graph73_78))
####
address = "./Results/Separate/vnew73-78.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end
vnew73_78 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    vnew73_78[parse(Int64, k)] = getkey(vnew73_78, parse(Int64, k), convert(Int64, temp[k]))
end
####
address = "./Results/Separate/real_comm73-78.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

real_comm73_78 = Dict{Int64, Vector{Int64}}()
for k in collect(keys(temp))
    real_comm73_78[parse(Int64, k)] = getkey(real_comm73_78, parse(Int64, k), convert(Vector{Int64}, temp[k]))
end
#######
address = "./Results/Separate/id_dict_inv73-78.json"
temp = Dict()
open(address, "r") do f
    global temp
    temp=JSON.parse(f)  # parse and transform data
end

id_dict_inv73_78 = Dict{Int64, Int64}()
for k in collect(keys(temp))
    id_dict_inv73_78[parse(Int64, k)] = getkey(id_dict_inv73_78, parse(Int64, k), convert(Int64, temp[k]))
end
#######
function real2graph(id, id_dict_inv, vnew)
    id_dict = Dict{Int64, Int64}()
    for v in collect(keys(id_dict_inv))
        id_dict[id_dict_inv[v]] = getkey(id_dict, id_dict_inv[v], v)
    end
    vnew_inv = Dict{Int64, Int64}()
    for i in 1:length(vnew)
        vnew_inv[vnew[i]] = getkey(vnew_inv, vnew[i], i)
    end
    vnew_inv[id_dict[id]]
end
function graph2real(id, id_dict_inv, vnew)
    id_dict_inv[vnew[id]]
end
real_comm73_78
println([length(real_comm73_78[c]) for c in collect(keys(real_comm73_78))])
id_name = Dict{Int64, String}()


completecases!(linktab)
vcat_authorids = vcat(linktab[:researcher_id_1], linktab[:researcher_id_2])
vcat_authorids = unique(vcat_authorids)
for id in vcat_authorids
    if !haskey(id_name, id)
        if !isempty(linktab[linktab[:researcher_id_2] .== id,:name_2])
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_2] .== id,:name_2][1])
        else
            id_name[id] = getkey(id_name, id,linktab[linktab[:researcher_id_1] .== id,:name_1][1])
        end
    end
end

function givemename(id, name_id::Dict{Int64, String})
    if isinteger(id)
        return name_id[id]
    else
        return name_id[parse(Int64, id)]
    end
end
cs = Int64[]
for k in collect(keys(real_comm73_78))
    if 41044 in real_comm73_78[k]
        push!(cs, k)
    end
end
graph73_78
adj = adjacency_matrix(graph73_78)
adjlist = zeros(Int64, (nnz(adj), 2))
count=1
for c in 1:size(adj, 2)
    for r in 1:size(adj, 1)
        if adj[r,c] == 1
            adjlist[count,1] = r
            adjlist[count,2] = c
            count+=1
        end
    end
end
writedlm("network73_78.tsv", adjlist, '\t')
#############
#############
name_dict73_78 = Dict{Int64, Vector{String}}()
for c in cs
    names_list = String[]
    for i in collect(values(real_comm73_78[c]))
        push!(names_list,givemename(i, id_name))
    end
    sort!(names_list)
    name_dict73_78[c] = getkey(name_dict73_78, c, names_list)
end
println(names_list)
println();println();println();println();


name_dict73_75
name_dict76_78
name_dict73_78
est76_78
est73_78
Plots.heatmap(est73_78, yflip=true)
real2graph(37551, id_dict_inv76_78, vnew76_78)
est76_78[3,:]

real2graph(41044, id_dict_inv73_78, vnew73_78)
sort(est76_78[240,:], rev=true)[1:5]
sort(est73_78[2,:], rev=true)[1:5]

tplot= hcat(sort(est76_78[3,:], rev=true)[1:5], sort(est73_78[2,:], rev=true)[1:5])'

Plots.heatmap(tplot,yflip = true)
