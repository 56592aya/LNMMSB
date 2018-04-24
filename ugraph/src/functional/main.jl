using ArgParse
using MiniLogging
using LightGraphs

    # - a short option(must consist of one single character)
    # -- long option
    #(nothing) is a positional arg
#some useful options
    #arge_type to set the type
    #default to set the default value if not parsed
#actions
    #
function main(args)
    root_logger = get_logger()
    basic_config(MiniLogging.INFO,"./logfile.log")
    @info(root_logger, "nothing done yet")

    s = ArgParseSettings(description="Arguments for the program.")
    @add_arg_table s begin
        "--file"            #File to read the network from
        "--findk"           #will determine the guess of no. of communities
            help = "a flag"
            action = "store_true"
        "-n"               #number of nodes
            help = "number of nodes"
            arg_type=Int
        "-k"               #number of communities
            help = "number of communities, not required if findk"
            arg_type=Int64
    end
    parsed_args = parse_args(s) ##result is a Dict{String, Any}
    println("Parsed args: ")
    for (k,v) in parsed_args
        println("  $k  =>  $(repr(v))")
    end
    file = parsed_args["file"]
    ##reading the netwokr, if it is a light graph, if it is a adj list
    findk = parsed_args["findk"]
    n = parsed_args["n"]
    k = parsed_args["k"]
    path2graph="/home/arashyazdiha/Dropbox/Arash/EUR/Workspace/CLSM/Julia Implementation/ProjectLN/ugraph/src/modules/Results/Separate/graph73-78.lgz"
    g = loadgraph(path2graph)
    network = adjacency_matrix(g)
    N = size(network, 1)
    include("types.jl")
    include("fun.jl")
    theta, beta =train(network, N)
    using Plots
    Plots.heatmap(theta', yflip = true)
    Plots.heatmap(sort_by_argmax!(theta)', yflip = true)


end
main(ARGS)
