module ParallelUtils

export @ensureworkers, sendto, getfrom

macro ensureworkers(numWorkers)
    :(if nworkers() < $numWorkers
        if nprocs() == 1
            addprocs($numWorkers)
            @everywhere import ParallelUtils
        else
            addprocs($numWorkers-nworkers())
            @everywhere import ParallelUtils
        end
    end)
end

# from http://stackoverflow.com/questions/27677399/julia-how-to-copy-data-to-another-processor-in-julia
function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, eval(Main, Expr(:(=), nm, val)))
    end
end
function sendto(ps::Vector{Int}; args...)
    for p in ps
        sendto(p; args...)
    end
end
getfrom(p::Int, nm::Symbol; mod=Main) = fetch(@spawnat(p, getfield(mod, nm)))



"Save the type and data of the passed dictionary as text."
function writedict(f::IOStream, dict::Dict)
    println(f, join(match(r"[A-z]+{([^,]+),([^}]+)}", string(typeof(dict))).captures, '\t'))
    for (k,v) in dict
        println(f, string(k)*"\t"*string(v))
    end
end
function writedict(fileName::AbstractString, dict::Dict)
    f = open(fileName, "w")
    writedict(f, dict)
    close(f)
end

"""
Read the type and data of the passed dictionary.
Numeric types are converted automatically, and string type are
preserved, but other types may fail to load.
"""
function readdict(f::IOStream; reversed=false)
    parts = split(strip(readline(f)), '\t')
    type1 = eval(parse(parts[1]))
    type2 = eval(parse(parts[2]))
    d = reversed ? Dict{type2,type1}() : Dict{type1,type2}()
    for line in eachline(f)
        k,v = split(strip(line), '\t', limit=2)
        if type1 <: Number
            k = parse(type1, k)
        end
        if type2 <: Number
            v = parse(type2, v)
        end

        reversed ? d[v] = k : d[k] = v
    end
    d
end
function readdict(fileName::AbstractString; reversed=false)
    f = open(fileName)
    d = readdict(f, reversed=reversed)
    close(f)
    d
end

end # modul
