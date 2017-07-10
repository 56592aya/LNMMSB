module main
include("AbstractTuple.jl")
include("AbstractDyad.jl")
include("AbstractMMSB.jl")
include("utils.jl")
inputtomodelgen=[150,4] ## N, K
include("modelgen.jl")

network=load("data/network.jld")["network"]
include("LNMMSB.jl")
##should move to LNMMSB
model=LNMMSB(network, inputtomodelgen[2])
include("modelutils.jl")

ho_dyaddict = Dict{Dyad,Bool}()
ho_linkdict = Dict{Link,Bool}()
ho_nlinkdict = Dict{NonLink,Bool}()
setholdout(model)
mb = MiniBatch()
mbsampling!(mb,model)
mb
train_degree!(train_out, train_in, model)
end
