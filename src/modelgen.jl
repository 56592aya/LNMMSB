using Distributions
using JLD:@save,@load
using Plots
using ArgParse
srand(4321)
K=Int64(inputtomodelgen[2])
m0            =zeros(Float64,K)
M0            =10.0*eye(Float64,K) #ones M0 matrix
l0            =(K+2.0)*1.0 #init the df l0
L0            =(.01/l0).*eye(Float64,K) #init the scale L0
η0            =9.0 #one the beta param
η1            =1.0 #one the beta param
truth_comm = [Int64[] for k in 1:K]
function gennetwork(N::Int64, K::Int64)
  network=utils.Network(N)
  θ=zeros(Float64, (N,K))
  ###note the scalar
  Λ = zeros(Float64, (K,K))
  for i in 1:N
    Λ .+= rand(Wishart(l0,L0))
  end



  Λ ./= N
  μ = rand(MvNormalCanon(M0*m0,M0))
  # μ -= mean(μ)

  # cov(1.0./(reshape(repeat(rand(MvNormalCanon(M0*m0,M0)), outer=[1000]),(4,1000)))')
  β = rand(Beta(η0, η1),K)
  ##We need to make sure that these probabilities are all positive
  for a in 1:N
    θ[a,:] = rand(MvNormalCanon(Λ*μ, Λ))
    # θ[a,:] -= mean(θ[a,:])
  end




  z_in=zeros(Float64, (N,N,K))
  z_out = zeros(Float64,(N,N,K))

  for a in 1:N
    θ[a,:]=utils.softmax(θ[a,:])
  end
  utils.sort_by_argmax!(θ)
  if isfile("data/true_thetas.txt")
    rm("data/true_thetas.txt")
    writedlm("data/true_thetas.txt",θ)
  else
    writedlm("data/true_thetas.txt",θ)
  end

  for a in 1:N
    for b in 1:N
      if a!= b
        z_out[a,b,:] = rand(Multinomial(1,θ[a,:]))
        z_in[a,b,:] = rand(Multinomial(1,θ[b,:]))
        if z_in[a,b,:] == z_out[a,b,:]
          push!(truth_comm[indmax(z_out[a,b,:])], a)
          push!(truth_comm[indmax(z_out[a,b,:])], b)
          network[a,b]=rand(Binomial(1,β[indmax(z_out[a,b,:])]),1)[1]
        else
          network[a,b]=rand(Binomial(1,EPSILON))[1]
        end
      end
    end
  end
  for k in 1:K
    truth_comm[k] = unique(truth_comm[k])
  end
	if isfile("data/true_mu.txt")
	    rm("data/true_mu.txt")
	    rm("data/true_Lambda.txt")
	    rm("data/true_beta.txt")
	    rm("data/true_m0.txt")
	    rm("data/true_BigM0.txt")
	    rm("data/true_l0.txt")
	    rm("data/true_BigL0.txt")
	    rm("data/true_eta0.txt")
	    rm("data/true_eta1.txt")
	end
  if isfile("data/truth_comm.txt")
    rm("data/truth_comm.txt")
  end
	if isfile("data/network.jld")
		rm("data/network.jld")
		JLD.@save("data/network.jld",network)
	else
		JLD.@save("data/network.jld",network)
	end
	writedlm("data/true_mu.txt", μ)
	writedlm("data/true_Lambda.txt", Λ)
	writedlm("data/true_beta.txt", β)
	writedlm("data/true_m0.txt", m0)
	writedlm("data/true_BigM0.txt", M0)
	writedlm("data/true_l0.txt", l0)
	writedlm("data/true_BigL0.txt", L0)
	writedlm("data/true_eta0.txt", η0)
	writedlm("data/true_eta1.txt", η1)
  open("./data/truth_comm.txt", "w") do f
    for k in 1:inputtomodelgen[2]
      for n in truth_comm[k]
        write(f, "$n ")
      end
      write(f, "\n")
    end
  end
end

isassigned(inputtomodelgen,2)?gennetwork(inputtomodelgen[1],inputtomodelgen[2]):println("you should set ARGS for gennetwork(N,K)")

# Plots.heatmap(Θ, yflip=true)
network=FileIO.load("data/network.jld")["network"]
Plots.heatmap(network, yflip=true)
Plots.heatmap(readdlm("data/true_thetas.txt"), yflip=true)
