using Distributions
using JLD:@save,@load
using Plots
using ArgParse
srand(1234)
K=4
m0            =zeros(Float64,K)
M0            =eye(Float64,K) #ones M0 matrix
l0            =K*1.0 #init the df l0
L0            =(.05).*eye(Float64,K) #init the scale L0
η0            =9.0 #one the beta param
η1            =1.0 #one the beta param
function gennetwork(N::Int64, K::Int64)
  network=Network(N)
  Θ=zeros(Float64, (N,K))
  ###note the scalar
  Λ = zeros(Float64, (K,K))
  for i in 1:N
    Λ .+= rand(Wishart(l0,L0))
  end


  Λ ./= N
  μ = rand(MvNormalCanon(M0*m0,M0))
  β = rand(Beta(η0, η1),K)
  ##We need to make sure that these probabilities are all positive
  for a in 1:N
    Θ[a,:] = rand(MvNormalCanon(Λ*μ, Λ))
    Θ[a,:] = expnormalize(Θ[a,:])
  end




  z_in=zeros(Float64, (N,N,K))
  z_out = zeros(Float64,(N,N,K))
  sort_by_argmax!(Θ)
  writedlm("data/true_theta.txt", Θ)

  for a in 1:N
    for b in 1:N
      if a!= b
        z_out[a,b,:] = rand(Multinomial(1,Θ[a,:]))
        z_in[a,b,:] = rand(Multinomial(1,Θ[b,:]))
        if z_in[a,b,:] == z_out[a,b,:]
          network[a,b]=rand(Binomial(1,β[indmax(z_out[a,b,:])]),1)[1]
        else
          network[a,b]=rand(Binomial(1,EPSILON))[1]
        end
      end
    end
  end

  JLD.@save("data/network.jld",network)
#   open("data/mu_true.txt", "w") do f
#     write(f, "mu=")
#     for k in 1:K
#       write(f, "$(μ[k])  ")
#     end
#     write(f, "\n")
#     write(f, "diag(Lambda)=")
#     for k in 1:K
#       write(f, "$(Λ[k,k])  ")
#     end
#     write(f, "\n")
#   end
  writedlm("data/true_mu.txt", μ)
  writedlm("data/true_Lambda.txt", Λ)
  writedlm("data/true_beta.txt", β)
end

if isfile("data/network.jld")
  println("There already exists a netwrok.jld, if you want to change it remove it first")
else
  isassigned(inputtomodelgen,2)?gennetwork(inputtomodelgen[1],inputtomodelgen[2]):println("you should set ARGS for gennetwork(N,K)")

  # Plots.heatmap(Θ, yflip=true)
  network=FileIO.load("data/network.jld")["network"]
  Plots.heatmap(network, yflip=true)
end
