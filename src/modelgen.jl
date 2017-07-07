using Distributions
using JLD
using Plots

function gennetwork(N::Int64, K::Int64)
  network=Network(N)
  global Θ=zeros(Float64, (N,K))
  ###note the scalar
  Λ = zeros(Float64, (K,K))
  for i in 1:N
    Λ .+= rand(Wishart(K,0.05*eye(Float64,K)))
  end
  Λ ./= N
  μ = rand(MvNormalCanon(eye(Float64,K)))

  β = rand(Beta(9, 1),K)
  for a in 1:N
    Θ[a,:] = rand(MvNormalCanon(Λ*μ, Λ))
    Θ[a,:] = expnormalize(Θ[a,:])
  end
  z_in=zeros(Float64, (N,N,K))
  z_out = zeros(Float64,(N,N,K))
  sort_by_argmax!(Θ)
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

  @save("data/network.jld",network)
end

# Plots.heatmap(Θ, yflip=true)
# network=load("data/network.jld")["network"]
# Plots.heatmap(network, yflip=true)
