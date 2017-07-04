using Distributions
function gennetwork(model::LNMMSB)
  for a in 1:model.N
    model.μ = rand(MvNormalCanon(model.M0*model.m0, model.M0))
    model.Λ = rand(Wishart(l0, L0))
    model.Θ[a,:] = rand(MvNormalCanon(model.Λ*model.μ, model.Λ))
    model.Θ[a,:] = expnormalize(model.Θ[a,:])
    model.β = rand(Beta(model.η0, model.η1))
    for b in 1:model.N
      if a!= b
        model.z_out[a,b,:] = rand(Multinomial(model.Θ[a,:],1))
        model.z_in[a,b,:] = rand(Multinomial(model.Θ[b,:],1))
        model.z_out[a,b,:]
        if model.z_in[a,b,:] == model.z_out[a,b,:]
          model.network[a,b]=rand(Binomial(1,model.β[indmax(model.z_out[a,b,:])]),1)
        else
          model.network[a,b]=rand(Binomial(1,EPSILON)]),1)
        end
      end
    end

  end

end
