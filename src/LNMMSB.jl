using StatsBase
#I should think whether it is still understandable for diagonal matrices to be represented as vectors?
# if any operation needs the matrice we can instead use the diagm(). Λ_var[a,k] and L[k] should be enough
# This includes M0,  Λ_var , L0
mutable struct LNMMSB <: AbstractMMSB
    K            ::Int64              #number of communities
    N            ::Int64              #number of individuals
    elbo         ::Float64            #ELBO
    oldelbo      ::Float64            #new ELBO
    μ            ::Vector{Float64}    #true mean of the MVN
    μ_var        ::Matrix2d{Float64}  #variational mean of the MVN
    μ_var_old    ::Matrix2d{Float64}  #variational mean of the MVN
    m0           ::Vector{Float64}    #hyperprior on mean
    m            ::Vector{Float64}    #variational hyperprior on mean
    m_old        ::Vector{Float64}    #variational hyperprior on mean
    M0           ::Matrix2d{Float64}    #hyperprior on precision diagonal
    M            ::Matrix2d{Float64}  #hyperprior on variational precision
    M_old        ::Matrix2d{Float64}  #hyperprior on variational precision
    Λ            ::Matrix2d{Float64}  #true precision
    Λ_var        ::Matrix2d{Float64}  #variational preicisoin diagonal
    Λ_var_old    ::Matrix2d{Float64}  #variational preicisoin diagonal
    l0           ::Float64            #df for Preicision in Wishart
    L0           ::Matrix2d{Float64}    #scale for precision in Wishart diagonal
    l            ::Float64            #variational df
    L            ::Matrix2d{Float64}  #variational scale diagonal
    L_old        ::Matrix2d{Float64}  #variational scale diagonal
    # ζ            ::Vector{Float64}    #additional variation param
    # ζ_old        ::Vector{Float64}    #additional variation param
    ϕlinoutsum   ::Vector{Float64}    #sum of phi products for links
    ϕnlinoutsum  ::Vector{Float64}    #sum of phi products for nonlinks
    # ϕbar         ::Matrix2d{Float64}  #average of phis to be used for the next round
    η0           ::Float64            #hyperprior on beta
    η1           ::Float64            #hyperprior on beta
    b0           ::Vector{Float64}    #variational param for beta
    b0_old       ::Vector{Float64}    #variational param for beta
    b1           ::Vector{Float64}    #variational param for beta
    b1_old       ::Vector{Float64}    #variational param for beta
    network      ::Network{Int64}     #sparse view network
    mbsize       ::Int64              #minibatch size
    mbids        ::Vector{Int64}      #minibatch node ids
    nho          ::Float64            # no. of validation links
    ho_dyaddict  ::Dict{Dyad,Bool}    #holdout dyad dictionary
    ho_linkdict  ::Dict{Link,Bool}    #holdout link dictionary
    ho_nlinkdict ::Dict{NonLink,Bool} #holdout nonlink dictionary
    train_out    ::Vector{Int64}      #outdeg of train and mb
    train_in     ::Vector{Int64}      #indeg of train and mb
    train_sinks  ::VectorList{Int64}  #sinks of train and mb
    train_sources::VectorList{Int64}  #sink sof train and mb
    ϕloutsum     ::Matrix2d{Float64}
    ϕnloutsum    ::Matrix2d{Float64}
    ϕlinsum      ::Matrix2d{Float64}
    ϕnlinsum     ::Matrix2d{Float64}
    elborecord   ::Vector{Float64}
    est_θ        ::Matrix2d{Float64}
    est_β        ::Vector{Float64}
    est_μ        ::Vector{Float64}
    est_Λ        ::Matrix2d{Float64}


 function LNMMSB(network::Network{Int64}, K::Int64)
  N             = size(network,1) #setting size of nodes
  elbo          =0.0 #init ELBO at zero
  oldelbo       =-Inf #init new ELBO at zero
  μ             =zeros(Float64,K) #zero the mu vector
  μ_var         =zeros(Float64, (N,K)) #zero the mu_var vector
  μ_var_old     = deepcopy(μ_var)
  m0            =zeros(Float64,K) #zero m0 vector
  m             =zeros(Float64,K) #zero m vector
  m_old         = deepcopy(m)
  M0            =eye(Float64,K) #ones M0 matrix
  M             =eye(Float64,K) #eye M matrix
  M_old         = deepcopy(M)
  Λ             =(1.0/K).*eye(Float64,K) #init Lambda matrix
  Λ_var         =zeros(Float64,(N,K));
  for a in 1:N
    Λ_var[a,:]  = rand(K)
  end
  Λ_var_old     = deepcopy(Λ_var)

  l0            =K*1.0 #init the df l0
  L0            =(0.05).*eye(Float64,K) #init the scale L0

  l             =K*1.0 #init the df l
  L             =(1.0/K).*eye(Float64,K) #zero the scale L
  L_old         = deepcopy(L)
  ϕlinoutsum    =zeros(Float64,K) #zero the phi link product sum
  ϕnlinoutsum   =zeros(Float64,K) #zero the phi nonlink product sum
  # ϕbar          =(1.0/K).*ones(Float64, (N,K)) ## to be used for other rounds as init
  # ζ             =ones(Float64, N) #one additional variational param
  # ζ_old         = deepcopy(ζ)
  η0            =9.0 #one the beta param
  η1            =1.0 #one the beta param
  b0            =9.0.*ones(Float64, K) #one the beta variational param
  b0_old        = deepcopy(b0)
  b1            =ones(Float64, K) #one the beta variational param
  b1_old        = deepcopy(b1)
  mbsize        =N #number of nodes in the minibatch
  mbids         =zeros(Int64,mbsize) # to be extended
  nho           =nnz(network)*0.025 #init nho
  ho_dyaddict   = Dict{Dyad,Bool}()
 	ho_linkdict   = Dict{Link,Bool}()
 	ho_nlinkdict  = Dict{NonLink,Bool}()
  train_out     = zeros(Int64, N)
 	train_in      = zeros(Int64, N)
  train_sinks   = VectorList{Int64}(N)
 	train_sources = VectorList{Int64}(N)
  train_sinks   = VectorList{Int64}(N)
 	train_sources = VectorList{Int64}(N)
  ϕloutsum      = zeros(Float64, (N,K))
  ϕnloutsum     = zeros(Float64, (N,K))
  ϕlinsum       = zeros(Float64, (N,K))
  ϕnlinsum      = zeros(Float64, (N,K))
  elborecord    = Vector{Float64}()
  est_θ         = zeros(Float64,(N,K))
  est_β         = zeros(Float64,K)
  est_μ         = zeros(Float64,K)
  est_Λ         = zeros(Float64,(K,K))
  model = new(K, N, elbo, oldelbo, μ, μ_var,μ_var_old, m0, m,m_old, M0, M,M_old, Λ, Λ_var,Λ_var_old, l0, L0, l,
   L,L_old, ϕlinoutsum, ϕnlinoutsum, η0, η1, b0,b0_old, b1,b1_old, network, mbsize, mbids,nho,  ho_dyaddict,ho_linkdict,    ho_nlinkdict,train_out,train_in,train_sinks,train_sources,ϕloutsum,  ϕnloutsum,  ϕlinsum,  ϕnlinsum,elborecord,est_θ, est_β, est_μ, est_Λ)
  return model
 end
end


println()
