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
    m_hist       ::Vector{Vector{Float64}}
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
    L_hist       ::Vector{Matrix2d{Float64}}
    L_old        ::Matrix2d{Float64}  #variational scale diagonal
    ϕlsum   ::Matrix2d{Float64}    #sum of phi products for links
    ϕnlinoutsum  ::Vector{Float64}    #sum of phi products for nonlinks
    η0           ::Float64            #hyperprior on beta
    η1           ::Float64            #hyperprior on beta
    b0           ::Vector{Float64}    #variational param for beta
    b0_hist      ::Vector{Vector{Float64}}
    b0_old       ::Vector{Float64}    #variational param for beta
    b1           ::Vector{Float64}    #variational param for beta
    b1_hist      ::Vector{Vector{Float64}}
    b1_old       ::Vector{Float64}    #variational param for beta
    network      ::Network{Int64}     #sparse view network
    mbsize       ::Int64              #minibatch size
    mbids        ::Vector{Int64}      #minibatch node ids
    nho          ::Float64            # no. of validation links
    ho_dyads     ::Vector{Dyad}       #holdout dyad vector
    ho_links     ::Vector{Link}       #holdout link vector
    ho_nlinks    ::Vector{NonLink}    #holdout nonlink vector
    ho_not     ::Vector{Int64}
    ho_fadj     ::Vector{Int64}
    trainnot   ::Vector{Int64}
    train_deg    ::Vector{Int64}      #outdeg of train and mb
    train_fadj  ::VectorList{Int64}  #sinks of train and mb
    ϕnloutsum    ::Matrix2d{Float64}
    ϕnlinsum     ::Matrix2d{Float64}
    ϕbar         ::Matrix2d{Float64}
    elborecord   ::Vector{Float64}
    est_θ        ::Matrix2d{Float64}
    est_β        ::Vector{Float64}
    est_μ        ::Vector{Float64}
    est_Λ        ::Matrix2d{Float64}
    visit_count  ::Vector{Int64}
    nl_partition ::Dict{Int64,Array{Array{Int64,1},1}}
    train_nonlinks :: Vector{NonLink}
    d            ::Array{Int64,2}
    Elogβ0       ::Vector{Float64}
    Elogβ1       ::Vector{Float64}
    mb_zeroer    ::MiniBatch
    link_set     ::Array{Array{Link,1},1}
    nonlink_setmap::Array{Array{Array{Int64,1},1},1}
    node_tnmap   ::Array{Array{Int64,1},1}
    fmap         ::Matrix2d{Int64}
    comm         ::VectorList{Int64}
    train_link_map:: Map{Int64, Set{Int64}}
    ho_map        :: Map{Dyad, Bool}
    test_map      :: Map{Dyad, Bool}
    minibatch_set :: Set{Dyad}
    linked_edges  :: Set{Dyad}
    num_peices    :: Int64
    sortedK       :: Array{Array{Int64,1},1}
    Active        :: Array{Array{Int64,1},1}
    Candidate     :: Array{Array{Int64,1},1}
    Bulk          :: Array{Array{Int64,1},1}
    stopAt        ::Array{Int64,1}

 function LNMMSB(network::Network{Int64}, K::Int64, minibatchsize::Int64)
  N             = size(network,1) #setting size of nodes
  elbo          =0.0 #init ELBO at zero
  oldelbo       =-Inf #init new ELBO at zero
  μ             =zeros(Float64,K) #zero the mu vector
  μ_var         =zeros(Float64, (N,K)) #zero the mu_var vector
  μ_var_old     = deepcopy(μ_var)
  m0            =zeros(Float64,K) #zero m0 vector
  m             =zeros(Float64,K) #zero m vector
  m_hist        =Vector{Vector{Float64}}()
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
  L_hist        = Vector{Matrix2d{Float64}}()
  L_old         = deepcopy(L)
  ϕlsum         = zeros(Float64, (N,K)) #zero the phi link product sum
  ϕnlinoutsum   = zeros(Float64,K) #zero the phi nonlink product sum
  η0            = true_eta0 #one the beta param
  η1            = 1.1 #one the beta param
  b0            = η0.*ones(Float64, K) #one the beta variational param
  b0_hist       = Vector{Vector{Float64}}()
  b0_old        = deepcopy(b0)
  b1            = η1*ones(Float64, K) #one the beta variational param
  b1_hist       = Vector{Vector{Float64}}()
  b1_old        = deepcopy(b1)
  mbsize        = minibatchsize#125#div(N,200)>1?div(N,200):div(N,3)#round(Int64, .05*N) #number of nodes in the minibatch
  mbids         = zeros(Int64,mbsize) # to be extended
  nho           = 0.01*(N*(N-1)) #init nho
  ho_dyads      = Vector{Dyad}()
 	ho_links      = Vector{Link}()
 	ho_nlinks     = Vector{NonLink}()
  ho_fadj       = zeros(Int64,N)
  ho_not        = zeros(Int64,N)
  trainnot      = zeros(Int64,N)
  train_deg     = zeros(Int64, N)
  train_fadj    = VectorList{Int64}(N)
  ϕnloutsum     = zeros(Float64, (N,K))
  ϕnlinsum      = zeros(Float64, (N,K))
  ϕbar          = zeros(Float64, (N,K))
  elborecord    = Vector{Float64}()
  est_θ         = zeros(Float64,(N,K))
  est_β         = zeros(Float64,K)
  est_μ         = zeros(Float64,K)
  est_Λ         = zeros(Float64,(K,K))
  visit_count   = zeros(Int64, N)
  nl_partition  = deepcopy(Dict{Int64, VectorList{Int64}}())
  train_nonlinks= Vector{NonLink}()
  d             = Matrix2d{Int64}(0,0)
  Elogβ0        =zeros(Float64, K)
  Elogβ1        =zeros(Float64, K)
  mb_zeroer     =MiniBatch()
  link_set      =Array{Array{Link,1},1}()
  nonlink_setmap=Array{Array{Array{Int64,1},1},1}()
  node_tnmap    =Array{Array{Int64,1},1}()
  fmap          = zeros(Float64, (N,K))
  comm          =[Int64[] for i in 1:K]
  train_link_map= Map{Int64, Set{Int64}}()
  ho_map        = Map{Dyad, Bool}()
  test_map      = Map{Dyad, Bool}()
  minibatch_set = Set{Dyad}()
  linked_edges  = Set{Dyad}()
  num_peices    = 10
  sortedK       = [sortperm(est_θ[i,:],rev=true) for  i in 1:N]
  Active        = [Int64[] for a in 1:N]
  Candidate     = [Int64[] for a in 1:N]
  Bulk          = [Int64[] for a in 1:N]
  stopAt        = ones(Int64, N)

  model = new(K, N, elbo, oldelbo, μ, μ_var,μ_var_old, m0, m,m_hist,m_old, M0, M,M_old, Λ, Λ_var,Λ_var_old, l0, L0, l,
   L,L_hist,L_old, ϕlsum, ϕnlinoutsum, η0, η1, b0,b0_hist,b0_old, b1,b1_hist,b1_old, network, mbsize, mbids,nho, ho_dyads, ho_links,
    ho_nlinks,ho_fadj,ho_not,trainnot,train_deg,train_fadj,
     ϕnloutsum,  ϕnlinsum,ϕbar,elborecord,est_θ, est_β, est_μ, est_Λ,visit_count,nl_partition,
     train_nonlinks,d,Elogβ0,Elogβ1,mb_zeroer,link_set,nonlink_setmap, node_tnmap, fmap,comm,
     train_link_map,ho_map,test_map,minibatch_set,linked_edges,num_peices,sortedK, Active,Candidate,Bulk,stopAt)
  return model
 end
end


print();
