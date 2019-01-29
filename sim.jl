using LinearAlgebra
using Distributions

# set random seed
Random.seed!(1234)

# model parameters

nbr_obs = 50
nbr_covaraites = 3

# gound-truth parameters

β_true = [1.2;1.3;4.2;2.5]
σ_true = 1.5

# set desing matrix

X = zeros(nbr_obs, nbr_covaraites+1)

X[:,1] = ones(nbr_obs)
X[:,2] = 5*rand(nbr_obs)
X[:,3] = 10*rand(nbr_obs)
X[:,4] = 2*rand(nbr_obs)

# generate V matrix (https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab)

V = rand(nbr_obs, nbr_obs)
V = 0.5*(V+V') + nbr_obs*Matrix{Float64}(I, nbr_obs, nbr_obs)

isposdef(V) # check V matrix

ϵ = rand(MvNormal(σ_true^2*V)) # generate epsilons

# Generate data from model

# we check easiest case when V = I

Y = X*β_true + ϵ

# Set constraints for LS estimation

# nbr rows of R > nbr rows of nbr_covaraites + 1

R = rand(7, nbr_covaraites + 1)

rank(R)

# nbr rows of R = nbr rows of nbr_covaraites + 1

R = rand(nbr_covaraites + 1, nbr_covaraites + 1)

rank(R)

# nbr rows of R < nbr rows of nbr_covaraites + 1

R = rand(3, nbr_covaraites + 1)

rank(R)

# compute restricted LS estimators

β_gls = inv(X'*X)*X'*Y
λ = inv(R*(X'*X)*R')*R*β_gls
β_hat = β_gls - inv((X'*X))*R'*λ

println("Estimated parameters:")
println(β_hat)


println("Grund-truth parameters:")
println(β_true)

# estimate sigma
