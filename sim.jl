
using LinearAlgebra
using Distributions
using Random

isinvertible(x) = issuccess(lu(x, check=false))

# test
isinvertible([1 0 ;0 1])
isinvertible([1 0 0  ; 0   1 0 ; 0 0 0])

# set random seed
Random.seed!(1234)

# model parameters

nbr_obs = 50
nbr_covaraites = 3

# gound-truth parameters

β_true = [1.2;1.3;4.2;2.5]
σ_true = 0.5

# set desing matrix (for non-transformed model)

X_tilde = zeros(nbr_obs, nbr_covaraites+1)

X_tilde[:,1] = ones(nbr_obs)
X_tilde[:,2] = 5*rand(nbr_obs)
X_tilde[:,3] = 10*rand(nbr_obs)
X_tilde[:,4] = 2*rand(nbr_obs)

# generate V matrix (https://math.stackexchange.com/questions/357980/how-to-generate-random-symmetric-positive-definite-matrices-using-matlab)

V = rand(nbr_obs, nbr_obs)
V = 0.5*(V+V') + nbr_obs*Matrix{Float64}(I, nbr_obs, nbr_obs)

isposdef(V) # check V matrix

isinvertible(V) # check that V is invertable

ϵ_tilde = rand(MvNormal(σ_true^2*V)) # generate epsilons

# Generate data from model

Y_tilde = X_tilde*β_true + ϵ_tilde


# transform model

P = cholesky(V).L

sum(diag(P) .== 0) # check that non of the diagonal elements are zero

X = P*X_tilde
Y = P*Y_tilde
ϵ = P*ϵ_tilde

# Set constraints for LS estimation

nbr_rows_R = 5

R = rand(nbr_rows_R, nbr_covaraites + 1)

# use this to reduce the rank of nbr_rows_R
R[3,:] = R[1,:]
R[4,:] = R[1,:]
R[5,:] = R[1,:]

rank(R)

# compute restricted LS estimators

isinvertible(X'*X)
isinvertible(R*inv(X'*X)*R')

β_gls = inv(X'*X)*X'*Y
λ = inv(R*inv(X'*X)*R')*R*β_gls # we need to use pinv if R*inv(X'*X)*R' is not invertable
β_hat = β_gls - inv((X'*X))*R'*λ

println("Estimated parameters (β_gls):")
println(β_gls)

println("Estimated parameters (β_hat):")
println(β_hat)

println("Grund-truth parameters:")
println(β_true)


# check restrictions

println("Restrictions (R*β_hat):")
println(R*β_hat)

# estimate sigma
