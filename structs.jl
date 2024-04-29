using Agents, LinearAlgebra
include("utils.jl")


Optional{T} = Union{Nothing,T}

@agent struct Neuron(ContinuousAgent{2,Float64})
    layer::Int
    weights::Vector{Float64}
    prev_t_weights::Optional{Vector{Float64}}
    delta_w::Optional{Vector{Float64}} # adjacency
    neighbors::Vector{Int} # adjacency indces
end

mutable struct Properties
    step::Int64
    batch_iter::Int64
    iter_per_batch::Int64
    adj_matrix::Matrix
    optim
    nn::Chain
    loader::Flux.DataLoader
    loss::Float64
    Properties(iter_per_batch, nn, optim, loader) = new(0, 0, iter_per_batch, to_adj_mat(nn), optim, nn, loader, 0)
end