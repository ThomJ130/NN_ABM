using Flux, LinearAlgebra, Agents
include("structs.jl")
include("utils.jl")

function setup_nn(num_layers::Int, width::Int)
    hidden_layers = []
    for i in 1:(num_layers-1)
        push!(hidden_layers, Dense(width => width, tanh))
        # push!(hidden_layers, BatchNorm(width))
    end

    nn = Chain(
        Dense(2 => width, tanh),   # activation function inside layer
        hidden_layers...,
        BatchNorm(width),
        Dense(width => 2),
        softmax
    ) |> gpu

    optim = Flux.setup(Flux.Adam(0.01), nn)  # will store optimiser momentum, etc.

    return nn, optim
end

function get_data()
    # We are going to constrain this analysis to traditional architectures.
    # We are doing XOR function
    noisy = rand(Float32, 2, 1000)                                    # 2×1000 Matrix{Float32}
    truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)]   # 1000-element Vector{Bool}
    target = Flux.onehotbatch(truth, [true, false])                   # 2×1000  OneHotMatrix

    # The network encapsulates parameters, randomly initialised. Its initial output is:
    # out1 = network(noisy |> gpu) |> cpu                                 # 2×1000 Matrix{Float32}

    loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true)
    # 16-element DataLoader with first element: (2×64 Matrix{Float32}, 2×64
    # OneHotMatrix)

    return loader, noisy, truth, target
end

function init_model(
    nn::Chain,
    optim,
    loader::Flux.DataLoader;
    iter_per_batch::Int=10,
    initial_r::Float64=1.0
)
    props = Properties(
        iter_per_batch,
        nn,
        optim,
        loader
    )

    weights = [Array(l.weight) for l in nn.layers if hasproperty(l, :weight)]
    nn_shape = get_shape(nn)
    bounds = (1 + length(weights), 1 + max(map(length, weights)...))
    space = ContinuousSpace(bounds; spacing=1, periodic=false)
    model = StandardABM(Neuron, space; properties=props, (model_step!)=model_step!)

    init_pos = get_initial_positions(nn_shape)
    l_index = 1
    l_n_index = 1
    neighbor_i = 1

    adj_mat = to_adj_mat(nn)
    for (n_indx, weights, pos) in zip(1:length(adj_mat), eachcol(adj_mat), init_pos)
        if l_n_index > nn_shape[l_index]
            neighbor_i = n_indx
            l_n_index = 1
            l_index += 1
        end

        neighbor_ids = [
            i
            for i in neighbor_i:(neighbor_i+nn_shape[l_index]-1)
            if i != n_indx
        ]

        add_agent!(
            pos,
            Neuron,
            model,
            layer=l_index,
            weights=Array(weights),
            prev_t_weights=nothing,
            delta_w=nothing,
            neighbors=neighbor_ids,
            vel=(0, 0) # not used
        )
        l_n_index += 1
    end

    return model
end