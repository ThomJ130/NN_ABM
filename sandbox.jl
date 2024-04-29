using Graphs, GraphMakie, GLMakie
using GraphMakie.NetworkLayout, SimpleWeightedGraphs
using Flux, CUDA, Statistics, ProgressMeter
using Agents
using LinearAlgebra, Observables
using DataFrames, CSV

include("forces.jl")
include("structs.jl")
include("utils.jl")
include("setup.jl")

NEURON_RADIUS = 0.1



function training_step!(nn::Chain, optim, x, y)
    # Takes a step in training and returns the loss
    loss, grads = Flux.withgradient(nn) do m
        # Evaluate nn and loss inside gradient context:
        y_hat = m(x)
        Flux.crossentropy(y_hat, y)
    end
    Flux.update!(optim, nn, grads[1])
    return loss
end


function update_connections!(
    agent::Neuron,
    model::StandardABM;
    prune_threshold=0.65
)
    model.adj_matrix = ((x) -> abs(x) < prune_threshold ? 0 : x).(model.adj_matrix)
end



function update_position!(
    agent::Neuron,
    model::StandardABM;
    max_vel::Float64=0.2
)
    agent.pos += calculate_position_delta(agent, model)
end

function apply_training!(model::StandardABM)
    ## Update agents based on trained weights from NN
    for (i, weights) in enumerate(eachcol(model.adj_matrix))
        neuron = model[i]
        neuron.delta_w = isnothing(neuron.prev_t_weights) ? nothing : weights - neuron.prev_t_weights
        neuron.prev_t_weights = weights
        neuron.weights = weights
    end
end

function update_nn!(model::StandardABM)
    layers = [l for l ∈ model.nn if hasproperty(l, :weight)]

    new_matrices = from_adj_mat(model.adj_matrix, get_shape(model.nn))

    for (l, w) in zip(layers, new_matrices)
        # print("Size p: ", size(p), " Size w: ", size(w))
        l.weight .= gpu(Float32.(w)) #convert(Matrix{Float32}, w)
        # Flux.params!(p, convert(CuArray{Float32}, w))
    end
end

function model_step!(model::StandardABM) # ?
    nn = model.nn
    optim = model.optim
    loader = model.loader

    if model.batch_iter > model.iter_per_batch || model.batch_iter == 0
        # update agents with next network weights
        ## Execute iteration of training
        (x, y), _ = iterate(loader)
        model.loss = training_step!(nn, optim, x, y)
        update_nn!(model)

        model.adj_matrix = to_adj_mat(nn)
        apply_training!(model)

        ## Reset batch iteration count
        model.batch_iter = 1
    end

    # for n in allagents(model)
    neurons = allagents(model)
    update_connections!.(neurons, [model])
    update_position!.(neurons, [model])
    # end

    model.batch_iter += 1
    model.step += 1
end

function history_ui(adf::DataFrame, mdf::DataFrame, n::Int; timescale=2)
    fig = Figure()
    ax = Axis(fig[1, 1])

    positions = adf[adf.time.==1, :pos]
    adj_matrix = mdf[mdf.time.==1, :adj_matrix][1]
    outflows, inflows, weights = extract_connections(adj_matrix)

    g = SimpleDiGraph(transpose(adj_matrix))

    p = graphplot!(
        ax,
        g;
        # layout=linearLayout([2, 3, 2]; pos=positions),
        node_size=40,
        edge_width=[abs(w) for w in weights],
        edge_color=[w >= 0 ? "blue" : "red" for w in weights]
    )

    display(fig)
    # print("Press enter to coninue ")
    # readline()

    for i in 1:n
        positions = adf[adf.time.==i, :pos]
        adj_matrix = mdf[mdf.time.==i, :adj_matrix][1]
        outflows, inflows, weights = extract_connections(transpose(adj_matrix))

        p[:node_pos][] = Point{2,Float32}.(positions)
        p[:edge_width][] = [abs(w) for w in weights]
        p[:edge_color][] = [w >= 0 ? "blue" : "red" for w in weights]

        # outflows, inflows, weights = extract_connections(adj_matrix)
        # edge_widths[] = [abs(w) for w in weights]
        # edge_colors[] = [(w > 0 ? "blue" : "red") for w in weights]

        # g[] = SimpleDiGraph(adj_matrix)

        display(fig)
        sleep(1 / timescale)
    end


end

function run_test(num_layers::Int, width::Int, loader, k::Int; index=1)
    nn, optim = setup_nn(num_layers, width)

    model = init_model(
        nn,
        optim,
        loader;
        iter_per_batch=1
    )

    adf, mdf = run!(
        model, k;
        mdata=[:adj_matrix, :loss],
        adata=[:pos]
    )

    data = Dict(
        :time => [],
        Dict(
            Symbol("n$i") => []
            for i in 1:size(mdf[1, :adj_matrix])[1]
        )...,
        :loss => [],
    )
    # data = []


    for (i, adj_matrix, loss) in eachrow(mdf)
        # row::Vector{Any} = [i, calculate_degree(adj_matrix)..., loss]
        # println("adj_mat $adj_matrix")
        # push!(row, )
        push!(data[:time], i)
        push!(data[:loss], loss)
        for (j, deg) in enumerate(calculate_degree(adj_matrix))
            push!(data[Symbol("n$j")], deg)
        end
    end

    df = DataFrame(data)

    CSV.write("results/raw/$(num_layers)_$(width)_$(k)-model_data-$(index).csv", mdf)
    CSV.write("results/raw/$(num_layers)_$(width)_$(k)-agent_data-$(index).csv", adf)
    CSV.write("results/$(num_layers)_$(width)_$(k)-results-$(index).csv", df)
    # history_ui(adf, mdf, k; timescale=1000)
end

function run_all_experiments()
    experiments = Iterators.product([1, 2, 3], [3, 5, 7])


    for i in 1:20

        loader, noisy, truth, target = get_data()

        for (num_layers, width) in experiments
            loader = Flux.DataLoader((noisy, target) |> gpu, batchsize=64, shuffle=true)
            run_test(num_layers, width, loader, length(target) * 10 - 1; index=i)
        end
    end
end

# nn, optim = setup_nn(1, 3)

# model = init_model(
#     nn,
#     optim,
#     loader;
#     iter_per_batch=1
# )

# run_test(1, 3, )
run_all_experiments()


# # g = SimpleWeightedDiGraph(transpose(adj_matrix * 10))
# sources, dests, weights = extract_connections(transpose(adj_matrix))
# g = SimpleWeightedDiGraph(sources, dests, weights)
# # graphplot(g; layout=linearLayout([2, 3, 2]), node_size=40, edge_width=[0.1 * i for i in 1:lengthgth(weights)])
# graphplot(g; layout=Stress(; dim=2), node_size=40