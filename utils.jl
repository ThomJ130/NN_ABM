using Agents, Flux

function to_adj_mat(nn::Chain)
    weights = get_weights(nn)
    # bounds = (length(weights), max(map(length, weights)...))
    n_count = sum([size(w)[1] for w in weights]) + size(weights[1])[2]
    # n_outputs = size(weights[end])[1]

    adj_matrix = zeros(n_count, n_count)
    start_index = 1
    for w in weights
        # Extract dimensions of the weight matrices
        num_neurons_next, num_neurons_prev = size(w)

        # Copy weights from the weight matrix to the adjacency matrix
        adj_matrix[start_index+num_neurons_prev:start_index+num_neurons_prev+num_neurons_next-1, start_index:start_index+num_neurons_prev-1] = w

        # Update start index for the next layer
        start_index += num_neurons_prev
    end

    return adj_matrix
end

function from_adj_mat(adj_matrix, shape::Vector{Int})
    layer = 1
    layer_i = 1

    start_index = 1
    weight_matrices = []

    for layer in 2:length(shape)
        num_neurons_prev = shape[layer-1]
        num_neurons_next = shape[layer]

        push!(
            weight_matrices,
            adj_matrix[start_index+num_neurons_prev:start_index+num_neurons_prev+num_neurons_next-1, start_index:start_index+num_neurons_prev-1]
        )

        start_index += num_neurons_prev
    end

    return weight_matrices
end

function get_shape(nn::Chain)::Vector{Int}
    weights = get_weights(nn)

    n_inputs = size(weights[1])[2]

    return [n_inputs, [size(w)[1] for w in weights]...]
end


function get_weights(nn::Chain)
    return [Array(l.weight) for l in nn.layers if hasproperty(l, :weight)]
end

function extract_connections(adjacency_matrix)
    outflows::Vector{Int} = []
    inflows::Vector{Int} = []
    weights::Vector{Float64} = []
    n = size(adjacency_matrix, 1)  # Assuming the adjacency matrix is square

    for i in 1:n
        for j in 1:n
            if adjacency_matrix[i, j] != 0
                # println("adj_mat[$i, $j]: ", adjacency_matrix[i, j])
                push!(outflows, i)
                push!(inflows, j)
                push!(weights, adjacency_matrix[i, j])
            end
        end
    end

    return outflows, inflows, weights
end

function get_initial_positions(nn_shape::Vector{Int}; r::Float64=1.0)
    pos = []
    for (i, width) in enumerate(nn_shape)
        for j in 1:width
            push!(
                pos,
                (i * r, j * r)
            )
        end
    end
    return pos
end

function calculate_degree(adj_matrix)
    return [length([w for w in c if w != 0]) for c in eachcol(adj_matrix)]
end