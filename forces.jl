using Agents, LinearAlgebra
include("structs.jl")

function force_to_vel(force)
    return [0, force[2] / 10]
end

function calculate_natural_force(
    neuron::Neuron,
    model::StandardABM,
    max_force,
    n_radius;
    cutoff=0.25
)
    force = zeros(2)

    # here we apply force-based position update rules.
    for nid in neuron.neighbors
        other = model[nid]
        s = neuron.pos - other.pos
        d = norm(s)

        # magnitude = 0
        ## Removed the attracting force due to excessive 
        ## jitter.
        magnitude = 1 / 25 * max_force * (1 - (d / n_radius)^2)
        # if d <= n_radius
        ## repelling force
        # else
        #     ## attracting force
        #     magnitude = -1 / 10 * min(1 / d^2, max_force)
        # end

        force += magnitude >= cutoff ? s / d * magnitude : zeros(2)
    end

    return force
end

function calculate_weight_force(
    neuron::Neuron,
    max_force,
    model::StandardABM;
    τ=0.01,
    scale=1
)
    force = zeros(2)

    for (nid, w) in enumerate(neuron.weights)
        other = model[nid]
        if other.layer != neuron.layer + 1
            continue
        end
        s = other.pos - neuron.pos
        d = norm(s)

        w = abs(w)
        # ‖f‖ = attracting when beyond τ
        # ‖f‖ = 0 at the τ
        # ‖f‖ = repelling when below τ
        magnitude = scale * (1 - 2 * τ / (w + τ))

        if magnitude != 0
            magnitude = magnitude / abs(magnitude) * min(abs(magnitude), max_force)
        end
        force += s / d * magnitude
    end

    return force
end

function calculate_net_force(
    neuron::Neuron,
    model::StandardABM,
    n_radius::Float64,
    max_force::Float64
)
    force = calculate_natural_force(
        neuron,
        model,
        max_force,
        n_radius
    )

    # cohesion forces
    force += calculate_weight_force(
        neuron, max_force, model;
        scale=2
    )

    return force
end

function calculate_position_delta(
    neuron::Neuron,
    model::StandardABM;
    n_radius::Float64=2.0,
    max_force::Float64=7.0
)
    return force_to_vel(
        calculate_net_force(neuron, model, n_radius, max_force)
    )
end